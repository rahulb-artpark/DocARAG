# Latency plan: graph, tree, and query retrieval

Supersedes the ordering in `scaling_plan.md` Phase 2/3. Baselines:
`scaling_baseline.md` and the completed 1,871-page run (16,092 tree nodes,
25,324 entities, 15,481 resolutions, 11,201 reranks, 14,561 model calls).

## Rule this plan follows

Measure before building. Four hypotheses about the graph stage were disproved by
data this cycle (unindexed Qdrant filter, descriptions growing on merge, the
reranker evicting the model, unbounded prompts), and the fix that actually
delivered ~7x was a one-line configuration change. Anything here costing more
than an afternoon is gated on a measurement that justifies it.

`rerank=11201 (85.4%)` counts *how often* a rerank happens, not time spent in
it. Rerank duration has never been measured. Phase 0 exists because of that.

## Phase 0 -- measure (done)

`PhaseTimings` accumulates seconds and calls per operation across entity
resolution, the tree stage, and query retrieval (commit 242afeb).

Results: `latency_measurements.md`. They killed both Phase 1 items and
re-pointed Phase 3, which is what this phase was for.

The headline: the graph stage is **entity resolution's model calls and nothing
else** -- 9,005 of them at ~2.9 s. Rerank is **1.6%** of its 434.6 minutes, not
the dominant cost the 85.4% figure was read as.

## Phase 1 -- withdrawn, both items

1. **Rerank batching -- rejected.** Measured 1.31x, not 3x, because the real ER
   documents are median 177 chars rather than near the 10,000-char truncation.
   Worth 2.6 minutes of a 7-hour stage, and not output-identical: against a
   bit-for-bit deterministic control, changing the batch reordered candidates in
   **21 of 40** calls, which changes merge decisions.
2. **Query-path thinking -- large but not free.** 23.7x on entity extraction and
   6.5x on section selection, so the latency claim held. But two of six
   comparisons regressed: an entity was dropped, and one selection returned
   `select_id=-1` where thinking chose a section. Unlike indexing -- where
   thinking consumed the budget and the model emitted nothing -- the reasoning
   here earns its cost. Revisit per call site against a real question set, not
   as a blanket switch.

## Phase 2 -- G4, batch entity adjudications (graph)

**This is the graph stage, not a step in it.** The completed run made 9,005
model calls at ~2.9 s each -- `select=5700`, `merge=3305` -- and reranking, the
previous suspect, is 1.6%. Both call sites are worth batching; selection is the
larger half. `multi=5044` means adjudication is genuinely needed rather than
avoidable.

Note also `direct=9738`: more resolutions already end in a model-free direct
merge than in any adjudicated outcome. Converting an adjudication into a direct
merge is worth more than making one faster.

Safe because entities sharing **no candidates** cannot influence each other:
Algorithm 1's sequential dependency is that one commit changes what the next
entity retrieves. Helpfully, a node's own entities are already independent --
`advanced_kg_refiner` defers `add_entities_to_vdb` until after its loop, so they
never see each other. Disjointness only has to guard merge side-effects on
shared existing candidates.

Gate: ER-decision diff identical on both fixtures. Watch the response-length
mismatch rate, which is the failure mode the tree mergers already guard against.

## Phase 3 -- named vectors + shared dedup (graph and query)

**Same design, different premise.** Phase 0 showed rerank is 2.5% of the graph
stage, so "make reranking cheaper" no longer justifies a migration. Better
candidates are still worth having, for the other reason in the design: they cut
**adjudications**, which are the 97%. Measure this one in model calls saved, not
rerank seconds. Entity search embeds the node name alone; descriptions sit in
the payload unembedded. Design and rationale: `entity_resolution_retrieval.md`.

Three things must ship together:

1. **Recreate the collection.** Qdrant cannot convert an unnamed vector to named
   vectors in place. Bump `graph_artifact_variant` so old and new indexes cannot
   be confused, and reindex.
2. **Every consumer passes `using=`.** An unqualified search fails once vectors
   are named; entity resolution and the query path change in one commit.
3. **One shared dedup helper, used by both paths.** A merged entity's facets
   cluster, so one entity can fill several search slots. In resolution that
   fakes a smooth gradient where 4.3.2 expects a sharp drop, so `case_a` fires
   and the entity is added instead of merged. On the query path it is worse:
   `Graph_Reasoning` runs PageRank and maps scores through the GT-Link matrix, so
   a duplicated entity is double-weighted and `Skyline_Ranker` sees inflated
   dimensions. Over-fetch, resolve to canonical via `entity_alias_map`, reduce to
   one row per entity by **max** score, then proceed.

Gate: Figure 6 metrics (entities, density, diameter, components) within
tolerance, ER-decision diff, and retrieval recall on the fixed question set.

## Phase 4 -- tree

**The larger target: 851.3 minutes, nearly twice the graph stage, and still
unbroken-down.** Phase 0's instrumentation covers MinerU parsing, content merge,
canonical binding, layout recheck, outline, construction and summaries; one run
turns the list below from four guesses into a ranking.

1. **Merger packing.** `MAX_PAIRS_PER_JUDGMENT = 3` caps a batch at three pairs
   regardless of remaining token budget, fixing 2,022 candidates into ~674
   calls. Filling the budget is roughly a 3x call reduction. Raise it while
   watching the mismatch rate.
2. **Sparse outline recheck.** Skip the model for titles the layout already
   classifies unambiguously (explicit numbering, the single height-max first-page
   title). Gate: deterministic assignments must agree with the model on the
   fixtures before the skip is trusted.
3. **Outline batch parallelism.** Batches are a sequential fold today -- each
   reads the accumulated outline. Feed every batch a precomputed level-0/level-1
   skeleton instead so they are independent, with deterministic parent repair in
   one post-pass. Gate: structural diff within reviewed tolerance.
4. **Summary depth.** Summaries are level-sequential and the old flat tree was
   the maximally parallel case, so the corrected hierarchy may have cost time
   here. Measure before and after; if it hurts, overlap independent subtrees --
   a node needs only its own children, not its whole level.

## Phase 5 -- query retrieval

Thinking is worth 23.7x on entity extraction and 6.5x on section selection, and
that is far too large to leave alone -- but the probe regressed two of six
comparisons, so it has to be resolved per call site against a real question set.
Phase 3's dedup protects PageRank and Skyline from double-weighting. Keep
answers diffed against current output so a latency win is never taken on
quality.

## What keeps this safe

- **G1 snapshots** mean a failed experiment on a large document costs one batch
  of work, not the run. That is what makes iterating at scale affordable now.
- **Every quality-affecting change is gated** on the ER-decision diff and the
  paper's own Figure 6 metrics, with baselines at both 28 pages and 1,871.
- **New behaviour arrives behind a value that defaults to today's**, the way
  `MAX_PAIRS_PER_JUDGMENT` and `indexing_llm_config` already do, so a phase can
  be measured and reverted independently.
- **Migrations bump the artifact variant** rather than mutating an index in
  place, so a half-migrated collection cannot be read as current.

## Order

Revised by Phase 0's results. Phase 1 is withdrawn, so:

**2** (the graph stage's only real lever, no migration), then **4** (the larger
stage, once one instrumented run ranks its phases), then **5**'s per-call-site
thinking question, then **3** only if Phase 2 leaves adjudications high enough
to justify the migration.

G8 -- parallelising resolution's read-only prep -- is now clearly not worth it.
It targeted embedding, search and rerank, which together are a small share of a
model-bound stage; and under `config/ollama-gbc.yaml` the reranker is one local
0.6B model on `cuda:0`, so concurrent calls serialise on the GIL and the CUDA
stream anyway. (Under the vLLM-backed configs the reranker is an HTTP service
and would overlap -- but there it is even less of the cost.)
