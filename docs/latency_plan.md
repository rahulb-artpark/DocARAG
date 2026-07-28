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

## Phase 0 -- measure (mandatory, no behaviour change)

Add duration accumulators beside the existing `EntityResolutionStats` counts:
embed, vector search, rerank, LLM select, LLM merge. Same for the tree stage's
phases and for query retrieval. Log with the per-batch snapshot already in place.

Deliverable: seconds per operation per stage. Every later phase cites it.

Risk: none. Without it, Phases 3 and 5 are guesses.

## Phase 1 -- output-identical wins

1. **Rerank batching.** `TextRerankerProvider.rerank(batch_size=4)`; entity
   resolution passes up to 10 candidates, so each rerank runs 3 forward passes
   where 1 would do. Raise the batch to cover the candidate count. Scores are
   per-document, so batching is pure chunking -- assert scores equal the
   unbatched result and the change is provably free.
2. **Query-path thinking.** `indexing_llm_config` disabled model thinking for
   tree and graph only; the query path still runs at the model's default. The
   measured effect on indexing was 114.3s to 8.7s on the same prompt. Untested
   on query schemas, and `parse_structured` already recovers a fenced answer.
   Largest single untouched win in the repository.

Gate: rerank scores identical; query answers spot-checked against the current
path on a fixed question set.

## Phase 2 -- G4, batch entity adjudications (graph)

14,561 model calls for 15,481 entities, and `multi=7,970` means adjudication is
genuinely needed rather than avoidable. Batch several independent decisions per
call.

Safe because entities sharing **no candidates** cannot influence each other:
Algorithm 1's sequential dependency is that one commit changes what the next
entity retrieves. Helpfully, a node's own entities are already independent --
`advanced_kg_refiner` defers `add_entities_to_vdb` until after its loop, so they
never see each other. Disjointness only has to guard merge side-effects on
shared existing candidates.

Gate: ER-decision diff identical on both fixtures. Watch the response-length
mismatch rate, which is the failure mode the tree mergers already guard against.

## Phase 3 -- named vectors + shared dedup (graph and query)

**Only if Phase 0 shows rerank is expensive.** Entity search embeds the node
name alone; descriptions sit in the payload unembedded, so the reranker carries
all semantic work. Design and rationale: `entity_resolution_retrieval.md`.

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

Phase 1's thinking fix and rerank batching apply directly. Phase 3's dedup
protects PageRank and Skyline from double-weighting. Re-measure query latency
after each, on the fixed question set, and keep answers diffed against current
output so a latency win is never taken on quality.

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

Phase 0, then 1 (free), then 2 (largest reduction, no migration), then 4 (tree,
independent of the graph work), then 3 only if Phase 0 justifies the migration,
then 5. G8 -- parallelising resolution's read-only prep -- stays last: it depends
on Phase 0's premise, and the reranker is one local model on `cuda:0`, so
concurrent calls serialise on the GIL and the CUDA stream regardless. Embedding
and search would overlap; rerank likely will not.
