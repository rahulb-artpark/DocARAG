# DocARAG Index Scaling Plan — Tree + Graph for Large Documents (BookRAG-based)

## Implementation status

Landed (each committed separately, full suite green at 1,274 tests):

| Item | Commit | Verified by |
|---|---|---|
| T1 outline `think_mode` | `2cc4677` | real facade: 114.3s exhausted → 8.7s, max level 1 → 4 |
| G2 merge atomicity + fallback | `4c5ace7` | 3 tests fail pre-fix with the production error |
| G1 resumable refinement | `f669a09` | snapshot/restore + resume-skip tests |
| T3 merger concurrency | `fd82809` | order preserved when completion order reverses |
| G9 per-entity GPU flush removed | `e3af77e` | source guard; provider self-manages on a threshold |
| G7 extraction/refinement overlap | `7a6afc2` | deterministic overlap test via an event |
| Phase 0 ER counters | `d370981` | per-outcome counts, logged against graph size |
| T2 packing cap named | `593a575` | value unchanged; lever made explicit |

**Blocked on a fixture, not on design.** Every remaining item (G3, G4, G5, G6,
G8, T4, T5, T6, Phase 6, Phase 7) changes merge decisions, tree structure, or
needs a measured baseline — and this plan's own gates require proving that
change is not a regression. That needs the golden + ~500pp scale fixtures from
open question 1. Landing them without that would trade the quality guarantee
for speed, which is the one thing the plan forbids.

**Status:** v2 replanned; items above implemented.
**Goal:** make tree + graph index construction complete, and complete in
industry-acceptable time, on 1000–5000 page documents — **without degrading**
tree-node, entity, or entity-resolution quality. Speed comes from durability,
call-count reduction, batching, and parallelism. Never from doing less semantic
work.

**Grounding — the real run** (`test.pdf`, 1870 pages), run dir
`runs/doc_df198aeb160bb6240f5843b978d6fbcc/ee3d1f8d…`:
- 18,312 blocks → 16,091 tree nodes (10,639 text / 3,165 title / 1,313 image /
  449 table / 525 equation).
- **Tree stage: 13h25m — completed.**
- **Graph stage: reached 672 / 16,091 nodes in 8.96h, then crashed. Never
  completed. Crashed 3+ times, each time losing all refinement work.**
- Backbone: **one** local GPU, Ollama `qwen3.6:35b`, `max_workers=4`,
  `max_in_flight=4`.

---

## 0. The framing this plan starts from: we are 44–90× beyond the paper's validated scale

BookRAG's design was validated on small documents (paper Table 4):

| Dataset | Docs | **Avg pages** | Entities (Fig. 6, basic) |
|---|---|---|---|
| MMLongBench | 85 | **42.16** | 1,327 |
| M3DocVQA | 500 | 8.52 | — |
| Qasper | 192 | 10.95 | 531 |

Our workload: **1,870 pages** (44× the paper's largest average) and a projected
**~118,700 entities** (90× the paper's largest graph).

This matters because BookRAG §4.3.2 chooses *incremental* ("clean") ER
specifically to avoid dirty-ER's O(n²) pairwise cost — an excellent trade at
1.3K entities. At 119K entities that same incremental loop becomes the dominant
cost, and — as measured in §1.1 — its **per-entity cost is not constant; it grows
as the graph fills.** Any plan that treats the paper's design as scale-neutral
will fail here. That is the core correction in this revision.

---

## 1. Measured profile (corrected)

### 1.1 Graph stage — the blocker, and it is *super-linear*

Measured from the full 8.96h graph run (`run_20260721_182104`), which committed
**672 of 16,091 nodes** (21 batches of 32) and extracted 5,180 entities /
3,731 relations across 702 checkpointed nodes (7.38 entities/node):

| ER outcome | Count | Cost per occurrence |
|---|---|---|
| distance-gate skip (`>0.2`, novel entity) | 1,679 | cheap — no rerank, no LLM |
| direct merge (same name/type) | 1,536 | cheap |
| gradient-select → **LLM ER-selection** | 1,665 | GPU rerank + **1 LLM call** |
| …of which also **LLM merge-summarize** | 968 | **+1 LLM call** |
| **Total refinement LLM calls** | **2,633** | **all serial** |

**Corrected headline numbers** (my earlier estimate of a ~10% slow path was
taken from a partial log fragment and was wrong):

- **53% of entities trigger at least one LLM call** (2,633 calls / 4,959 entities).
- **3.92 serial LLM calls per tree node.**
- These 2,633 serial calls account for essentially the whole 8.96h
  (≈12.3s/call), while extraction runs concurrently underneath and is far cheaper.
- **Projected full document: 118,700 entities → ~62,900 serial LLM calls →
  ~214 hours ≈ 8.9 days.**

#### The super-linear finding (new, and decisive)

Slow-path rate across five equal segments of that same run:

| Segment | gate-skips | direct | gradient-select | **slow-path %** |
|---|---|---|---|---|
| 1 | 927 | 278 | 361 | **23.1%** |
| 2 | 153 | 316 | 305 | **39.4%** |
| 3 | 211 | 351 | 301 | **34.9%** |
| 4 | 196 | 291 | 355 | **42.2%** |
| 5 | 192 | 300 | 343 | **41.1%** |

Cheap distance-gate skips **collapse from 927 → ~190** while the expensive path
nearly **doubles (23% → 41%)** — and this happened within the first **4%** of the
document. Mechanism: as the entity VDB fills, almost every new entity finds a
near-neighbour inside the 0.2 gate
([kg_refiner.py:480](../retrieval_engine/indexing/graph/refiner.py#L480)), so it proceeds to
rerank + LLM instead of being admitted cheaply. The slow-path share keeps rising
with document size.

**Consequence: per-entity ER cost grows with graph size, so total graph cost is
super-linear in document length.** Parallelism alone cannot fix a super-linear
curve — it shifts it down by a constant. The fix must **reduce the number of LLM
calls**, not merely run them concurrently. This is the single biggest change in
priority from v1 of this plan.

#### Why it has never finished (also new, also decisive)

**Refinement has zero durability.** `build_knowledge_graph` does not persist the
graph (the call is commented out,
[kg_builder.py:172](../retrieval_engine/indexing/graph/builder.py#L172)); the only
`graph_index.save_graph()` on the real pipeline path runs in
[construct_index.py](../retrieval_engine/indexing/pipeline.py) **after `build_knowledge_graph`
fully returns** — i.e. after all extraction *and* `refine_entities()` *and*
`refine_relation()` complete.

So: extraction is checkpointed per node (702 files survived), but **every hour of
refinement is lost on any crash.** Three crashes ⇒ three total losses of
refinement. Combined with ~63K serial LLM calls and a fail-fast
`entity_merge` ([kg_refiner.py:242](../retrieval_engine/indexing/graph/refiner.py#L242)), the
probability of ever reaching the end is effectively zero.

**Even a 10× speedup leaves a ~21h unresumable run, which will still fail.
Durability is therefore item #1, ahead of all optimization.**

### 1.2 Tree stage — completes, but 68% is needlessly serial

| Phase | Wall | % | LLM calls | Today |
|---|---|---|---|---|
| MinerU parse + canonical bind (187 shards) | 2h20m | 17% | VLM enrich | ✅ sharded |
| **text_merger** | 3h56m | 29% | ~674 | ❌ serial |
| **label_merger** | 1h24m | 10% | ~318 | ❌ serial |
| **table_merger** | 55m | 7% | 210 | ❌ serial |
| **outline extraction** | 2h54m | 22% | 282 | ❌ serial **& 100% failed → flat tree** |
| node summary | 1h55m | 14% | 6,562 | ✅ concurrent + cached |

---

## 2. Three distinct failure modes at scale

Structuring the plan around *what breaks as documents grow*, rather than what is
slow today:

| # | Failure mode | Where | Fix class |
|---|---|---|---|
| **F1** | **Cannot complete** — no refinement durability; fail-fast crash paths | graph | checkpoint + resume + atomicity |
| **F2** | **Super-linear growth** — ER slow-path rate rises with graph size (23%→41%) | graph | reduce LLM-call *count*: blocking, batching, deterministic fast paths |
| **F3** | **Serial constant factor** — independent work run one-at-a-time | tree mergers, outline fold, ER commit | batching + parallelism |
| **F4** | **Whole-document memory** — structures sized by the whole document | `pdf_list`, `tree.pkl`, outline fold | streaming / sharding |

On a **single GPU**, decode throughput is the hard floor, so **F2 (fewer calls)
outranks F3 (concurrent calls)** — a point v1 of this plan had backwards.

---

## 3. Cross-cutting: quality-preservation harness (Phase 0 — gates everything)

"Same quality" is operational, not asserted. Nothing merges until this exists:

1. **Golden documents:** one mid-size real PDF (~100–200pp) as fixture, plus the
   20-page BookRAG paper PDF for fast iteration. Never the Celestron tenant.
   **Plus a scale fixture** (~500pp) — because §1.1 proves small documents do not
   exercise the failure that matters.
2. **Tree invariants:** node count, type distribution, max depth, full parent_id
   map. Pure-reordering changes must be **byte-identical**; correctness changes
   (T1) are judged on depth + spot-check.
3. **Graph invariants — BookRAG Fig. 6 metrics:** #entities, density, diameter of
   largest CC, #connected components. Any ER change must land within tight
   tolerance of the serial baseline on all four. This is the paper's own ER
   yardstick.
4. **ER decision diff:** for the same input, the set of merge decisions
   (which entity merged into which) must match the serial baseline — the
   strongest possible check that parallelism/blocking changed nothing semantic.
5. **Retrieval recall** on a fixed ~30-question set (paper's primary metric).
6. **Per-phase token/latency ledger** — every phase reports measured before/after.

---

## Phase 0 — Instrumentation, durability harness, baseline (no behavior change)

- Per-phase wall + LLM-call + token counters (extend `TokenTracker.record_stage`).
- **ER counters** in `search_similar_entities`
  ([kg_refiner.py:454](../retrieval_engine/indexing/graph/refiner.py#L454)): gate-skips, rerank
  calls, gradient outcomes, LLM-select calls, LLM-merge calls — **bucketed by
  graph size** so the super-linear curve in §1.1 is tracked continuously, not
  reconstructed from logs afterwards.
- Golden + scale fixtures; ER-decision-diff tooling (harness item 4).

**Risk:** none. **Output:** baseline table for both stages, and the slow-path
growth curve.

---

## PHASE 1 — GRAPH: make it finish (F1). Nothing else matters until this lands.

### G1. Durable, resumable refinement  ← **the new #1 item**
- **Problem:** all refinement state (graph, entity VDB, alias map) is in-memory
  until the single end-of-stage `save_graph()`
  ([construct_index.py](../retrieval_engine/indexing/pipeline.py), after
  `build_knowledge_graph` returns). Any crash ⇒ total loss.
- **Change:** periodically and atomically snapshot the refinement triple
  *(graph, entity-VDB state, `entity_alias_map`)* at a batch boundary, keyed by
  the last committed node_idx, so a restart resumes from the last snapshot
  instead of node 0. Must be consistent across all three stores — a snapshot with
  a graph ahead of the VDB is worse than none.
- **Quality:** unchanged — this only adds persistence.
- **Why first:** converts "guaranteed failure" into "eventually finishes", and
  makes every later optimization measurable on real scale instead of a 4% prefix.

### G2. `entity_merge` atomicity + resilience
- **Problem:** one LLM budget-exhaustion in `entity_merge`
  ([kg_refiner.py:242](../retrieval_engine/indexing/graph/refiner.py#L242)) kills the whole
  stage (fail-fast, unlike the pdf_refiner mergers).
- **Care:** `delete_entity_from_vdb(old_entity)`
  ([kg_refiner.py:200](../retrieval_engine/indexing/graph/refiner.py#L200)) runs **before** the
  risky LLM call. A naive try/except-skip leaves the graph half-mutated. Stage the
  delete and apply it only on LLM success (or make the pair transactional).
- Also set `think_mode=False` on merge/select calls — same budget-exhaustion root
  cause as the outline bug (T1).
- **Quality:** unchanged — failure handling only.

---

## PHASE 2 — GRAPH: cut LLM-call count (F2 — the super-linear fix)

Target: reduce the ~63K projected serial ER calls **without changing merge
decisions**. Each item is gated on harness item 4 (ER-decision diff).

### G3. Candidate blocking / canopy before the gate
- Add a cheap deterministic pre-filter (normalized-name / type-compatible /
  n-gram or trigram blocking) so entities that cannot plausibly be coreferent
  never reach rerank + LLM. Complements — does not replace — the vector gate.
- Directly attacks the 23%→41% slow-path climb: as the VDB fills, most
  above-gate neighbours are cheap-to-reject near-misses, not real aliases.
- **Quality gate:** ER-decision diff must be **identical** on the golden + scale
  fixtures; any blocked pair that the baseline would have merged is a bug.

### G4. Batch multiple ER decisions per LLM call
- Today: one LLM call per candidate entity ([:584](../retrieval_engine/indexing/graph/refiner.py#L584))
  and one per merge ([:242](../retrieval_engine/indexing/graph/refiner.py#L242)) — up to 2
  round-trips per entity. Batch several independent ER adjudications into a single
  structured call (as the tree mergers already do with pairs).
- Only entities whose candidate sets are disjoint may share a call — otherwise
  Algorithm 1's sequential semantics break.
- **Quality gate:** ER-decision diff identical; watch for the length-mismatch
  failure mode the mergers already guard against.

### G5. Widen the deterministic fast paths
- 1,536 of 4,959 entities already merge cheaply by exact name/type
  ([:198-215](../retrieval_engine/indexing/graph/refiner.py#L198)). Extend deterministic
  handling to unambiguous normalizations (case/whitespace/punctuation, known
  abbreviation pairs) that today pay a full LLM adjudication.
- **Quality gate:** each new deterministic rule must show ~100% agreement with
  the LLM's decision on the fixtures before it is trusted.

### G6. `g` and distance-gate tuning (measured — BookRAG §4.3.2)
- **Mental model:** order is `search → distance-gate (>0.2 ⇒ skip, no rerank) →
  rerank all topk → gradient-select with g → LLM select → LLM merge`. The reranker
  fires **before `g` is consulted**, so **`g` controls LLM-merge frequency, not
  rerank cost.** To cut *reranking*, tune `distance_threshold` / `topk`; to cut
  *LLM calls*, tune `g`.
- **Reciprocal caveat:** code is `score ≥ prev·g`
  ([kg_refiner.py:533](../retrieval_engine/indexing/graph/refiner.py#L533)); paper Alg. 1 line 6
  is `S[v_c] > score/g`. In code, **lowering `g` is *more* permissive** — opposite
  to the paper's notation. Confirm intent before touching `g=0.6`.
- Tune only against Fig. 6 metrics + recall. This is the one ER change that
  legitimately alters the KG, so it is gated hardest and sequenced last.

---

## PHASE 3 — GRAPH: parallelism (F3)

### G7. Decouple extraction from refinement
- Driver interleaves them ([kg_builder.py:189-205](../retrieval_engine/indexing/graph/builder.py#L189)):
  32 nodes extract concurrently, then refine **serially** before the next batch
  extracts — so extraction is throttled by ER.
- Let extraction stream to completion as its own resumable pass (already
  checkpointed per node), then refine as a separate phase.
- **Quality:** identical — only the scheduling boundary moves; the driver already
  sorts results by `node_idx` for deterministic refinement order.

### G8. Parallelize the read-only half of ER (preserve Algorithm 1 exactly)
- Algorithm 1 is sequential **only in the commit**. The prep — embed + VDB search
  + rerank ([:475-520](../retrieval_engine/indexing/graph/refiner.py#L475)) — is read-only.
- Compute prep for a window of K pending entities concurrently; **commit
  serially**; recompute an entity's candidate set only if an earlier commit in the
  window touched its neighbourhood.
- Safe because `advanced_kg_refiner` defers VDB adds until after the per-node loop
  ([:763](../retrieval_engine/indexing/graph/refiner.py#L763)) — **this "no ER within the same
  node" invariant (BookRAG §4.3.2) must be preserved explicitly.**
- **Quality gate:** ER-decision diff identical to serial baseline.

### G9. Reranker: batch, and stop flushing per entity
- `self.reranker.clean_cache()` runs **once per entity**
  ([:516](../retrieval_engine/indexing/graph/refiner.py#L516)) — a GPU cache flush ~119K times.
  Move to per-window/threshold.
- The provider already batches documents per query
  ([rerank.py:351](../retrieval_engine/providers/reranking.py#L351)); with G8's window, batch
  across entities too.
- **Quality:** none affected (non-semantic).

---

## PHASE 4 — TREE: correctness first (BookRAG §4.2.2)

### T1. Outline `think_mode` fix — restores the hierarchy BookRAG depends on
- **Symptom:** 94/94 outline chunks raise `empty structured response` → 3 retries
  → fallback → **`Max level in outline: 1` (a flat tree)**. ~1.41M completion
  tokens produce nothing. The current index is effectively the paper's
  *Tree-Traverse baseline*, **not BookRAG's hierarchical BookIndex**.
- **Cause:** `qwen3.6:35b` spends its whole `max_tokens=5000` on internal
  thinking and emits no JSON. `get_json_completion` accepts `think_mode`
  ([llm.py:665](../retrieval_engine/providers/language_model.py#L665)); it is never set on this path.
- **Fix:** set `think_mode=False` (or raise the completion budget) at
  [outline_extractor.py:320](../retrieval_engine/indexing/tree/outline.py#L320);
  reproduce the empty response in isolation first.
- **Effect:** real hierarchy (**quality ↑**) **and** reclaims ~2h54m.

> ⚠️ **New risk this revision surfaces — T1 will make summarization slower.**
> `generate_tree_node_summary` is **level-sequential** (a section prompt consumes
> its children's summaries); only nodes *within* a level run concurrently
> ([tree_node_summary.py:218-224](../retrieval_engine/indexing/tree/summary.py#L218)).
> Today's flat tree (max level 1) is therefore the *maximally parallel* case — one
> huge level. A correct deep hierarchy introduces more sequential level passes.
> Leaves still dominate the node count, so the effect should be modest, but it
> **must be measured** (T6), and T1's speed saving must be reported net of it.

---

## PHASE 5 — TREE: fewer calls, then parallelism (F3)

### T2. Raise merger batch packing (call-count reduction — do before T3)
- `get_json_str_text` hard-caps **3 pairs per call**
  (`number_of_pairs < 3`, [pdf_refiner.py:797](../retrieval_engine/parsing/pdf_refinement.py#L797))
  independently of the token budget. 2,022 text candidates ⇒ ~674 calls.
- Packing to the actual token budget (~10/call) is a **~3× call-count reduction**
  that *composes* with T3's parallelism — and on one GPU, fewer calls beats
  concurrent calls.
- **Quality gate:** the existing response-length-mismatch check
  ([:868](../retrieval_engine/parsing/pdf_refinement.py#L868)) already guards the failure mode;
  measure mismatch/retry rate as packing rises and stop before it degrades.
  Merged `pdf_list` must match baseline.

### T3. Parallelize the mergers (byte-identical)
- text/label/table judging is a **pure map** — batches independent, merging is a
  separate reverse pass ([:893](../retrieval_engine/parsing/pdf_refinement.py#L893)).
- Replace the serial loop in `_llm_stitching_judge`
  ([:843-870](../retrieval_engine/parsing/pdf_refinement.py#L843)) and the table-pair loop
  ([:1257-1270](../retrieval_engine/parsing/pdf_refinement.py#L1257)) with the in-tree
  `llm.iter_completion_results` (already used by `tree_node_summary`).
- **Subtlety:** `found_remove_text`'s failure-path list mutation must move to
  after collection.
- **Quality:** byte-identical (same prompts, packing, schema; ordered iterator).

### T4. Sparse LLM recheck (BookRAG §4.2.2 — "recheck only where needed")
- The recheck is already scoped to Title blocks only (text/image/table never hit
  the LLM). Go finer: skip the LLM where layout is unambiguous — explicit
  numbering (`^\d+(\.\d+)*`), the single height-max first-page title, clean
  height-band matches — and send only ambiguous titles.
- **Quality gate:** deterministic assignments must agree ~100% with the
  (post-T1, working) LLM on the fixtures before the skip is trusted.

### T5. Parallelize outline batches
- ⚠️ Outline batching is a **sequential fold, not a map**: each chunk reads
  `final_outline` accumulated so far
  ([outline_extractor.py:253-280](../retrieval_engine/indexing/tree/outline.py#L253)),
  so T3's trick does not apply directly.
- **Change:** precompute a cheap global skeleton (level-0 + level-1 candidates via
  numbering/height), feed every batch that *fixed* skeleton instead of the running
  fold ⇒ batches independent ⇒ parallelizable. Deterministic parent repair in one
  post-pass (`outline_refine` already exists).
- **Preserves** the paper's l=1 context-window intent, but from a stable source.
- **Quality gate:** structural diff vs the sequential-fold version within reviewed
  tolerance. Medium risk; if it regresses, keep T4 only.

### T6. Summary level-depth mitigation (follows from T1's risk)
- Measure summary wall time before/after T1. If added level-sequencing hurts,
  overlap independent subtrees across levels (a node only needs *its own*
  children, not the whole level) — the existing per-node checkpoint cache
  (`summary-checkpoints.sqlite3`) already makes this safe to retry.

---

## PHASE 6 — Whole-document memory (F4) — required for 5000-page targets

Sized by the whole document today, and untouched by v1 of this plan:
- `pdf_list` accumulates every block across all shards for
  `pdf_info_refiner` / outline / tree construction.
- `tree.pkl` / `tree.json` — whole-tree pickle at 16K nodes (≈160K at 10×).
- `construct_tree_index` and outline extraction both need the complete ordered
  block list to assign hierarchy.

The underlying invariant (a title's parent has an earlier `pdf_id`) is
streaming-compatible, and `docarag.tree_node`'s schema is explicitly designed for
incremental writes — but making `construct_tree_index` itself incremental is a
real redesign. **Scoped here, deliberately after Phases 1–5**, and gated on
measured peak RSS at the 500pp fixture: fix it when it is the binding
constraint, not on speculation.

---

## PHASE 7 — Provider ceiling (measure, do not assume)

`max_workers=4` on **one** GPU. Raising it (and `OLLAMA_NUM_PARALLEL`) helps only
if there is decode-batching headroom. Measure aggregate tok/s at 4 vs 6 vs 8
in-flight on the fixtures before committing a value.

**Ceiling:** ~3.4M completion tokens/stage at the observed ~70 tok/s aggregate is
decode-bound on a single local 35B. This is the arithmetic reason Phases 2 and 5's
**call-count reductions outrank every parallelism item**, and why T1 (deleting
1.4M wasted tokens) is the best single change in the tree stage.

---

## Sequencing, risk, and what each phase buys

| Phase | Item | Risk | Quality | Buys |
|---|---|---|---|---|
| 0 | instrument + fixtures + ER-diff tool | none | none | makes everything measurable |
| **1** | **G1 durable/resumable refinement** | low | none | **graph can finish at all** |
| **1** | **G2 merge atomicity + resilience** | med | none | stops total-loss crashes |
| 2 | G3 blocking/canopy | med | ER-diff identical | attacks super-linear curve |
| 2 | G4 batch ER decisions | med | ER-diff identical | fewer round-trips |
| 2 | G5 deterministic fast paths | low | ~100% agreement gate | fewer LLM calls |
| 2 | G6 `g` / gate tuning | med | Fig. 6 + recall gated | measured |
| 3 | G7 decouple extract/refine | low | identical | unblocks extraction |
| 3 | G8 parallel ER prep | med | ER-diff identical | ER speedup |
| 3 | G9 rerank batch + no flush | none | none | free GPU win |
| **4** | **T1 outline `think_mode`** | low | **↑ (real hierarchy)** | correctness + ~2h54m |
| 5 | T2 merger packing | low | mismatch-rate gated | ~3× fewer calls |
| 5 | T3 merger parallelism | low | byte-identical | ~4h40m at 4-way |
| 5 | T4 sparse outline recheck | low-med | ~100% agreement gate | fewer calls |
| 5 | T5 outline batch parallelism | med | structural diff gated | outline speedup |
| 5 | T6 summary depth mitigation | low | none | offsets T1's risk |
| 6 | whole-document memory | high | none | 5000-page headroom |
| 7 | provider ceiling | low | none | measured |

**Honest expectations.** Tree: T1+T2+T3 take 13h25m toward roughly 4–5h, net of
T1's summary-depth effect. Graph: no credible wall-clock promise can be made from
a 4% prefix — Phase 1 exists precisely so that Phase 2/3 numbers can be measured
on a completing run instead of extrapolated. What *is* firm: ~63K serial LLM calls
and a super-linear slow-path curve mean **call-count reduction (Phase 2), not
parallelism, is the deciding factor.**

---

## Appendix A — BookRAG fidelity: deviations found in the code

| # | Deviation | Location | Call needed |
|---|---|---|---|
| A1 | **Hierarchy absent — §4.2.2 effectively not realised.** All outline chunks fail ⇒ `Max level: 1`. Current index ≈ paper's Tree-Traverse baseline, not BookRAG. | [outline_extractor.py:320](../retrieval_engine/indexing/tree/outline.py#L320) | fixed by **T1** |
| A2 | **`g` reciprocal to the paper.** Code `score ≥ prev·g`; paper Alg. 1 `S > score/g`. Lowering `g` is *more* permissive in code. | [kg_refiner.py:533](../retrieval_engine/indexing/graph/refiner.py#L533) | confirm intent before **G6** |
| A3 | **`mink` seed not in the paper.** Code seeds selection with `ranked_results[:mink]` (mink=1); paper seeds `Ec[0]` only. Equivalent at 1; flagged for clarity. | [kg_refiner.py:528](../retrieval_engine/indexing/graph/refiner.py#L528) | note only |
| A4 | **`max_gleaning=0`.** Paper's extraction permits multi-round gleaning; disabled here. Fine for speed — noted so a future quality tweak does not silently add serial calls. | [kg_extractor.py:386](../retrieval_engine/indexing/graph/extractor.py#L386) | note only |
| A5 | **Scale gap.** Paper validated ≤42 avg pages / ~1.3K entities; we run 1,870 pages / ~119K entities. Incremental "clean ER" was chosen for the small-graph regime. | §0 | drives **Phase 2** |

## Appendix B — BookRAG invariants the scaling must not break

| Paper concern | Where | Protected by |
|---|---|---|
| §4.3.2 Alg. 1 **sequential commit** (each insert changes the next lookup) | [kg_refiner.py:526-541](../retrieval_engine/indexing/graph/refiner.py#L526) | G8 parallelizes read-only prep only; commit stays serial; ER-decision diff |
| §4.3.2 **no ER within the same node** (a node's own entities aren't resolved against each other) | VDB add deferred to after the loop, [:763](../retrieval_engine/indexing/graph/refiner.py#L763) | explicit invariant in G7/G8 |
| §4.3.2 **GT-Link** (origin node per entity, aggregated on merge) | `add_and_link(tree_node_id=…)` | link-integrity check in harness |
| §4.3.1 **modality-dependent extraction** (LLM text / VLM image / typed table entity) | [kg_extractor.py:472](../retrieval_engine/indexing/graph/extractor.py#L472), [:767](../retrieval_engine/indexing/graph/extractor.py#L767) | G7 moves scheduling only |
| §4.2.2 **batched filtering with l=1 context window** | [outline_extractor.py:253-280](../retrieval_engine/indexing/tree/outline.py#L253) | T5 keeps the level-0/1 signal from a stable skeleton |
| §4.2.2 **type recheck τ′ (Title→Text)** | [doc_tree_builder.py:1346-1353](../retrieval_engine/indexing/tree/builder.py#L1346) → [tree_node_builder.py:75](../retrieval_engine/indexing/tree/nodes.py#L75) | T4's agreement gate |
| Fig. 6 **ER quality metrics** (#entities, density, diameter, #CC) | — | harness item 3, gate for every ER change |

---

## Open questions for review

1. **Fixtures:** a ~100–200pp doc and a ~500pp doc you're happy to check in, or
   should I select public ones? The 500pp one matters most — §1.1 shows small docs
   do not reproduce the failure.
2. **Phase 1 scope:** G1 (durable refinement) is the unblock, but it touches three
   stores (graph, VDB, alias map) that must snapshot consistently. Land G1 alone
   first and prove a completing run on the 500pp fixture, before any Phase-2 work?
3. **Phase 2 ordering:** G3 (blocking) and G4 (batched adjudication) both cut call
   count but are the two changes most able to alter merge decisions. Prefer G5
   (deterministic fast paths, safest) first to establish the ER-diff harness, then
   G3, then G4?
4. **Phase 6:** defer whole-document memory until a 500pp run shows RSS pressure,
   or is 5000-page capability an immediate requirement?
