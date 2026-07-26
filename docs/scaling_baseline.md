# Phase 2 baseline — measured 2026-07-26

Captured after the eight landed items (`2cc4677`..`593a575`), on fresh runs
(1 revision, 1 attempt, zero reused summaries or extraction checkpoints). Every
Phase 2 change (G3 blocking, G4 batched adjudication, G5 fast paths, G6 `g`)
must be diffed against these numbers before it is accepted.

## Wall clock

| Document | Tree | Graph | Total | Speedup |
|---|---|---|---|---|
| Basic_computer_concept.pdf | 5m22s → **2m31s** | 30m26s → **9m53s** | 35m48s → **12m24s** | **2.9×** |
| physics_N.pdf (was 312_LG_E_L3) | 3m53s → **43s** | 16m58s → **3m23s** | 20m51s → **4m6s** | **5.1×** |

All three stages (`tree`, `graph`, `vdb`) reached `succeeded` on both.

## Graph quality — the BookRAG Figure 6 metrics

These are the paper's own entity-resolution yardstick. An ER change that moves
them has changed output quality, not just cost.

| Document | Entities | Relations | Density | Components | Diameter (LCC) | GT-links |
|---|---|---|---|---|---|---|
| physics_N.pdf | 127 | 123 | 0.01537 | 21 | 10 | 42 |
| Basic_computer_concept.pdf | 539 | 454 | 0.00313 | 210 | 13 | 94 |

## Entity-resolution cost curve (from `d370981`'s counters)

Basic_computer_concept.pdf, one row per batch boundary:

| graph_nodes | resolved | rerank | rerank share | llm_calls | select | merge | fallback | gate | direct | case_a |
|---|---|---|---|---|---|---|---|---|---|---|
| 267 | 266 | 33 | **12.4%** | 45 | 31 | 14 | 0 | 233 | 24 | 1 |
| 434 | 452 | 80 | **17.7%** | 105 | 72 | 33 | 0 | 372 | 53 | 1 |
| 539 | 579 | 130 | **22.5%** | 170 | 115 | 55 | 0 | 449 | 88 | 3 |

physics_N.pdf ends at: graph_nodes=127 resolved=141 rerank=45 (31.9%)
llm_calls=66 (select=42 merge=24 fallback=0) gate=96 direct=19 case_a=0.

Two facts this pins down:

1. **The super-linear curve reproduces at small scale.** Rerank share climbs
   12.4% → 17.7% → 22.5% as the graph fills, the same mechanism measured on the
   1,870-page run (23% → 41%). Phase 2 targets exactly this slope.
2. **`fallback=0` everywhere.** G2's direct-merge fallback never fired, so no
   merge lost its synthesized name. The resilience path is insurance, not a
   quality cost, on these documents.

## G1 resume validation

Interrupted `build_knowledge_graph` in a scratch save_path against a throwaway
Qdrant collection (`extraction_batch_size=8` for more snapshot points):

- **Attempt 1** killed ~63s in, one batch committed. Snapshot held
  8 refined node ids, 23 graph nodes, 8 alias entries.
- **Attempt 2** restarted and logged
  `Resuming refinement from …: 8 nodes already refined, 23 graph nodes`, then
  `Rebuilt the entity vector store with 23 entities`, committed the remaining
  5 batches, finished at 131 entities / 132 relations, and cleared its snapshot.

The resumed graph is not identical to an uninterrupted one (131/132 versus
120/120 at `extraction_batch_size=32`). That is inherent to BookRAG's
incremental resolution rather than a resume defect: batch size decides when an
entity becomes visible in the vector store to later entities, and the model is
not deterministic. Resume restores the committed state exactly — what it cannot
do is make incremental ER order-independent.

## What is still missing

A ~500pp scale fixture. At 539 entities these documents sit at the bottom of the
cost curve; G3 and G4 target the regime where the slow path is 40%+, which only
appears on a much larger document.
