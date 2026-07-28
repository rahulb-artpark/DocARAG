# Phase 0 results: where the time actually goes

Measured 2026-07-28 against revision 26 (`test.pdf`, 1,871 pages) and its real
entity text. Every number here replaces an estimate in `latency_plan.md`.

## Stage wall clock (Postgres `stage_state`)

| stage | ran | duration | on current code? |
| --- | --- | --- | --- |
| tree | 07-20 22:38 - 07-21 12:49 | 851.3 min | **no -- pre-fix** |
| graph | 07-27 19:04 - 07-28 02:19 | 434.6 min | yes |
| vdb | 07-28 03:22 - 03:30 | 8.0 min | yes |

**The tree number is not comparable to the other two.** All three tree fixes
landed on 2026-07-26 -- `2cc4677` (thinking), `fd82809` (concurrent merger
passes), `593a575` (packing cap) -- five days after that run, and the tree
artifacts are stamped 07-21 17:33, so they were never rebuilt. The graph stage
ran 07-27, after.

Rescaling it is not straightforward either. The fixes measured 2.9x and 5.1x,
but on small documents, and they touch only the model-driven phases. MinerU
parsing is untouched and scales with page count, so on 1,871 pages it is a much
larger share of the stage and the realized speedup will be lower. What the tree
stage costs today is unknown; `PhaseTimings` now separates `mineru_parse` from
the phases the fixes actually changed, so one run answers it.

## What the graph stage actually did

The final ER snapshot of the completed run (the plan quoted an earlier partial
run's counters; these supersede them):

```text
graph_nodes=25325 resolved=8201 rerank=7007 (85.4%)
llm_calls=9005 (select=5700 merge=3305 fallback=0)
cheap(gate=1194 low_score=325 direct=9738)
gradient(case_a=982 single=656 multi=5044)
```

The stage's log opens with `Resuming refinement from ...` and one line from
`kg_extractor`: extraction was reused from `kg_extractor_res/`, so those 434.6
minutes were **entity resolution almost end to end**. 9,005 model calls over
26,076 seconds is **~2.9 s per call**, which is the stage.

## Reranking is not the graph stage's cost

`rerank=7007 (85.4%)` was call frequency. Measured on the real reranker
(Qwen3-Reranker-0.6B, local, cuda:0) over 40 calls of 10 candidates drawn from
the run's own 25,081 described entities:

- a rerank costs **58 ms** (median 57 ms)
- 7,007 reranks is **6.8 min**, or **1.6% of the 434.6-minute graph stage**

The document sizes explain it. The plan assumed descriptions near the
10,000-char truncation; the real distribution is **median 177 chars, p90 472,
max 5,740**. The truncation almost never binds, and a 0.6B model on inputs that
short is launch-bound, not compute-bound.

## Rerank batching: rejected

The plan claimed 10 candidates at `batch_size=4` run 3 forward passes where 1
would do, and that batching is output-identical. Both halves are wrong.

| | mean per call | over 7,007 reranks |
| --- | --- | --- |
| `batch_size=4` | 58 ms | 6.8 min |
| `batch_size=10` | 44 ms | 5.2 min |

**1.31x, worth 1.6 minutes of a 7-hour stage.** And it is not free:

| comparison | max delta | reordered | crossed the 0.5 floor |
| --- | --- | --- | --- |
| 4 vs 4 (control) | 0 | **0/40** | 0/400 |
| 4 vs 10 | 2.3e-02 | **21/40** | 0/400 |

The control is the point. The reranker is bit-for-bit deterministic at a fixed
batch size, so the 53% of calls whose candidate ordering changes are caused by
batching, not by pre-existing noise. Ordering feeds both the 4.3.2 gradient and
the candidate list the model adjudicates, so those are changed merge decisions.

Mathematically the scores should be invariant -- padding is left-side, the
attention mask covers it, and RoPE is relative, so a uniform shift cancels.
What remains is float non-determinism from differently-shaped batched matmuls,
and near-tied candidates are dense enough that it reorders them anyway.

0.4% of the stage for perturbed merge decisions is a bad trade. Not doing it.

## Query-path thinking: large, but not free

`indexing_llm_config` disabled thinking for tree and graph only; the query path
still runs qwen3.6:35b at its default. Measured on the query path's own prompts
and schemas, 3 queries each:

| call site | think=default | think=False | speedup |
| --- | --- | --- | --- |
| query entity extraction | 17.4 s | 0.7 s | **23.7x** |
| section selection | 6.6 s | 1.0 s | **6.5x** |

The latency is as large as the plan expected. The quality is not free:

- entity extraction dropped `optical design` from one query (2 entities to 1)
  and relabelled `chief ray` / `marginal ray` from `SCIENTIFIC_THEORY` to
  `CONCEPT` -- the type is half the node-name key, so it moves entity linking
- section selection returned **`select_id=-1`** where thinking selected
  section 2, which changes the retrieved subtree wholesale
- the other four of six comparisons were identical

This differs from indexing for a concrete reason. There, thinking consumed the
whole `num_predict` and the model emitted **no content at all** -- disabling it
recovered an answer that was otherwise lost. Here the model answers either way
and the reasoning is doing real work.

So: no blanket `think=False` on the query path. Worth pursuing per call site,
against a real question set rather than three queries, since two of the six
comparisons regressed and a 23.7x saving is too large to abandon on a probe.

## What this implies

The graph stage is **9,005 model calls at ~2.9 s each, and nothing else that
matters**. Rerank is 1.6%; vector search and embedding are smaller still.

- **Phase 2 (batch ER adjudications) is the graph stage.** Not a step in it.
  The two call sites are `er_selection_by_llm` (5,700) and `entity_merge`'s
  summarization (3,305); both are worth batching, and the split says selection
  is the larger half.
- **`direct=9738` is the shape to exploit.** More resolutions end in a direct
  merge, costing no model call, than in any adjudicated outcome. Anything that
  converts an adjudication into a direct merge is worth more than making the
  adjudication faster.
- **Phase 3 (named vectors) keeps its design but loses its stated reason.** It
  was justified as making reranking cheaper, which is 1.6% of the stage. Better
  candidates are still worth having -- because they cut *adjudications*, which
  are effectively the other 98%. Measure it in model calls saved.
- **G8 is dead.** It parallelised embed, search and rerank: together a rounding
  error here.
- **The graph stage is the top target on current code.** It is the only stage
  above with a post-fix number. The tree stage's 851.3 min is pre-fix and cannot
  rank against it until one instrumented run replaces it.
