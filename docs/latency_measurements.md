# Phase 0 results: where the time actually goes

Measured 2026-07-28 against revision 26 (`test.pdf`, 1,871 pages) and its real
entity text. Every number here replaces an estimate in `latency_plan.md`.

## Stage wall clock (Postgres `stage_state`)

| stage | duration |
| --- | --- |
| tree | 851.3 min |
| graph | 434.6 min |
| vdb | 8.0 min |

## Reranking is not the graph stage's cost

`rerank=11201 (85.4%)` was call frequency. Measured on the real reranker
(Qwen3-Reranker-0.6B, local, cuda:0) over 40 calls of 10 candidates drawn from
the run's own 25,081 described entities:

- a rerank costs **58 ms** (median 57 ms)
- 11,201 reranks is **10.9 min**, or **2.5% of the 434.6-minute graph stage**

The document sizes explain it. The plan assumed descriptions near the
10,000-char truncation; the real distribution is **median 177 chars, p90 472,
max 5,740**. The truncation almost never binds, and a 0.6B model on inputs that
short is launch-bound, not compute-bound.

## Rerank batching: rejected

The plan claimed 10 candidates at `batch_size=4` run 3 forward passes where 1
would do, and that batching is output-identical. Both halves are wrong.

| | mean per call | over 11,201 reranks |
| --- | --- | --- |
| `batch_size=4` | 58 ms | 10.9 min |
| `batch_size=10` | 44 ms | 8.3 min |

**1.31x, worth 2.6 minutes of a 7-hour stage.** And it is not free:

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

0.6% of the stage for perturbed merge decisions is a bad trade. Not doing it.

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

The graph stage is **model-bound**: ~30,000 LLM calls (extraction over 16,092
tree nodes, plus 14,561 for resolution) against 434.6 minutes is roughly 0.85 s
per call, and rerank accounts for 2.5%. Cutting model calls is the only change
that moves it.

- **Phase 2 (batch ER adjudications) is the whole graph opportunity**, not a
  secondary one. 14,561 calls is measured, not estimated.
- **Phase 3 (named vectors) keeps its value but loses its stated reason.** It
  was justified as making reranking cheaper, which is 2.5% of the stage. Better
  candidates are still worth having -- because they cut *adjudications*, which
  is the 97%. Same design, different premise, and it must be measured in model
  calls saved rather than rerank time.
- **The tree stage is the larger target and remains unmeasured at 851.3 min.**
  Phase 0's instrumentation is in place; it needs one run to break that down.
