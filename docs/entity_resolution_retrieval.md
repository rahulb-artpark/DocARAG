# Phase 2 (revised): retrieve entities better instead of reranking more

Supersedes G3/G4/G8's ordering in `scaling_plan.md`. Baselines to diff against:
`scaling_baseline.md`, plus the full-scale run (25,324 entities, rerank share
85.4%, 11,201 reranks, 14,561 model calls).

## Why

Entity resolution embeds only the node name:

```python
embed_texts.append(node_name)   # "Name: {name}\nType: {type}"
metadatas.append(...)           # description is payload, never embedded
```

So retrieval compares surface names, and the reranker does all the semantic
work -- up to 10 candidates with descriptions cut at 10,000 chars each. At
11,201 reranks that is the dominant cost of the graph stage. Better candidates
beat faster reranking: fewer, more precise candidates cut rerank input *and*
the LLM adjudications behind `multi=7,970`.

## Design

**Two named vectors per entity.** `name` keeps today's behaviour and is what
distinguishes entities whose descriptions are near-identical ("chief ray" vs
"marginal ray"). `context` embeds name + type + description and is what finds
"LLM" = "Large Language Model". Keep both; description-only retrieval drifts
toward topical similarity.

**Merges do not touch vectors.** A and B merging into C leaves both their
vectors in place, each still meaning what it meant when written. No
re-embedding, and no vector can go stale. A later entity matching A's phrasing
matches A's own vector rather than a blurred centroid, and resolves to C.

**Canonical identity stays client-side (option B).** Payloads keep the
*original* entity id. `entity_alias_map` / `get_latest_entity_name` already
resolves chains with cycle detection, so it remains the single source of truth:
no payload writes on merge, nothing to drift. `delete_entity_from_vdb` becomes a
no-op on merge.

**Dedup before the gradient.** Search returns nearest *points*, and a merged
entity's facets cluster, so one entity can fill several slots. Over-fetch
(~30 points for 10 distinct entities), resolve each to its canonical entity,
reduce to one row per entity taking the **max** score across its facets, then
run the gradient. Skipping this breaks BookRAG 4.3.2 quietly: adjacent
near-identical facet scores manufacture a smooth run where a sharp drop should
be, so `case_a` fires and the entity is added as new instead of merged.

**`refine_entities` mints C's synthesized vector and keeps the facets.** The
synthesis is a summary and loses specific phrasing, so an entity stays findable
by any surface form plus its consolidated meaning. ER is over by then, so this
only affects query-time retrieval, where more facets is better.

## Sequence

1. Add the `context` vector + dedup-before-gradient. Measure precision@k and
   whether `similar_search_topk` can drop from 10.
2. Lower `topk`. The ER counters are the measurement: `rerank_calls` should fall
   while `entities_resolved` holds, against the 85.4% baseline.
3. Add a sparse vector and fusion only if aliases are still missed -- alias
   matching is lexical, which is where dense embeddings are weakest.

## Gates

Changes which candidates reach the model, so it changes merge decisions, and a
wrong merge silently redirects every edge through `process_relationships`.
Requires the ER-decision diff and the paper's Figure 6 metrics (entities,
density, diameter, components) within tolerance of the baselines above.

## Cost

`Core/provider/vdb.py` configures a single unnamed vector with no sparse index,
so named vectors are provider work, not configuration.
