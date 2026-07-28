# Retrieve entities with Qdrant's native hybrid search

Supersedes G3/G4/G8's ordering in `scaling_plan.md`, and supersedes this
document's own earlier "two named vectors, dedup client-side" design. Baselines
to diff against: `scaling_baseline.md`, plus the completed 1,871-page run
(25,324 entities, `rerank=7007`, `llm_calls=9005`, `direct=9738`).

## Why

Entity resolution embeds only the node name:

```python
embed_texts.append(node_name)   # "Name: {name}\nType: {type}"
metadatas.append(...)           # description is payload, never embedded
```

Retrieval therefore compares surface forms, and the reranker carries every
semantic judgement. That is why "LLM" does not retrieve "Large Language Model"
on its own.

The cost of fixing it is **not** rerank time. Phase 0 measured reranking at
1.6% of the graph stage (`latency_measurements.md`). The reason to retrieve
better is that the stage is **9,005 model adjudications at ~2.9 s each**, and
precise candidates are what remove adjudications. `direct=9738` already shows
the cheap path dominating; the goal is to move more resolutions into it.

## What Qdrant actually supports

Verified against the live server (1.18.2) with a throwaway collection, not
assumed from documentation:

- named dense vectors and a sparse vector in **one** collection
- `Modifier.IDF` -- BM25-style scoring computed **server-side** from collection
  statistics, so no model and no vocabulary to persist
- `Prefetch` + `FusionQuery(fusion=Fusion.RRF)` -- fusion in **one round trip**
- scope filters still apply to a fused query (a foreign-tenant point was
  correctly excluded)

A sparse-only probe for `aperture stop ray` ranked both ray entities at 3.5019,
confirming IDF works on raw term frequencies sent from the client.

## Design

**Three vectors per entity, one point.**

| vector | kind | embeds | catches |
| --- | --- | --- | --- |
| `name` | dense | `Name: x\nType: y` | today's behaviour, preserved exactly |
| `context` | dense | name + type + description | "LLM" = "Large Language Model" |
| `lexical` | sparse, IDF | tokenized name + description | "Qwen3-Reranker-4B" = "Qwen3 Reranker 4B" |

Dense and sparse fail in opposite directions, which is the whole point. Dense
embeddings are weakest on exactly what alias matching needs -- short strings,
punctuation, model numbers, casing -- and that is what the sparse vector covers.
Semantic equivalence with no shared tokens is what dense covers. The earlier
plan listed a sparse vector as step 3, "only if aliases are still missed"; the
server computing IDF makes it cheap enough to include from the start.

Keeping `name` matters for the quality bar. It is today's entire behaviour, so
retaining it means the change adds evidence rather than replacing it, and a
regression in candidate quality is attributable to fusion rather than to having
discarded the only signal that was ever tested.

**One point per entity.** This is the concrete gain over the superseded
multiple-points design: no facet duplicates in the result set, no over-fetch to
compensate, and fusion happens in Qdrant rather than in application code.

**Merges do not rewrite vectors.** A and B merging into C leaves their vectors
untouched, each still meaning what it meant when written. A later entity
matching A's phrasing matches A's own vector rather than a blurred centroid, and
resolves to C through the alias map.

**Canonical identity stays client-side (option B).** Payloads keep the
*original* entity id. `entity_alias_map` / `get_latest_entity_name` already
resolves chains with cycle detection, so it remains the single source of truth:
no payload writes on merge, nothing to drift. `delete_entity_from_vdb` becomes a
no-op on merge.

**Dedup before the gradient, still required.** Native hybrid removes the *facet*
duplicates but not the *merge* duplicates: after merges, several distinct
entities' points resolve to one canonical entity. Left in place, adjacent
near-identical scores manufacture a smooth run where BookRAG 4.3.2 expects a
sharp drop, so `case_a` fires and the entity is added instead of merged. On the
query path it is worse -- `Graph_Reasoning` runs PageRank through the GT-Link
matrix, so a duplicated entity is double-weighted and `Skyline_Ranker` sees
inflated dimensions. Over-fetch, resolve each hit to its canonical entity,
reduce to one row per entity by **max** score, then proceed. One shared helper,
used by entity resolution and the query path both.

## What this costs in the provider

`Core/provider/vdb.py` today configures a single unnamed dense vector and
actively rejects named ones:

```python
raise RuntimeError(
    "DocARAG requires one unnamed dense vector; named-vector collection "
    "configuration is incompatible"
)
```

So this is provider work, not configuration:

1. The vector schema becomes a property of the `VectorStore` instance. The
   entity store opts into the hybrid schema; the tree-node store keeps the
   unnamed dense schema. `VectorStore` is shared by both, and confining the
   change to the entity collection is what keeps the query path's own index out
   of this migration.
2. `_vector_params` must validate a named-vector collection instead of refusing
   one.
3. Writes supply all three vectors; reads issue three prefetches and one
   `FusionQuery`.
4. The entity collection is recreated -- Qdrant cannot convert an unnamed vector
   in place -- and the entity attestation schema version is bumped so a
   half-migrated index cannot be read as current.

## Sequence

1. Provider: hybrid schema, named-vector validation, fused search. Prove it on
   the entity collection alone, with the tree-node collection untouched.
2. Refiner and query path: write three vectors, read through the shared dedup.
3. Measure `llm_selection_calls` against the 5,700 baseline with
   `entities_resolved` holding. That is the number this change exists to move --
   not rerank time, which is 1.6%.
4. Only then consider lowering `similar_search_topk` from 10.

## Gates

This changes which candidates reach the model, so it changes merge decisions,
and a wrong merge silently redirects every edge through `process_relationships`.
Requires the ER-decision diff and the paper's Figure 6 metrics (entities,
density, diameter, components) within tolerance of the baselines above, at both
28 pages and 1,871.
