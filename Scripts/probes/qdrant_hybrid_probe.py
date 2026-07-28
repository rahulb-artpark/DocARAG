"""Prove Qdrant's native hybrid shape before building the entity store on it.

Checks, against the live server, the four things the design depends on:
named dense vectors coexisting with a sparse vector, server-side IDF, RRF
fusion in one round trip, and that a scope filter still applies to a fused
query. Uses a throwaway collection; touches nothing real.
"""

from __future__ import annotations

import re
from qdrant_client import QdrantClient, models

COLLECTION = "probe-hybrid-entities"
DIM = 8

client = QdrantClient(url="http://127.0.0.1:6335")


def sparse(text: str) -> models.SparseVector:
    """Term frequencies; Qdrant applies IDF from collection statistics."""

    counts: dict[int, float] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        # Stable non-negative 32-bit token id, no vocabulary to persist.
        token_id = hash(token) & 0x7FFFFFFF
        counts[token_id] = counts.get(token_id, 0.0) + 1.0
    return models.SparseVector(
        indices=list(counts), values=[float(v) for v in counts.values()]
    )


def dense(seed: float) -> list[float]:
    return [seed] * DIM


if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "name": models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        "context": models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "lexical": models.SparseVectorParams(
            index=models.SparseIndexParams(),
            modifier=models.Modifier.IDF,
        )
    },
)
print("created a collection with 2 named dense vectors + 1 sparse")

client.create_payload_index(
    collection_name=COLLECTION,
    field_name="tenant_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True,
)

ENTITIES = [
    (1, "large language model", "a transformer trained on text", 0.9, "t1"),
    (2, "llm", "an abbreviation for large language model", 0.85, "t1"),
    (3, "chief ray", "the ray through the centre of the aperture stop", 0.2, "t1"),
    (4, "marginal ray", "the ray through the edge of the aperture stop", 0.21, "t1"),
    (5, "large language model", "an entity in another tenant", 0.9, "t2"),
]

client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(
            id=pid,
            vector={
                "name": dense(seed),
                "context": dense(seed),
                "lexical": sparse(f"{name} {desc}"),
            },
            payload={"entity_name": name, "node_id": f"n{pid}", "tenant_id": tenant},
        )
        for pid, name, desc, seed, tenant in ENTITIES
    ],
    wait=True,
)
print(f"upserted {len(ENTITIES)} points")

scope = models.Filter(
    must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value="t1"))]
)

query_text = "llm"
response = client.query_points(
    collection_name=COLLECTION,
    prefetch=[
        models.Prefetch(query=dense(0.87), using="name", filter=scope, limit=10),
        models.Prefetch(query=dense(0.87), using="context", filter=scope, limit=10),
        models.Prefetch(query=sparse(query_text), using="lexical", filter=scope, limit=10),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=scope,
    limit=5,
    with_payload=True,
)

print(f"\nfused RRF results for {query_text!r} (scoped to t1):")
for hit in response.points:
    print(f"  score={hit.score:.4f}  {hit.payload['entity_name']!r}  {hit.payload['node_id']}")

returned_tenants = {h.payload["tenant_id"] for h in response.points}
print(f"\ntenants in result: {returned_tenants}  (filter held: {returned_tenants == {'t1'}})")

sparse_only = client.query_points(
    collection_name=COLLECTION,
    query=sparse("aperture stop ray"),
    using="lexical",
    query_filter=scope,
    limit=3,
    with_payload=True,
)
print("\nsparse-only 'aperture stop ray' (IDF should rank the two ray entities):")
for hit in sparse_only.points:
    print(f"  score={hit.score:.4f}  {hit.payload['entity_name']!r}")

client.delete_collection(COLLECTION)
print("\nprobe collection deleted")
