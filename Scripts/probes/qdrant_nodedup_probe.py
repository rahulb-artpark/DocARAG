"""Can Qdrant guarantee one row per entity, so no client-side dedup exists?

Two candidates, tested rather than argued:

  A. Native multivector (MAX_SIM). A merge appends the absorbed entity's
     vectors to the surviving point. One point per canonical entity by
     construction, scored by its best vector.
  B. Server-side group_by on a canonical payload field with group_size=1.

Both must survive the thing that matters: combining with sparse + RRF fusion,
which is what the entity store actually issues.
"""

from __future__ import annotations

import re
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://127.0.0.1:6335")
DIM = 4

# Deliberately far apart so a max-vs-mean confusion is unmissable.
NEAR = [1.0, 0.0, 0.0, 0.0]
FAR = [0.0, 0.0, 0.0, 1.0]
QUERY = [1.0, 0.0, 0.0, 0.0]


def sparse(text: str) -> models.SparseVector:
    counts: dict[int, float] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tid = hash(token) & 0x7FFFFFFF
        counts[tid] = counts.get(tid, 0.0) + 1.0
    return models.SparseVector(indices=list(counts), values=list(counts.values()))


def fresh(name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)


# ---------------------------------------------------------------- candidate A
print("=" * 70)
print("A. native multivector, MAX_SIM")
print("=" * 70)
COL_A = "probe-multivector"
fresh(COL_A)
client.create_collection(
    collection_name=COL_A,
    vectors_config={
        "name": models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        )
    },
    sparse_vectors_config={
        "lexical": models.SparseVectorParams(
            index=models.SparseIndexParams(), modifier=models.Modifier.IDF
        )
    },
)
client.upsert(
    collection_name=COL_A,
    points=[
        # C absorbed A: the point carries both identities' vectors.
        models.PointStruct(
            id=1,
            vector={"name": [FAR, NEAR], "lexical": sparse("entity c absorbed a")},
            payload={"node_id": "C", "label": "C (absorbed A)"},
        ),
        # A lone entity whose only vector is far from the query.
        models.PointStruct(
            id=2,
            vector={"name": [FAR], "lexical": sparse("unrelated entity d")},
            payload={"node_id": "D", "label": "D (unrelated)"},
        ),
    ],
    wait=True,
)
res = client.query_points(
    collection_name=COL_A, query=[QUERY], using="name", limit=5, with_payload=True
)
print("  dense multivector query:")
for h in res.points:
    print(f"    score={h.score:.4f}  {h.payload['label']}")
print("  -> C scores on its best vector, not an average:",
      abs(res.points[0].score - 1.0) < 1e-5)
print("  -> rows returned:", len(res.points), "(one per entity, no duplicates)")

fused = client.query_points(
    collection_name=COL_A,
    prefetch=[
        models.Prefetch(query=[QUERY], using="name", limit=10),
        models.Prefetch(query=sparse("absorbed"), using="lexical", limit=10),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
    with_payload=True,
)
print("  multivector + sparse, fused with RRF:")
for h in fused.points:
    print(f"    score={h.score:.4f}  {h.payload['label']}")
print("  -> multivector coexists with sparse RRF fusion: OK")

# ---------------------------------------------------------------- candidate B
print()
print("=" * 70)
print("B. server-side group_by on a canonical payload field")
print("=" * 70)
COL_B = "probe-groupby"
fresh(COL_B)
client.create_collection(
    collection_name=COL_B,
    vectors_config={"name": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
    sparse_vectors_config={
        "lexical": models.SparseVectorParams(
            index=models.SparseIndexParams(), modifier=models.Modifier.IDF
        )
    },
)
client.create_payload_index(
    collection_name=COL_B,
    field_name="canonical_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True,
)
client.upsert(
    collection_name=COL_B,
    points=[
        models.PointStruct(
            id=1, vector={"name": NEAR, "lexical": sparse("entity a phrasing")},
            payload={"node_id": "A", "canonical_id": "C", "label": "A's point -> C"},
        ),
        models.PointStruct(
            id=2, vector={"name": NEAR, "lexical": sparse("entity c phrasing")},
            payload={"node_id": "C", "canonical_id": "C", "label": "C's own point"},
        ),
        models.PointStruct(
            id=3, vector={"name": FAR, "lexical": sparse("unrelated entity d")},
            payload={"node_id": "D", "canonical_id": "D", "label": "D (unrelated)"},
        ),
    ],
    wait=True,
)
plain = client.query_points(
    collection_name=COL_B, query=NEAR, using="name", limit=5, with_payload=True
)
print("  ungrouped (the duplicate problem):")
for h in plain.points:
    print(f"    score={h.score:.4f}  {h.payload['label']}")

try:
    grouped = client.query_points_groups(
        collection_name=COL_B,
        query=NEAR,
        using="name",
        group_by="canonical_id",
        group_size=1,
        limit=5,
        with_payload=True,
    )
    print("  grouped by canonical_id, group_size=1:")
    for g in grouped.groups:
        top = g.hits[0]
        print(f"    id={g.id!r}  score={top.score:.4f}  {top.payload['label']}")
    print("  -> groups returned:", len(grouped.groups), "(one per canonical entity)")
except Exception as exc:
    print(f"  group_by plain FAILED: {type(exc).__name__}: {exc}")

try:
    grouped_fused = client.query_points_groups(
        collection_name=COL_B,
        prefetch=[
            models.Prefetch(query=NEAR, using="name", limit=10),
            models.Prefetch(query=sparse("phrasing"), using="lexical", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        group_by="canonical_id",
        group_size=1,
        limit=5,
        with_payload=True,
    )
    print("  grouped + hybrid RRF fusion:")
    for g in grouped_fused.groups:
        top = g.hits[0]
        print(f"    id={g.id!r}  score={top.score:.4f}  {top.payload['label']}")
    print("  -> group_by works with fusion: OK")
except Exception as exc:
    print(f"  group_by + fusion FAILED: {type(exc).__name__}: {exc}")

fresh(COL_A)
fresh(COL_B)
print("\nprobe collections deleted")
