"""Exercise the hybrid entity schema end to end against the live Qdrant.

Fakes cannot validate a collection schema, a fusion query or server-side
grouping, so the three claims that matter are checked against a throwaway
collection: the semantic win the design exists for, one row per merged entity,
and the gate still refusing an unrelated probe.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/rahul/DocARAG")

from retrieval_engine.indexing.graph.entity_facets import (
    CANONICAL_FIELD,
    context_text,
    lexical_terms,
    lexical_text,
)
from retrieval_engine.providers.embeddings import TextEmbeddingProvider
from retrieval_engine.indexes.vector_store import VectorStore

SCOPE = {
    "tenant_id": "probe-tenant",
    "corpus_id": "probe-corpus",
    "document_version_id": "probe-doc",
    "source_sha256": "0" * 64,
    "index_kind": "entity",
}

ENTITIES = [
    ("Name: large language model\nType: METHOD",
     "a neural network trained on very large text corpora to generate language"),
    ("Name: qwen3 reranker 4b\nType: PRODUCT",
     "a cross-encoder reranking model with four billion parameters"),
    ("Name: chief ray\nType: CONCEPT",
     "the ray passing through the centre of the aperture stop"),
    ("Name: marginal ray\nType: CONCEPT",
     "the ray passing through the edge of the aperture stop"),
]


def record(node_name: str, description: str, canonical: str | None = None):
    return {
        "name_text": node_name,
        "context_text": context_text(node_name, description),
        "lexical_terms": lexical_terms(lexical_text(node_name, description)),
        "metadata": {
            "entity_name": node_name,
            "description": description,
            "node_id": node_name,
            CANONICAL_FIELD: canonical or node_name,
            **SCOPE,
        },
    }


embedder = TextEmbeddingProvider(
    model_name="qwen3-embedding:0.6b",
    backend="ollama",
    api_base="http://localhost:11434",
    max_length=2048,
)
store = VectorStore(
    embedding_model=embedder,
    qdrant_url="http://127.0.0.1:6335",
    collection_name="probe-hybrid-entity-store",
    scope_filter=SCOPE,
    allow_collection_reset=True,
    hybrid_entity_schema=True,
)

try:
    store.reset()
    ids = store.add_entity_points([record(n, d) for n, d in ENTITIES])
    print(f"wrote {len(ids)} entity points\n")

    store.client.create_payload_index(
        collection_name=store.collection_name,
        field_name=CANONICAL_FIELD,
        field_schema=store._models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    def search(name_text: str, description: str, top_k: int = 4):
        return store.search_entities(
            name_text=name_text,
            context_text=context_text(name_text, description),
            lexical_terms=lexical_terms(lexical_text(name_text, description)),
            top_k=top_k,
            group_by=CANONICAL_FIELD,
        )

    print("1. the semantic case name-only search could never reach")
    print("   probe: 'Name: llm\\nType: METHOD'")
    for hit in search("Name: llm\nType: METHOD", "an abbreviation used for LLMs"):
        print(f"     {hit['metadata']['entity_name']!r}")

    print("\n2. the lexical case dense embeddings are worst at")
    print("   probe: 'Name: Qwen3-Reranker-4B\\nType: PRODUCT'")
    for hit in search("Name: Qwen3-Reranker-4B\nType: PRODUCT", "reranking model"):
        print(f"     {hit['metadata']['entity_name']!r}")

    print("\n3. a merged entity returns one row, not two")
    store.set_canonical_id(
        field_name=CANONICAL_FIELD,
        current_value="Name: marginal ray\nType: CONCEPT",
        canonical_value="Name: chief ray\nType: CONCEPT",
    )
    hits = search("Name: chief ray\nType: CONCEPT", "ray through the aperture stop")
    rays = [h for h in hits if "ray" in h["metadata"]["entity_name"]]
    print(f"   ray entities returned: {len(rays)} (both points now resolve to one)")
    for hit in rays:
        print(f"     canonical={hit['metadata'][CANONICAL_FIELD]!r}"
              f"  matched={hit['metadata']['node_id']!r}")

    print("\n4. the gate still refuses something unrelated")
    gated = store.search_entities(
        name_text="Name: photosynthesis\nType: CONCEPT",
        context_text=context_text("Name: photosynthesis\nType: CONCEPT",
                                  "plants converting light into chemical energy"),
        lexical_terms=lexical_terms("photosynthesis chlorophyll plants"),
        top_k=4,
        group_by=CANONICAL_FIELD,
        name_score_threshold=0.8,
        context_score_threshold=0.8,
    )
    print(f"   candidates above the 0.8 floor: {len(gated)} (0 == gate skip)")
finally:
    try:
        store.client.delete_collection(store.collection_name)
        print("\nprobe collection deleted")
    finally:
        store.close()
