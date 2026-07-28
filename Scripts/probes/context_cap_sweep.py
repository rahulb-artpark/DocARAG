"""Does the 120-word context cap help, hurt, or do nothing?

The cap has a real justification -- both producers must cut identically or the
same entity gets different vectors -- but 120 itself was a guess sitting above
the p90 of observed descriptions. This measures what the number actually buys,
on real entities from the 1,871-page run, including the long merged
descriptions where the cap is the only thing that bites.

Two things matter and they pull against each other:
  recall        an alias must still retrieve its entity
  discrimination near-identical entities must stay apart, which is what a
                blurred, everything-matches vector destroys
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rahul/DocARAG")

from Core.pipelines.entity_facets import CANONICAL_FIELD, lexical_terms, lexical_text
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore

GRAPH = Path(
    "/home/rahul/DocARAG/runs/doc_df198aeb160bb6240f5843b978d6fbcc/"
    "ee3d1f8d2bbfb5aef8ee657346835d15264b666647c4a757954f29e36f2de245/graph_data.json"
)
SCOPE = {
    "tenant_id": "p", "corpus_id": "p", "document_version_id": "d",
    "source_sha256": "0" * 64, "index_kind": "entity",
}
CAPS = [60, 120, 300, None]


def capped_context(node_name: str, description: str, cap: int | None) -> str:
    words = (description or "").split()
    if cap is not None and len(words) > cap:
        words = words[:cap]
    summary = " ".join(words)
    return f"{node_name}\nDescription: {summary}" if summary else node_name


# Entities whose descriptions are long enough for the cap to matter, plus the
# near-identical pairs that discrimination is measured on.
PROBES = [
    ("Name: llm\nType: METHOD", "abbreviation for large language model",
     "large language model"),
    ("Name: Qwen3-Reranker-4B\nType: PRODUCT", "a reranking model",
     "qwen3 reranker 4b"),
    ("Name: chief ray\nType: CONCEPT",
     "the ray from an off-axis object point that passes through the centre of "
     "the aperture stop", "chief ray"),
    ("Name: marginal ray\nType: CONCEPT",
     "the ray from an on-axis object point that passes through the edge of the "
     "aperture stop", "marginal ray"),
]


def main() -> None:
    nodes = json.loads(GRAPH.read_text())["graph"]["nodes"]
    described = [n for n in nodes if n.get("entity_name") and n.get("description")]
    described.sort(key=lambda n: -len(n["description"]))
    # The longest real descriptions are exactly where a cap changes the vector.
    corpus = described[:60]
    lengths = sorted(len(n["description"].split()) for n in corpus)
    print(f"corpus: {len(corpus)} entities, description words "
          f"median={lengths[len(lengths)//2]} max={lengths[-1]}")

    seeded = [
        ("Name: large language model\nType: METHOD",
         "a neural network trained on very large text corpora to generate text"),
        ("Name: qwen3 reranker 4b\nType: PRODUCT",
         "a cross-encoder reranking model with four billion parameters"),
        ("Name: chief ray\nType: CONCEPT",
         "the ray passing through the centre of the aperture stop from an "
         "off-axis point in the object field"),
        ("Name: marginal ray\nType: CONCEPT",
         "the ray passing through the edge of the aperture stop from an "
         "on-axis point in the object field"),
    ] + [(f"Name: {n['entity_name']}\nType: {n['entity_type']}", n["description"])
         for n in corpus]

    embedder = TextEmbeddingProvider(
        model_name="qwen3-embedding:0.6b", backend="ollama",
        api_base="http://localhost:11434", max_length=2048,
    )
    print(f"\n{'cap':>8} | {'alias recall':>12} | {'mean margin':>11} | "
          f"{'worst margin':>12} | {'ray pair separated':>18}")
    print("-" * 78)

    for cap in CAPS:
        store = VectorStore(
            embedding_model=embedder, qdrant_url="http://127.0.0.1:6335",
            collection_name="probe-cap-sweep", scope_filter=SCOPE,
            allow_collection_reset=True, hybrid_entity_schema=True,
        )
        try:
            store.reset()
            store.add_entity_points([
                {
                    "name_text": name,
                    "context_text": capped_context(name, desc, cap),
                    "lexical_terms": lexical_terms(lexical_text(name, desc)),
                    "metadata": {"entity_name": name, "description": desc,
                                 "node_id": name, CANONICAL_FIELD: name, **SCOPE},
                }
                for name, desc in seeded
            ])

            hits_correct = 0
            margins = []
            for probe_name, probe_desc, expected in PROBES:
                results = store.search_entities(
                    name_text=probe_name,
                    context_text=capped_context(probe_name, probe_desc, cap),
                    lexical_terms=lexical_terms(
                        lexical_text(probe_name, probe_desc)
                    ),
                    top_k=5, group_by=CANONICAL_FIELD,
                )
                if not results:
                    margins.append(-1.0)
                    continue
                names = [r["metadata"]["entity_name"].splitlines()[0][6:]
                         for r in results]
                if names[0] == expected:
                    hits_correct += 1
                # Fused distance: lower is better, so the margin to the
                # runner-up is how confidently the right entity won.
                top = results[0]["distance"]
                second = results[1]["distance"] if len(results) > 1 else top + 1.0
                margins.append(second - top if names[0] == expected else -(second - top))

            # The pair whose descriptions are near-identical: if one retrieves
            # the other first, the context vector has blurred them together.
            ray_ok = margins[2] > 0 and margins[3] > 0
            print(f"{str(cap):>8} | {hits_correct}/{len(PROBES):<10} | "
                  f"{sum(margins)/len(margins):>11.4f} | {min(margins):>12.4f} | "
                  f"{'yes' if ray_ok else 'NO':>18}")
        finally:
            try:
                store.client.delete_collection(store.collection_name)
            finally:
                store.close()


if __name__ == "__main__":
    main()
