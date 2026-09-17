"""Retrieval asks each named vector, and keeps which one answered.

A node's readings share a point now, so the dense leg splits into one leg per
vector name and a sparse leg alongside them. The results are fused client-side
rather than by Qdrant: which vector matched is the signal this schema exists to
carry, and a server-side fusion would return one opaque list while the RRF trace
went on claiming an arithmetic it never performed.

Lexical search moves into Qdrant, where it runs over the whole document scope
and can reach a section the selector did not pick -- something the in-process
BM25 could never do, scoring only the subtree it was handed. That leg is kept
for the one case that still needs it: if Qdrant is unreachable, an outage would
otherwise take dense and lexical together and leave no answer at all.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from retrieval_engine.rag.gbc.retrieval import HybridRetrievalBinding, Retriever
from retrieval_engine.indexes.tree import NodeType


def _node(index_id, content, *, node_type=NodeType.TEXT):
    """A node with the canonical evidence bindings the contract requires."""

    return SimpleNamespace(
        index_id=index_id,
        type=node_type,
        meta_info=SimpleNamespace(
            content=content,
            caption=None,
            footnote=None,
            table_body=None,
            canonical_block_id=f"block-{index_id}",
            canonical_revision_id=f"block-{index_id}:r1",
            derived_evidence_id=None,
            source_canonical_revision_ids=None,
            canonical_representation_ids=[f"repr-{index_id}"],
            canonical_asset_ids=[],
            selected_representation_ids=[f"repr-{index_id}"],
            selected_asset_ids=[],
        ),
    )


def _binding():
    return HybridRetrievalBinding(
        tenant_id="tenant-a",
        corpus_id="corpus-a",
        document_id="doc-a",
        document_version_id="docv-a",
        source_sha256="a" * 64,
        canonical_artifact_sha256="b" * 64,
        tree_revision_id="tree-r1",
        dense_index_revision_id="dense-r1",
    )


class _Reranker:
    def rerank(self, query, documents, **_kwargs):
        return [(index, 1.0) for index, _ in enumerate(documents)]


class _Embedder:
    def compute_texts_sim(self, *_args, **_kwargs):
        return 1.0


def _row(node_id, vector_name, distance=0.1):
    return {
        "id": f"record_node_abc#{vector_name}",
        "distance": distance,
        "content": "text",
        "metadata": {
            "node_id": node_id,
            "vector_name": vector_name,
            "modality": "image" if vector_name == "image" else "text",
        },
    }


class _NamedStore:
    """A tree store that answers per vector name."""

    def __init__(self, rows_by_name, *, record=None):
        self.rows_by_name = rows_by_name
        self.calls = record if record is not None else []

    def search_tree_vectors(self, query, *, top_k, lexical_terms=None, **_kwargs):
        self.calls.append({"query": query, "top_k": top_k, "lexical": lexical_terms})
        return self.rows_by_name


def _retriever(store, **kwargs):
    return Retriever(
        variant="wo_graph",
        reranker=_Reranker(),
        embedder=_Embedder(),
        topk=4,
        candidate_pool_size=4,
        tree_vdb=store,
        retrieval_binding=_binding(),
        **kwargs,
    )


def _traces(retriever):
    return {t.leg_id: t for t in retriever.last_fusion_result.leg_traces}


class EachVectorIsItsOwnLeg(unittest.TestCase):
    def test_every_named_vector_becomes_a_leg(self):
        nodes = [_node(1, "alpha"), _node(2, "beta")]
        store = _NamedStore(
            {
                "text": [_row(1, "text")],
                "context": [_row(2, "context")],
                "image": [_row(1, "image")],
                "lexical": [_row(2, "lexical")],
            }
        )
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike surface")

        self.assertEqual(
            {"dense-tree-text", "dense-tree-context", "dense-tree-image",
             "lexical-tree"},
            set(_traces(retriever)),
        )

    def test_the_sparse_leg_is_declared_sparse_not_dense(self):
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")], "lexical": [_row(1, "lexical")]})
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike")
        traces = _traces(retriever)

        self.assertEqual("sparse", traces["lexical-tree"].leg_kind)
        self.assertEqual("qdrant-tree-sparse", traces["lexical-tree"].backend_id)
        self.assertEqual("dense", traces["dense-tree-text"].leg_kind)

    def test_the_query_terms_reach_the_sparse_leg(self):
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")]})
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike surface")

        self.assertTrue(store.calls[0]["lexical"])

    def test_one_round_trip_serves_every_leg(self):
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")], "lexical": [_row(1, "lexical")]})
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike")

        self.assertEqual(1, len(store.calls))

    def test_a_node_hit_on_two_vectors_is_one_candidate(self):
        # dedup_key excludes modality on purpose: the picture and the prose are
        # the same evidence, and fusion should collect both rather than count
        # the node twice.
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")], "image": [_row(1, "image")]})
        retriever = _retriever(store)

        selected = retriever._hybrid_candidates(nodes, "zernike")

        self.assertEqual([1], [n.index_id for n in selected])
        modalities = {
            m
            for candidate in retriever.last_fusion_result.candidates
            for m in candidate.modalities
        }
        self.assertEqual({"text", "image"}, modalities)


class LexicalSearchMovesIntoQdrant(unittest.TestCase):
    def test_the_in_process_bm25_leg_is_not_run_when_qdrant_answers(self):
        # Two lexical legs would double-count keyword evidence in the RRF.
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")], "lexical": [_row(1, "lexical")]})
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike")

        self.assertNotIn("bm25-tree", _traces(retriever))

    def test_bm25_still_runs_when_qdrant_returns_no_lexical_leg(self):
        # An unnamed store has no sparse vector, so the in-process leg is the
        # only lexical evidence there is.
        nodes = [_node(1, "alpha lexical evidence")]
        store = SimpleNamespace(
            search=lambda query_text, top_k: [_row(1, "text")]
        )
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "lexical evidence")

        self.assertIn("bm25-tree", _traces(retriever))

    def test_a_qdrant_outage_still_answers_through_bm25(self):
        def fail(*_args, **_kwargs):
            raise RuntimeError("qdrant unavailable")

        nodes = [_node(1, "resilient lexical evidence")]
        retriever = _retriever(SimpleNamespace(search_tree_vectors=fail))

        selected = retriever._hybrid_candidates(nodes, "resilient lexical evidence")
        traces = _traces(retriever)

        self.assertEqual([1], [n.index_id for n in selected])
        self.assertEqual("success", traces["bm25-tree"].status)

    def test_an_outage_is_visible_on_every_leg_it_took_down(self):
        # A failure the trace does not mention is one nobody can see afterwards.
        def fail(*_args, **_kwargs):
            raise RuntimeError("qdrant unavailable")

        nodes = [_node(1, "resilient lexical evidence")]
        retriever = _retriever(SimpleNamespace(search_tree_vectors=fail))

        retriever._hybrid_candidates(nodes, "resilient lexical evidence")
        traces = _traces(retriever)

        for leg_id in ("dense-tree-text", "dense-tree-image", "lexical-tree"):
            with self.subTest(leg=leg_id):
                self.assertEqual("error", traces[leg_id].status)
                self.assertEqual("backend_error", traces[leg_id].error_code)


class TheLegsStayDistinguishable(unittest.TestCase):
    def test_hits_from_different_vectors_never_share_a_hit_id(self):
        nodes = [_node(1, "alpha")]
        store = _NamedStore({"text": [_row(1, "text")], "image": [_row(1, "image")]})
        retriever = _retriever(store)

        retriever._hybrid_candidates(nodes, "zernike")
        hit_ids = [
            hit.hit_id
            for trace in retriever.last_fusion_result.leg_traces
            for hit in getattr(trace, "eligible_hits", ()) or ()
        ]

        self.assertEqual(len(hit_ids), len(set(hit_ids)))

    def test_the_vector_names_are_part_of_the_retrieval_config(self):
        nodes = [_node(1, "alpha")]
        one = _retriever(_NamedStore({"text": [_row(1, "text")]}))
        both = _retriever(
            _NamedStore({"text": [_row(1, "text")], "image": [_row(1, "image")]})
        )

        one._hybrid_candidates(nodes, "zernike")
        both._hybrid_candidates(nodes, "zernike")

        self.assertNotEqual(
            one.last_fusion_result.scope.retrieval_config_sha256,
            both.last_fusion_result.scope.retrieval_config_sha256,
        )


if __name__ == "__main__":
    unittest.main()
