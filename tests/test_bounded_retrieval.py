from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
from pydantic import ValidationError

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.configuration.strategies.gbc import GBCRAGConfig
from retrieval_engine.rag.gbc.agent import GBCRAG
from retrieval_engine.rag.gbc.retrieval import HybridRetrievalBinding, Retriever
from retrieval_engine.rag.gbc.context import calculate_skyline


class _RecordingReranker:
    max_length = 2048

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.cache_clean_count = 0

    def rerank(self, *, query, documents, instruction):
        self.document_batches.append(list(documents))
        return [float(index) for index in range(len(documents))]

    def clean_cache(self) -> None:
        self.cache_clean_count += 1


class _ResultReranker(_RecordingReranker):
    def __init__(self, result=None, error=None, *, max_length=2048) -> None:
        super().__init__()
        self.result = result
        self.error = error
        self.max_length = max_length

    def rerank(self, *, query, documents, instruction):
        self.document_batches.append(list(documents))
        if self.error is not None:
            raise self.error
        return self.result


class _Embedder:
    def compute_texts_sim(self, _left, _right):
        return 1.0


def _text_node(index_id: int):
    return SimpleNamespace(
        index_id=index_id,
        type=NodeType.TEXT,
        meta_info=SimpleNamespace(content=f"document node {index_id}"),
    )


def _canonical_node(
    index_id: int,
    content: str,
    *,
    node_type: NodeType = NodeType.TEXT,
    assets: tuple[str, ...] = (),
):
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
            canonical_asset_ids=list(assets),
            selected_representation_ids=[f"repr-{index_id}"],
            selected_asset_ids=list(assets),
        ),
    )


def _binding() -> HybridRetrievalBinding:
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


class SkylineBoundTests(unittest.TestCase):
    def test_anti_correlated_frontier_has_deterministic_hard_cap(self):
        scores = {
            node_id: [float(node_id), float(9 - node_id)]
            for node_id in reversed(range(10))
        }

        skyline = calculate_skyline(scores, max_results=4)

        self.assertEqual([9, 8, 7, 6], [node["node_id"] for node in skyline])
        self.assertEqual(4, len(skyline))

    def test_equal_scores_are_exact_and_independent_of_insertion_order(self):
        first = {3: [1.0, 1.0], 4: [0.0, 0.0], 1: [1.0, 1.0], 2: [1.0, 1.0]}
        second = dict(reversed(list(first.items())))

        first_ids = [node["node_id"] for node in calculate_skyline(first)]
        second_ids = [node["node_id"] for node in calculate_skyline(second)]

        self.assertEqual([1, 2, 3], first_ids)
        self.assertEqual(first_ids, second_ids)

    def test_standard_fallback_fills_exact_topk_in_stable_order(self):
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=4,
            candidate_pool_size=4,
        )
        retriever.graph_reranker = lambda *_args: (
            [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)],
            [],
        )
        retriever.text_reranker = lambda *_args: [
            (1, 0.95),
            (2, 0.8),
            (3, 0.7),
            (4, 0.6),
        ]

        first_ids, _ = retriever.skyline_filter("query", [_text_node(1)], nx.Graph(), {})
        second_ids, _ = retriever.skyline_filter("query", [_text_node(1)], nx.Graph(), {})

        self.assertEqual([1, 2, 3, 4], first_ids)
        self.assertEqual(first_ids, second_ids)

    def test_confidence_filter_drops_a_real_score_cliff(self):
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=10,
        )

        ranked = [(1, 0.95), (2, 0.9), (3, 0.88), (4, 0.1), (5, 0.05)]

        self.assertEqual(
            [1, 2, 3],
            [node_id for node_id, _ in retriever._confidence_filtered(ranked)],
        )

    def test_confidence_filter_keeps_a_closely_scored_cluster_up_to_topk(self):
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=10,
        )

        ranked = [(i, 0.9 - i * 0.01) for i in range(10)]

        self.assertEqual(10, len(retriever._confidence_filtered(ranked)))

    def test_confidence_filter_never_drops_below_topk_min(self):
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=10,
            topk_min=3,
        )

        ranked = [(1, 0.5), (2, 0.05), (3, 0.04), (4, 0.03)]

        self.assertEqual(
            [1, 2, 3],
            [node_id for node_id, _ in retriever._confidence_filtered(ranked)],
        )

    def test_confidence_filter_never_exceeds_the_unfiltered_selection(self):
        # An adaptive floor must only ever shrink what a fixed cutoff would
        # have returned -- it can never surface a candidate topk excluded.
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=3,
        )

        ranked = [(i, 1.0) for i in range(10)]  # every score identical

        kept = [node_id for node_id, _ in retriever._confidence_filtered(ranked)]
        self.assertEqual([0, 1, 2], kept)

    def test_wo_graph_variant_applies_the_confidence_floor(self):
        retriever = Retriever(
            variant="wo_graph",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=5,
            topk_min=2,
        )
        retriever.text_reranker = lambda *_args: [
            (1, 0.9),
            (2, 0.85),
            (3, 0.1),
            (4, 0.05),
        ]

        tree_node_ids, _ = retriever.skyline_filter(
            "query", [_text_node(1)], nx.Graph(), {}
        )

        self.assertEqual([1, 2], tree_node_ids)

    def test_fallback_padding_below_the_floor_is_dropped_even_under_topk(self):
        # The skyline alone is one dominant node; the rest is quota padding.
        # A genuinely irrelevant padding candidate must not ride along just
        # to hit topk -- topk is a ceiling, not a promise of that many nodes.
        retriever = Retriever(
            variant="standard",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=4,
            topk_min=1,
            candidate_pool_size=4,
        )
        retriever.graph_reranker = lambda *_args: (
            [(1, 0.9), (2, 0.5), (3, 0.4), (4, 0.3)],
            [],
        )
        retriever.text_reranker = lambda *_args: [
            (1, 0.95),
            (2, 0.5),
            (3, 0.4),
            (4, 0.02),
        ]

        tree_node_ids, _ = retriever.skyline_filter(
            "query", [_text_node(1)], nx.Graph(), {}
        )

        self.assertIn(1, tree_node_ids)
        self.assertNotIn(4, tree_node_ids)


class CandidatePoolTests(unittest.TestCase):
    def test_hybrid_candidate_contract_fuses_visual_dense_and_bm25(self):
        nodes = [
            _canonical_node(1, "unique quasar requirements"),
            _canonical_node(
                2,
                "approval workflow",
                node_type=NodeType.IMAGE,
                assets=("asset-2",),
            ),
        ]
        tree_vdb = SimpleNamespace(
            search=lambda **_kwargs: [
                {
                    "id": "record-image-2",
                    "distance": 0.05,
                    "content": "/immutable/workflow.png",
                    "metadata": {"node_id": 2, "modality": "image"},
                }
            ]
        )
        retriever = Retriever(
            variant="wo_graph",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=2,
            candidate_pool_size=2,
            tree_vdb=tree_vdb,
            retrieval_binding=_binding(),
        )

        candidates = retriever._hybrid_candidates(nodes, "unique quasar")

        self.assertEqual({1, 2}, {node.index_id for node in candidates})
        fusion = retriever.last_fusion_result
        self.assertIsNotNone(fusion)
        traces = {trace.leg_id: trace for trace in fusion.leg_traces}
        self.assertEqual("success", traces["dense-tree"].status)
        self.assertEqual("success", traces["bm25-tree"].status)
        self.assertIn(
            "image",
            {
                modality
                for candidate in fusion.candidates
                for modality in candidate.modalities
            },
        )
        self.assertIn(
            "asset-2",
            {
                asset
                for candidate in fusion.candidates
                for asset in candidate.selected_asset_ids
            },
        )
        rag = object.__new__(GBCRAG)
        rag.retriever = retriever
        tree_data = [{"index_id": 1}, {"index_id": 2}]
        rag._attach_candidate_fusion_trace(tree_data)
        traced = [item for item in tree_data if "candidate_fusion" in item]
        self.assertEqual(2, len(traced))
        self.assertTrue(
            all(item["candidate_fusion"]["fusion_id"] == fusion.fusion_id for item in traced)
        )

    def test_dense_failure_is_traced_and_bm25_still_retrieves(self):
        node = _canonical_node(1, "resilient lexical evidence")

        def fail_search(**_kwargs):
            raise RuntimeError("qdrant unavailable")

        retriever = Retriever(
            variant="wo_graph",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=1,
            candidate_pool_size=1,
            tree_vdb=SimpleNamespace(search=fail_search),
            retrieval_binding=_binding(),
        )

        candidates = retriever._hybrid_candidates(
            [node], "resilient lexical evidence"
        )

        self.assertEqual([1], [item.index_id for item in candidates])
        traces = {
            trace.leg_id: trace for trace in retriever.last_fusion_result.leg_traces
        }
        self.assertEqual("error", traces["dense-tree"].status)
        self.assertEqual("backend_error", traces["dense-tree"].error_code)
        self.assertEqual("success", traces["bm25-tree"].status)

    def test_legacy_tree_uses_bounded_compatibility_fusion(self):
        retriever = Retriever(
            variant="wo_graph",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=1,
            candidate_pool_size=1,
            tree_vdb=SimpleNamespace(
                search=lambda **_kwargs: [
                    {
                        "id": "legacy-hit",
                        "distance": 0.1,
                        "content": "legacy",
                        "metadata": {"node_id": 3, "modality": "text"},
                    }
                ]
            ),
        )
        node = _text_node(3)

        candidates = retriever._hybrid_candidates([node], "document")

        self.assertEqual([3], [item.index_id for item in candidates])
        self.assertIsNone(retriever.last_fusion_result)

    def test_text_reranker_receives_at_most_candidate_pool_size(self):
        reranker = _RecordingReranker()
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=3,
            candidate_pool_size=5,
        )

        selected_ids, _ = retriever.skyline_filter(
            "bounded query",
            [_text_node(index) for index in range(20)],
            nx.Graph(),
            {},
        )

        self.assertEqual(1, len(reranker.document_batches))
        self.assertEqual(5, len(reranker.document_batches[0]))
        self.assertEqual(3, len(selected_ids))
        self.assertGreater(max(selected_ids), 4)
        self.assertEqual(1, reranker.cache_clean_count)

    def test_tail_lexical_match_survives_bounded_preselection(self):
        reranker = _RecordingReranker()
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=2,
            candidate_pool_size=4,
        )
        nodes = [_text_node(index) for index in range(20)]
        nodes[-1].meta_info.content = "the unique quasar specification"

        candidates = retriever._bounded_text_candidates(nodes, "unique quasar")

        self.assertEqual(4, len(candidates))
        self.assertIn(19, [node.index_id for node in candidates])
        self.assertEqual(
            [node.index_id for node in candidates],
            [
                node.index_id
                for node in retriever._bounded_text_candidates(
                    list(nodes), "unique quasar"
                )
            ],
        )

    def test_empty_visual_content_is_safe_and_uses_caption(self):
        node = SimpleNamespace(
            index_id=8,
            type=NodeType.IMAGE,
            meta_info=SimpleNamespace(
                content=None,
                caption="workflow approval diagram",
                footnote=None,
            ),
        )

        self.assertEqual(
            "workflow approval diagram", Retriever._node_rerank_text(node)
        )

    def test_candidate_pool_is_explicit_and_validated(self):
        config = GBCRAGConfig()
        self.assertEqual(512, config.candidate_pool_size)

        with self.assertRaises(ValidationError):
            GBCRAGConfig(topk=10, candidate_pool_size=9)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(candidate_pool_size=4097)

    def test_graph_only_drops_nodes_without_source_evidence(self):
        graph = nx.Graph()
        graph.add_node("entity-a", source_ids={1})
        graph.add_node("entity-b", source_ids=set())
        retriever = Retriever(
            variant="wo_text",
            reranker=_RecordingReranker(),
            embedder=_Embedder(),
            topk=2,
            candidate_pool_size=2,
        )

        with (
            patch(
                "retrieval_engine.rag.gbc.retrieval.enhance_graph_with_semantic_links",
                return_value=graph,
            ),
            patch(
                "retrieval_engine.rag.gbc.retrieval.nx.pagerank",
                return_value={"entity-a": 0.6, "entity-b": 0.4},
            ),
        ):
            ranked, _ = retriever.graph_reranker(
                graph,
                {"question entity": ["entity-a"]},
                [_text_node(1), _text_node(2)],
            )

        self.assertEqual([(1, 0.6)], ranked)

    def test_text_reranker_rejects_score_cardinality_mismatch_and_cleans_cache(self):
        reranker = _ResultReranker([0.5])
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=2,
            candidate_pool_size=2,
        )

        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            retriever.text_reranker([_text_node(1), _text_node(2)], "query")

        self.assertEqual(1, reranker.cache_clean_count)

    def test_text_reranker_rejects_unsized_results_and_cleans_cache(self):
        reranker = _ResultReranker(None)
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=1,
            candidate_pool_size=1,
        )

        with self.assertRaisesRegex(ValueError, "sized sequence"):
            retriever.text_reranker([_text_node(1)], "query")

        self.assertEqual(1, reranker.cache_clean_count)

    def test_text_reranker_rejects_nonfinite_and_nonnumeric_scores(self):
        for invalid_score in (float("nan"), float("inf"), "0.5", True):
            with self.subTest(invalid_score=invalid_score):
                reranker = _ResultReranker([invalid_score])
                retriever = Retriever(
                    variant="wo_graph",
                    reranker=reranker,
                    embedder=_Embedder(),
                    topk=1,
                    candidate_pool_size=1,
                )

                with self.assertRaisesRegex(ValueError, "non-finite numeric score"):
                    retriever.text_reranker([_text_node(1)], "query")

                self.assertEqual(1, reranker.cache_clean_count)

    def test_text_reranker_cleans_cache_when_backend_raises(self):
        reranker = _ResultReranker(error=RuntimeError("backend unavailable"))
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=1,
            candidate_pool_size=1,
        )

        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            retriever.text_reranker([_text_node(1)], "query")

        self.assertEqual(1, reranker.cache_clean_count)

    def test_text_reranker_rejects_invalid_effective_token_length(self):
        for max_length in (None, True, 1000, 999):
            with self.subTest(max_length=max_length):
                reranker = _ResultReranker([0.5], max_length=max_length)
                retriever = Retriever(
                    variant="wo_graph",
                    reranker=reranker,
                    embedder=_Embedder(),
                    topk=1,
                    candidate_pool_size=1,
                )

                with self.assertRaisesRegex(ValueError, "reranker.max_length"):
                    retriever.text_reranker([_text_node(1)], "query")

                self.assertEqual([], reranker.document_batches)

    def test_text_reranker_empty_candidates_is_a_noop(self):
        reranker = _ResultReranker([], max_length=None)
        retriever = Retriever(
            variant="wo_graph",
            reranker=reranker,
            embedder=_Embedder(),
            topk=1,
            candidate_pool_size=1,
        )

        self.assertEqual([], retriever.text_reranker([], "query"))
        self.assertEqual([], reranker.document_batches)
        self.assertEqual(0, reranker.cache_clean_count)


if __name__ == "__main__":
    unittest.main()
