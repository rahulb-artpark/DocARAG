from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.rag.strategies.gbc_vanilla import GBCVanillaRAG
from retrieval_engine.rag.strategies.graph import GraphRAG
from retrieval_engine.rag.grounding import NO_RETRIEVED_EVIDENCE_ANSWER
from retrieval_engine.rag.strategies.multimodal import MMRAG
from retrieval_engine.rag.strategies.traverse import TraverseAgent
from retrieval_engine.rag.strategies.vanilla import VanillaRAG


class EmptyRetrievalGroundingTests(unittest.TestCase):
    def test_vanilla_does_not_call_llm_without_retrieved_evidence(self):
        rag = object.__new__(VanillaRAG)
        rag.cfg = SimpleNamespace(retrieval_method="vector")
        rag.topk = 3
        rag.vdb = SimpleNamespace(search=Mock(return_value=[]))
        rag.llm = Mock()

        answer, ids = rag.generation("question", Path("unused"))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()

    def test_vanilla_rejects_nonempty_but_blank_or_malformed_evidence(self):
        rag = object.__new__(VanillaRAG)
        rag.cfg = SimpleNamespace(retrieval_method="vector")
        rag.topk = 3
        rag.vdb = SimpleNamespace(
            search=Mock(
                return_value=[
                    {"id": "blank", "content": " \n\t", "metadata": {}},
                    {"id": "missing-content", "metadata": {}},
                    "not-a-retrieval-record",
                ]
            )
        )
        rag.llm = Mock()

        answer, ids = rag.generation("question", Path("unused"))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()

    def test_multimodal_does_not_call_models_without_retrieved_evidence(self):
        rag = object.__new__(MMRAG)
        rag.topk = 3
        rag.vdb = SimpleNamespace(search=Mock(return_value=[]))
        rag.llm = Mock()
        rag.vlm = Mock()

        answer, ids = rag.generation("question", Path("unused"))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()
        rag.vlm.generate.assert_not_called()

    def test_gbc_vanilla_does_not_call_llm_without_any_leg(self):
        rag = object.__new__(GBCVanillaRAG)
        rag.topk = 3
        rag.topk_ent = 3
        rag.tree_vdb = SimpleNamespace(search=Mock(return_value=[]))
        rag.graph_vdb = SimpleNamespace(search=Mock(return_value=[]))
        rag.llm = Mock()

        answer, ids = rag.generation("question", Path("unused"))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()

    def test_traverse_does_not_call_models_without_tree_nodes(self):
        rag = object.__new__(TraverseAgent)
        rag._retrieve = Mock(return_value=[])
        rag.llm = Mock()
        rag.vlm = Mock()

        answer, ids = rag.generation("question", Path("unused"))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()
        rag.vlm.generate.assert_not_called()

    def test_graph_rag_requires_source_tree_evidence(self):
        rag = object.__new__(GraphRAG)
        rag.max_retry = 1
        rag._retrieve = Mock(
            return_value={"TreeNode_ids": [], "EntNode_name": ["entity"]}
        )
        rag.llm = Mock()
        rag.vlm = Mock()

        with tempfile.TemporaryDirectory() as directory:
            answer, ids = rag.generation("question", Path(directory))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()
        rag.vlm.generate.assert_not_called()

    def test_graph_rag_rejects_stale_tree_links_before_generation(self):
        tree = SimpleNamespace(
            get_nodes_data=Mock(side_effect=KeyError("stale tree node 999"))
        )
        rag = object.__new__(GraphRAG)
        rag.max_retry = 1
        rag.gbc_index = SimpleNamespace(TreeIndex=tree)
        rag._retrieve = Mock(
            return_value={"TreeNode_ids": [999], "EntNode_name": ["entity"]}
        )
        rag.llm = Mock()
        rag.vlm = Mock()

        with tempfile.TemporaryDirectory() as directory:
            answer, ids = rag.generation("question", Path(directory))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        tree.get_nodes_data.assert_called_once_with([999])
        rag.llm.get_completion.assert_not_called()
        rag.vlm.generate.assert_not_called()

    def test_graph_rag_rejects_nonempty_but_unusable_tree_rows(self):
        tree_rows = [
            {"index_id": 1, "type": NodeType.TEXT, "content": "  "},
            {
                "index_id": 2,
                "type": NodeType.IMAGE,
                "content": "",
                "img_path": "/missing/diagram.png",
            },
            {"index_id": 3, "type": NodeType.TABLE, "table_body": ""},
        ]
        tree = SimpleNamespace(get_nodes_data=Mock(return_value=tree_rows))
        rag = object.__new__(GraphRAG)
        rag.max_retry = 1
        rag.gbc_index = SimpleNamespace(TreeIndex=tree)
        rag._retrieve = Mock(
            return_value={"TreeNode_ids": [1, 2, 3], "EntNode_name": ["entity"]}
        )
        rag.llm = Mock()
        rag.vlm = Mock()

        with tempfile.TemporaryDirectory() as directory:
            answer, ids = rag.generation("question", Path(directory))

        self.assertEqual(NO_RETRIEVED_EVIDENCE_ANSWER, answer)
        self.assertEqual([], ids)
        rag.llm.get_completion.assert_not_called()
        rag.vlm.generate.assert_not_called()

    def test_graph_entity_mapping_handles_an_empty_entity_index(self):
        search = Mock(return_value=[])
        rag = object.__new__(GraphRAG)
        rag.threshold_e = 0.3
        rag.gbc_index = SimpleNamespace(
            entity_vdb=SimpleNamespace(search_entities=search),
            GraphIndex=Mock(),
        )
        entity = SimpleNamespace(entity_name="missing", entity_type="PRODUCT")

        result = rag._entity_map([entity])

        self.assertEqual({}, dict(result))
        rag.gbc_index.GraphIndex.get_node_name_from_str.assert_not_called()

    def test_graph_entity_mapping_gates_on_similarity_not_on_fused_rank(self):
        # Fusion scores rank candidates against each other, so the nearest one
        # scores well even when the document contains nothing related. The
        # threshold has to reach the vectors as a floor or it rejects nothing.
        search = Mock(return_value=[])
        rag = object.__new__(GraphRAG)
        rag.threshold_e = 0.3
        rag.gbc_index = SimpleNamespace(
            entity_vdb=SimpleNamespace(search_entities=search),
            GraphIndex=Mock(),
        )

        rag._entity_map([SimpleNamespace(entity_name="a", entity_type="B")])

        floor = search.call_args.kwargs
        self.assertAlmostEqual(0.7, floor["name_score_threshold"])
        self.assertAlmostEqual(0.7, floor["context_score_threshold"])


if __name__ == "__main__":
    unittest.main()
