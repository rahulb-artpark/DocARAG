from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from Core.Index.Tree import NodeType
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.rag.gbc_plan import Filter, PlanResult
from Core.rag.gbc_rag import GBCRAG
from Core.rag.gbc_utils import GBCRAGContext, filter_tree_nodes, global_scope_chars


def _global_plan(operation: str) -> PlanResult:
    return PlanResult(
        query_type="global",
        original_query="Inspect every table",
        filters=[Filter(filter_type="table")],
        operation=operation,
    )


def _rag(*, limit: int, tree, char_budget: int = 10_000_000) -> GBCRAG:
    rag = object.__new__(GBCRAG)
    # The char budget defaults out of the way so a test that is about the item
    # limit is not silently also testing the size gate.
    rag.cfg = SimpleNamespace(
        global_evidence_limit=limit, global_evidence_char_budget=char_budget
    )
    rag.gbc_index = SimpleNamespace(TreeIndex=tree)
    rag.answer = Mock()
    rag.evidence_hydrator = None
    rag.retrieval_callback = None
    return rag


class GlobalQueryBoundTests(unittest.TestCase):
    def test_limit_is_strictly_validated(self):
        self.assertEqual(512, GBCRAGConfig().global_evidence_limit)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(global_evidence_limit=0)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(global_evidence_limit=4097)

    def test_char_budget_is_strictly_validated(self):
        self.assertEqual(240_000, GBCRAGConfig().global_evidence_char_budget)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(global_evidence_char_budget=999)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(global_evidence_char_budget=4_000_001)

    def test_structural_filter_values_are_strictly_validated(self):
        invalid_filters = (
            {"filter_type": "page"},
            {"filter_type": "page", "filter_value": ""},
            {"filter_type": "page", "filter_value": "0"},
            {"filter_type": "page", "filter_value": "3-2"},
            {"filter_type": "page", "filter_value": "1-2-3"},
            {"filter_type": "section"},
            {"filter_type": "section", "filter_value": "  "},
        )
        for payload in invalid_filters:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                Filter(**payload)

        self.assertEqual("4", Filter(filter_type="page", filter_value="4").filter_value)
        self.assertEqual(
            "4 - 7",
            Filter(filter_type="page", filter_value="4 - 7").filter_value,
        )
        self.assertEqual(
            "Methods",
            Filter(filter_type="section", filter_value="Methods").filter_value,
        )
        for filter_type in ("document", "title", "image", "table", "equation"):
            with self.subTest(filter_type=filter_type):
                self.assertIsNone(Filter(filter_type=filter_type).filter_value)
                with self.assertRaises(ValidationError):
                    Filter(filter_type=filter_type, filter_value="ignored")

    def test_explicit_whole_document_and_structural_type_filters(self):
        nodes = [
            SimpleNamespace(index_id=1, type=NodeType.TITLE),
            SimpleNamespace(index_id=2, type=NodeType.TEXT),
            SimpleNamespace(index_id=3, type=NodeType.EQUATION),
        ]
        tree = SimpleNamespace(get_nodes=Mock(return_value=nodes))

        self.assertEqual(
            nodes,
            filter_tree_nodes(tree, [Filter(filter_type="document")]),
        )
        self.assertEqual(
            [nodes[0]],
            filter_tree_nodes(tree, [Filter(filter_type="title")]),
        )
        self.assertEqual(
            [nodes[2]],
            filter_tree_nodes(tree, [Filter(filter_type="equation")]),
        )

    def test_invalid_structural_filter_cannot_broaden_count_at_execution_boundary(self):
        nodes = [
            SimpleNamespace(
                index_id=index,
                type=NodeType.TABLE,
                meta_info=SimpleNamespace(page_idx=index - 1),
            )
            for index in range(1, 4)
        ]
        tree = SimpleNamespace(
            get_nodes=Mock(return_value=nodes),
            get_nodes_data=Mock(),
        )
        rag = _rag(limit=32, tree=tree)
        invalid_filters = (
            (
                Filter.model_construct(
                    filter_type="page",
                    filter_value="not-a-page",
                ),
                "page filter_value",
            ),
            (
                Filter.model_construct(
                    filter_type="section",
                    filter_value="  ",
                ),
                "section filter_value",
            ),
            (
                Filter.model_construct(
                    filter_type="future-unvalidated-filter",
                    filter_value=None,
                ),
                "unknown global filter type",
            ),
        )
        for invalid_filter, message in invalid_filters:
            with self.subTest(filter_type=invalid_filter.filter_type):
                plan = PlanResult.model_construct(
                    query_type="global",
                    original_query="Count tables in a malformed structural scope",
                    filters=[invalid_filter],
                    operation="COUNT",
                )
                context = GBCRAGContext(query=plan.original_query)

                with self.assertRaisesRegex(ValueError, message):
                    rag.process_analysis(context, plan)

                self.assertEqual("", context.final_answer)
                with self.assertRaisesRegex(ValueError, message):
                    filter_tree_nodes(tree, [invalid_filter])

        tree.get_nodes_data.assert_not_called()
        rag.answer.answer_global_question.assert_not_called()

    def test_count_is_exact_without_hydration_or_model_calls(self):
        nodes = [SimpleNamespace(index_id=index) for index in range(1, 101)]
        tree = SimpleNamespace(get_nodes_data=Mock())
        rag = _rag(limit=2, tree=tree)
        context = GBCRAGContext(query="How many tables?")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("COUNT"))

        self.assertIn("100 items", context.final_answer)
        step = context.iterations[0]
        self.assertEqual(100, step.global_filtered_count)
        self.assertFalse(step.global_budget_exhausted)
        self.assertEqual([], step.retrieval_nodes)
        tree.get_nodes_data.assert_not_called()
        rag.answer.answer_global_question.assert_not_called()

    def test_oversized_global_scope_fails_without_partial_sampling(self):
        nodes = [SimpleNamespace(index_id=index) for index in range(1, 6)]
        tree = SimpleNamespace(get_nodes_data=Mock())
        rag = _rag(limit=3, tree=tree)
        context = GBCRAGContext(query="Summarize every table")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        step = context.iterations[0]
        self.assertTrue(step.global_budget_exhausted)
        self.assertEqual(5, step.global_filtered_count)
        self.assertEqual(3, step.global_evidence_limit)
        self.assertIn("no partial answer was generated", context.final_answer)
        self.assertEqual(context.final_answer, step.generated_answer)
        tree.get_nodes_data.assert_not_called()
        rag.answer.answer_global_question.assert_not_called()

    def test_in_budget_scope_is_hydrated_and_answered(self):
        nodes = [SimpleNamespace(index_id=1), SimpleNamespace(index_id=2)]
        tree_data = [
            {"index_id": 1, "type": NodeType.TEXT, "content": "first"},
            {"index_id": 2, "type": NodeType.TEXT, "content": "second"},
        ]
        tree = SimpleNamespace(get_nodes_data=Mock(return_value=tree_data))
        rag = _rag(limit=2, tree=tree)
        rag.answer.answer_global_question.return_value = (
            "bounded answer",
            [
                {
                    "answer_citation_validation": {
                        "status": "not_required",
                        "valid": True,
                    }
                }
            ],
            [],
        )
        context = GBCRAGContext(query="Summarize every table")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        self.assertEqual("bounded answer", context.final_answer)
        step = context.iterations[0]
        self.assertFalse(step.global_budget_exhausted)
        self.assertEqual(tree_data, step.retrieval_nodes)
        self.assertEqual("not_required", context.answer_validation["status"])
        tree.get_nodes_data.assert_called_once_with([1, 2])
        rag.answer.answer_global_question.assert_called_once()

class TheBudgetMeasuresSizeNotPieceCount(unittest.TestCase):
    """A whole-document question about a short document is a fair question.

    "what are all the defects in vision and causes" matched 43 of the 44 nodes in
    a physics chapter -- essentially the whole chapter -- and was refused for
    exceeding a 32-item limit. That entire scope is about 5,200 tokens, which any
    answer model reads without effort. Node count is a poor proxy for cost in both
    directions: it refused that, and would have admitted a scope of half as many
    but far larger nodes.
    """

    @staticmethod
    def _nodes(count, chars_each):
        return [
            SimpleNamespace(
                index_id=index,
                summary="",
                meta_info=SimpleNamespace(
                    content="x" * chars_each, caption=None, table_body=None
                ),
            )
            for index in range(1, count + 1)
        ]

    def test_many_small_nodes_are_answered(self):
        # The reported case: lots of pieces, very little text.
        nodes = self._nodes(43, 500)
        tree = SimpleNamespace(get_nodes_data=Mock(return_value=[]))
        rag = _rag(limit=512, tree=tree, char_budget=240_000)
        context = GBCRAGContext(query="what are all the defects in vision and causes")
        rag.answer.answer_global_question.return_value = (
            "answer",
            [{"answer_citation_validation": {"status": "not_required", "valid": True}}],
            [],
        )

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        self.assertFalse(context.iterations[0].global_budget_exhausted)
        tree.get_nodes_data.assert_called_once()

    def test_few_but_enormous_nodes_are_refused(self):
        # The direction the old rule missed: well under any item limit, far over
        # what can be read in one request.
        nodes = self._nodes(4, 100_000)
        tree = SimpleNamespace(get_nodes_data=Mock())
        rag = _rag(limit=512, tree=tree, char_budget=240_000)
        context = GBCRAGContext(query="Summarize everything")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        step = context.iterations[0]
        self.assertTrue(step.global_budget_exhausted)
        self.assertIn("characters of evidence", context.final_answer)
        self.assertIn("no partial answer was generated", context.final_answer)
        tree.get_nodes_data.assert_not_called()
        rag.answer.answer_global_question.assert_not_called()

    def test_the_refusal_still_names_an_item_overflow_as_items(self):
        nodes = self._nodes(9, 10)
        tree = SimpleNamespace(get_nodes_data=Mock())
        rag = _rag(limit=3, tree=tree, char_budget=10_000_000)
        context = GBCRAGContext(query="Summarize everything")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        self.assertIn("9 evidence items", context.final_answer)

    def test_the_trace_records_both_measurements(self):
        nodes = self._nodes(4, 100_000)
        tree = SimpleNamespace(get_nodes_data=Mock())
        rag = _rag(limit=512, tree=tree, char_budget=240_000)
        context = GBCRAGContext(query="Summarize everything")

        with patch("Core.rag.gbc_rag.filter_tree_nodes", return_value=nodes):
            rag.process_analysis(context, _global_plan("SUMMARIZE"))

        partial = context.iterations[0].partial_answers[0]
        self.assertEqual(partial["matched_count"], 4)
        self.assertEqual(partial["matched_chars"], 400_000)
        self.assertEqual(partial["char_budget"], 240_000)


class ScopeSizeIsMeasuredFromWhatIsSent(unittest.TestCase):
    def test_an_empty_scope_costs_nothing(self):
        self.assertEqual(global_scope_chars([]), 0)

    def test_every_text_channel_counts(self):
        node = SimpleNamespace(
            summary="s" * 5,
            meta_info=SimpleNamespace(content="c" * 10, caption="p" * 3, table_body="t" * 7),
        )
        self.assertEqual(global_scope_chars([node]), 25)

    def test_a_node_with_no_text_does_not_raise(self):
        node = SimpleNamespace(summary=None, meta_info=None)
        self.assertEqual(global_scope_chars([node]), 0)


if __name__ == "__main__":
    unittest.main()
