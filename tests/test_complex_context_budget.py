from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from retrieval_engine.configuration.strategies.gbc import GBCRAGConfig
from retrieval_engine.rag.gbc.answer import AnswerAgent
from retrieval_engine.rag.gbc.planner import TaskPlanner


class ComplexContextBudgetTests(unittest.TestCase):
    def agent(self, complete, window=None):
        return AnswerAgent(
            llm=SimpleNamespace(
                config=SimpleNamespace(num_ctx=window, max_tokens=100),
                get_completion=complete,
            ),
            vlm=None,
            complex_synthesis_input_tokens=2000,
        )

    def test_default_is_sixteen(self):
        self.assertEqual(16, GBCRAGConfig().planner_max_subquestions)
        self.assertEqual(16, TaskPlanner(None).max_subquestions)

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    def test_fitting_input_needs_no_extra_calls(self, _):
        complete = Mock()
        self.assertEqual(('short', 0), self.agent(complete)._fit_complex_findings('short', lambda x: x))
        complete.assert_not_called()

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    @patch('retrieval_engine.rag.gbc.answer.TextProcessor.split_text_into_chunks')
    def test_all_chunks_are_compacted_before_synthesis(self, split, _):
        split.return_value = ['a' * 1000 + ' [Q1-E1]', 'b' * 1000 + ' [Q2-E1]']
        complete = Mock(side_effect=['fact A [Q1-E1]', 'fact B [Q2-E1]'])
        findings = '\n'.join(split.return_value)
        result, passes = self.agent(complete)._fit_complex_findings(findings, lambda x: x)
        self.assertEqual('fact A [Q1-E1]\n\nfact B [Q2-E1]', result)
        self.assertEqual(1, passes)
        self.assertEqual(2, complete.call_count)
        for call in complete.call_args_list:
            self.assertLessEqual(len(call.args[0]), 2000)

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    @patch('retrieval_engine.rag.gbc.answer.TextProcessor.split_text_into_chunks')
    def test_missing_citation_stops_compaction(self, split, _):
        split.return_value = ['fact [Q1-E1]']
        with self.assertRaisesRegex(ValueError, 'changed evidence labels'):
            self.agent(Mock(return_value='fact'))._fit_complex_findings('x' * 2100, lambda x: x)

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    def test_output_reservation_leaves_no_room(self, _):
        complete = Mock()
        with self.assertRaisesRegex(ValueError, 'no room'):
            self.agent(complete, window=700)._fit_complex_findings('facts', lambda x: x)
        complete.assert_not_called()

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    @patch('retrieval_engine.rag.gbc.answer.TextProcessor.split_text_into_chunks')
    def test_nonshrinking_summary_is_rejected(self, split, _):
        split.return_value = ['x' * 1000, 'x' * 1000]
        with self.assertRaisesRegex(ValueError, 'did not reduce'):
            self.agent(Mock(return_value='x' * 1200))._fit_complex_findings('x' * 2100, lambda x: x)

    @patch('retrieval_engine.rag.gbc.answer.num_tokens', side_effect=len)
    def test_context_failure_returns_explicit_status_without_generation(self, _):
        complete = Mock()
        answer, validation = self.agent(complete, window=700).answer_complex_question_with_validation(
            original_query='Compare the findings',
            sub_question_plan=[],
            sub_question_results=[{'question': 'Find a value', 'answer': '42'}],
        )
        self.assertEqual('context_budget_exceeded', validation['status'])
        self.assertFalse(validation['valid'])
        self.assertIn('context window', answer)
        complete.assert_not_called()
