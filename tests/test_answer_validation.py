from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from Core.Index.Tree import NodeType
from Core.rag.gbc_answer import (
    CITATION_INSTRUCTION,
    CITATION_VALIDATION_FAILURE_ANSWER,
    AnswerAgent,
)
from Core.rag.gbc_plan import PlanResult, SubQuestion
from Core.rag.gbc_rag import GBCRAG
from Core.rag.gbc_utils import GBCRAGContext
from docarag.answer_validation import validate_answer_citations


def _node():
    return {
        "index_id": 1,
        "type": NodeType.TEXT,
        "page": 0,
        "content": "The verified value is 42.",
        "document_version_id": "document-v1",
        "canonical_revision_id": "revision-v1",
        "citation": {"citation_id": "citation_" + "a" * 32},
    }


def _agent(answer: str, captured: list):
    def complete(memory):
        captured.append(memory)
        return answer

    return AnswerAgent(
        llm=SimpleNamespace(
            config=SimpleNamespace(max_tokens=2000),
            get_completion=complete,
        ),
        vlm=SimpleNamespace(config=SimpleNamespace(max_tokens=2000)),
    )


class AnswerCitationValidationTests(unittest.TestCase):
    def test_structural_validator_rejects_missing_and_unknown_labels(self):
        missing = validate_answer_citations(
            "The value is 42.", {"E1": "citation_" + "a" * 32}, required=True
        )
        unknown = validate_answer_citations(
            "The value is 42 [E2].",
            {"E1": "citation_" + "a" * 32},
            required=True,
        )

        self.assertEqual("missing_citations", missing.status)
        self.assertFalse(missing.valid)
        self.assertEqual("unknown_citations", unknown.status)
        self.assertEqual(("E2",), unknown.unknown_labels)

    def test_structural_validator_rejects_compound_ranges_and_malformed_markers(self):
        bindings = {
            "E1": "citation_" + "a" * 32,
            "E2": "citation_" + "b" * 32,
            "Q1-E1": "citation_" + "c" * 32,
            "Q2-E1": "citation_" + "d" * 32,
        }
        markers = (
            "[E1, E2]",
            "[E1-E2]",
            "[Q1-E1, Q2-E1]",
            "[ E1 ]",
            "[E0]",
        )

        for marker in markers:
            with self.subTest(marker=marker):
                validation = validate_answer_citations(
                    f"Unsupported reference {marker}",
                    bindings,
                    required=True,
                )
                self.assertEqual("malformed_citations", validation.status)
                self.assertFalse(validation.valid)
                self.assertEqual((marker,), validation.malformed_markers)

    def test_structural_validator_rejects_citation_only_output(self):
        validation = validate_answer_citations(
            "- [E1]\n- [E2]",
            {
                "E1": "citation_" + "a" * 32,
                "E2": "citation_" + "b" * 32,
            },
            required=True,
        )

        self.assertEqual("citation_only_answer", validation.status)
        self.assertTrue(validation.citation_only)
        self.assertFalse(validation.valid)

    def test_structural_validator_rejects_raw_citation_id_but_not_ordinary_brackets(self):
        citation_id = "citation_" + "a" * 32
        raw_id = validate_answer_citations(
            f"The value is 42 {citation_id}.",
            {"E1": citation_id},
            required=True,
        )
        ordinary_brackets = validate_answer_citations(
            "Table metadata [Table 2] reports 42 [E1].",
            {"E1": citation_id},
            required=True,
        )

        self.assertEqual("malformed_citations", raw_id.status)
        self.assertEqual((citation_id,), raw_id.malformed_markers)
        self.assertEqual("verified", ordinary_brackets.status)

    def test_strict_answer_accepts_only_supplied_evidence_label(self):
        captured = []
        agent = _agent("The value is 42 [E1].", captured)

        answer, partials = agent.answer_simple_question(
            "What is the value?", [_node()], require_citations=True
        )

        self.assertEqual("The value is 42 [E1].", answer)
        validation = partials[0]["answer_citation_validation"]
        self.assertTrue(validation["valid"])
        self.assertEqual("verified", validation["status"])
        self.assertIn(CITATION_INSTRUCTION, captured[0].storage[0].content)

    def test_strict_answer_fails_closed_when_model_omits_citation(self):
        agent = _agent("The value is 42.", [])

        answer, partials = agent.answer_simple_question(
            "What is the value?", [_node()], require_citations=True
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        validation = partials[0]["answer_citation_validation"]
        self.assertFalse(validation["valid"])
        self.assertEqual("missing_citations", validation["status"])

    def test_strict_answer_fails_closed_on_compound_marker(self):
        agent = _agent("The value is 42 [E1, E2].", [])

        answer, partials = agent.answer_simple_question(
            "What is the value?", [_node()], require_citations=True
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        validation = partials[0]["answer_citation_validation"]
        self.assertEqual("malformed_citations", validation["status"])
        self.assertEqual(["[E1, E2]"], validation["malformed_markers"])

    def test_legacy_answer_is_reported_but_not_rejected(self):
        agent = _agent("Compatibility answer.", [])

        answer, partials = agent.answer_simple_question(
            "What is the value?", [_node()]
        )

        self.assertEqual("Compatibility answer.", answer)
        self.assertEqual(
            "not_required",
            partials[0]["answer_citation_validation"]["status"],
        )

    def test_simple_answer_can_use_a_complex_query_evidence_scope(self):
        captured = []
        agent = _agent("The value is 42 [Q1-E1].", captured)

        answer, partials = agent.answer_simple_question(
            "What is the value?",
            [_node()],
            require_citations=True,
            evidence_scope="Q1",
        )

        self.assertEqual("The value is 42 [Q1-E1].", answer)
        validation = partials[0]["answer_citation_validation"]
        self.assertEqual("verified", validation["status"])
        self.assertEqual(
            [{
                "evidence_label": "Q1-E1",
                "citation_id": "citation_" + "a" * 32,
            }],
            validation["allowed_bindings"],
        )
        self.assertEqual(["[Q1-E1]"], partials[0]["evidence_labels"])
        self.assertIn("Evidence: [Q1-E1]", captured[0].storage[1].content)

    def test_complex_synthesis_scopes_colliding_local_labels(self):
        captured = []
        agent = _agent("The values are 42 [Q1-E1] and 7 [Q2-E1].", captured)
        plan = [SubQuestion(question="Compare the values", type="synthesis")]
        results = [
            {
                "question": "Find the first value",
                "answer": "The first value is 42 [E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {"E1": "citation_" + "a" * 32},
            },
            {
                "question": "Find the second value",
                "answer": "The second value is 7 [E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {"E1": "citation_" + "b" * 32},
            },
        ]

        answer, validation = agent.answer_complex_question_with_validation(
            "Compare both values",
            plan,
            results,
            require_citations=True,
        )

        self.assertEqual("The values are 42 [Q1-E1] and 7 [Q2-E1].", answer)
        self.assertEqual("verified", validation["status"])
        self.assertTrue(validation["valid"])
        self.assertFalse(validation["semantic_entailment_verified"])
        self.assertEqual(
            ["citation_" + "a" * 32, "citation_" + "b" * 32],
            validation["resolved_citation_ids"],
        )
        prompt = captured[0]
        self.assertIn("Sub-question answer:\nThe first value is 42 [Q1-E1].", prompt)
        self.assertIn("Sub-question answer:\nThe second value is 7 [Q2-E1].", prompt)
        self.assertIn("[Q1-E1] -> citation_" + "a" * 32, prompt)
        self.assertIn("[Q2-E1] -> citation_" + "b" * 32, prompt)

    def test_complex_synthesis_fails_closed_on_bare_or_unknown_label(self):
        agent = _agent("The value is 42 [E1].", [])
        results = [
            {
                "question": "Find the value",
                "answer": "The value is 42 [Q1-E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {
                    "Q1-E1": "citation_" + "a" * 32,
                },
            }
        ]

        answer, validation = agent.answer_complex_question_with_validation(
            "What is the value?",
            [],
            results,
            require_citations=True,
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        self.assertEqual("unknown_citations", validation["status"])
        self.assertEqual(["E1"], validation["unknown_labels"])

    def test_complex_synthesis_rejects_compound_scoped_marker(self):
        agent = _agent("Combined [Q1-E1, Q2-E1].", [])
        results = [
            {
                "question": "Find the first value",
                "answer": "First [Q1-E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {
                    "Q1-E1": "citation_" + "a" * 32,
                },
            },
            {
                "question": "Find the second value",
                "answer": "Second [Q2-E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {
                    "Q2-E1": "citation_" + "b" * 32,
                },
            },
        ]

        answer, validation = agent.answer_complex_question_with_validation(
            "Combine both values",
            [],
            results,
            require_citations=True,
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        self.assertEqual("malformed_citations", validation["status"])
        self.assertEqual(["[Q1-E1, Q2-E1]"], validation["malformed_markers"])
        self.assertFalse(validation["semantic_entailment_verified"])

    def test_complex_synthesis_rejects_cross_scope_binding(self):
        captured = []
        agent = _agent("The value is 42 [Q2-E1].", captured)
        results = [
            {
                "question": "Find the value",
                "answer": "The value is 42 [Q2-E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {
                    "Q2-E1": "citation_" + "a" * 32,
                },
            }
        ]

        answer, validation = agent.answer_complex_question_with_validation(
            "What is the value?",
            [],
            results,
            require_citations=True,
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        self.assertEqual("invalid_evidence_bindings", validation["status"])
        self.assertTrue(validation["binding_errors"])
        self.assertEqual([], captured)

    def test_complex_synthesis_fails_closed_when_generation_raises(self):
        def fail(_prompt):
            raise RuntimeError("model unavailable")

        agent = AnswerAgent(
            llm=SimpleNamespace(
                config=SimpleNamespace(max_tokens=2000),
                get_completion=fail,
            ),
            vlm=SimpleNamespace(config=SimpleNamespace(max_tokens=2000)),
        )
        results = [
            {
                "question": "Find the value",
                "answer": "The value is 42 [Q1-E1].",
                "answer_validation": {"valid": True, "status": "verified"},
                "evidence_bindings": {
                    "Q1-E1": "citation_" + "a" * 32,
                },
            }
        ]

        answer, validation = agent.answer_complex_question_with_validation(
            "What is the value?",
            [],
            results,
            require_citations=True,
        )

        self.assertEqual(CITATION_VALIDATION_FAILURE_ANSWER, answer)
        self.assertEqual("generation_failed", validation["status"])
        self.assertFalse(validation["valid"])

    def test_complex_context_persists_final_validation_and_scoped_bindings(self):
        rag = object.__new__(GBCRAG)
        rag.answer = Mock()
        rag.evidence_hydrator = object()
        rag.retrieval_callback = None
        rag._retrieve = Mock()
        first_validation = {
            "status": "verified",
            "valid": True,
            "allowed_bindings": [
                {
                    "evidence_label": "Q1-E1",
                    "citation_id": "citation_" + "a" * 32,
                }
            ],
        }
        second_validation = {
            "status": "verified",
            "valid": True,
            "allowed_bindings": [
                {
                    "evidence_label": "Q2-E1",
                    "citation_id": "citation_" + "b" * 32,
                }
            ],
        }
        rag.answer.answer_simple_question.side_effect = [
            (
                "First [Q1-E1].",
                [{"answer_citation_validation": first_validation}],
            ),
            (
                "Second [Q2-E1].",
                [{"answer_citation_validation": second_validation}],
            ),
        ]
        final_validation = {
            "status": "verified",
            "valid": True,
            "validation_scope": "complex_final_synthesis",
            "semantic_entailment_verified": False,
        }
        rag.answer.answer_complex_question_with_validation.return_value = (
            "Combined [Q1-E1] [Q2-E1].",
            final_validation,
        )
        plan = PlanResult(
            query_type="complex",
            original_query="Combine both facts",
            sub_questions=[
                SubQuestion(question="Find first", type="retrieval"),
                SubQuestion(question="Find second", type="retrieval"),
                SubQuestion(question="Combine them", type="synthesis"),
            ],
        )
        context = GBCRAGContext(query=plan.original_query)

        rag.process_analysis(context, plan)

        self.assertEqual(final_validation, context.answer_validation)
        self.assertEqual("Combined [Q1-E1] [Q2-E1].", context.final_answer)
        simple_calls = rag.answer.answer_simple_question.call_args_list
        self.assertEqual("Q1", simple_calls[0].kwargs["evidence_scope"])
        self.assertEqual("Q2", simple_calls[1].kwargs["evidence_scope"])
        complex_results = (
            rag.answer.answer_complex_question_with_validation.call_args.kwargs[
                "sub_question_results"
            ]
        )
        self.assertEqual(
            {"Q1-E1": "citation_" + "a" * 32},
            complex_results[0]["evidence_bindings"],
        )
        self.assertEqual(
            {"Q2-E1": "citation_" + "b" * 32},
            complex_results[1]["evidence_bindings"],
        )
        self.assertTrue(
            rag.answer.answer_complex_question_with_validation.call_args.kwargs[
                "require_citations"
            ]
        )


if __name__ == "__main__":
    unittest.main()
