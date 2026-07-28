from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.prompts.gbc_prompt import (
    QuestionEntity,
    QuestionEntityExtraction,
    SecEXPSelection,
)
from Core.rag.gbc_plan import (
    ComplexResult,
    Filter,
    GlobalResult,
    PlanResult,
    PlanningBudgetExceeded,
    QueryTypeResult,
    SubQuestion,
    TaskPlanner,
)
from Core.rag.gbc_rag import GBCRAG, SectionSelectionBudgetExceeded
from Core.rag.gbc_utils import GBCRAGContext, SubStep


class _SequencedLLM:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def get_json_completion(self, prompt, schema):
        self.calls.append((prompt, schema))
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class PlannerFanoutBoundTests(unittest.TestCase):
    def test_long_query_is_rejected_before_any_model_call(self):
        llm = _SequencedLLM()
        planner = TaskPlanner(llm, max_query_chars=8)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "query contains 9"):
            planner.analyze("x" * 9)

        self.assertEqual([], llm.calls)

    def test_normal_complex_plan_is_preserved(self):
        expected = [
            SubQuestion(question="Find alpha", type="retrieval"),
            SubQuestion(question="Use alpha", type="synthesis"),
        ]
        llm = _SequencedLLM(
            QueryTypeResult(query_type="complex"),
            ComplexResult(sub_questions=expected),
        )
        planner = TaskPlanner(llm, max_subquestions=2)

        plan = planner.analyze("Explain alpha")

        self.assertEqual(expected, plan.sub_questions)
        self.assertEqual(2, len(llm.calls))

    def test_oversized_complex_plan_is_rejected_before_retrieval(self):
        questions = [
            SubQuestion(question=f"Find item {index}", type="retrieval")
            for index in range(3)
        ]
        llm = _SequencedLLM(
            QueryTypeResult(query_type="complex"),
            ComplexResult(sub_questions=questions),
        )
        planner = TaskPlanner(llm, max_subquestions=2)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "3 sub-questions"):
            planner.analyze("Compare the items")

        # Classification plus decomposition are the only calls.  No plan is
        # returned for downstream retrieval/model fan-out.
        self.assertEqual(2, len(llm.calls))

    def test_global_prompt_lists_the_document_sections(self):
        # Without the real headings the planner invents a section name from the
        # query's topic, and a section filter matching no heading reduces the
        # retrieval scope to nothing.
        llm = _SequencedLLM(
            QueryTypeResult(query_type="global"),
            GlobalResult(
                filters=[Filter(filter_type="document")], operation="ANALYZE"
            ),
        )
        planner = TaskPlanner(llm)

        planner.analyze(
            "What drives the margin gap?",
            section_titles=["Why now", "Root cause", "Pricing"],
        )

        global_prompt = llm.calls[1][0]
        self.assertIn("- Why now", global_prompt)
        self.assertIn("- Root cause", global_prompt)
        self.assertIn("- Pricing", global_prompt)
        self.assertIn(
            "A topic the query merely asks about is NOT a section",
            global_prompt,
        )

    def test_global_prompt_forbids_section_filters_without_headings(self):
        llm = _SequencedLLM(
            QueryTypeResult(query_type="global"),
            GlobalResult(
                filters=[Filter(filter_type="document")], operation="SUMMARIZE"
            ),
        )
        planner = TaskPlanner(llm)

        planner.analyze("Summarize it", section_titles=[])

        global_prompt = llm.calls[1][0]
        self.assertIn("no detected sections", global_prompt)
        self.assertIn("MUST NOT emit", global_prompt)

    def test_section_titles_are_bounded_and_deduplicated(self):
        llm = _SequencedLLM(
            QueryTypeResult(query_type="global"),
            GlobalResult(
                filters=[Filter(filter_type="document")], operation="COUNT"
            ),
        )
        planner = TaskPlanner(llm)

        planner.analyze(
            "Count the tables",
            section_titles=["Repeated", "Repeated", "x" * 400, 7, None],
        )

        global_prompt = llm.calls[1][0]
        self.assertEqual(1, global_prompt.count("- Repeated"))
        self.assertNotIn("x" * 200, global_prompt)

    def test_oversized_global_filter_plan_is_rejected(self):
        filters = [Filter(filter_type="page", filter_value=str(index + 1)) for index in range(3)]
        llm = _SequencedLLM(
            QueryTypeResult(query_type="global"),
            GlobalResult(filters=filters, operation="COUNT"),
        )
        planner = TaskPlanner(llm, max_filters=2)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "3 filters"):
            planner.analyze("Count across these pages")

        self.assertEqual(2, len(llm.calls))

    def test_execution_boundary_revalidates_a_deserialized_plan(self):
        planner = TaskPlanner(_SequencedLLM(), max_subquestions=1)
        plan = PlanResult(
            query_type="complex",
            original_query="Compare",
            sub_questions=[
                SubQuestion(question="First", type="retrieval"),
                SubQuestion(question="Second", type="retrieval"),
            ],
        )
        rag = GBCRAG.__new__(GBCRAG)
        rag.planner = planner
        rag._retrieve = Mock()
        context = GBCRAGContext(query=plan.original_query)

        with self.assertRaises(PlanningBudgetExceeded):
            rag.process_analysis(context, plan)

        rag._retrieve.assert_not_called()
        self.assertEqual([], context.iterations)

    def test_config_rejects_incoherent_and_unbounded_limits(self):
        with self.assertRaises(ValidationError):
            GBCRAGConfig(planner_max_subquestions=33)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(query_entity_limit=4, query_entity_candidate_limit=5)
        with self.assertRaises(ValidationError):
            GBCRAGConfig(
                section_selection_candidate_limit=4,
                section_selection_supplementary_limit=5,
            )

    def test_plan_schema_rejects_irrelevant_empty_fields(self):
        invalid_plans = (
            {
                "query_type": "simple",
                "original_query": "Find alpha",
                "filters": [],
            },
            {
                "query_type": "complex",
                "original_query": "Compare alpha",
                "sub_questions": [
                    {"question": "Find alpha", "type": "retrieval"}
                ],
                "filters": [],
            },
            {
                "query_type": "global",
                "original_query": "Count tables",
                "filters": [{"filter_type": "table"}],
                "operation": "COUNT",
                "sub_questions": [],
            },
        )
        for payload in invalid_plans:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                PlanResult(**payload)


class _EntityStore:
    def __init__(self, callback=None) -> None:
        self.callback = callback or (lambda _query, _top_k: [])
        self.calls = []

    def search_entities(self, *, name_text, top_k, **kwargs):
        self.calls.append((name_text, top_k))
        return self.callback(name_text, top_k)


class EntityFanoutBoundTests(unittest.TestCase):
    @staticmethod
    def _rag(store, llm, *, entity_limit=3, candidate_limit=2):
        rag = GBCRAG.__new__(GBCRAG)
        rag.cfg = SimpleNamespace(
            planner_max_query_chars=32,
            query_entity_limit=entity_limit,
            query_entity_candidate_limit=candidate_limit,
            query_entity_name_chars=32,
            query_entity_type_chars=32,
        )
        rag.gbc_index = SimpleNamespace(
            entity_vdb=store,
            GraphIndex=SimpleNamespace(
                get_node_name_from_str=lambda name, entity_type: f"{entity_type}:{name}",
                get_node_name_from_entity=lambda entity: (
                    f"{entity.entity_type}:{entity.entity_name}"
                ),
            ),
        )
        rag.llm = llm
        rag.threshold_e = 0.5
        return rag

    def test_long_query_makes_no_store_or_model_call(self):
        store = _EntityStore()
        llm = _SequencedLLM()
        rag = self._rag(store, llm)

        with self.assertRaises(PlanningBudgetExceeded):
            rag._get_query_entity("x" * 33)

        self.assertEqual([], store.calls)
        self.assertEqual([], llm.calls)

    def test_store_contract_overflow_is_read_only_to_limit_plus_one(self):
        yielded = []

        def overflow(_query, _top_k):
            def values():
                for index in range(1000):
                    yielded.append(index)
                    yield {"metadata": {"entity_name": str(index), "entity_type": "X"}}

            return values()

        store = _EntityStore(overflow)
        llm = _SequencedLLM()
        rag = self._rag(store, llm, candidate_limit=2)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "candidate search"):
            rag._get_query_entity("bounded query")

        self.assertEqual([0, 1, 2], yielded)
        self.assertEqual([], llm.calls)

    def test_model_entity_overflow_does_not_fan_out_vector_calls(self):
        entities = [
            QuestionEntity(entity_name=f"item-{index}", entity_type="PRODUCT")
            for index in range(4)
        ]
        llm = _SequencedLLM(QuestionEntityExtraction(entities=entities))
        store = _EntityStore()
        rag = self._rag(store, llm, entity_limit=3)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "entity extraction"):
            rag._get_query_entity("find items")

        # One bounded candidate lookup occurred; no per-entity lookup did.
        self.assertEqual(1, len(store.calls))
        self.assertEqual(1, len(llm.calls))

    def test_normal_entity_plan_has_a_strict_store_call_upper_bound(self):
        entities = [
            QuestionEntity(entity_name=f"item-{index}", entity_type="PRODUCT")
            for index in range(3)
        ]
        llm = _SequencedLLM(QuestionEntityExtraction(entities=entities))
        store = _EntityStore()
        rag = self._rag(store, llm, entity_limit=3)

        self.assertEqual({}, rag._get_query_entity("find items"))

        # One candidate search plus at most one mapping search per entity.
        self.assertEqual(1 + 3, len(store.calls))
        self.assertEqual(1, len(llm.calls))


class _SectionTree:
    def __init__(self, sections) -> None:
        self.sections = sections
        self.by_id = {section.index_id: section for section in sections}

    def get_nodes_at_depth(self, _depth):
        return list(self.sections)

    def get_path_from_root(self, node_id):
        return [self.by_id[node_id]]


def _section(index: int, title: str):
    return SimpleNamespace(
        index_id=index,
        meta_info=SimpleNamespace(content=title),
    )


class SectionSelectionBoundTests(unittest.TestCase):
    @staticmethod
    def _rag(sections, llm, *, candidate_limit=4, supplementary_limit=2):
        rag = GBCRAG.__new__(GBCRAG)
        rag.cfg = SimpleNamespace(
            planner_max_query_chars=256,
            section_selection_candidate_limit=candidate_limit,
            section_selection_supplementary_limit=supplementary_limit,
            section_selection_title_chars=64,
            section_selection_path_depth=4,
            section_selection_prompt_chars=65536,
        )
        rag.gbc_index = SimpleNamespace(TreeIndex=_SectionTree(sections))
        rag.llm = llm
        rag.select_depth = 2
        return rag

    def test_late_lexical_section_survives_bounded_source_aware_policy(self):
        sections = [_section(index, f"General topic {index}") for index in range(1, 20)]
        sections.append(_section(20, "Quantum latency results"))
        llm = _SequencedLLM(
            SecEXPSelection(
                supplementary_ids=[20],
                explanation="The heading directly matches the question.",
            )
        )
        rag = self._rag(sections, llm, candidate_limit=4)
        rag.link_section = lambda _nodes: {2: ["latency"]}
        step = SubStep(sub_query="quantum latency", sub_number=1)

        rag.llm_section_selection("quantum latency", [], step)

        self.assertIn(20, step.section_prompt_candidate_ids)
        self.assertEqual([2, 20], step.retrieval_sec_ids)
        self.assertEqual(20, step.section_total_count)
        self.assertEqual(4, step.section_prompt_candidate_count)
        self.assertEqual(16, step.section_omitted_count)
        self.assertEqual("selected", step.section_selection_status)
        self.assertEqual(1, len(llm.calls))
        prompt = llm.calls[0][0]
        self.assertIn("Quantum latency results", prompt)

    def test_normal_in_budget_sections_are_all_represented(self):
        sections = [_section(1, "Alpha"), _section(2, "Beta"), _section(3, "Gamma")]
        llm = _SequencedLLM(
            SecEXPSelection(supplementary_ids=[3], explanation="Gamma is useful")
        )
        rag = self._rag(sections, llm, candidate_limit=4)
        rag.link_section = lambda _nodes: {1: ["alpha"]}
        step = SubStep(sub_query="gamma", sub_number=1)

        rag.llm_section_selection("gamma", [], step)

        self.assertEqual([1, 2, 3], step.section_prompt_candidate_ids)
        self.assertEqual(0, step.section_omitted_count)
        self.assertEqual([1, 3], step.retrieval_sec_ids)

    def test_linked_evidence_overflow_fails_before_selector_call(self):
        sections = [_section(index, f"Section {index}") for index in range(1, 8)]
        llm = _SequencedLLM()
        rag = self._rag(sections, llm, candidate_limit=3)
        rag.link_section = lambda _nodes: {
            1: ["a"],
            2: ["b"],
            3: ["c"],
            4: ["d"],
        }
        step = SubStep(sub_query="question", sub_number=1)

        with self.assertRaisesRegex(
            SectionSelectionBudgetExceeded, "no linked evidence was silently dropped"
        ):
            rag.llm_section_selection("question", [], step)

        self.assertTrue(step.section_selection_budget_exhausted)
        self.assertEqual("linked_budget_exceeded", step.section_selection_status)
        self.assertEqual([], llm.calls)

    def test_selector_cannot_reach_an_omitted_section_id(self):
        sections = [_section(index, f"Section {index}") for index in range(1, 10)]
        llm = _SequencedLLM(
            SecEXPSelection(supplementary_ids=[9], explanation="hallucinated scope")
        )
        rag = self._rag(sections, llm, candidate_limit=2)
        rag.link_section = lambda _nodes: {1: ["a"]}
        step = SubStep(sub_query="unrelated", sub_number=1)

        # Determine a definitely omitted ID; the fake output is adjusted after
        # the deterministic policy has been inspected without a model call.
        candidates = rag._bounded_section_candidates(
            query="unrelated",
            remain_sections=sections[1:],
            capacity=1,
            source_positions={section.index_id: i for i, section in enumerate(sections)},
            linked_positions=[0],
        )
        omitted = next(
            section.index_id
            for section in sections[1:]
            if section.index_id != candidates[0].index_id
        )
        llm.responses[0] = SecEXPSelection(
            supplementary_ids=[omitted], explanation="outside pool"
        )

        with self.assertRaisesRegex(ValueError, "outside the bounded"):
            rag.llm_section_selection("unrelated", [], step)

        self.assertEqual("invalid_out_of_pool_model_ids", step.section_selection_status)
        self.assertEqual(1, len(llm.calls))

    def test_selector_output_count_is_rejected_not_truncated(self):
        sections = [_section(index, f"Section {index}") for index in range(1, 6)]
        llm = _SequencedLLM(
            SecEXPSelection(
                supplementary_ids=[2, 3, 4],
                explanation="too many",
            )
        )
        rag = self._rag(
            sections,
            llm,
            candidate_limit=5,
            supplementary_limit=2,
        )
        rag.link_section = lambda _nodes: {1: ["a"]}
        step = SubStep(sub_query="question", sub_number=1)

        with self.assertRaisesRegex(PlanningBudgetExceeded, "3 IDs"):
            rag.llm_section_selection("question", [], step)

        self.assertEqual("model_selection_budget_exceeded", step.section_selection_status)
        self.assertEqual([], step.supplementary_ids)
        self.assertEqual(1, len(llm.calls))


if __name__ == "__main__":
    unittest.main()
