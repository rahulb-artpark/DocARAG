"""Batching adjudications must not change a single merge decision.

Adjudication is 181.7 of the graph stage's 434.6 minutes and is strictly
sequential, so batching independent decisions is the one lever that moves it.
Algorithm 1's sequential dependency is that one commit changes what the next
entity retrieves; disjoint candidate sets are exactly when that cannot happen,
and everything here pins that boundary.
"""

from __future__ import annotations

import unittest
from collections import defaultdict

from Core.Index.Graph import Entity, Graph
from Core.pipelines.kg_refiner import (
    EntityResolutionStats,
    KGRefiner,
    _Decision,
)


class _LLM:
    """Records prompts and returns queued replies in order."""

    def __init__(self, replies=None, error: Exception | None = None):
        self.replies = list(replies or [])
        self.error = error
        self.prompts: list[str] = []

    def get_json_completion(self, prompt=None, schema=None, **kwargs):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        if not self.replies:
            raise AssertionError("the refiner made more model calls than queued")
        return self.replies.pop(0)


class _Batch:
    def __init__(self, decisions):
        self.decisions = decisions


class _Decided:
    def __init__(self, task_id, select_id, explanation=""):
        self.task_id = task_id
        self.select_id = select_id
        self.explanation = explanation


class _Single:
    def __init__(self, select_id, explanation=""):
        self.select_id = select_id
        self.explanation = explanation


def _entity(name: str, type_: str = "METHOD", source: int = 1) -> Entity:
    return Entity(
        entity_name=name,
        entity_type=type_,
        description=f"{name} description",
        source_ids={source},
    )


def _refiner(llm: _LLM, *seed: Entity) -> tuple[KGRefiner, Graph]:
    refiner = object.__new__(KGRefiner)
    graph = Graph(save_path=None, variant=None)
    refiner.graph_index = graph
    refiner.llm = llm
    refiner.entity_to_vdb_id = defaultdict(str)
    refiner.entity_alias_map = defaultdict(str)
    refiner.er_stats = EntityResolutionStats()
    for index, entity in enumerate(seed):
        graph.add_and_link(tree_node_id=index + 1, entities=[entity])
    return refiner, graph


class BatchGroupingTests(unittest.TestCase):
    """Which tasks may share a call."""

    def setUp(self):
        self.candidates = {}

        def fake_search(entity):
            return self.candidates.get(entity.entity_name, [])

        self.fake_search = fake_search

    def _prepared(self, refiner, entities):
        refiner.search_similar_entities = self.fake_search
        return refiner._adjudicate_batched(entities)

    def test_entities_with_disjoint_candidates_share_one_call(self):
        seed_a, seed_b = _entity("alpha"), _entity("beta")
        llm = _LLM([_Batch([_Decided(0, -1), _Decided(1, -1)])])
        refiner, _graph = _refiner(llm, seed_a, seed_b)
        new_a, new_b = _entity("alpha prime"), _entity("beta prime")
        self.candidates = {"alpha prime": [seed_a], "beta prime": [seed_b]}

        self._prepared(refiner, [new_a, new_b])

        self.assertEqual(1, len(llm.prompts))
        self.assertEqual(2, refiner.er_stats.batched_selections)

    def test_a_shared_candidate_forces_separate_calls(self):
        # One commit would change what the other should have seen, which is
        # exactly Algorithm 1's dependency.
        shared = _entity("shared")
        llm = _LLM([_Single(-1), _Single(-1)])
        refiner, _graph = _refiner(llm, shared)
        new_a, new_b = _entity("first"), _entity("second")
        self.candidates = {"first": [shared], "second": [shared]}

        self._prepared(refiner, [new_a, new_b])

        self.assertEqual(2, len(llm.prompts))
        self.assertEqual(0, refiner.er_stats.batched_selections)

    def test_a_batch_never_exceeds_the_configured_size(self):
        seeds = [_entity(f"seed{i}") for i in range(12)]
        llm = _LLM(
            [
                _Batch([_Decided(i, -1) for i in range(5)]),
                _Batch([_Decided(i, -1) for i in range(5)]),
                _Batch([_Decided(i, -1) for i in range(2)]),
            ]
        )
        refiner, _graph = _refiner(llm, *seeds)
        news = [_entity(f"new{i}") for i in range(12)]
        self.candidates = {f"new{i}": [seeds[i]] for i in range(12)}

        self._prepared(refiner, news)

        self.assertEqual(3, len(llm.prompts))
        self.assertEqual(12, refiner.er_stats.batched_selections)

    def test_entities_needing_no_model_call_never_enter_a_batch(self):
        llm = _LLM([])
        refiner, _graph = _refiner(llm)
        news = [_entity("a"), _entity("b")]
        self.candidates = {}

        decisions = self._prepared(refiner, news)

        self.assertEqual([], llm.prompts)
        self.assertTrue(all(d.selected is None for d in decisions.values()))

    def test_an_entity_already_in_the_graph_is_left_to_the_serial_path(self):
        existing = _entity("llm")
        llm = _LLM([])
        refiner, _graph = _refiner(llm, existing)

        decisions = self._prepared(refiner, [_entity("llm")])

        self.assertEqual({}, decisions)
        self.assertEqual([], llm.prompts)


class MalformedResponseTests(unittest.TestCase):
    """A batch that cannot be trusted is discarded whole, never in part."""

    def _two_tasks(self, llm):
        seed_a, seed_b = _entity("alpha"), _entity("beta")
        refiner, _graph = _refiner(llm, seed_a, seed_b)
        return refiner, [
            (_entity("alpha prime"), [seed_a]),
            (_entity("beta prime"), [seed_b]),
        ]

    def test_too_few_decisions_falls_back_to_deciding_singly(self):
        llm = _LLM([_Batch([_Decided(0, 0)]), _Single(0), _Single(-1)])
        refiner, tasks = self._two_tasks(llm)

        results = refiner.er_selection_by_llm_batch(tasks)

        self.assertEqual(3, len(llm.prompts))
        self.assertEqual(0, refiner.er_stats.batched_selections)
        self.assertEqual("alpha", results[0].entity_name)
        self.assertIsNone(results[1])

    def test_a_repeated_task_id_discards_the_whole_response(self):
        # A duplicated id means the model lost the mapping, so no decision in
        # the response can be attributed to the right entity.
        llm = _LLM([_Batch([_Decided(0, 0), _Decided(0, 0)]), _Single(-1), _Single(-1)])
        refiner, tasks = self._two_tasks(llm)

        refiner.er_selection_by_llm_batch(tasks)

        self.assertEqual(3, len(llm.prompts))
        self.assertEqual(0, refiner.er_stats.batched_selections)

    def test_an_invented_task_id_discards_the_whole_response(self):
        llm = _LLM([_Batch([_Decided(0, 0), _Decided(7, 0)]), _Single(-1), _Single(-1)])
        refiner, tasks = self._two_tasks(llm)

        refiner.er_selection_by_llm_batch(tasks)

        self.assertEqual(0, refiner.er_stats.batched_selections)

    def test_a_failed_batch_call_falls_back_rather_than_raising(self):
        llm = _LLM(error=RuntimeError("budget exhausted"))
        refiner, tasks = self._two_tasks(llm)

        results = refiner.er_selection_by_llm_batch(tasks)

        # Every call raises here, so both singles fail too and return None --
        # the stage continues rather than aborting.
        self.assertEqual([None, None], results)

    def test_an_out_of_range_selection_becomes_no_match(self):
        llm = _LLM([_Batch([_Decided(0, 9), _Decided(1, -1)])])
        refiner, tasks = self._two_tasks(llm)

        results = refiner.er_selection_by_llm_batch(tasks)

        self.assertEqual([None, None], results)

    def test_a_single_task_uses_the_single_prompt(self):
        seed = _entity("alpha")
        llm = _LLM([_Single(0)])
        refiner, _graph = _refiner(llm, seed)

        results = refiner.er_selection_by_llm_batch([(_entity("alpha prime"), [seed])])

        self.assertEqual("alpha", results[0].entity_name)
        self.assertEqual(1, refiner.er_stats.llm_selection_calls)


class StaleDecisionTests(unittest.TestCase):
    """A decision is only applied while the graph still matches it."""

    def test_a_decision_holds_while_its_candidates_are_untouched(self):
        seed = _entity("alpha")
        refiner, _graph = _refiner(_LLM([]), seed)

        self.assertTrue(
            refiner._decision_still_holds(_Decision(candidates=[seed], selected=seed))
        )

    def test_a_merged_away_candidate_invalidates_the_decision(self):
        seed = _entity("alpha")
        refiner, graph = _refiner(_LLM([]), seed)
        refiner.entity_alias_map[graph.get_node_name_from_entity(seed)] = "other"

        self.assertFalse(
            refiner._decision_still_holds(_Decision(candidates=[seed], selected=seed))
        )

    def test_a_candidate_removed_from_the_graph_invalidates_the_decision(self):
        seed = _entity("alpha")
        refiner, _graph = _refiner(_LLM([]), seed)
        missing = _entity("vanished")

        self.assertFalse(
            refiner._decision_still_holds(
                _Decision(candidates=[missing], selected=missing)
            )
        )

    def test_a_stale_decision_is_re_adjudicated_and_counted(self):
        seed = _entity("alpha")
        llm = _LLM([_Single(-1)])
        refiner, graph = _refiner(llm, seed)
        refiner.search_similar_entities = lambda entity: []
        refiner.entity_alias_map[graph.get_node_name_from_entity(seed)] = "other"

        refiner.entity_resolution(
            _entity("beta", source=2),
            decision=_Decision(candidates=[seed], selected=seed),
        )

        self.assertEqual(1, refiner.er_stats.batch_revalidations)

    def test_a_valid_decision_skips_retrieval_entirely(self):
        seed = _entity("alpha")
        refiner, _graph = _refiner(_LLM([]), seed)

        def fail(_entity):
            raise AssertionError("a held decision must not search again")

        refiner.search_similar_entities = fail

        resolved = refiner.entity_resolution(
            _entity("alpha prime", source=2),
            decision=_Decision(candidates=[seed], selected=None),
        )

        self.assertEqual("alpha prime", resolved.entity_name)
        self.assertEqual(0, refiner.er_stats.batch_revalidations)


if __name__ == "__main__":
    unittest.main()
