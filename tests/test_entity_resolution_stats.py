"""Counters that separate cheap resolution outcomes from expensive ones.

Tuning entity resolution needs the rerank count and the model-call count kept
apart: the distance gate and topk govern reranking, while g governs how often a
model adjudicates. Reranking happens before g is consulted, so one number
cannot stand in for the other.
"""

from __future__ import annotations

import unittest
from collections import defaultdict

from Core.Index.Graph import Entity, Graph
from Core.pipelines.kg_refiner import EntityResolutionStats, KGRefiner


class _VDB:
    def __init__(self, hits):
        self._hits = hits

    def search(self, text, top_k):
        return self._hits


class _Reranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def rerank(self, query, documents, instruction):
        self.calls += 1
        return self.scores[: len(documents)]


def _hit(name: str, distance: float) -> dict:
    return {
        "distance": distance,
        "metadata": {
            "entity_name": name,
            "entity_type": "METHOD",
            "description": f"{name} description",
        },
    }


def _refiner(hits, scores, *seeded: str) -> KGRefiner:
    refiner = object.__new__(KGRefiner)
    graph = Graph(save_path=None, variant=None)
    for name in seeded:
        graph.add_and_link(
            tree_node_id=1,
            entities=[
                Entity(
                    entity_name=name,
                    entity_type="METHOD",
                    description="d",
                    source_ids={1},
                )
            ],
        )
    refiner.graph_index = graph
    refiner.vdb = _VDB(hits)
    refiner.reranker = _Reranker(scores)
    refiner.entity_to_vdb_id = defaultdict(str)
    refiner.entity_alias_map = defaultdict(str)
    refiner.er_stats = EntityResolutionStats()
    refiner.similar_search_topk = 10
    refiner.similar_search_distance_threshold = 0.2
    refiner.similar_search_mink = 1
    refiner.similar_search_g = 0.6
    return refiner


def _probe(name: str = "new entity") -> Entity:
    return Entity(
        entity_name=name, entity_type="METHOD", description="d", source_ids={9}
    )


class StatsShapeTests(unittest.TestCase):
    def test_llm_calls_sums_the_two_model_paths(self):
        stats = EntityResolutionStats(
            llm_selection_calls=5, llm_summarized_merges=3
        )
        self.assertEqual(8, stats.llm_calls)

    def test_as_dict_exposes_every_counter(self):
        keys = EntityResolutionStats().as_dict()
        for expected in (
            "entities_resolved",
            "gate_skips",
            "rerank_calls",
            "llm_selection_calls",
            "summarization_fallbacks",
        ):
            self.assertIn(expected, keys)


class ResolutionCountingTests(unittest.TestCase):
    def test_a_distance_gate_skip_costs_no_rerank(self):
        refiner = _refiner([_hit("other", 0.9)], [0.9])

        self.assertEqual([], refiner.search_similar_entities(_probe()))
        self.assertEqual(1, refiner.er_stats.entities_resolved)
        self.assertEqual(1, refiner.er_stats.gate_skips)
        self.assertEqual(0, refiner.er_stats.rerank_calls)
        self.assertEqual(0, refiner.reranker.calls)

    def test_a_low_reranker_score_is_counted_apart_from_the_gate(self):
        refiner = _refiner([_hit("other", 0.1)], [0.4])

        self.assertEqual([], refiner.search_similar_entities(_probe()))
        self.assertEqual(1, refiner.er_stats.rerank_calls)
        self.assertEqual(1, refiner.er_stats.low_score_skips)
        self.assertEqual(0, refiner.er_stats.gate_skips)

    def test_no_gradient_counts_as_case_a(self):
        # Every candidate clears the gradient, so the scores did not discriminate.
        refiner = _refiner(
            [_hit("a", 0.1), _hit("b", 0.1)], [0.9, 0.8], "a", "b"
        )

        self.assertEqual([], refiner.search_similar_entities(_probe()))
        self.assertEqual(1, refiner.er_stats.case_a_no_gradient)
        self.assertEqual(0, refiner.er_stats.case_b_single_alias)

    def test_a_sharp_drop_counts_as_a_single_alias(self):
        refiner = _refiner(
            [_hit("a", 0.1), _hit("b", 0.1)], [0.9, 0.1], "a", "b"
        )

        found = refiner.search_similar_entities(_probe())
        self.assertEqual(1, len(found))
        self.assertEqual(1, refiner.er_stats.case_b_single_alias)
        self.assertEqual(0, refiner.er_stats.case_a_no_gradient)

    def test_several_close_candidates_count_as_multiple_aliases(self):
        refiner = _refiner(
            [_hit("a", 0.1), _hit("b", 0.1), _hit("c", 0.1)],
            [0.9, 0.8, 0.1],
            "a",
            "b",
            "c",
        )

        found = refiner.search_similar_entities(_probe())
        self.assertEqual(2, len(found))
        self.assertEqual(1, refiner.er_stats.case_b_multiple_aliases)
        self.assertEqual(0, refiner.er_stats.case_b_single_alias)

    def test_log_snapshot_reports_without_raising_on_an_empty_run(self):
        EntityResolutionStats().log_snapshot(0)


if __name__ == "__main__":
    unittest.main()
