from __future__ import annotations

import inspect
import math
import unittest
from types import SimpleNamespace

from Core.Index.Graph import Entity, Relationship
from Core.configs.graph_config import GraphConfig
from Core.pipelines.kg_extractor import KGExtractor, LLMExtractor, LocalExtractor
from Core.pipelines.kg_refiner import EntityResolutionStats, KGRefiner


class GraphConfigValidationTests(unittest.TestCase):
    def test_refinement_mode_is_closed_over_supported_values(self):
        for refine_type in ("basic", "advanced"):
            with self.subTest(refine_type=refine_type):
                self.assertEqual(
                    GraphConfig(refine_type=refine_type).refine_type,
                    refine_type,
                )

        for refine_type in ("", "ADVANCED", "experimental"):
            with self.subTest(refine_type=refine_type):
                with self.assertRaisesRegex(ValueError, "refine_type"):
                    GraphConfig(refine_type=refine_type)

    def test_gradient_threshold_must_be_finite_and_bounded(self):
        for g in (0.01, 0.6, 1.0):
            with self.subTest(g=g):
                self.assertEqual(GraphConfig(g=g).g, g)

        for g in (0.0, -0.1, 1.01, math.inf, -math.inf, math.nan, True, "0.6"):
            with self.subTest(g=g):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    GraphConfig(g=g)


class KGExtractionCorrectnessTests(unittest.TestCase):
    def test_local_extractor_returns_graph_models_accepted_by_repair(self):
        local = object.__new__(LocalExtractor)
        local.nlp = lambda _text: object()
        local._extract_entities = lambda _doc: [
            {
                "text": "Alice",
                "label": "PERSON",
                "start_char": 0,
                "end_char": 5,
            },
            {
                "text": "Acme",
                "label": "ORG",
                "start_char": 14,
                "end_char": 18,
            },
        ]
        local._extract_relations = lambda _doc: [("Alice", "join", "Acme")]
        node = SimpleNamespace(
            index_id=17,
            meta_info=SimpleNamespace(content="Alice joined Acme."),
        )

        result = local.extract(node)

        self.assertTrue(all(isinstance(item, Entity) for item in result["entities"]))
        self.assertTrue(
            all(isinstance(item, Relationship) for item in result["relations"])
        )
        self.assertEqual(result["entities"][0].source_ids, {17})
        self.assertEqual(result["relations"][0].source_ids, {17})

        repaired = KGExtractor.res_repair(object.__new__(KGExtractor), result)
        self.assertEqual(
            [entity.entity_name for entity in repaired["entities"]],
            ["alice", "acme"],
        )
        self.assertEqual(repaired["relations"][0].src_entity_name, "alice")
        self.assertEqual(repaired["relations"][0].tgt_entity_name, "acme")

    def test_image_fallback_is_returned_as_evidence(self):
        extractor = object.__new__(LLMExtractor)
        extractor.graph_config = SimpleNamespace(image_description_force=True)
        extractor.vlm = SimpleNamespace(
            generate_json=lambda **_kwargs: SimpleNamespace(entities=[])
        )
        node = SimpleNamespace(
            index_id=23,
            meta_info=SimpleNamespace(
                content="Revenue workflow for the current quarter",
                img_path="page-1.png",
            ),
        )

        entities, relationships = extractor._extract_kg_from_image(node)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, "IMAGE")
        self.assertEqual(entities[0].source_ids, {23})
        self.assertEqual(relationships, [])

    def test_duplicate_entity_aggregation_is_linear_and_does_not_skip(self):
        entities = [
            Entity(
                entity_name="  Same   Name ",
                entity_type="test type",
                description=f"description-{index}",
                source_ids={index},
            )
            for index in range(4)
        ]
        entities.append(
            Entity(
                entity_name="Different",
                entity_type="OTHER",
                description="kept",
                source_ids={9},
            )
        )
        result = {"entities": entities, "relations": [], "node_idx": 1}

        repaired = KGExtractor.res_repair(object.__new__(KGExtractor), result)

        self.assertEqual(len(repaired["entities"]), 2)
        aggregate = repaired["entities"][0]
        self.assertEqual(aggregate.entity_name, "same name")
        self.assertEqual(aggregate.entity_type, "TEST_TYPE")
        self.assertEqual(
            aggregate.description,
            "description-0 description-1 description-2 description-3",
        )
        self.assertEqual(aggregate.source_ids, {0, 1, 2, 3})


class KGRefinementCorrectnessTests(unittest.TestCase):
    def test_alias_resolution_handles_deep_chains_and_compresses_paths(self):
        refiner = object.__new__(KGRefiner)
        chain_length = 5_000
        refiner.entity_alias_map = {
            f"entity-{index}": f"entity-{index + 1}"
            for index in range(chain_length)
        }
        refiner.entity_alias_map[f"entity-{chain_length}"] = (
            f"entity-{chain_length}"
        )

        latest = refiner.get_latest_entity_name("entity-0")

        self.assertEqual(latest, f"entity-{chain_length}")
        self.assertEqual(refiner.entity_alias_map["entity-0"], latest)
        self.assertEqual(refiner.entity_alias_map["entity-2499"], latest)

    def test_alias_resolution_rejects_cycles_and_dangling_destinations(self):
        refiner = object.__new__(KGRefiner)
        refiner.entity_alias_map = {"a": "b", "b": "c", "c": "a"}
        with self.assertRaisesRegex(ValueError, "alias cycle.*a -> b -> c -> a"):
            refiner.get_latest_entity_name("a")

        refiner.entity_alias_map = {"a": "missing"}
        with self.assertRaisesRegex(ValueError, "points to missing entity"):
            refiner.get_latest_entity_name("a")

    def test_all_gradient_candidates_returns_no_ambiguous_shortlist(self):
        refiner = object.__new__(KGRefiner)
        refiner.er_stats = EntityResolutionStats()
        refiner.similar_search_topk = 10
        refiner.similar_search_distance_threshold = 0.2
        refiner.similar_search_mink = 1
        refiner.similar_search_g = 0.5
        refiner.entity_alias_map = {}
        refiner.vdb = SimpleNamespace(
            search_entities=lambda **_kwargs: [
                {
                    "distance": 0.1,
                    "metadata": {
                        "entity_name": "first",
                        "entity_type": "TYPE",
                        "description": "first",
                        "node_id": "first:TYPE",
                        "canonical_id": "first:TYPE",
                    },
                },
                {
                    "distance": 0.11,
                    "metadata": {
                        "entity_name": "second",
                        "entity_type": "TYPE",
                        "description": "second",
                        "node_id": "second:TYPE",
                        "canonical_id": "second:TYPE",
                    },
                },
            ]
        )
        refiner.reranker = SimpleNamespace(
            rerank=lambda **_kwargs: [0.9, 0.8],
            clean_cache=lambda: None,
        )

        class GraphIndex:
            kg = {}

            @staticmethod
            def get_node_name_from_entity(entity):
                return f"{entity.entity_name}:{entity.entity_type}"

            @staticmethod
            def get_entity(*_args, **_kwargs):
                raise AssertionError("all candidates should return an empty shortlist")

        refiner.graph_index = GraphIndex()
        entity = Entity(
            entity_name="candidate",
            entity_type="TYPE",
            description="candidate",
            source_ids={1},
        )

        self.assertEqual(refiner.search_similar_entities(entity), [])

    def test_hot_membership_paths_do_not_allocate_all_graph_nodes(self):
        source = inspect.getsource(KGRefiner)
        self.assertNotIn("get_all_nodes()", source)

    def test_resolution_does_not_force_a_gpu_collection_per_entity(self):
        # rerank() empties its own cache on a content threshold; calling it per
        # entity ran a full gc plus a cuda sync ~119k times on a large document.
        source = inspect.getsource(KGRefiner)
        self.assertNotIn("clean_cache()", source)


if __name__ == "__main__":
    unittest.main()
