from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from Core.Index.Graph import Entity
from Core.Index.Tree import NodeType
from Core.configs.graph_config import GraphConfig
from Core.pipelines.kg_extractor import KGExtractionError, KGExtractor
from Core.pipelines.kg_refiner import KGRefinementError, KGRefiner


def _node(index: int):
    return SimpleNamespace(index_id=index, type=NodeType.TEXT)


class BoundedKGExtractionTests(unittest.TestCase):
    def make_extractor(self, worker, **config_overrides):
        config = GraphConfig(
            extraction_max_workers=2,
            extraction_max_in_flight=3,
            extraction_batch_size=4,
            **config_overrides,
        )
        extractor = object.__new__(KGExtractor)
        extractor.cfg_graph = config
        extractor.extract_kg = worker
        return extractor

    def test_extraction_is_ordered_and_source_consumption_is_bounded(self):
        release = threading.Event()
        two_started = threading.Event()
        three_consumed = threading.Event()
        lock = threading.Lock()
        consumed = 0
        active = 0
        maximum_active = 0

        def source():
            nonlocal consumed
            for index in range(10):
                with lock:
                    consumed += 1
                    if consumed == 3:
                        three_consumed.set()
                yield _node(index)

        def worker(node):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == 2:
                    two_started.set()
            self.assertTrue(release.wait(2))
            if node.index_id % 2 == 0:
                time.sleep(0.005)
            with lock:
                active -= 1
            return {"node_idx": node.index_id, "entities": [], "relations": []}

        extractor = self.make_extractor(worker)
        results = []
        errors = []

        def consume():
            try:
                results.extend(extractor.iter_extract_kg(source()))
            except Exception as error:  # pragma: no cover - assertion aid
                errors.append(error)

        thread = threading.Thread(target=consume)
        thread.start()
        self.assertTrue(two_started.wait(2))
        self.assertTrue(three_consumed.wait(2))
        with lock:
            self.assertEqual(consumed, 3)
            self.assertEqual(maximum_active, 2)
        release.set()
        thread.join(3)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([result["node_idx"] for result in results], list(range(10)))

    def test_extraction_failure_skips_node_without_aborting(self):
        def worker(node):
            if node.index_id == 2:
                raise LookupError("provider unavailable")
            return {"node_idx": node.index_id, "entities": [], "relations": []}

        extractor = self.make_extractor(worker)
        results = list(extractor.iter_extract_kg(_node(index) for index in range(8)))

        self.assertEqual([r["node_idx"] for r in results], list(range(8)))
        skipped = next(r for r in results if r["node_idx"] == 2)
        # The empty result carries the failure so the builder can tell it apart
        # from a node that simply named no entity; without that flag the loss is
        # silent and the node is never re-attempted.
        self.assertEqual(
            skipped,
            {
                "entities": [],
                "relations": [],
                "node_idx": 2,
                "extraction_failed": True,
            },
        )


class BoundedKGRefinementTests(unittest.TestCase):
    def make_refiner(self, entities, synthesizer):
        refiner = object.__new__(KGRefiner)
        refiner.graph_config = GraphConfig(
            refinement_max_workers=2,
            refinement_max_in_flight=3,
        )
        refiner.scope_filter = {
            "tenant_id": "tenant-a",
            "corpus_id": "corpus-a",
            "document_version_id": "revision-a",
            "source_sha256": "a" * 64,
            "index_kind": "entity",
        }
        refiner.entity_alias_map = {
            entity.entity_name: entity.entity_name for entity in entities
        }
        refiner.get_latest_entity_name = lambda name: name

        class GraphIndex:
            def __init__(self):
                self.entities = {entity.entity_name: entity for entity in entities}
                self.updates = []

            def get_entity_by_node_name(self, name):
                return self.entities[name]

            def get_node_name_from_entity(self, entity):
                return f"{entity.entity_name}:{entity.entity_type}"

            def update_entity(self, **kwargs):
                self.updates.append(kwargs)

        refiner.graph_index = GraphIndex()
        refiner._synthesize_entity_description = synthesizer
        refiner.entity_to_vdb_id = {
            refiner.graph_index.get_node_name_from_entity(entity): f"vdb-{index}"
            for index, entity in enumerate(entities)
        }

        class VDB:
            def __init__(self):
                self.updates = []

            def update_text_metadatas(self, ids, metadatas):
                self.updates.append((list(ids), list(metadatas)))

        refiner.vdb = VDB()
        refiner.deleted = []
        refiner.delete_entity_from_vdb = lambda entity: refiner.deleted.append(
            entity.entity_name
        )
        refiner.added_batches = []
        refiner.add_entities_to_vdb = lambda batch: refiner.added_batches.append(
            [entity.entity_name for entity in batch]
        )
        refiner._debug_check_num = lambda: None
        return refiner

    def test_refinement_applies_state_serially_without_reembedding(self):
        entities = [
            Entity(
                entity_name=f"entity-{index}",
                entity_type="TEST",
                description="left<SEP>right",
                source_ids={index},
            )
            for index in range(5)
        ]
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def synthesize(entity):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return f"refined {entity.entity_name}"

        refiner = self.make_refiner(entities, synthesize)
        refiner.refine_entities()

        self.assertEqual(maximum_active, 2)
        self.assertEqual(refiner.added_batches, [])
        self.assertEqual(refiner.deleted, [])
        self.assertEqual(
            [update[0][0] for update in refiner.vdb.updates],
            [f"vdb-{index}" for index in range(5)],
        )
        self.assertEqual(len(refiner.graph_index.updates), 5)
        self.assertEqual(
            [entity.description for entity in entities],
            [f"refined entity-{index}" for index in range(5)],
        )

    def test_refinement_failure_is_typed_and_not_written_as_content(self):
        entities = [
            Entity(
                entity_name=f"entity-{index}",
                entity_type="TEST",
                description="left<SEP>right",
                source_ids={index},
            )
            for index in range(3)
        ]

        def synthesize(entity):
            if entity.entity_name == "entity-1":
                raise TimeoutError("model timeout")
            return f"refined {entity.entity_name}"

        refiner = self.make_refiner(entities, synthesize)
        with self.assertRaises(KGRefinementError) as raised:
            refiner.refine_entities()

        self.assertEqual(raised.exception.entity_name, "entity-1")
        self.assertNotIn("model timeout", entities[1].description)
        self.assertEqual(refiner.added_batches, [])

    def test_metadata_update_failure_rolls_back_graph_description(self):
        entity = Entity(
            entity_name="entity",
            entity_type="TEST",
            description="original",
            source_ids={1},
        )
        refiner = self.make_refiner([entity], lambda _entity: "refined")

        def fail_update(*_args, **_kwargs):
            raise OSError("vector store unavailable")

        refiner.vdb.update_text_metadatas = fail_update
        with self.assertRaises(OSError):
            refiner._apply_refined_description(entity, "refined")

        self.assertEqual(entity.description, "original")
        self.assertEqual(len(refiner.graph_index.updates), 2)
        self.assertEqual(
            refiner.graph_index.updates[-1]["new_entity"].description,
            "original",
        )


class GraphExecutionConfigTests(unittest.TestCase):
    def test_limits_accept_small_memory_profiles_and_reject_zero(self):
        config = GraphConfig(
            extraction_max_workers=8,
            extraction_max_in_flight=2,
            extraction_batch_size=1,
            refinement_max_workers=4,
            refinement_max_in_flight=1,
        )
        self.assertEqual(config.extraction_max_in_flight, 2)
        self.assertEqual(config.refinement_max_in_flight, 1)

        for field in (
            "extraction_max_workers",
            "extraction_max_in_flight",
            "extraction_batch_size",
            "refinement_max_workers",
            "refinement_max_in_flight",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    GraphConfig(**{field: 0})


if __name__ == "__main__":
    unittest.main()
