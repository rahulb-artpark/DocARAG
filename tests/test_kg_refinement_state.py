from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Core.Index.Graph import Entity, Graph
from Core.pipelines.kg_refiner import EntityResolutionStats, KGRefiner
from Core.pipelines.kg_refinement_state import (
    SCHEMA_VERSION,
    clear_state,
    load_state,
    save_state,
    state_path,
)


def _entity(name: str, type_: str = "METHOD", source_ids=(1,)) -> Entity:
    return Entity(
        entity_name=name,
        entity_type=type_,
        description=f"{name} description",
        source_ids=set(source_ids),
    )


def _graph(save_path: str, *entities: Entity) -> Graph:
    graph = Graph(save_path=save_path, variant="v1")
    for index, entity in enumerate(entities):
        graph.add_and_link(tree_node_id=index + 1, entities=[entity])
    return graph


class GraphPayloadRoundTripTests(unittest.TestCase):
    def test_entities_edges_and_gt_links_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _graph(tmp, _entity("llm"), _entity("rag", "CONCEPT"))
            payload = graph.to_payload()

            restored = Graph.from_payload(payload, save_path=tmp, variant="v1")

            self.assertEqual(graph.get_all_nodes(), restored.get_all_nodes())
            # tree2kg is BookRAG's GT-Link; losing it unbinds entities from the tree.
            self.assertEqual(dict(graph.tree2kg), dict(restored.tree2kg))

    def test_source_ids_come_back_as_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _graph(tmp, _entity("llm", source_ids=(1, 2)))
            restored = Graph.from_payload(
                graph.to_payload(), save_path=tmp, variant="v1"
            )
            for _name, data in restored.kg.nodes(data=True):
                self.assertIsInstance(data["source_ids"], set)


class SaveAndLoadTests(unittest.TestCase):
    def test_a_snapshot_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _graph(tmp, _entity("llm"), _entity("rag", "CONCEPT"))
            save_state(
                tmp,
                graph=graph,
                alias_map={"Name: a\nType: T": "Name: b\nType: T"},
                refined_node_ids={1, 2, 7},
                variant="v1",
            )

            state = load_state(tmp, variant="v1")

            self.assertIsNotNone(state)
            self.assertEqual({1, 2, 7}, state.refined_node_ids)
            self.assertEqual(
                {"Name: a\nType: T": "Name: b\nType: T"}, state.alias_map
            )
            self.assertEqual(graph.get_all_nodes(), state.graph.get_all_nodes())

    def test_absent_snapshot_reads_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_state(tmp, variant="v1"))

    def test_a_snapshot_from_another_variant_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_state(
                tmp,
                graph=_graph(tmp, _entity("llm")),
                alias_map={},
                refined_node_ids={1},
                variant="v1",
            )
            self.assertIsNone(load_state(tmp, variant="v2"))

    def test_a_snapshot_from_another_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_state(
                tmp,
                graph=_graph(tmp, _entity("llm")),
                alias_map={},
                refined_node_ids={1},
                variant="v1",
            )
            target = state_path(tmp)
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["schema_version"] = "something.else.v9"
            target.write_text(json.dumps(payload), encoding="utf-8")

            self.assertIsNone(load_state(tmp, variant="v1"))

    def test_unreadable_and_corrupt_snapshots_read_as_none(self):
        for body in ("{not json", json.dumps({"schema_version": SCHEMA_VERSION})):
            with self.subTest(body=body[:12]), tempfile.TemporaryDirectory() as tmp:
                state_path(tmp).write_text(body, encoding="utf-8")
                self.assertIsNone(load_state(tmp, variant=None))

    def test_saving_twice_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _graph(tmp, _entity("llm"))
            for refined in ({1}, {1, 2}):
                save_state(
                    tmp,
                    graph=graph,
                    alias_map={},
                    refined_node_ids=refined,
                    variant="v1",
                )
            self.assertEqual(
                [state_path(tmp).name],
                sorted(p.name for p in Path(tmp).iterdir()),
            )
            self.assertEqual({1, 2}, load_state(tmp, variant="v1").refined_node_ids)

    def test_clear_state_removes_the_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_state(
                tmp,
                graph=_graph(tmp, _entity("llm")),
                alias_map={},
                refined_node_ids={1},
                variant="v1",
            )
            clear_state(tmp)
            self.assertIsNone(load_state(tmp, variant="v1"))
            clear_state(tmp)  # idempotent


class _AddOnlyVDB:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.batches: list[int] = []

    def add_entity_points(self, records, **kwargs):
        self.batches.append(len(records))
        self.added.extend(r["name_text"] for r in records)
        return [
            f"vdb-{index}"
            for index in range(len(self.added) - len(records), len(self.added))
        ]


class RebuildVDBFromGraphTests(unittest.TestCase):
    def _refiner(self, graph: Graph, vdb: _AddOnlyVDB) -> KGRefiner:
        refiner = object.__new__(KGRefiner)
        refiner.graph_index = graph
        refiner.vdb = vdb
        refiner.entity_to_vdb_id = {}
        refiner.scope_filter = {"tenant_id": "t"}
        refiner.er_stats = EntityResolutionStats()
        return refiner

    def test_every_graph_entity_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _graph(tmp, _entity("llm"), _entity("rag", "CONCEPT"))
            vdb = _AddOnlyVDB()
            refiner = self._refiner(graph, vdb)

            restored = refiner.rebuild_vdb_from_graph()

            self.assertEqual(2, restored)
            self.assertEqual(set(graph.kg.nodes), set(vdb.added))
            self.assertEqual(set(graph.kg.nodes), set(refiner.entity_to_vdb_id))

    def test_an_empty_graph_restores_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            refiner = self._refiner(_graph(tmp), _AddOnlyVDB())
            self.assertEqual(0, refiner.rebuild_vdb_from_graph())

    def test_entities_are_sent_in_bounded_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            entities = [_entity(f"entity-{index}") for index in range(1100)]
            graph = _graph(tmp, *entities)
            vdb = _AddOnlyVDB()
            refiner = self._refiner(graph, vdb)

            restored = refiner.rebuild_vdb_from_graph()

            self.assertEqual(1100, restored)
            self.assertEqual([512, 512, 76], vdb.batches)


if __name__ == "__main__":
    unittest.main()
