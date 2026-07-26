from __future__ import annotations

import unittest
from collections import defaultdict

from Core.Index.Graph import Entity, Graph
from Core.pipelines.kg_refiner import KGRefiner


class _RecordingVDB:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_text_by_ids(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


class _LLM:
    """Returns a queued reply, raises a queued error, and records ordering."""

    def __init__(self, reply=None, error: Exception | None = None, on_call=None):
        self.reply = reply
        self.error = error
        self.on_call = on_call
        self.calls = 0

    def get_json_completion(self, prompt=None, schema=None, **kwargs):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return self.reply


def _entity(name: str, type_: str, description: str, source_ids=(1,)) -> Entity:
    return Entity(
        entity_name=name,
        entity_type=type_,
        description=description,
        source_ids=set(source_ids),
    )


def _refiner(llm: _LLM, *seed: Entity) -> tuple[KGRefiner, Graph, _RecordingVDB]:
    """A KGRefiner with only what entity_merge touches, over a real graph."""

    refiner = object.__new__(KGRefiner)
    graph = Graph(save_path=None, variant=None)
    vdb = _RecordingVDB()
    refiner.graph_index = graph
    refiner.llm = llm
    refiner.vdb = vdb
    refiner.entity_to_vdb_id = defaultdict(str)
    refiner.entity_alias_map = defaultdict(str)
    for index, entity in enumerate(seed):
        graph.add_and_link(tree_node_id=index + 1, entities=[entity])
        refiner.entity_to_vdb_id[graph.get_node_name_from_entity(entity)] = (
            f"vdb-{entity.entity_name}"
        )
    return refiner, graph, vdb


class DirectMergeTests(unittest.TestCase):
    def test_same_node_name_never_calls_the_model(self):
        old = _entity("llm", "METHOD", "a decoder", source_ids=(1,))
        new = _entity("llm", "METHOD", "a transformer", source_ids=(2,))
        llm = _LLM()
        refiner, _graph, _vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new)

        self.assertEqual(0, llm.calls)
        self.assertEqual("llm", merged.entity_name)
        self.assertEqual("a decoder<SEP>a transformer", merged.description)
        self.assertEqual({1, 2}, set(merged.source_ids))

    def test_merged_to_old_entity_keeps_the_old_identity(self):
        old = _entity("large language model", "METHOD", "spelled out")
        new = _entity("llm", "METHOD", "abbreviated", source_ids=(2,))
        llm = _LLM()
        refiner, _graph, _vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new, merged_to_old_entity=True)

        self.assertEqual(0, llm.calls)
        self.assertEqual("large language model", merged.entity_name)
        self.assertEqual("spelled out<SEP>abbreviated", merged.description)


class LLMMergeTests(unittest.TestCase):
    def test_the_synthesized_identity_is_adopted(self):
        old = _entity("llm", "METHOD", "one")
        new = _entity("large language model", "CONCEPT", "two", source_ids=(2,))
        llm = _LLM(
            reply=Entity(
                entity_name="Large Language Model",
                entity_type="method or technique",
                description="ignored -- both source descriptions are kept",
                source_ids=set(),
            )
        )
        refiner, _graph, _vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new)

        self.assertEqual(1, llm.calls)
        self.assertEqual("large language model", merged.entity_name)
        self.assertEqual("METHOD_OR_TECHNIQUE", merged.entity_type)
        self.assertEqual("one<SEP>two", merged.description)
        self.assertEqual({1, 2}, set(merged.source_ids))

    def test_alias_map_points_both_inputs_at_the_merged_entity(self):
        old = _entity("llm", "METHOD", "one")
        new = _entity("large language model", "CONCEPT", "two")
        llm = _LLM(
            reply=Entity(
                entity_name="llm",
                entity_type="METHOD",
                description="d",
                source_ids=set(),
            )
        )
        refiner, graph, _vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new)

        merged_name = graph.get_node_name_from_entity(merged)
        for entity in (old, new, merged):
            self.assertEqual(
                merged_name,
                refiner.entity_alias_map[graph.get_node_name_from_entity(entity)],
            )


class SummarizationFailureTests(unittest.TestCase):
    """A budget-exhausted summarization must not corrupt or abort the stage."""

    def test_nothing_is_mutated_before_the_model_replies(self):
        old = _entity("llm", "METHOD", "one")
        new = _entity("large language model", "CONCEPT", "two")
        seen: dict[str, object] = {}

        refiner_box: list[KGRefiner] = []

        def snapshot() -> None:
            refiner = refiner_box[0]
            seen["deleted"] = list(refiner.vdb.deleted)
            seen["nodes"] = set(refiner.graph_index.get_all_nodes())
            seen["aliases"] = dict(refiner.entity_alias_map)

        llm = _LLM(error=RuntimeError("budget exhausted"), on_call=snapshot)
        refiner, graph, _vdb = _refiner(llm, old)
        refiner_box.append(refiner)
        nodes_before = set(graph.get_all_nodes())

        refiner.entity_merge(old, new)

        self.assertEqual([], seen["deleted"])
        self.assertEqual(nodes_before, seen["nodes"])
        self.assertEqual({}, seen["aliases"])

    def test_a_failed_summarization_still_merges(self):
        old = _entity("llm", "METHOD", "one", source_ids=(1,))
        new = _entity("large language model", "CONCEPT", "two", source_ids=(2,))
        llm = _LLM(error=RuntimeError("budget exhausted"))
        refiner, _graph, vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new)

        # Falls back to the old entity's identity, losing only the synthesized
        # name -- the coreference decision and both descriptions survive.
        self.assertEqual("llm", merged.entity_name)
        self.assertEqual("METHOD", merged.entity_type)
        self.assertEqual("one<SEP>two", merged.description)
        self.assertEqual({1, 2}, set(merged.source_ids))
        self.assertEqual(["vdb-llm"], vdb.deleted)

    def test_the_graph_still_holds_the_merged_entity_after_a_failure(self):
        old = _entity("llm", "METHOD", "one")
        new = _entity("large language model", "CONCEPT", "two")
        llm = _LLM(error=RuntimeError("budget exhausted"))
        refiner, graph, _vdb = _refiner(llm, old)

        merged = refiner.entity_merge(old, new)

        self.assertIn(graph.get_node_name_from_entity(merged), graph.get_all_nodes())


class ExistingTargetEntityTests(unittest.TestCase):
    def test_an_existing_third_entity_is_folded_in(self):
        old = _entity("llm", "METHOD", "one", source_ids=(1,))
        new = _entity("large language model", "CONCEPT", "two", source_ids=(2,))
        entity_c = _entity("gpt", "METHOD", "three", source_ids=(3,))
        llm = _LLM(
            reply=Entity(
                entity_name="gpt",
                entity_type="METHOD",
                description="d",
                source_ids=set(),
            )
        )
        refiner, _graph, vdb = _refiner(llm, old, entity_c)

        merged = refiner.entity_merge(old, new)

        self.assertEqual("one<SEP>two<SEP>three", merged.description)
        self.assertEqual({1, 2, 3}, set(merged.source_ids))
        self.assertEqual({"vdb-llm", "vdb-gpt"}, set(vdb.deleted))


if __name__ == "__main__":
    unittest.main()
