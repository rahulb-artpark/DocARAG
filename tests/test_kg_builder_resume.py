from __future__ import annotations

import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval_engine.indexes.graph import Entity, Graph
from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.configuration.knowledge_graph import GraphConfig
from retrieval_engine.configuration.vector_store import VDBConfig
from retrieval_engine.indexing.graph import builder as kg_builder
from retrieval_engine.indexing.graph.refinement_state import load_state, save_state, state_path


def _tree(node_count: int) -> SimpleNamespace:
    root = SimpleNamespace(index_id=0, type=NodeType.ROOT)
    nodes = [root] + [
        SimpleNamespace(index_id=index, type=NodeType.TEXT)
        for index in range(1, node_count + 1)
    ]
    return SimpleNamespace(root_node=root, nodes=nodes)


def _config(directory: str, batch_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        save_path=directory,
        graph=GraphConfig(extraction_batch_size=batch_size, refine_type="advanced"),
        vdb=VDBConfig(),
        llm=object(),
        vlm=object(),
    )


class _Extractor:
    """Records which nodes were asked for, and yields empty extractions."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def iter_extract_kg(self, nodes):
        results = []
        for node in nodes:
            self.seen.append(node.index_id)
            results.append(
                {"entities": [], "relations": [], "node_idx": node.index_id}
            )
        return results

    def iter_extract_titles(self, work):
        return []


def _run(config, tree, extractor, refiner):
    scope = {
        "tenant_id": "test",
        "corpus_id": "test",
        "document_version_id": "document-v1",
        "source_sha256": "a" * 64,
        "index_kind": "tree",
    }
    with (
        patch.object(kg_builder, "LLM", return_value=object()),
        patch.object(kg_builder, "VLM", return_value=object()),
        patch.object(kg_builder, "KGExtractor", return_value=extractor),
        patch.object(kg_builder, "KGRefiner", return_value=refiner),
        patch.object(kg_builder, "tree_vector_scope", return_value=scope),
    ):
        return kg_builder.build_knowledge_graph(tree, config)


def _refiner_mock() -> Mock:
    refiner = Mock()
    refiner.entity_alias_map = {}
    return refiner


class RefinementResumeTests(unittest.TestCase):
    def test_a_completed_stage_leaves_no_snapshot_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            _run(config, _tree(4), _Extractor(), _refiner_mock())
            self.assertFalse(state_path(directory).exists())

    def test_a_snapshot_is_written_as_batches_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, batch_size=2)
            captured: list[set[int]] = []
            real_save = save_state

            def spy(save_path, **kwargs):
                captured.append(set(kwargs["refined_node_ids"]))
                real_save(save_path, **kwargs)

            with patch.object(kg_builder, "save_refinement_state", spy):
                _run(config, _tree(4), _Extractor(), _refiner_mock())

            self.assertEqual([{1, 2}, {1, 2, 3, 4}], captured)

    def test_a_resumed_run_skips_already_refined_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            variant = kg_builder.graph_artifact_variant(config.graph)
            graph = Graph(save_path=directory, variant=variant)
            graph.add_and_link(
                tree_node_id=1,
                entities=[
                    Entity(
                        entity_name="llm",
                        entity_type="METHOD",
                        description="d",
                        source_ids={1},
                    )
                ],
            )
            save_state(
                directory,
                graph=graph,
                alias_map={"Name: a\nType: T": "Name: b\nType: T"},
                refined_node_ids={1, 2},
                variant=variant,
            )

            extractor = _Extractor()
            refiner = _refiner_mock()
            _run(config, _tree(4), extractor, refiner)

            # Nodes 1 and 2 were already refined, so only 3 and 4 are re-run.
            self.assertEqual([3, 4], extractor.seen)
            refiner.rebuild_vdb_from_graph.assert_called_once_with()
            self.assertEqual(
                {"Name: a\nType: T": "Name: b\nType: T"}, refiner.entity_alias_map
            )

    def test_a_resumed_run_keeps_the_restored_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            variant = kg_builder.graph_artifact_variant(config.graph)
            graph = Graph(save_path=directory, variant=variant)
            graph.add_and_link(
                tree_node_id=1,
                entities=[
                    Entity(
                        entity_name="llm",
                        entity_type="METHOD",
                        description="d",
                        source_ids={1},
                    )
                ],
            )
            save_state(
                directory,
                graph=graph,
                alias_map={},
                refined_node_ids={1, 2, 3, 4},
                variant=variant,
            )

            result = _run(config, _tree(4), _Extractor(), _refiner_mock())

            self.assertEqual(graph.get_all_nodes(), result.get_all_nodes())

    def test_a_snapshot_for_another_variant_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            graph = Graph(save_path=directory, variant="other-variant")
            save_state(
                directory,
                graph=graph,
                alias_map={},
                refined_node_ids={1, 2, 3, 4},
                variant="other-variant",
            )

            extractor = _Extractor()
            refiner = _refiner_mock()
            _run(config, _tree(4), extractor, refiner)

            self.assertEqual([1, 2, 3, 4], extractor.seen)
            refiner.rebuild_vdb_from_graph.assert_not_called()

    def test_a_fresh_run_does_not_rebuild_the_vector_store(self):
        with tempfile.TemporaryDirectory() as directory:
            refiner = _refiner_mock()
            _run(_config(directory), _tree(2), _Extractor(), refiner)
            refiner.rebuild_vdb_from_graph.assert_not_called()


class ExtractionPrefetchTests(unittest.TestCase):
    def test_the_next_batch_extracts_while_the_current_one_is_refined(self):
        second_extraction = threading.Event()

        class _SignallingExtractor(_Extractor):
            def __init__(self) -> None:
                super().__init__()
                self.batches = 0

            def iter_extract_kg(self, nodes):
                self.batches += 1
                if self.batches == 2:
                    second_extraction.set()
                return super().iter_extract_kg(nodes)

        overlapped: list[bool] = []
        refiner = _refiner_mock()
        refiner.advanced_kg_refiner.side_effect = lambda *a, **k: overlapped.append(
            second_extraction.wait(timeout=5)
        )

        with tempfile.TemporaryDirectory() as directory:
            _run(
                _config(directory, batch_size=2),
                _tree(4),
                _SignallingExtractor(),
                refiner,
            )

        # Refining the first batch must not have to wait for the second batch's
        # extraction to be requested -- that is the whole point of the prefetch.
        self.assertTrue(overlapped[0])

    def test_refinement_still_runs_in_node_order(self):
        refiner = _refiner_mock()
        order: list[int] = []
        refiner.advanced_kg_refiner.side_effect = lambda **kwargs: order.append(
            kwargs["source_id"]
        )

        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory, batch_size=2), _tree(6), _Extractor(), refiner)

        self.assertEqual([1, 2, 3, 4, 5, 6], order)

    def test_an_extraction_failure_still_propagates(self):
        extractor = Mock()
        extractor.iter_extract_kg.side_effect = RuntimeError("model failed")
        extractor.iter_extract_titles.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                _run(
                    _config(directory), _tree(4), extractor, _refiner_mock()
                )


if __name__ == "__main__":
    unittest.main()
