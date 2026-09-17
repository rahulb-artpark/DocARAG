"""A node lost to extraction must not leave a graph that claims to be whole.

A 267-node document published a knowledge graph in which every chart and the one
diagram had no entities at all. Nothing in the run said so. Three separate
mechanisms had to line up for that:

* Page furniture reached ``_Extractor.extract``, which raises for it, so the
  failure path was the *normal* path for 34 nodes. Any count of failures was
  therefore meaningless, and nobody kept one.
* ``iter_extract_kg`` yields an empty result for a node that failed, which is
  shaped exactly like a node that named no entity. The builder counted it as
  extracted, logged "267/267", published, and cleared the resume state -- so the
  six content nodes could never be re-attempted.
* Every log record the extractor emitted was dropped. ``iter_bounded_ordered``
  copies the context of whichever thread submits work, and the builder submits
  from its own plain ``ThreadPoolExecutor`` pump thread, where the run-scope
  contextvar was never set. The six failures were logged and discarded.

These tests hold all three shut.
"""

from __future__ import annotations

import contextvars
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.configuration.knowledge_graph import GraphConfig
from retrieval_engine.configuration.vector_store import VDBConfig
from retrieval_engine.indexing.graph import builder as kg_builder
from retrieval_engine.indexing.graph.refinement_state import state_path


_PROBE: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="unset")


def _tree(*types: NodeType) -> SimpleNamespace:
    root = SimpleNamespace(index_id=0, type=NodeType.ROOT)
    nodes = [root] + [
        SimpleNamespace(index_id=index, type=node_type)
        for index, node_type in enumerate(types, start=1)
    ]
    return SimpleNamespace(root_node=root, nodes=nodes)


def _config(directory: str, batch_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        save_path=directory,
        graph=GraphConfig(extraction_batch_size=batch_size, refine_type="advanced"),
        vdb=VDBConfig(),
        llm=object(),
        vlm=object(),
    )


class _Extractor:
    """Yields empty extractions, failing the nodes it is told to fail."""

    def __init__(self, fail_ids=(), fail_times=None):
        self.seen: list[int] = []
        self.observed_context: list[str] = []
        self._fail_ids = set(fail_ids)
        self._remaining = {
            node_id: (fail_times if fail_times is not None else 99)
            for node_id in self._fail_ids
        }

    def iter_extract_kg(self, nodes):
        self.observed_context.append(_PROBE.get())
        results = []
        for node in nodes:
            self.seen.append(node.index_id)
            result = {"entities": [], "relations": [], "node_idx": node.index_id}
            if self._remaining.get(node.index_id, 0) > 0:
                self._remaining[node.index_id] -= 1
                result["extraction_failed"] = True
            results.append(result)
        return results

    def iter_extract_titles(self, work):
        return []


class _Stage:
    """Stands in for the durable stage, capturing what it was told."""

    def __init__(self):
        self.checkpoints = []

    def checkpoint(self, name, payload=None):
        self.checkpoints.append((name, dict(payload or {})))


def _run(config, tree, extractor, refiner=None, stage=None):
    scope = {
        "tenant_id": "test",
        "corpus_id": "test",
        "document_version_id": "document-v1",
        "source_sha256": "a" * 64,
        "index_kind": "tree",
    }
    refiner = refiner or _refiner_mock()
    with (
        patch.object(kg_builder, "LLM", return_value=object()),
        patch.object(kg_builder, "VLM", return_value=object()),
        patch.object(kg_builder, "KGExtractor", return_value=extractor),
        patch.object(kg_builder, "KGRefiner", return_value=refiner),
        patch.object(kg_builder, "tree_vector_scope", return_value=scope),
    ):
        return kg_builder.build_knowledge_graph(tree, config, stage=stage)


def _refiner_mock() -> Mock:
    refiner = Mock()
    refiner.entity_alias_map = {}
    return refiner


class FurnitureIsSkippedByDecisionNotByError(unittest.TestCase):
    def test_furniture_never_reaches_the_extractor(self):
        tree = _tree(
            NodeType.TEXT,
            NodeType.HEADER,
            NodeType.FOOTER,
            NodeType.TEXT,
            NodeType.ASIDE,
            NodeType.DISCARDED,
        )
        extractor = _Extractor()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, extractor)
        self.assertEqual([1, 4], extractor.seen)

    def test_content_that_looks_like_furniture_is_still_extracted(self):
        # table, equation and title were once declared "not extracted" although
        # every one of them produced a checkpoint on a real document.
        tree = _tree(NodeType.TABLE, NodeType.EQUATION, NodeType.CHART, NodeType.DIAGRAM)
        extractor = _Extractor()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, extractor)
        self.assertEqual([1, 2, 3, 4], extractor.seen)

    def test_the_progress_count_is_out_of_extractable_nodes_only(self):
        tree = _tree(NodeType.TEXT, NodeType.HEADER, NodeType.FOOTER)
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(kg_builder.log, "info", lambda msg, *a: messages.append(msg % a if a else msg)):
                _run(_config(directory), tree, _Extractor())
        committed = [m for m in messages if m.startswith("Committed KG")]
        self.assertTrue(committed)
        self.assertTrue(
            committed[-1].endswith("1/1 tree nodes"),
            f"progress counted furniture it never extracts: {committed[-1]!r}",
        )


class AFailedNodeIsRetriedThenRecordedAsAGap(unittest.TestCase):
    """Stopping the stage was worse than the loss it guarded against.

    Six figures a provider deterministically refused left a whole document with
    no index at all, and resuming could not help because the refusal repeated.
    The graph is published without them; what must not happen is the loss going
    unrecorded, which is the bug this file exists for.
    """

    def _gaps(self, stage):
        """The gap record as the run states it: a graph-stage event payload."""
        for name, payload in stage.checkpoints:
            if name == kg_builder.EXTRACTION_GAPS_EVENT:
                return payload
        return None

    def test_a_node_that_fails_once_is_retried_and_succeeds(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART, NodeType.TEXT)
        extractor = _Extractor(fail_ids={2}, fail_times=1)
        stage = _Stage()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, extractor, stage=stage)
            # A completed stage still clears its resume state.
            self.assertFalse(state_path(directory).exists())
        self.assertEqual(0, self._gaps(stage)["missing_count"])
        self.assertEqual([1, 2, 3, 2], extractor.seen)

    def test_a_node_that_keeps_failing_does_not_stop_the_stage(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART, NodeType.DIAGRAM)
        stage = _Stage()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, _Extractor(fail_ids={2, 3}), stage=stage)
        self.assertEqual(2, self._gaps(stage)["missing_count"])

    def test_the_record_names_the_nodes_that_were_lost(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART, NodeType.DIAGRAM)
        stage = _Stage()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, _Extractor(fail_ids={2, 3}), stage=stage)
        missing = self._gaps(stage)["missing"]
        self.assertEqual([2, 3], [entry["node_idx"] for entry in missing])
        self.assertEqual(["chart", "diagram"], [entry["node_type"] for entry in missing])

    def test_every_run_states_its_gaps_so_none_is_inherited(self):
        # The record belongs to one stage run, so a later run cannot be read
        # through an earlier run's leftovers -- which a file beside the graph
        # allowed.
        tree = _tree(NodeType.TEXT, NodeType.CHART)
        first, second = _Stage(), _Stage()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, _Extractor(fail_ids={2}), stage=first)
            _run(_config(directory), tree, _Extractor(), stage=second)
        self.assertEqual(1, self._gaps(first)["missing_count"])
        self.assertEqual(0, self._gaps(second)["missing_count"])

    def test_the_loss_is_reported_at_error_level(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART)
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                kg_builder.log, "error", lambda msg, *a: errors.append(msg % a if a else msg)
            ):
                _run(_config(directory), tree, _Extractor(fail_ids={2}), stage=_Stage())
        self.assertTrue(any("missing" in message for message in errors))

    def test_a_failed_node_is_not_recorded_as_refined(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART)
        captured: list[set[int]] = []

        def spy(save_path, **kwargs):
            captured.append(set(kwargs["refined_node_ids"]))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(kg_builder, "save_refinement_state", spy):
                _run(_config(directory), tree, _Extractor(fail_ids={2}))

        self.assertTrue(captured)
        for snapshot in captured:
            self.assertNotIn(2, snapshot, "a failed node must not pass as extracted")

    def test_a_failed_node_is_not_refined_into_the_graph(self):
        tree = _tree(NodeType.TEXT, NodeType.CHART)
        refiner = _refiner_mock()
        with tempfile.TemporaryDirectory() as directory:
            _run(_config(directory), tree, _Extractor(fail_ids={2}), refiner)
        refined_sources = [
            call.kwargs.get("source_id") for call in refiner.advanced_kg_refiner.call_args_list
        ]
        self.assertNotIn(2, refined_sources)


class TheExtractorsLogsSurviveThePumpThread(unittest.TestCase):
    def test_the_batch_runs_in_the_callers_context(self):
        # The run-log scope filters every record whose context does not carry the
        # run id. Losing the context in the pump thread silences the extractor
        # entirely -- which is why 40 failures left no trace anywhere.
        tree = _tree(NodeType.TEXT, NodeType.TEXT)
        extractor = _Extractor()
        token = _PROBE.set("inside-the-run")
        try:
            with tempfile.TemporaryDirectory() as directory:
                _run(_config(directory), tree, extractor)
        finally:
            _PROBE.reset(token)
        self.assertEqual(["inside-the-run"], extractor.observed_context)


if __name__ == "__main__":
    unittest.main()
