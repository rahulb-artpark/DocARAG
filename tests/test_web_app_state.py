"""AppState's tree/graph payload caching must not outlive the run's own state.

A request made while a stage was still writing its evidence must not be
served forever afterward as if that were the complete, final answer --
exactly what happened for a real run where the graph panel showed nothing
long after the graph had actually finished building successfully.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from docarag.tree_store import NodeMeta, TreeNodeView

REPO = Path(__file__).resolve().parents[1]
WEB_PATH = REPO / "Scripts" / "web.py"


def load_web_module():
    spec = importlib.util.spec_from_file_location("docarag_test_web_app_state", WEB_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTree:
    """Records call counts so the test can assert whether a re-query happened."""

    def __init__(self, node_count: int, entity_count: int):
        self.node_count = node_count
        self.entity_count = entity_count
        self.nodes_calls = 0
        self.graph_calls = 0

    def nodes(self, node_ids=None):
        self.nodes_calls += 1
        views = [
            TreeNodeView(
                index_id=0, type="root", depth=0, meta_info=NodeMeta(), parent_id=None,
            )
        ]
        for i in range(1, self.node_count + 1):
            views.append(
                TreeNodeView(
                    index_id=i,
                    type="text",
                    depth=1,
                    meta_info=NodeMeta(content=f"node {i}"),
                    evidence={"page": 0, "content": f"node {i}"},
                    parent_id=0,
                )
            )
        return views

    def structures_for_nodes(self, node_ids):
        return {}

    def graph_entities(self, limit=8192):
        self.graph_calls += 1
        return [
            {
                "canonical": f"entity-{i}",
                "entity_kind": "CONCEPT",
                "description": "",
                "node_ids": [],
            }
            for i in range(self.entity_count)
        ]

    def graph_statements(self, limit=8192):
        return []


class _FakeStore:
    def __init__(self, tree: _FakeTree):
        self.tree = tree

    def close(self):
        pass


class AppStateCacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.web = load_web_module()

    def _state_with_store(self, tree: _FakeTree):
        state = self.web.AppState(Path("/does/not/matter"))
        state._store = _FakeStore(tree)
        return state

    def test_global_state_rejects_document_scoped_operations_until_selection(self):
        state = self.web.AppState()

        self.assertIsNone(state.run_dir)
        with self.assertRaisesRegex(
            self.web.RunNotReadyError, "Select a document from the Library"
        ):
            state.tree_payload()
        with self.assertRaisesRegex(
            self.web.RunNotReadyError, "Select a document from the Library"
        ):
            state.graph_payload()
        with self.assertRaisesRegex(
            self.web.RunNotReadyError, "Select a document from the Library"
        ):
            state.ensure_rag()

    def test_global_state_can_bind_to_a_run_after_startup(self):
        state = self.web.AppState()
        selected = Path("/a/selected/run")

        state.switch_run(selected)

        self.assertEqual(selected, state.run_dir)

    def test_tree_payload_is_not_reused_after_the_runs_state_changes(self):
        tree = _FakeTree(node_count=1, entity_count=0)
        state = self._state_with_store(tree)
        with patch.object(self.web, "query_readiness_stat_key", return_value=("key-1",)):
            state.tree_payload()
            state.tree_payload()
        self.assertEqual(1, tree.nodes_calls, "second call within the same key must be cached")

        with patch.object(self.web, "query_readiness_stat_key", return_value=("key-2",)):
            state.tree_payload()
        self.assertEqual(2, tree.nodes_calls, "a changed key must trigger a fresh query")

    def test_tree_payload_advertises_crop_published_only_in_tree_json(self):
        tree = _FakeTree(node_count=1, entity_count=0)
        state = self._state_with_store(tree)
        published = {
            "nodes": [
                {"index_id": 1, "meta_info": {"img_path": "vlm/images/crop.jpg"}}
            ]
        }
        with (
            patch.object(self.web, "query_readiness_stat_key", return_value=("key",)),
            patch.object(self.web, "_read_json_if_present", return_value=published),
        ):
            payload = state.tree_payload()

        node = next(item for item in payload["nodes"] if item["id"] == 1)
        self.assertTrue(node["hasImage"])

    def test_graph_payload_reflects_data_that_only_appeared_after_the_first_query(self):
        # Simulates the actual incident: the first request lands while the
        # graph stage is still running (no entities persisted yet), the
        # second lands after it has finished (entities now exist in
        # Postgres). The stale cache must not keep serving the empty result.
        tree = _FakeTree(node_count=1, entity_count=0)
        state = self._state_with_store(tree)
        with patch.object(self.web, "query_readiness_stat_key", return_value=("building",)):
            empty_payload = state.graph_payload()
        self.assertEqual([], empty_payload["nodes"])

        tree.entity_count = 5
        with patch.object(self.web, "query_readiness_stat_key", return_value=("building",)):
            still_cached = state.graph_payload()
        self.assertEqual([], still_cached["nodes"], "same key must still be served from cache")

        with patch.object(self.web, "query_readiness_stat_key", return_value=("finished",)):
            fresh_payload = state.graph_payload()
        self.assertEqual(5, len(fresh_payload["nodes"]))
        self.assertEqual(2, tree.graph_calls)

    def test_switch_run_clears_both_cache_keys_along_with_the_cached_payloads(self):
        tree = _FakeTree(node_count=1, entity_count=1)
        state = self._state_with_store(tree)
        with patch.object(self.web, "query_readiness_stat_key", return_value=("key-1",)):
            state.tree_payload()
            state.graph_payload()
        self.assertIsNotNone(state._tree_cache_key)
        self.assertIsNotNone(state._graph_cache_key)

        state.switch_run(Path("/a/different/run"))

        self.assertIsNone(state._tree_cache_key)
        self.assertIsNone(state._graph_cache_key)
        self.assertIsNone(state._tree_cache)
        self.assertIsNone(state._graph_cache)




class AnswerTimeBudgetsComeFromTheCurrentConfig(unittest.TestCase):
    """A run's config is authoritative for what was built, not for how it is read.

    Embedder, collection and graph variant must come from the run: they describe
    artifacts on disk. An evidence budget does not -- it bounds what the answer
    model reads now. Freezing it meant a 43-node section query kept being refused
    against a limit of 32 pinned into the run long before that limit was found to
    be the wrong measure, and no correction could ever reach an indexed document.
    """

    @classmethod
    def setUpClass(cls):
        cls.web = load_web_module()

    @staticmethod
    def _cfg(limit, char_budget):
        return SimpleNamespace(
            rag=SimpleNamespace(
                strategy_config=SimpleNamespace(
                    global_evidence_limit=limit,
                    global_evidence_char_budget=char_budget,
                    topk=7,
                )
            )
        )

    def test_a_stale_pinned_budget_is_replaced(self):
        cfg = self._cfg(32, 1000)
        current = self._cfg(512, 48_000)
        # Patch where it is imported from: the helper imports it inside the
        # function, so patching the web module would be a no-op that still
        # passes whenever the real config happens to match.
        with patch("Core.configs.system_config.load_system_config", return_value=current):
            self.web.refresh_answer_time_budgets(cfg)
        self.assertEqual(cfg.rag.strategy_config.global_evidence_limit, 512)
        self.assertEqual(cfg.rag.strategy_config.global_evidence_char_budget, 48_000)

    def test_nothing_else_in_the_run_config_is_touched(self):
        # Only the answer-time budgets travel; the rest of the run's config
        # describes what was built and must survive untouched.
        cfg = self._cfg(32, 1000)
        current = self._cfg(512, 48_000)
        current.rag.strategy_config.topk = 999
        # Patch where it is imported from: the helper imports it inside the
        # function, so patching the web module would be a no-op that still
        # passes whenever the real config happens to match.
        with patch("Core.configs.system_config.load_system_config", return_value=current):
            self.web.refresh_answer_time_budgets(cfg)
        self.assertEqual(cfg.rag.strategy_config.topk, 7)

    def test_an_unreadable_current_config_leaves_the_run_s_values(self):
        cfg = self._cfg(32, 1000)
        with patch(
            "Core.configs.system_config.load_system_config", side_effect=OSError("gone")
        ):
            self.web.refresh_answer_time_budgets(cfg)
        self.assertEqual(cfg.rag.strategy_config.global_evidence_limit, 32)

    def test_a_config_without_a_strategy_section_is_ignored(self):
        cfg = SimpleNamespace(rag=None)
        self.web.refresh_answer_time_budgets(cfg)  # must not raise

class DeletingARunStopsTheWorkProducingIt(unittest.TestCase):
    """A deleted run whose indexer keeps going holds the GPU against everything.

    Deleting a document removed its directory and left its driver alive: still
    holding the indexing/GPU admission lock, still writing MinerU output into a
    path whose manifest, inputs and logs had just been deleted, and unable to
    finish usefully. Every document submitted afterwards queued behind a lock
    held by a run that no longer existed.
    """

    @classmethod
    def setUpClass(cls):
        cls.web = load_web_module()

    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.run_dir, ignore_errors=True)

    def _state(self, **fields):
        (self.run_dir / "index-state.json").write_text(json.dumps(fields), encoding="utf-8")

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _detached_driver(self):
        """A driver leading its own group with a child, naming its run in argv."""

        proc = subprocess.Popen(
            ["bash", "-c", f"sleep 60 & echo $! > {self.run_dir}/child.pid; wait"],
            start_new_session=True,
        )
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        for _ in range(50):
            if (self.run_dir / "child.pid").is_file():
                break
            time.sleep(0.1)
        child = int((self.run_dir / "child.pid").read_text().strip())
        return proc, child

    def test_the_whole_process_group_is_stopped_not_just_the_driver(self):
        # Terminating only the driver leaves the stage worker holding the GPU,
        # which is the state that had to be cleaned up by hand.
        proc, child = self._detached_driver()
        self._state(pid=proc.pid, overall_status="running")
        result = self.web.stop_indexing_run(self.run_dir)
        proc.wait(timeout=15)
        self.assertTrue(result["stopped"])
        self.assertFalse(self._alive(child))

    def test_a_finished_run_has_nothing_to_stop(self):
        self._state(pid=os.getpid(), overall_status="succeeded")
        self.assertIsNone(self.web.stop_indexing_run(self.run_dir))

    def test_a_run_with_no_state_is_not_an_error(self):
        self.assertIsNone(self.web.stop_indexing_run(self.run_dir))

    def test_a_pid_that_is_not_this_run_s_driver_is_never_signalled(self):
        # Process ids are reused. Signalling on a bare pid match would
        # eventually kill a stranger that merely inherited the number.
        other = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(other.kill)
        self._state(pid=other.pid, overall_status="running")
        self.assertIsNone(self.web.stop_indexing_run(self.run_dir))
        self.assertTrue(self._alive(other.pid))

    def test_a_nonsense_pid_is_ignored(self):
        for pid in (0, 1, -5, None, True, "1234"):
            with self.subTest(pid=pid):
                self._state(pid=pid, overall_status="running")
                self.assertIsNone(self.web.stop_indexing_run(self.run_dir))

    def test_a_zombie_does_not_read_as_running(self):
        # kill(pid, 0) succeeds on a zombie, so a liveness check built on
        # signalling can never observe the process it just killed go away.
        proc = subprocess.Popen(["true"])
        for _ in range(50):
            if proc.poll() is None:
                time.sleep(0.05)
            else:
                break
        # Not reaped yet from the OS's view until wait(); check before reaping.
        self.assertFalse(self.web._process_is_running(proc.pid))
        proc.wait()


if __name__ == "__main__":
    unittest.main()
