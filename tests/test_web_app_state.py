"""AppState's tree/graph payload caching must not outlive the run's own state.

A request made while a stage was still writing its evidence must not be
served forever afterward as if that were the complete, final answer --
exactly what happened for a real run where the graph panel showed nothing
long after the graph had actually finished building successfully.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
