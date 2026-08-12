from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import tempfile
import unittest
from unittest.mock import ANY, Mock, patch

from Core import construct_index
from Core.provider.TokenTracker import TokenTracker
from Core.configs.graph_config import (
    GraphConfig,
    entity_attestation_directory,
    graph_artifact_variant,
)


class _CapturedStage:
    enabled = True
    replayed = False

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __enter__(self):
        self.calls.append(("enter", self.name))
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def checkpoint(self, name, payload=None):
        self.calls.append((f"checkpoint:{self.name}", name))

    def record_artifact(self, path, *, artifact_role, **kwargs):
        self.calls.append((f"artifact:{self.name}", artifact_role))

    def publish_index_revision(self, **kwargs):
        self.calls.append((f"publication:{self.name}", kwargs["qdrant_point_count"]))


class ConstructIndexStageTests(unittest.TestCase):
    def test_concurrent_index_runs_persist_isolated_token_stages(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        barrier = Barrier(2)
        saved = {}
        saved_lock = Lock()

        def build_tree(config, *, parsing_provider=None):
            barrier.wait()
            tracker.add_usage(config.prompt_tokens, 1)
            tracker.record_stage("tree")
            return object()

        def save_stats(*, save_path, new_stats):
            with saved_lock:
                saved[save_path] = new_stats

        left = SimpleNamespace(save_path="left", prompt_tokens=4)
        right = SimpleNamespace(save_path="right", prompt_tokens=40)
        with (
            patch.object(construct_index, "build_tree_from_pdf", side_effect=build_tree),
            patch.object(construct_index, "save_indexing_stats", side_effect=save_stats),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(construct_index.construct_GBC_index, left, True),
                executor.submit(construct_index.construct_GBC_index, right, True),
            ]
            for future in futures:
                future.result()

        self.assertEqual(
            5,
            saved["left"]["token_stage_history"]["tree"]["total_tokens"],
        )
        self.assertEqual(
            41,
            saved["right"]["token_stage_history"]["tree"]["total_tokens"],
        )
        self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_nested_index_run_reuses_outer_scope_without_resetting_totals(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        outer = tracker.scope(name="request")
        saved = {}

        def build_tree(config, *, parsing_provider=None):
            tracker.add_usage(5, 1)
            tracker.record_stage("tree")
            return object()

        with outer:
            tracker.add_usage(2, 1)
            tracker.record_stage("request_setup")
            with (
                patch.object(construct_index, "build_tree_from_pdf", side_effect=build_tree),
                patch.object(
                    construct_index,
                    "save_indexing_stats",
                    side_effect=lambda *, save_path, new_stats: saved.update(new_stats),
                ),
            ):
                construct_index.construct_GBC_index(
                    SimpleNamespace(save_path="run"),
                    tree_only=True,
                )
            self.assertIs(outer, tracker.active_scope())

        self.assertEqual(9, outer.get_usage()["total_tokens"])
        self.assertEqual(6, outer.stage_history["tree"]["total_tokens"])
        self.assertEqual(6, saved["token_stage_history"]["tree"]["total_tokens"])

    def test_injected_ingestion_provider_reaches_tree_boundary(self):
        config = SimpleNamespace(save_path="/tmp/run")
        provider = object()
        tracker = Mock(stage_history=[])

        with (
            patch.object(
                construct_index,
                "build_tree_from_pdf",
                return_value=object(),
            ) as build_tree,
            patch.object(
                construct_index.TokenTracker,
                "get_instance",
                return_value=tracker,
            ),
            patch.object(construct_index, "save_indexing_stats"),
        ):
            construct_index.construct_GBC_index(
                config,
                tree_only=True,
                parsing_provider=provider,
            )

        build_tree.assert_called_once_with(
            config,
            parsing_provider=provider,
        )

    def test_graph_only_publishes_graph_and_entity_index_together(self):
        config = SimpleNamespace(save_path="/tmp/run")
        tree = object()
        graph = SimpleNamespace(
            kg=SimpleNamespace(
                number_of_edges=lambda: 2,
                number_of_nodes=lambda: 3,
            ),
            save_graph=Mock(),
        )
        gbc = Mock()
        gbc.entity_attestation_path = "/tmp/run/entity-attestation"
        tracker = Mock(stage_history=[])
        stage_calls = []

        with (
            patch(
                "docarag.vdb_tree_source.StoreTreeSource.open", return_value=tree
            ),
            patch.object(
                construct_index, "build_knowledge_graph", return_value=graph
            ),
            patch.object(construct_index, "GBC", return_value=gbc) as gbc_factory,
            patch.object(construct_index.TokenTracker, "get_instance", return_value=tracker),
            patch.object(construct_index, "save_indexing_stats") as save_stats,
            patch.object(
                construct_index,
                "durable_stage",
                side_effect=lambda name, **_kwargs: _CapturedStage(name, stage_calls),
            ),
            patch.object(
                construct_index,
                "_persist_graph_evidence",
                side_effect=lambda _path: stage_calls.append(("persist", "graph")),
            ) as persist,
        ):
            construct_index.construct_GBC_index(config, graph_only=True)

        gbc_factory.assert_called_once_with(
            config=config,
            graph_index=graph,
            TreeIndex=tree,
        )
        graph.save_graph.assert_called_once_with()
        gbc.save_gbc_index.assert_not_called()
        gbc.rebuild_vdb.assert_called_once_with()
        gbc.close.assert_called_once_with()
        save_stats.assert_called_once()
        self.assertEqual(
            ["graph", "entity"],
            [value for action, value in stage_calls if action == "enter"],
        )
        # The entity index is derived from the store, so the graph's rows have
        # to be there before it is built. The orchestrator persists a stage's
        # evidence after that stage's process exits, which is one process too
        # late for the entity stage running inside it.
        persist.assert_called_once_with(config.save_path)
        order = [
            index
            for index, (action, value) in enumerate(stage_calls)
            if (action, value) in (("persist", "graph"), ("enter", "entity"))
        ]
        self.assertEqual(sorted(order), order)
        self.assertLess(
            stage_calls.index(("persist", "graph")),
            stage_calls.index(("enter", "entity")),
        )
        self.assertIn(
            ("artifact:graph", "bookrag-knowledge-graph"), stage_calls
        )
        self.assertIn(
            ("artifact:entity", "qdrant-entity-index-attestation"), stage_calls
        )

    def test_graph_artifact_and_entity_attestation_names_share_one_policy(self):
        basic = GraphConfig(refine_type="basic")
        self.assertEqual("basic", graph_artifact_variant(basic))
        self.assertEqual(
            "/index/entity_attestation_basic",
            entity_attestation_directory("/index", basic),
        )

        custom = GraphConfig(g=0.7)
        custom.embedding_config.model_name = "vendor/custom model"
        self.assertEqual(
            "0.7_vendor_custom_model",
            graph_artifact_variant(custom),
        )
        self.assertIn(
            "entity_attestation_0.7_vendor_custom model_",
            entity_attestation_directory("/index", custom),
        )

    def test_existing_attestation_repair_is_delegated_to_bound_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            attestation = run / "vector_attestation"
            attestation.mkdir(parents=True)
            stale = attestation / "partial.bin"
            stale.write_bytes(b"partial")
            config = SimpleNamespace(
                save_path=str(run),
                index_type="gbc",
                vdb=SimpleNamespace(
                    manifest_dir_name="vector_attestation",
                    collection_name="bookrag-tree",
                    force_rebuild=False,
                ),
            )
            tracker = Mock(stage_history=[])
            stage_calls = []

            with (
                patch.object(construct_index.TokenTracker, "get_instance", return_value=tracker),
                patch(
                    "docarag.vdb_tree_source.StoreTreeSource.open",
                    return_value=SimpleNamespace(nodes=[], store_seal=None, meta_info=None),
                ),
                patch.object(construct_index, "build_vdb_index") as build_vdb,
                patch.object(construct_index, "save_indexing_stats"),
                patch.object(
                    construct_index,
                    "read_vector_index_attestation",
                    return_value={"point_count": 7, "identity_sha256": "a" * 64},
                ),
                patch.object(
                    construct_index,
                    "durable_stage",
                    side_effect=lambda name, **_kwargs: _CapturedStage(
                        name, stage_calls
                    ),
                ),
            ):
                construct_index.construct_vdb(config)

            self.assertTrue(stale.exists())
            build_vdb.assert_called_once_with(
                ANY,
                config.vdb,
                reuse_if_ready=True,
            )
            self.assertIn(("enter", "vector"), stage_calls)
            self.assertIn(
                ("artifact:vector", "qdrant-tree-index-attestation"),
                stage_calls,
            )
            self.assertIn(("publication:vector", 7), stage_calls)


if __name__ == "__main__":
    unittest.main()
