import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
WEB_PATH = REPO / "Scripts" / "web.py"
PROGRESS_PATH = REPO / "Scripts" / "progress.sh"


def load_web_module():
    spec = importlib.util.spec_from_file_location("docarag_test_web", WEB_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def graph_payload(variant=None, node_count=1):
    return {
        "graph": {
            "nodes": [
                {
                    "id": f"entity-{index}",
                    "entity_name": f"Entity {index}",
                    "source_ids": [],
                }
                for index in range(node_count)
            ],
            "links": [],
        },
        "tree2kg": {},
        "variant": variant,
    }


class WebDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = load_web_module()

    def setUp(self):
        self.web._GRAPH_VALIDATION_CACHE.clear()
        self.web._ROLE_CACHE.clear()
        self.web.TokenTracker.get_instance().reset()

    def test_runs_root_and_discovery_honor_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "external-runs"
            indexed = runs / "group" / "run-a"
            write_json(indexed / "run-manifest.json", {"revision_id": "r1"})
            with patch.dict(os.environ, {"DOCARAG_RUNS_DIR": str(runs)}):
                self.assertEqual(self.web.runs_root(), runs.resolve())
                self.assertEqual(self.web.find_runs(), [indexed])

    def test_explicit_run_cannot_escape_configured_runs_root(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            runs = base / "runs"
            runs.mkdir()
            outside = base / "outside"
            outside.mkdir()
            with patch.dict(os.environ, {"DOCARAG_RUNS_DIR": str(runs)}):
                with self.assertRaises(SystemExit):
                    self.web.pick_run(outside)

    def test_compatibility_web_is_loopback_only(self):
        self.assertTrue(self.web.is_loopback_host("127.0.0.1"))
        self.assertTrue(self.web.is_loopback_host("::1"))
        self.assertFalse(self.web.is_loopback_host("0.0.0.0"))

    def test_web_query_traces_are_unique_and_owner_only(self):
        tracker = self.web.TokenTracker.get_instance()

        class Agent:
            def generation(self, question, query_dir):
                tracker.add_usage(3, 1)
                return f"answer: {question}", [1]

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            state = self.web.AppState(run)
            state.rag_agent = Agent()
            with patch.object(
                self.web,
                "inspect_run_readiness",
                return_value=SimpleNamespace(ready=True, missing=()),
            ):
                first = state.ask("first")
                second = state.ask("second")

            self.assertNotEqual(first["queryId"], second["queryId"])
            for result in (first, second):
                execution = Path(result["executionTrace"])
                saved_result = Path(result["resultTrace"])
                self.assertEqual(0o600, stat.S_IMODE(execution.stat().st_mode))
                self.assertEqual(0o600, stat.S_IMODE(saved_result.stat().st_mode))
                self.assertEqual(0o700, stat.S_IMODE(execution.parent.stat().st_mode))
                trace_payload = json.loads(saved_result.read_text(encoding="utf-8"))
                self.assertIn("provenance", trace_payload)
                self.assertEqual(
                    "persistent_web_process",
                    trace_payload["agent_lifecycle"]["mode"],
                )
                self.assertEqual(
                    4,
                    trace_payload["token_usage"]["totals"]["total_tokens"],
                )
                self.assertEqual(
                    trace_payload["token_usage"],
                    result["tokenUsage"],
                )
                self.assertEqual(trace_payload["run_log"], result["runLog"])
                self.assertEqual(
                    0o600,
                    stat.S_IMODE(Path(result["runLog"]).stat().st_mode),
                )
            self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_failed_web_query_persists_its_isolated_token_usage(self):
        tracker = self.web.TokenTracker.get_instance()

        class Agent:
            def generation(self, question, query_dir):
                tracker.add_usage(7, 2)
                raise RuntimeError("generation failed")

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            state = self.web.AppState(run)
            state.rag_agent = Agent()
            with (
                patch.object(
                    self.web,
                    "inspect_run_readiness",
                    return_value=SimpleNamespace(ready=True, missing=()),
                ),
                self.assertRaisesRegex(RuntimeError, "generation failed"),
            ):
                state.ask("failure")

            traces = list((run / "web_chat").glob("*/execution.json"))
            self.assertEqual(1, len(traces))
            payload = json.loads(traces[0].read_text(encoding="utf-8"))
            self.assertEqual("failed", payload["status"])
            self.assertEqual(
                {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                payload["token_usage"]["totals"],
            )
            self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_chat_refuses_tree_only_run_before_loading_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            state = self.web.AppState(run)
            with patch.object(
                self.web,
                "inspect_run_readiness",
                return_value=SimpleNamespace(
                    ready=False, missing=("persistent retrieval vector store",)
                ),
            ):
                with self.assertRaises(self.web.RunNotReadyError):
                    state.ensure_rag()

    def test_request_body_validation_distinguishes_bad_and_oversized_json(self):
        handler = object.__new__(self.web.Handler)
        handler.headers = {"Content-Length": str(self.web.MAX_REQUEST_BYTES + 1)}
        handler.rfile = io.BytesIO(b"")
        with self.assertRaises(self.web.RequestBodyError) as oversized:
            handler.read_json_body()
        self.assertEqual(413, oversized.exception.status)

        handler.headers = {"Content-Length": "1"}
        handler.rfile = io.BytesIO(b"{")
        with self.assertRaises(self.web.RequestBodyError) as invalid:
            handler.read_json_body()
        self.assertEqual(400, invalid.exception.status)

    def test_graph_selection_prefers_configured_variant_then_fresh_valid_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "run_config.yaml").write_text(
                "graph:\n  refine_type: basic\n", encoding="utf-8"
            )
            generic = run / "graph_data.json"
            basic = run / "graph_data_basic.json"
            write_json(basic, graph_payload("basic"))
            write_json(generic, graph_payload(None))
            os.utime(generic, (generic.stat().st_atime, generic.stat().st_mtime + 10))
            self.assertEqual(self.web.find_graph_artifact(run), basic)

            basic.unlink()
            fresh = run / "graph_data_experiment.json"
            write_json(fresh, graph_payload("experiment"))
            os.utime(fresh, (fresh.stat().st_atime, generic.stat().st_mtime + 10))
            (run / "graph_data_broken.json").write_text("{", encoding="utf-8")
            self.assertEqual(self.web.find_graph_artifact(run), fresh)

    def test_graph_selection_uses_published_role_and_ignores_stale_neighbor(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            stale = run / "graph_data.json"
            selected = run / "graphs" / "graph_data_0.7_qwen-custom.json"
            write_json(stale, graph_payload("stale"))
            write_json(selected, graph_payload("published"))
            write_json(
                run / "artifact-manifest.json",
                {"roles": {"graph": str(selected.relative_to(run))}},
            )

            self.assertEqual(self.web.find_graph_artifact(run), selected.resolve())

    def test_published_graph_role_rejects_symlink_even_inside_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            target = run / "graph_data_real.json"
            link = run / "graph_data_selected.json"
            write_json(target, graph_payload("real"))
            link.symlink_to(target.name)
            write_json(
                run / "artifact-manifest.json",
                {"roles": {"graph": link.name}},
            )

            self.assertIsNone(self.web.find_graph_artifact(run))

    def test_published_graph_role_and_validation_are_stat_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            selected = run / "graph_data_custom.json"
            write_json(selected, graph_payload("custom"))
            write_json(
                run / "artifact-manifest.json",
                {"roles": {"graph": selected.name}},
            )

            with patch.object(
                self.web, "read_json", wraps=self.web.read_json
            ) as reader:
                self.assertEqual(self.web.find_graph_artifact(run), selected.resolve())
                self.assertEqual(self.web.find_graph_artifact(run), selected.resolve())

            selected_reads = [
                call
                for call in reader.call_args_list
                if Path(call.args[0]) == selected
            ]
            self.assertEqual(len(selected_reads), 1)

    def test_chat_readiness_is_cached_until_published_identity_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            state = self.web.AppState(run)
            result = SimpleNamespace(ready=True, missing=())
            with patch.object(
                self.web, "inspect_run_readiness", return_value=result
            ) as inspect:
                self.assertIs(state.chat_readiness(), result)
                self.assertIs(state.chat_readiness(), result)
                self.assertEqual(inspect.call_count, 1)

                write_json(run / "index-state.json", {"overall_status": "running"})
                self.assertIs(state.chat_readiness(), result)
                self.assertEqual(inspect.call_count, 2)

    def test_retrieval_artifact_selection_uses_config_and_attestations(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "run_config.yaml").write_text(
                "vdb:\n  manifest_dir_name: indexes/custom-tree\n",
                encoding="utf-8",
            )
            expected = (run / "indexes" / "custom-tree").resolve()
            self.assertEqual(self.web.find_retrieval_artifact_dir(run), expected)

            (run / "run_config.yaml").unlink()
            retrieval = run / "document_attestation"
            entity = run / "entity_attestation_entities"
            write_json(retrieval / "docarag-qdrant-index.ready.json", {"ready": True})
            write_json(entity / "docarag-qdrant-index.ready.json", {"ready": True})
            self.assertEqual(
                self.web.find_retrieval_artifact_dir(run), retrieval.resolve()
            )

    def test_graph_payload_is_built_from_the_store(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(
                run / "run-manifest.json",
                {"source": {"path": "/data/sample.pdf"}},
            )
            state = self.web.AppState(run)
            state._tree_cache = {
                "docName": "sample.pdf",
                "runDir": str(run),
                "rootId": 0,
                "nodes": [
                    {"id": 2, "content": "Speaker text", "summary": ""},
                ],
            }
            tree = SimpleNamespace(
                graph_entities=lambda: [
                    {
                        "canonical": "speaker",
                        "entity_kind": "DEVICE",
                        "description": "outputs sound",
                        "node_ids": [2],
                    },
                    {
                        "canonical": "sound",
                        "entity_kind": "CONCEPT",
                        "description": "",
                        "node_ids": [],
                    },
                ],
                graph_statements=lambda: [
                    {
                        "source": "speaker",
                        "target": "sound",
                        "predicate": "produces",
                        "weight": 2.0,
                        "description": "a speaker produces sound",
                        "node_ids": [2],
                    }
                ],
            )
            state._store = SimpleNamespace(tree=tree, close=lambda: None)

            payload = state.graph_payload()

            self.assertEqual(2, len(payload["nodes"]))
            speaker = payload["nodes"][0]
            self.assertEqual("speaker", speaker["id"])
            self.assertEqual("DEVICE", speaker["type"])
            self.assertEqual(1, speaker["degree"])
            self.assertEqual(["Speaker text"], speaker["sourceSnippets"])
            link = payload["links"][0]
            self.assertEqual("produces", link["name"])
            self.assertEqual([2], link["sourceIds"])
            self.assertEqual("sample.pdf", payload["docName"])


class ProgressScriptTests(unittest.TestCase):
    def run_progress(self, run, runs_root):
        env = os.environ.copy()
        env["DOCARAG_RUNS_DIR"] = str(runs_root)
        return subprocess.run(
            ["bash", str(PROGRESS_PATH), str(run)],
            cwd=REPO,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_state_and_configured_artifacts_drive_progress_for_quoted_path(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "custom-runs"
            run = runs / "odd'run"
            (run / "logs").mkdir(parents=True)
            write_json(
                run / "tree.json",
                {"meta_info": {"file_name": "quoted.pdf"}, "nodes": [{"index_id": 0}]},
            )
            (run / "tree.pkl").write_bytes(b"tree")
            (run / "run_config.yaml").write_text(
                "index_type: bm25\n"
                "graph:\n  refine_type: basic\n"
                "vdb:\n  manifest_dir_name: vectors/custom\n",
                encoding="utf-8",
            )
            write_json(run / "graph_data_basic.json", graph_payload("basic", 2))
            vdb = run / "vectors" / "custom"
            vdb.mkdir(parents=True)
            (vdb / "bm25_index.pkl").write_bytes(b"lexical-index")
            write_json(
                run / "index-state.json",
                {
                    "schema_version": 1,
                    "overall_status": "succeeded",
                    "phase": "finished",
                    "current_stage": None,
                    "started_at": "2026-07-14T10:00:00+00:00",
                    "ended_at": "2026-07-14T10:01:00+00:00",
                    "stages": {
                        name: {
                            "status": "succeeded",
                            "started_at": "2026-07-14T10:00:00+00:00",
                            "ended_at": "2026-07-14T10:01:00+00:00",
                        }
                        for name in ("tree", "graph", "vdb")
                    },
                },
            )
            result = self.run_progress(run, runs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("doc : quoted.pdf", result.stdout)
            self.assertIn("graph_data_basic.json", result.stdout)
            self.assertIn("(bm25_index.pkl)", result.stdout)
            self.assertIn("index complete", result.stdout)

    def test_legacy_incomplete_run_is_unknown_not_failed_or_running(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"
            run = runs / "legacy"
            (run / "logs").mkdir(parents=True)
            (run / "logs" / "index-driver.log").write_text(
                "[driver] Tue Jul 14 10:00:00 IST 2026 starting tree\n",
                encoding="utf-8",
            )
            result = self.run_progress(run, runs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("legacy run activity is unknown", result.stdout)
            self.assertNotIn("tree   ❌ FAILED", result.stdout)
            self.assertNotIn("tree   ⏳ running", result.stdout)

    def test_running_state_wins_over_stale_graph_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"
            run = runs / "active"
            (run / "logs").mkdir(parents=True)
            write_json(run / "tree.json", {"nodes": [{"index_id": 0}]})
            (run / "tree.pkl").write_bytes(b"tree")
            write_json(run / "graph_data.json", graph_payload(None, 1))
            write_json(
                run / "index-state.json",
                {
                    "schema_version": 1,
                    "overall_status": "running",
                    "phase": "indexing",
                    "current_stage": "graph",
                    "started_at": "2026-07-14T10:00:00+00:00",
                    "stages": {
                        "tree": {"status": "succeeded"},
                        "graph": {"status": "running"},
                        "vdb": {"status": "pending"},
                    },
                },
            )
            result = self.run_progress(run, runs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("graph  ⏳ extracting", result.stdout)
            self.assertNotIn("graph  ✅ done", result.stdout)
            self.assertNotIn("pgrep", PROGRESS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
