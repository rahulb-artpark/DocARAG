from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from threading import Barrier
import types
import unittest
from unittest.mock import patch

from Core.provider.TokenTracker import TokenTracker
from docarag import cli
from docarag import compat
from docarag.settings import (
    RuntimeSettings,
    discover_settings_path,
    initialize_settings,
    load_settings,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROUTED_PROFILE = (
    REPOSITORY_ROOT / "config" / "routed-balanced.example.toml"
)


def qualified_routed_profile_text(*, token_env: str | None = None) -> str:
    value = EXAMPLE_ROUTED_PROFILE.read_text(encoding="utf-8")
    value = value.replace(
        'expected_adapter_version = "replace-with-qualified-version"',
        'expected_adapter_version = "1.1.0"',
        1,
    )
    value = value.replace(
        'expected_adapter_version = "replace-with-qualified-version"',
        'expected_adapter_version = "1.0.0"',
    )
    value = value.replace("0" * 64, "a" * 64)
    if token_env is not None:
        value = value.replace(
            '# bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
            f'bearer_token_env = "{token_env}"',
            1,
        )
    return value


def make_run(root: Path, name: str = "run-1", *, snapshot: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "tree.pkl").write_bytes(b"legacy-pickle-placeholder")
    (run / "tree.json").write_text('{"nodes": [{}]}', encoding="utf-8")
    (run / "graph_data.json").write_text(
        '{"graph": {"nodes": [{"id": "entity"}], "links": []}, "tree2kg": {}}',
        encoding="utf-8",
    )
    if snapshot:
        (run / "run_config.yaml").write_text("index_type: gbc\n", encoding="utf-8")
    return run


def fake_runtime_modules(agent, events: list[str]) -> dict[str, types.ModuleType]:
    core = types.ModuleType("Core")
    core.__path__ = []
    configs = types.ModuleType("Core.configs")
    configs.__path__ = []
    system_config = types.ModuleType("Core.configs.system_config")
    rag = types.ModuleType("Core.rag")
    provider = types.ModuleType("Core.provider")
    provider.__path__ = []
    token_tracker = types.ModuleType("Core.provider.TokenTracker")
    utils = types.ModuleType("Core.utils")
    utils.__path__ = []
    resource_loader = types.ModuleType("Core.utils.resource_loader")

    def load_config(path):
        events.append("load_config")
        return types.SimpleNamespace(
            save_path=None,
            rag=types.SimpleNamespace(strategy_config=object()),
            llm=object(),
            vlm=object(),
        )

    def prepare(*, cfg):
        events.append("prepare_dependencies")
        return {"prepared": True}

    def create(**kwargs):
        events.append("create_agent")
        return agent

    system_config.load_system_config = load_config
    resource_loader.prepare_rag_dependencies = prepare
    rag.create_rag_agent = create
    token_tracker.TokenTracker = TokenTracker
    return {
        "Core": core,
        "Core.configs": configs,
        "Core.configs.system_config": system_config,
        "Core.rag": rag,
        "Core.provider": provider,
        "Core.provider.TokenTracker": token_tracker,
        "Core.utils": utils,
        "Core.utils.resource_loader": resource_loader,
    }


class SettingsTests(unittest.TestCase):
    def test_initialization_and_relative_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_path = root / "docarag.toml"
            initialize_settings(settings_path, project_root=root)

            settings = load_settings(settings_path)

            self.assertEqual(settings.project_root, root.resolve())
            self.assertEqual(settings.system_config, root / "config/ollama-gbc.yaml")
            self.assertEqual(settings.runs_dir, root / "runs")
            self.assertEqual(settings.conda_environment, "gbc-rag")

    def test_initialization_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            initialize_settings(target)
            with self.assertRaises(FileExistsError):
                initialize_settings(target)

    def test_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text("schema_version = 99\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_settings(target)

    def test_unknown_runtime_key_is_rejected_instead_of_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text(
                "schema_version = 1\n"
                "[runtime]\n"
                'conda_enviroment = "typo-env"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conda_enviroment"):
                load_settings(target)

    def test_unimplemented_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text(
                'schema_version = 1\nprofile = "production"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "production"):
                load_settings(target)

    def test_blank_runtime_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text(
                'schema_version = 1\n[runtime]\nconda_environment = ""\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conda_environment"):
                load_settings(target)

    def test_generated_toml_safely_quotes_project_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unusual_root = root / 'quoted "project" with \\ slash'
            settings_path = root / "docarag.toml"

            initialize_settings(settings_path, project_root=unusual_root)
            settings = load_settings(settings_path)

            self.assertEqual(unusual_root.resolve(), settings.project_root)
            self.assertIn('\\"project\\"', settings_path.read_text(encoding="utf-8"))
            self.assertIn("\\\\ slash", settings_path.read_text(encoding="utf-8"))

    def test_settings_file_precedence_is_explicit_then_env_then_cwd_then_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd = root / "cwd"
            repo = root / "repo"
            cwd.mkdir()
            repo.mkdir()
            explicit = root / "explicit.toml"
            environment = root / "environment.toml"
            cwd_file = cwd / "docarag.toml"
            repo_file = repo / "docarag.toml"
            for path in (explicit, environment, cwd_file, repo_file):
                path.write_text("schema_version = 1\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {"DOCARAG_SETTINGS": str(environment)}),
                patch("docarag.settings.repository_root", return_value=repo),
            ):
                self.assertEqual(explicit.resolve(), discover_settings_path(explicit, cwd=cwd))
                self.assertEqual(environment.resolve(), discover_settings_path(cwd=cwd))

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("docarag.settings.repository_root", return_value=repo),
            ):
                self.assertEqual(cwd_file.resolve(), discover_settings_path(cwd=cwd))
                cwd_file.unlink()
                self.assertEqual(repo_file.resolve(), discover_settings_path(cwd=cwd))


class CompatibilityTests(unittest.TestCase):
    def test_ask_requires_complete_legacy_book_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            partial = root / "partial"
            partial.mkdir(parents=True)
            (partial / "tree.json").write_text(
                '{"nodes": [{}]}', encoding="utf-8"
            )

            with self.assertRaisesRegex(FileNotFoundError, "tree.pkl") as raised:
                compat.resolve_run(root, str(partial), purpose="ask")

            message = str(raised.exception)
            self.assertIn("graph_data", message)
            self.assertIn("ready scoped retrieval index", message)
            self.assertIn("ready scoped entity index", message)
            self.assertEqual(
                partial.resolve(),
                compat.resolve_run(root, str(partial), purpose="web"),
            )

    def test_run_must_resolve_below_configured_runs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "runs"
            root.mkdir()
            outside = make_run(base / "outside-runs")

            with self.assertRaisesRegex(ValueError, "descendant"):
                compat.resolve_run(root, str(outside), purpose="ask")

    def test_relative_run_name_and_snapshot_are_resolved_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root, "nested/run-a")
            fallback = Path(directory) / "current.yaml"
            fallback.write_text("index_type: other\n", encoding="utf-8")

            with patch.object(
                compat,
                "_unpublished_store_readiness",
                return_value=(True, True, ()),
            ):
                selected = compat.resolve_run(root, "nested/run-a", purpose="ask")
            config = compat.resolve_run_config(
                selected,
                fallback,
                allow_legacy_fallback=False,
            )

            self.assertEqual(run.resolve(), selected)
            self.assertEqual((run / "run_config.yaml").resolve(), config)

    def test_legacy_config_fallback_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root, snapshot=False)
            fallback = Path(directory) / "current.yaml"
            fallback.write_text("index_type: gbc\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "--allow-legacy-config"):
                compat.resolve_run_config(
                    run, fallback, allow_legacy_fallback=False
                )
            self.assertEqual(
                fallback.resolve(),
                compat.resolve_run_config(
                    run, fallback, allow_legacy_fallback=True
                ),
            )

    def test_hyphenated_index_state_can_declare_ask_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root, "current-run")
            (run / "index-state.json").write_text(
                json.dumps(
                    {
                        "overall_status": "succeeded",
                        "ended_at": "2026-07-14T10:01:00Z",
                        "stages": {
                            stage: {
                                "status": "succeeded",
                                "ended_at": "2026-07-14T10:01:00Z",
                            }
                            for stage in ("tree", "graph", "vdb")
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                compat,
                "_unpublished_store_readiness",
                return_value=(True, True, ()),
            ):
                readiness = compat.inspect_run_readiness(run, purpose="ask")

            self.assertTrue(readiness.ready)
            self.assertEqual(
                "state:index-state.json+verified_artifacts", readiness.source
            )
            with patch.object(
                compat,
                "_unpublished_store_readiness",
                return_value=(True, True, ()),
            ):
                self.assertEqual(
                    run.resolve(), compat.resolve_run(root, None, purpose="ask")
                )

    def test_success_state_does_not_hide_missing_or_deleted_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = root / "incomplete-run"
            run.mkdir(parents=True)
            (run / "index-state.json").write_text(
                json.dumps(
                    {
                        "overall_status": "succeeded",
                        "stages": {
                            stage: {"status": "succeeded"}
                            for stage in ("tree", "graph", "vdb")
                        },
                    }
                ),
                encoding="utf-8",
            )

            readiness = compat.inspect_run_readiness(run, purpose="ask")

            self.assertFalse(readiness.ready)
            self.assertIn("tree.pkl", readiness.missing)
            self.assertEqual(
                "state:index-state.json+verified_artifacts", readiness.source
            )

    def test_running_or_failed_state_blocks_queries_even_if_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            for status in ("running", "failed"):
                run = make_run(root, status)
                (run / "index-state.json").write_text(
                    json.dumps(
                        {
                            "overall_status": status,
                            "stages": {
                                "tree": {"status": "succeeded"},
                                "graph": {"status": "succeeded"},
                                "vdb": {"status": "running"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                readiness = compat.inspect_run_readiness(run, purpose="ask")

                self.assertFalse(readiness.ready)
                self.assertIn(status, readiness.missing[0])

    def test_default_run_order_ignores_query_created_directory_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            older = make_run(root, "older")
            newer = make_run(root, "newer")
            for artifact in (older / "tree.pkl", older / "tree.json"):
                os.utime(artifact, (100, 100))
            for artifact in (newer / "tree.pkl", newer / "tree.json"):
                os.utime(artifact, (200, 200))

            # Query traces mutate the directory but not publication identity.
            (older / "cli_queries").mkdir()
            os.utime(older, (300, 300))

            with patch.object(
                compat,
                "_unpublished_store_readiness",
                return_value=(True, True, ()),
            ):
                runs = compat.find_runs(root, purpose="ask")

            self.assertEqual([newer.resolve(), older.resolve()], runs)

    def test_end_to_end_timing_starts_before_runtime_and_index_loading(self):
        events: list[str] = []

        class Agent:
            def generation(self, question, query_dir):
                events.append("generation")
                return "answer", [4]

            def close(self):
                events.append("close")

        def clock():
            events.append("clock")
            return 10.0 if events.count("clock") == 1 else 13.5

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            config = Path(directory) / "run_config.yaml"
            config.write_text("index_type: gbc\n", encoding="utf-8")
            (run / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "revision_id": "revision-1",
                        "document_id": "document-1",
                        "source": {
                            "path": "/source/document.pdf",
                            "sha256": "source-digest",
                            "snapshot_path": "/snapshots/source-digest.pdf",
                        },
                        "controller": {
                            "code_fingerprint": {"sha256": "code-digest"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            modules = fake_runtime_modules(Agent(), events)
            with (
                patch.dict(sys.modules, modules),
                patch("docarag.compat.time.monotonic", side_effect=clock),
            ):
                result = compat.ask(
                    run_dir=run,
                    system_config=config,
                    question="question",
                )

            self.assertEqual("clock", events[0])
            self.assertLess(events.index("clock"), events.index("load_config"))
            self.assertEqual(3.5, result["elapsed_seconds"])
            self.assertEqual(
                ["load_config", "prepare_dependencies", "create_agent", "generation", "close"],
                [event for event in events if event != "clock"],
            )
            trace = json.loads(Path(result["execution_trace"]).read_text(encoding="utf-8"))
            saved_result = json.loads(
                Path(result["result_trace"]).read_text(encoding="utf-8")
            )
            self.assertEqual("succeeded", trace["status"])
            self.assertTrue(trace["cleanup"]["succeeded"])
            self.assertEqual(0, trace["token_usage"]["totals"]["total_tokens"])
            self.assertEqual(
                trace["token_usage"],
                saved_result["token_usage"],
            )
            self.assertEqual("revision-1", trace["provenance"]["revision_id"])
            self.assertEqual(
                hashlib.sha256(config.read_bytes()).hexdigest(),
                trace["system_config_sha256"],
            )
            self.assertEqual("question", saved_result["question"])
            self.assertEqual("answer", saved_result["answer"])
            self.assertEqual([4], saved_result["node_ids"])
            self.assertEqual(result["run_log"], trace["run_log"])
            self.assertTrue(Path(result["run_log"]).is_file())
            self.assertEqual(
                0o600,
                stat.S_IMODE(Path(result["run_log"]).stat().st_mode),
            )

    def test_query_failure_closes_agent_and_writes_structured_trace(self):
        events: list[str] = []

        class Agent:
            def generation(self, question, query_dir):
                events.append("generation")
                TokenTracker.get_instance().add_usage(9, 2)
                raise RuntimeError("model endpoint unavailable")

            def close(self):
                events.append("close")

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            config = Path(directory) / "run_config.yaml"
            config.write_text("index_type: gbc\n", encoding="utf-8")
            modules = fake_runtime_modules(Agent(), events)

            with (
                patch.dict(sys.modules, modules),
                self.assertRaises(compat.CompatibilityAskError) as raised,
            ):
                compat.ask(
                    run_dir=run,
                    system_config=config,
                    question="question",
                )

            self.assertEqual("generate_answer", raised.exception.details["stage"])
            self.assertEqual("RuntimeError", raised.exception.details["error"]["type"])
            self.assertIn("close", events)
            failure_path = Path(raised.exception.details["failure_trace"])
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual("failed", failure["status"])
            self.assertTrue(failure["cleanup"]["succeeded"])
            self.assertEqual(
                {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
                failure["token_usage"]["totals"],
            )

    def test_concurrent_compatibility_queries_have_isolated_usage_traces(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        barrier = Barrier(2)
        events: list[str] = []

        class Agent:
            def generation(self, question, query_dir):
                barrier.wait()
                prompt_tokens = 4 if question == "left" else 40
                TokenTracker.get_instance().add_usage(prompt_tokens, 1)
                return f"answer-{question}", [prompt_tokens]

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            config = Path(directory) / "run_config.yaml"
            config.write_text("index_type: gbc\n", encoding="utf-8")
            modules = fake_runtime_modules(Agent(), events)

            with patch.dict(sys.modules, modules):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    left_future = executor.submit(
                        compat.ask,
                        run_dir=run,
                        system_config=config,
                        question="left",
                    )
                    right_future = executor.submit(
                        compat.ask,
                        run_dir=run,
                        system_config=config,
                        question="right",
                    )
                    left = left_future.result()
                    right = right_future.result()

        self.assertEqual(5, left["token_usage"]["totals"]["total_tokens"])
        self.assertEqual(41, right["token_usage"]["totals"]["total_tokens"])
        self.assertEqual(
            5,
            left["token_usage"]["stages"]["query"]["total_tokens"],
        )
        self.assertEqual(
            41,
            right["token_usage"]["stages"]["query"]["total_tokens"],
        )
        self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_keyboard_interrupt_closes_agent_records_trace_and_preserves_exit_code(self):
        events: list[str] = []

        class Agent:
            def generation(self, question, query_dir):
                raise KeyboardInterrupt()

            def close(self):
                events.append("close")

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            config = Path(directory) / "run_config.yaml"
            config.write_text("index_type: gbc\n", encoding="utf-8")
            modules = fake_runtime_modules(Agent(), events)

            with (
                patch.dict(sys.modules, modules),
                self.assertRaises(compat.CompatibilityAskError) as raised,
            ):
                compat.ask(run_dir=run, system_config=config, question="question")

            self.assertEqual(130, raised.exception.exit_code)
            self.assertEqual("close", events[-1])
            self.assertTrue(Path(raised.exception.details["failure_trace"]).is_file())


class CLITests(unittest.TestCase):
    def test_help_does_not_import_heavy_runtime(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(output):
            cli.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("doctor", output.getvalue())
        self.assertIn("routed-profile", output.getvalue())
        self.assertIn("verify", output.getvalue())
        self.assertIn("$DOCARAG_SETTINGS", output.getvalue())

    def test_routed_profile_help_explains_index_selection(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(output):
            cli.main(["routed-profile", "--help"])

        self.assertEqual(0, raised.exception.code)
        self.assertIn("--routed-profile", output.getvalue())
        self.assertIn("validate", output.getvalue())
        self.assertIn("show", output.getvalue())

    def test_example_routed_profile_is_valid_but_fails_configuration_readiness(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(
                [
                    "routed-profile",
                    "validate",
                    str(EXAMPLE_ROUTED_PROFILE),
                    "--json",
                ]
            )

        self.assertEqual(1, code)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["configuration_ready"])
        self.assertFalse(payload["runtime_compatibility_checked"])
        self.assertFalse(payload["live_readiness_checked"])
        self.assertTrue(payload["routed_indexing"]["enabled"])
        self.assertEqual(6, len(payload["issues"]))

    def test_qualified_static_pins_validate_without_claiming_live_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "routed.toml"
            profile_path.write_text(
                qualified_routed_profile_text(), encoding="utf-8"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(
                    ["routed-profile", "validate", str(profile_path), "--json"]
                )

        self.assertEqual(0, code)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["configuration_ready"])
        self.assertFalse(payload["runtime_compatibility_checked"])
        self.assertFalse(payload["live_readiness_checked"])
        self.assertTrue(payload["routed_indexing"]["enabled"])

    def test_routed_profile_show_never_resolves_or_prints_bearer_value(self):
        secret = "must-not-appear-in-routed-profile-output"
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "routed.toml"
            profile_path.write_text(
                qualified_routed_profile_text(token_env="DOCARAG_TEST_TOKEN"),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch.dict(os.environ, {"DOCARAG_TEST_TOKEN": secret}),
                contextlib.redirect_stdout(output),
            ):
                code = cli.main(["routed-profile", "show", str(profile_path)])

        self.assertEqual(0, code)
        rendered = output.getvalue()
        payload = json.loads(rendered)
        self.assertIn("DOCARAG_TEST_TOKEN", rendered)
        self.assertNotIn(secret, rendered)
        self.assertTrue(payload["routed_indexing"]["enabled"])

    def test_routed_profile_cli_never_echoes_rejected_parse_extra_secrets(self):
        sentinel = "rejected-profile-secret-must-never-serialize"
        for field_name in ("accessToken", "apiKey", "clientSecret"):
            with self.subTest(field_name=field_name), tempfile.TemporaryDirectory() as directory:
                profile_path = Path(directory) / "routed.toml"
                value = qualified_routed_profile_text().replace(
                    'timeout_seconds = 300.0',
                    'timeout_seconds = 300.0\n'
                    f'extra = {{ {field_name} = "{sentinel}" }}',
                    1,
                )
                profile_path.write_text(value, encoding="utf-8")
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = cli.main(
                        ["routed-profile", "show", str(profile_path)]
                    )

                rendered = stdout.getvalue() + stderr.getvalue()
                self.assertEqual(2, code)
                self.assertNotIn(sentinel, rendered)
                self.assertIn("literal secret field", rendered)

    def test_doctor_dispatches_explicit_profile_for_index_readiness(self):
        settings = RuntimeSettings(project_root=REPOSITORY_ROOT).resolved()
        report = types.SimpleNamespace(
            ok=True,
            as_dict=lambda: {
                "overall_status": "amber",
                "routed_indexing_enabled": True,
            },
        )
        output = io.StringIO()
        with (
            patch("docarag.cli._load", return_value=settings),
            patch(
                "docarag.diagnostics.doctor_for_routed_profile",
                return_value=report,
            ) as routed_doctor,
            patch("docarag.diagnostics.doctor_for_settings") as legacy_doctor,
            contextlib.redirect_stdout(output),
        ):
            code = cli.main(
                [
                    "doctor",
                    "--routed-profile",
                    str(EXAMPLE_ROUTED_PROFILE),
                    "--offline",
                    "--json",
                ]
            )

        self.assertEqual(0, code)
        self.assertTrue(json.loads(output.getvalue())["routed_indexing_enabled"])
        routed_doctor.assert_called_once()
        self.assertFalse(routed_doctor.call_args.kwargs["check_endpoints"])
        legacy_doctor.assert_not_called()

    def test_unreadable_settings_are_reported_without_a_traceback(self):
        stderr = io.StringIO()
        with (
            patch("docarag.cli.load_settings", side_effect=PermissionError("denied")),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(["settings"])

        self.assertEqual(2, code)
        self.assertIn("error: denied", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_verify_reports_missing_published_manifest_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = make_run(runs)
            settings = RuntimeSettings(
                project_root=root,
                runs_dir=runs,
                system_config=root / "current.yaml",
                python_executable=sys.executable,
            ).resolved()
            output = io.StringIO()
            with (
                patch("docarag.cli._load", return_value=settings),
                contextlib.redirect_stdout(output),
            ):
                code = cli.main(["verify", "--run", str(run), "--json"])

            self.assertEqual(1, code)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["valid"])
            self.assertIn("artifact-manifest", payload["errors"][0])

    def test_index_rejects_missing_document_without_subprocess(self):
        with patch("docarag.cli._run") as runner:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli.main(["index", "/definitely/missing.pdf"])
        self.assertEqual(code, 2)
        self.assertIn("Document not found", stderr.getvalue())
        runner.assert_not_called()

    def test_ask_rejects_blank_question_without_loading_settings(self):
        with patch("docarag.cli._load") as loader, contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = cli.main(["ask", "   "])
        self.assertEqual(2, code)
        loader.assert_not_called()

    def test_ask_delegates_run_validation_once_to_runtime_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = make_run(runs)
            settings = RuntimeSettings(
                project_root=root,
                runs_dir=runs,
                system_config=root / "current.yaml",
                python_executable="/configured/python",
            ).resolved()

            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                code = cli.main(["ask", "question", "--run", str(run)])

            self.assertEqual(0, code)
            command = runner.call_args.args[0]
            self.assertEqual("/configured/python", command[0])
            self.assertEqual(
                str(settings.system_config),
                command[command.index("--fallback-system-config") + 1],
            )
            self.assertEqual(
                str(runs.resolve()),
                command[command.index("--runs-dir") + 1],
            )
            self.assertEqual(str(run), command[command.index("--run") + 1])

    def test_ask_legacy_fallback_authority_is_delegated_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = make_run(runs, snapshot=False)
            fallback = root / "current.yaml"
            fallback.write_text("index_type: gbc\n", encoding="utf-8")
            settings = RuntimeSettings(
                project_root=root,
                runs_dir=runs,
                system_config=fallback,
                python_executable=sys.executable,
            ).resolved()

            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                delegated_without_fallback = cli.main(
                    ["ask", "question", "--run", str(run)]
                )
                delegated_with_fallback = cli.main(
                    [
                        "ask",
                        "question",
                        "--run",
                        str(run),
                        "--allow-legacy-config",
                    ]
                )

            self.assertEqual(0, delegated_without_fallback)
            self.assertEqual(0, delegated_with_fallback)
            self.assertEqual(2, runner.call_count)
            first_command = runner.call_args_list[0].args[0]
            second_command = runner.call_args_list[1].args[0]
            self.assertNotIn("--allow-legacy-config", first_command)
            self.assertIn("--allow-legacy-config", second_command)
            self.assertEqual(
                str(fallback.resolve()),
                second_command[second_command.index("--fallback-system-config") + 1],
            )

    def test_internal_ask_bridge_selects_run_snapshot_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "run"
            run.mkdir(parents=True)
            snapshot = run / "run_config.yaml"
            snapshot.write_text("index_type: gbc\n", encoding="utf-8")
            fallback = root / "fallback.yaml"
            fallback.write_text("index_type: other\n", encoding="utf-8")
            output = io.StringIO()

            with (
                patch("docarag.compat.resolve_run", return_value=run) as resolver,
                patch(
                    "docarag.compat.resolve_run_config", return_value=snapshot
                ) as config_resolver,
                patch("docarag.compat.ask", return_value={"answer": "ok"}) as asker,
                contextlib.redirect_stdout(output),
            ):
                code = compat._main(
                    [
                        "ask",
                        "--run",
                        str(run),
                        "--runs-dir",
                        str(root / "runs"),
                        "--fallback-system-config",
                        str(fallback),
                        "--json",
                        "question",
                    ]
                )

            self.assertEqual(0, code)
            resolver.assert_called_once_with(root / "runs", str(run), purpose="ask")
            config_resolver.assert_called_once_with(
                run,
                fallback,
                allow_legacy_fallback=False,
            )
            asker.assert_called_once_with(
                run_dir=run,
                system_config=snapshot,
                question="question",
            )

    def test_web_uses_snapshot_and_receives_configured_runs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = make_run(runs)
            settings = RuntimeSettings(
                project_root=root,
                runs_dir=runs,
                system_config=root / "current.yaml",
                python_executable="/configured/python",
            ).resolved()

            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                code = cli.main(["web", "--run", str(run)])

            self.assertEqual(0, code)
            environment = runner.call_args.kwargs["env"]
            self.assertEqual(str(runs.resolve()), environment["DOCARAG_RUNS_DIR"])
            self.assertEqual(
                str((run / "run_config.yaml").resolve()),
                environment["BOOKRAG_CONFIG"],
            )

    def test_web_rejects_non_loopback_host_before_loading_settings(self):
        with (
            patch("docarag.cli._load") as loader,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(["web", "--host", "0.0.0.0"])

        self.assertEqual(2, code)
        loader.assert_not_called()

    def test_web_accepts_highest_valid_tcp_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = make_run(runs)
            settings = RuntimeSettings(
                project_root=root,
                runs_dir=runs,
                system_config=root / "current.yaml",
                python_executable="/configured/python",
            ).resolved()

            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                code = cli.main(
                    ["web", "--run", str(run), "--port", "65535"]
                )

            self.assertEqual(0, code)
            command = runner.call_args.args[0]
            self.assertEqual("65535", command[command.index("--port") + 1])


if __name__ == "__main__":
    unittest.main()
