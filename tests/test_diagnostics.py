import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from docarag.diagnostics import (
    CheckType,
    CommandOutput,
    CommandSpec,
    ConfigurationValueSpec,
    DiagnosticProbes,
    DiagnosticsOptions,
    EndpointSpec,
    GpuSpec,
    HttpResponse,
    ModelEndpointSpec,
    PathProbeResult,
    PythonSpec,
    Status,
    WritablePathSpec,
    doctor_for_settings,
    options_from_docarag_config,
    render_report,
    run_diagnostics,
)


DOCTOR_COMMANDS = {
    "bash": "/bin/bash",
    "python3": "/usr/bin/python3",
    "flock": "/usr/bin/flock",
    "setsid": "/usr/bin/setsid",
    "nohup": "/usr/bin/nohup",
}


def make_probes(
    *,
    python_version=(3, 12, 3),
    paths=None,
    commands=None,
    command_runner=None,
    http_get=None,
):
    paths = paths or {}
    commands = commands or {}

    def path_probe(path):
        return paths.get(
            str(path), PathProbeResult(True, str(path), path.exists())
        )

    return DiagnosticProbes(
        python_version=lambda: python_version,
        which=lambda executable: commands.get(executable),
        run_command=command_runner
        or (lambda command, timeout: CommandOutput(0, "ok\n", "")),
        http_get=http_get
        or (lambda url, headers, timeout: HttpResponse(200, b"{}")),
        writable_path=path_probe,
        monotonic=lambda: 10.0,
        timestamp=lambda: "2026-07-14T12:00:00+00:00",
    )


class DiagnosticsTest(unittest.TestCase):
    def test_report_is_json_serializable_and_human_readable(self):
        options = DiagnosticsOptions(
            python=PythonSpec(minimum=(3, 10), recommended=(3, 12)),
            configuration=(
                ConfigurationValueSpec("llm.model_name", "model-a"),
            ),
        )

        report = run_diagnostics(options, probes=make_probes())

        self.assertEqual(Status.GREEN, report.overall_status)
        self.assertEqual(0, report.exit_code)
        self.assertEqual("green", json.loads(report.to_json())["overall_status"])
        rendered = report.render()
        self.assertIn("DocARAG diagnostics: GREEN", rendered)
        self.assertIn("[GREEN] python", rendered)
        self.assertIn("2 green, 0 amber, 0 red", rendered)

    def test_old_python_is_red_and_supported_nonrecommended_is_amber(self):
        spec = PythonSpec(minimum=(3, 10), recommended=(3, 12))

        old = run_diagnostics(
            DiagnosticsOptions(python=spec),
            probes=make_probes(python_version=(3, 9, 19)),
        )
        supported = run_diagnostics(
            DiagnosticsOptions(python=spec),
            probes=make_probes(python_version=(3, 11, 9)),
        )

        self.assertEqual(Status.RED, old.checks[0].status)
        self.assertEqual(1, old.exit_code)
        self.assertEqual(Status.AMBER, supported.checks[0].status)
        self.assertEqual(0, supported.exit_code)

    def test_configuration_placeholder_is_red_without_exposing_secret(self):
        report = run_diagnostics(
            DiagnosticsOptions(
                configuration=(
                    ConfigurationValueSpec("llm.api_base", "TODO"),
                    ConfigurationValueSpec(
                        "llm.api_key", "super-secret", secret=True
                    ),
                )
            ),
            probes=make_probes(),
        )

        self.assertEqual(Status.RED, report.overall_status)
        serialized = report.to_json()
        self.assertNotIn("super-secret", serialized)
        self.assertIn("<redacted>", serialized)

    def test_writable_path_uses_injected_probe_and_reports_remediation(self):
        target = "/readonly/index"
        probes = make_probes(
            paths={
                target: PathProbeResult(
                    False, "/readonly", False, "permission denied"
                )
            }
        )
        options = DiagnosticsOptions(
            writable_paths=(WritablePathSpec("index", target),)
        )

        report = run_diagnostics(options, probes=probes)
        result = report.checks[1]

        self.assertEqual(CheckType.PATH, result.check_type)
        self.assertEqual(Status.RED, result.status)
        self.assertIn("grant", result.remediation)

    def test_real_writable_path_probe_leaves_no_doctor_file(self):
        with tempfile.TemporaryDirectory() as directory:
            options = DiagnosticsOptions(
                writable_paths=(WritablePathSpec("temporary", directory),)
            )

            report = run_diagnostics(options)

            self.assertEqual(Status.GREEN, report.checks[1].status)
            self.assertEqual([], list(Path(directory).iterdir()))

    def test_required_and_optional_missing_commands_have_different_status(self):
        options = DiagnosticsOptions(
            commands=(
                CommandSpec("required", "required-tool"),
                CommandSpec("optional", "optional-tool", required=False),
            )
        )

        report = run_diagnostics(options, probes=make_probes())

        self.assertEqual(Status.RED, report.checks[1].status)
        self.assertEqual(Status.AMBER, report.checks[2].status)

    def test_command_probe_captures_version_without_unbounded_output(self):
        options = DiagnosticsOptions(
            commands=(CommandSpec("docker", "docker", args=("--version",)),)
        )
        probes = make_probes(
            commands={"docker": "/usr/bin/docker"},
            command_runner=lambda command, timeout: CommandOutput(
                0, "Docker version 28.0\nsecond line", ""
            ),
        )

        result = run_diagnostics(options, probes=probes).checks[1]

        self.assertEqual(Status.GREEN, result.status)
        self.assertEqual("Docker version 28.0", result.details["output"])

    def test_endpoint_checks_reachability_and_http_status(self):
        def http_get(url, headers, timeout):
            if "healthy" in url:
                return HttpResponse(204)
            return HttpResponse(503)

        options = DiagnosticsOptions(
            endpoints=(
                EndpointSpec("healthy", "http://localhost:9000/healthy"),
                EndpointSpec(
                    "optional",
                    "http://localhost:9001/not-ready",
                    required=False,
                ),
            )
        )

        report = run_diagnostics(
            options, probes=make_probes(http_get=http_get)
        )

        self.assertEqual(Status.GREEN, report.checks[1].status)
        self.assertEqual(Status.AMBER, report.checks[2].status)

    def test_openai_and_ollama_model_inventory_checks(self):
        requested_urls = []

        def http_get(url, headers, timeout):
            requested_urls.append(url)
            if url.endswith("/api/tags"):
                return HttpResponse(
                    200, b'{"models":[{"name":"vision-model:latest"}]}'
                )
            return HttpResponse(200, b'{"data":[{"id":"text-model"}]}')

        options = DiagnosticsOptions(
            models=(
                ModelEndpointSpec(
                    "llm", "http://localhost:8000/v1", "text-model"
                ),
                ModelEndpointSpec(
                    "vlm",
                    "http://localhost:11434",
                    "vision-model",
                    backend="ollama",
                ),
            )
        )

        report = run_diagnostics(
            options, probes=make_probes(http_get=http_get)
        )

        self.assertEqual(Status.GREEN, report.checks[1].status)
        self.assertEqual(Status.GREEN, report.checks[2].status)
        self.assertEqual(
            [
                "http://localhost:8000/v1/models",
                "http://localhost:11434/api/tags",
            ],
            requested_urls,
        )

    def test_missing_configured_model_is_red(self):
        options = DiagnosticsOptions(
            models=(
                ModelEndpointSpec(
                    "llm", "http://localhost:8000/v1", "wanted-model"
                ),
            )
        )
        probes = make_probes(
            http_get=lambda url, headers, timeout: HttpResponse(
                200, b'{"data":[{"id":"different-model"}]}'
            )
        )

        result = run_diagnostics(options, probes=probes).checks[1]

        self.assertEqual(Status.RED, result.status)
        self.assertIn("wanted-model", result.remediation)

    def test_gpu_check_parses_all_memory_fields_and_enforces_free_memory(self):
        output = "0, RTX A, 49140, 43000, 6140\n1, RTX B, 24576, 4096, 20480\n"
        probes = make_probes(
            commands={"nvidia-smi": "/usr/bin/nvidia-smi"},
            command_runner=lambda command, timeout: CommandOutput(0, output, ""),
        )
        options = DiagnosticsOptions(
            gpu=GpuSpec(
                required=True,
                minimum_total_memory_mib=20_000,
                minimum_free_memory_mib=10_000,
            )
        )

        result = run_diagnostics(options, probes=probes).checks[1]

        self.assertEqual(Status.GREEN, result.status)
        self.assertIn("GPU 1", result.message)
        self.assertEqual(43_000, result.details["gpus"][0]["memory_used_mib"])

    def test_gpu_memory_shortage_is_red_when_gpu_is_required(self):
        probes = make_probes(
            commands={"nvidia-smi": "/usr/bin/nvidia-smi"},
            command_runner=lambda command, timeout: CommandOutput(
                0, "0, RTX, 49140, 48000, 1140\n", ""
            ),
        )

        result = run_diagnostics(
            DiagnosticsOptions(
                gpu=GpuSpec(required=True, minimum_free_memory_mib=8_000)
            ),
            probes=probes,
        ).checks[1]

        self.assertEqual(Status.RED, result.status)
        self.assertIn("Stop unused GPU workloads", result.remediation)

    def test_config_adapter_discovers_remote_models_mineru_and_save_path(self):
        config = {
            "save_path": "/tmp/docarag-index",
            "llm": {
                "backend": "openai",
                "api_base": "http://localhost:8003/v1",
                "api_key": "token",
                "model_name": "answer-model",
            },
            "vlm": {
                "backend": "ollama",
                "api_base": "http://localhost:11434",
                "model_name": "vision-model",
            },
            "mineru": {
                "backend": "vlm-sglang-client",
                "server_url": "http://localhost:30000",
            },
            "rag": {
                "reranker_config": {
                    "backend": "local",
                    "model_name": "local-reranker",
                }
            },
        }

        options = options_from_docarag_config(config)

        self.assertEqual(1, len(options.writable_paths))
        self.assertEqual(2, len(options.models))
        self.assertEqual(1, len(options.endpoints))
        self.assertEqual(
            "http://localhost:30000/health", options.endpoints[0].url
        )
        self.assertEqual("answer-model", options.models[0].model_name)
        self.assertEqual("Bearer token", options.models[0].headers["Authorization"])
        config_names = {item.name for item in options.configuration}
        self.assertIn("rag.reranker_config.model_name", config_names)
        self.assertNotIn("rag.reranker_config.api_base", config_names)

    def test_qdrant_diagnostics_use_the_named_environment_override_safely(self):
        options = options_from_docarag_config(
            {
                "vdb": {
                    "qdrant_url": "http://configured-qdrant:6333",
                    "qdrant_url_env": "TEST_QDRANT_URL",
                    "api_key_env": "TEST_QDRANT_API_KEY",
                    "timeout_seconds": 4.0,
                }
            },
            environment={
                "TEST_QDRANT_URL": "https://qdrant.internal.example:7443",
                "TEST_QDRANT_API_KEY": "never-serialize-this-key",
            },
        )

        self.assertEqual(1, len(options.endpoints))
        endpoint = options.endpoints[0]
        self.assertEqual(
            "https://qdrant.internal.example:7443/healthz", endpoint.url
        )
        self.assertEqual("never-serialize-this-key", endpoint.headers["api-key"])
        rendered_configuration = json.dumps(
            [item.value for item in options.configuration]
        )
        self.assertNotIn("never-serialize-this-key", rendered_configuration)
        self.assertIn("TEST_QDRANT_URL", rendered_configuration)

    def test_default_config_placeholders_are_exposed_as_failed_config_checks(self):
        options = options_from_docarag_config(
            {
                "llm": {
                    "backend": "openai",
                    "api_base": "TODO",
                    "model_name": "TODO",
                }
            }
        )

        report = run_diagnostics(options, probes=make_probes())
        config_results = [
            result
            for result in report.checks
            if result.check_type is CheckType.CONFIGURATION
        ]

        self.assertEqual(3, len(config_results))
        self.assertEqual(2, sum(item.status is Status.RED for item in config_results))
        self.assertEqual(0, len(options.models))

    def test_import_does_not_load_model_frameworks(self):
        code = (
            "import sys; import docarag.diagnostics; "
            "print(','.join(n for n in "
            "('torch','paddle','transformers','mineru') if n in sys.modules))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual("", completed.stdout.strip())

    def test_doctor_for_settings_supports_cli_contract_and_offline_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            (root / "config").mkdir()
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "Scripts/index_pdf.sh").write_text("", encoding="utf-8")
            config = root / "system.yaml"
            config.write_text(
                """
llm:
  backend: openai
  api_base: http://localhost:8000/v1
  model_name: answer-model  # inline comments are accepted
""".lstrip(),
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                project_root=root,
                system_config=config,
                runs_dir=root / "runs",
                conda_environment="gbc-rag",
                python_executable="/runtime/python",
            )
            http_calls = []
            probes = make_probes(
                commands={
                    **DOCTOR_COMMANDS,
                    "/runtime/python": "/runtime/python",
                },
                http_get=lambda url, headers, timeout: http_calls.append(url)
                or HttpResponse(200, b'{"data":[{"id":"answer-model"}]}'),
            )

            report = doctor_for_settings(
                settings, check_endpoints=False, probes=probes
            )

            self.assertEqual([], http_calls)
            self.assertTrue(report.ok)
            self.assertEqual(report.to_dict(), report.as_dict())
            self.assertIn("DocARAG diagnostics", render_report(report))
            check_ids = {item.check_id for item in report.checks}
            self.assertIn("config.project_root", check_ids)
            self.assertIn("config.system_config", check_ids)
            self.assertIn("path.runs_dir", check_ids)
            self.assertNotIn("model.llm", check_ids)
            gpu = next(item for item in report.checks if item.check_id == "gpu.nvidia")
            self.assertEqual(6 * 1024, gpu.details["minimum_free_memory_mib"])

    def test_doctor_for_settings_checks_models_when_online(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            (root / "config").mkdir()
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "Scripts/index_pdf.sh").write_text("", encoding="utf-8")
            config = root / "system.yaml"
            config.write_text(
                """
llm:
  backend: ollama
  api_base: http://localhost:11434
  model_name: answer-model
""".lstrip(),
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                project_root=root,
                system_config=config,
                runs_dir=root / "runs",
                conda_environment="gbc-rag",
                python_executable="/runtime/python",
            )
            probes = make_probes(
                commands={
                    **DOCTOR_COMMANDS,
                    "/runtime/python": "/runtime/python",
                },
                http_get=lambda url, headers, timeout: HttpResponse(
                    200, b'{"models":[{"name":"answer-model:latest"}]}'
                ),
            )

            report = doctor_for_settings(settings, probes=probes)
            model_result = next(
                item for item in report.checks if item.check_id == "model.llm"
            )

            self.assertEqual(Status.GREEN, model_result.status)

    def test_doctor_rejects_unsupported_configured_runtime_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            (root / "config").mkdir()
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "Scripts/index_pdf.sh").write_text("", encoding="utf-8")
            config = root / "system.yaml"
            config.write_text("llm:\n  backend: local\n", encoding="utf-8")
            settings = SimpleNamespace(
                project_root=root,
                system_config=config,
                runs_dir=root / "runs",
                conda_environment="unused",
                python_executable="/runtime/python",
            )

            def command_runner(command, timeout):
                joined = " ".join(command)
                if "runtime_python=" in joined:
                    return CommandOutput(1, "runtime_python=3.13.1 required=3.12.x\n")
                if "--query-gpu" in joined:
                    return CommandOutput(0, "0, Test GPU, 24000, 0, 24000\n")
                return CommandOutput(0, "ok\n")

            probes = make_probes(
                commands={
                    **DOCTOR_COMMANDS,
                    "/runtime/python": "/runtime/python",
                    "nvidia-smi": "/usr/bin/nvidia-smi",
                },
                command_runner=command_runner,
            )

            report = doctor_for_settings(
                settings, check_endpoints=False, probes=probes
            )
            runtime = next(
                item
                for item in report.checks
                if item.check_id == "command.runtime_python_version"
            )
            self.assertEqual(Status.RED, runtime.status)
            self.assertFalse(report.ok)

    def test_doctor_requires_gpu_when_selected_config_requests_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            (root / "config").mkdir()
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "Scripts/index_pdf.sh").write_text("", encoding="utf-8")
            config = root / "system.yaml"
            config.write_text(
                "vdb:\n  embedding_config:\n    device: cuda:0\n",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                project_root=root,
                system_config=config,
                runs_dir=root / "runs",
                conda_environment="unused",
                python_executable="/runtime/python",
            )
            probes = make_probes(
                commands={**DOCTOR_COMMANDS, "/runtime/python": "/runtime/python"}
            )

            report = doctor_for_settings(
                settings, check_endpoints=False, probes=probes
            )
            gpu = next(item for item in report.checks if item.check_id == "gpu.nvidia")

            self.assertTrue(gpu.details["required"])
            self.assertEqual(Status.RED, gpu.status)
            self.assertFalse(report.ok)

    def test_doctor_rejects_directory_that_is_not_a_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "system.yaml"
            config.write_text("llm:\n  backend: local\n", encoding="utf-8")
            settings = SimpleNamespace(
                project_root=root,
                system_config=config,
                runs_dir=root / "runs",
                conda_environment="gbc-rag",
                python_executable="/runtime/python",
            )
            probes = make_probes(
                commands={**DOCTOR_COMMANDS, "/runtime/python": "/runtime/python"}
            )

            report = doctor_for_settings(
                settings, check_endpoints=False, probes=probes
            )
            layout = next(
                item for item in report.checks if item.check_id == "config.project_layout"
            )

            self.assertEqual(Status.RED, layout.status)
            self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
