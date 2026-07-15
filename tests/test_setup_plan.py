from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

from docarag import cli
from docarag.routed_profile import load_routed_runtime_profile
from docarag.settings import RuntimeSettings
from docarag.setup_plan import (
    SETUP_PLAN_SCHEMA_VERSION,
    build_setup_plan,
    render_setup_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROUTED_PROFILE = REPOSITORY_ROOT / "config/routed-balanced.example.toml"


def make_checkout(root: Path, *, environment_name: str = "gbc-rag") -> RuntimeSettings:
    scripts = root / "Scripts"
    config = root / "config"
    scripts.mkdir(parents=True)
    config.mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    (scripts / "index_pdf.sh").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='docarag'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    (root / "compose.yaml").write_text(
        "services:\n  postgres:\n    image: postgres:18\n"
        "  qdrant:\n    image: qdrant/qdrant:v1.18.2\n",
        encoding="utf-8",
    )
    (root / "environment.yml").write_text(
        f"name: {environment_name}\ndependencies:\n  - python=3.12\n",
        encoding="utf-8",
    )
    system_config = config / "system.yaml"
    system_config.write_text("llm:\n  backend: ollama\n", encoding="utf-8")
    settings_path = root / "docarag.toml"
    settings_path.write_text("schema_version = 1\n", encoding="utf-8")
    return RuntimeSettings(
        project_root=root,
        system_config=system_config,
        runs_dir=root / "runs",
        conda_environment=environment_name,
        python_executable=sys.executable,
        settings_path=settings_path,
    ).resolved()


class SetupPlanTests(unittest.TestCase):
    def test_controller_supports_vendor_worker_python_313_but_not_314(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = make_checkout(Path(directory))
            with patch("docarag.setup_plan.sys.version_info", (3, 13, 7)):
                supported = build_setup_plan(settings, environment={})
            with patch("docarag.setup_plan.sys.version_info", (3, 14, 0)):
                unsupported = build_setup_plan(settings, environment={})

        def controller(plan):
            return next(
                item
                for item in plan["static_readiness"]["prerequisites"]
                if item["check_id"] == "controller_python"
            )

        self.assertEqual("ready", controller(supported)["status"])
        self.assertEqual("blocked", controller(unsupported)["status"])

    def test_plan_is_deterministic_read_only_and_uses_argv_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_checkout(root)
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

            with (
                patch("subprocess.run") as process_probe,
                patch("urllib.request.urlopen") as network_probe,
            ):
                first = build_setup_plan(settings, environment={})
            process_probe.assert_not_called()
            network_probe.assert_not_called()
            second = build_setup_plan(settings, environment={})
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(SETUP_PLAN_SCHEMA_VERSION, first["schema_version"])
        self.assertEqual(
            {
                "writes_files": False,
                "starts_subprocesses": False,
                "contacts_network": False,
                "starts_or_stops_services": False,
                "changes_gpu_state": False,
                "resolves_secret_values": False,
            },
            first["report_behavior"],
        )
        self.assertTrue(first["static_readiness"]["ready_for_offline_doctor"])
        for command in first["commands"]:
            self.assertIsInstance(command["argv"], list)
            self.assertTrue(all(isinstance(item, str) for item in command["argv"]))
            self.assertNotIn("shell", command)
            self.assertNotIn("command", command)

    def test_plan_preserves_bookrag_core_and_reports_worker_qualification_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_setup_plan(make_checkout(Path(directory)), environment={})

        self.assertEqual(
            {
                "profile": "BookRAG with optional routed canonical ingestion",
                "core": "BookRAG",
                "indexing_enabled": True,
                "routed_indexing_enabled": True,
            },
            plan["supported_runtime"],
        )
        components = {item["component_id"]: item for item in plan["components"]}
        self.assertFalse(components["controller-cli"]["production_verified"])
        self.assertTrue(
            components["bookrag-compatibility-runtime"][
                "available_to_supported_index_command"
            ]
        )
        for worker_id in (
            "docling-worker",
            "mineru-worker",
            "paddleocr-vl-worker",
        ):
            worker = components[worker_id]
            self.assertTrue(worker["available_to_supported_index_command"])
            self.assertFalse(worker["production_verified"])
            self.assertEqual("fixture-only", worker["verification_level"])
            self.assertTrue(worker["uncertified_artifacts"])

    def test_first_static_blocker_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_checkout(root)
            settings.system_config.unlink()

            plan = build_setup_plan(settings, environment={})

        self.assertFalse(plan["static_readiness"]["ready_for_offline_doctor"])
        self.assertEqual("remediate_static_blocker", plan["next_action"]["kind"])
        self.assertEqual("system_config", plan["next_action"]["check_id"])
        self.assertEqual(1, plan["static_readiness"]["counts"]["blocked"])

    def test_environment_name_drift_blocks_ambiguous_conda_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_checkout(root, environment_name="declared-env")
            settings = RuntimeSettings(
                project_root=settings.project_root,
                system_config=settings.system_config,
                runs_dir=settings.runs_dir,
                conda_environment="configured-env",
                python_executable=None,
                settings_path=settings.settings_path,
            ).resolved()

            plan = build_setup_plan(
                settings,
                environment={},
                executable_lookup=lambda name: "/usr/bin/conda",
            )

        checks = {
            item["check_id"]: item
            for item in plan["static_readiness"]["prerequisites"]
        }
        self.assertEqual("blocked", checks["compatibility_environment_name"]["status"])
        self.assertEqual("declared-env", checks["compatibility_environment_name"]["details"]["declared"])
        self.assertEqual("configured-env", checks["compatibility_environment_name"]["details"]["configured"])

    def test_routed_profile_reports_only_secret_name_and_enables_explicit_indexing(self):
        secret = "setup-plan-secret-must-not-appear"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_checkout(root)
            profile_path = root / "routed.toml"
            profile_text = EXAMPLE_ROUTED_PROFILE.read_text(encoding="utf-8").replace(
                '# bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
                'bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
                1,
            )
            profile_path.write_text(profile_text, encoding="utf-8")
            profile = load_routed_runtime_profile(profile_path)

            plan = build_setup_plan(
                settings,
                routed_profile=profile,
                environment={"DOCARAG_DOCLING_TOKEN": secret},
            )
            rendered = json.dumps(plan, sort_keys=True)

        self.assertNotIn(secret, rendered)
        self.assertIn("DOCARAG_DOCLING_TOKEN", rendered)
        self.assertTrue(plan["routed_runtime"]["indexing_enabled"])
        self.assertTrue(plan["routed_runtime"]["selected"])
        self.assertFalse(plan["routed_runtime"]["configuration_ready"])
        self.assertFalse(plan["routed_runtime"]["live_readiness_checked"])
        self.assertEqual(6, len(plan["routed_runtime"]["configuration_issues"]))
        self.assertFalse(plan["static_readiness"]["ready_for_offline_doctor"])
        self.assertEqual(
            "routed_profile_static_configuration",
            plan["next_action"]["check_id"],
        )
        self.assertTrue(
            plan["routed_runtime"]["workers"][0]["bearer_token_configured"]
        )

    def test_cli_json_exit_code_reflects_static_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = make_checkout(Path(directory))
            output = io.StringIO()
            with (
                patch("docarag.cli._load", return_value=settings),
                contextlib.redirect_stdout(output),
            ):
                code = cli.main(["setup-plan", "--json"])

            payload = json.loads(output.getvalue())
            settings.system_config.unlink()
            with (
                patch("docarag.cli._load", return_value=settings),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                blocked_code = cli.main(["setup-plan", "--json"])

        self.assertEqual(0, code)
        self.assertEqual(SETUP_PLAN_SCHEMA_VERSION, payload["schema_version"])
        self.assertEqual(1, blocked_code)

    def test_human_render_keeps_next_command_as_json_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_checkout(root)
            settings = RuntimeSettings(
                project_root=settings.project_root,
                system_config=settings.system_config,
                runs_dir=settings.runs_dir,
                conda_environment=settings.conda_environment,
                python_executable=settings.python_executable,
                settings_path=None,
            ).resolved()
            plan = build_setup_plan(settings, environment={})

        rendered = render_setup_plan(plan)
        self.assertIn("READY FOR OFFLINE DOCTOR", rendered)
        self.assertIn('argv: ["docarag", "init"', rendered)
        self.assertIn("no writes, network calls, subprocess calls", rendered)

    def test_import_does_not_load_model_frameworks(self):
        code = (
            "import json,sys; import docarag.setup_plan; "
            "roots=('torch','paddle','paddleocr','docling','mineru','transformers','vllm'); "
            "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in roots)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("[]", completed.stdout.strip())


if __name__ == "__main__":
    unittest.main()
