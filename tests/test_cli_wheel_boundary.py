from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class ControllerWheelBoundaryTests(unittest.TestCase):
    def test_cli_help_settings_and_storage_parser_do_not_import_core_or_yaml(self):
        program = textwrap.dedent(
            """
            import contextlib
            import importlib.abc
            import io
            import json
            import sys

            class CheckoutImportBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.partition(".")[0]
                    if root in {"Core", "retrieval_engine", "yaml"}:
                        raise AssertionError(f"forbidden controller import: {fullname}")
                    return None

            sys.meta_path.insert(0, CheckoutImportBlocker())

            from document_services import cli

            assert "document_services.runtime.bookrag_bridge" not in sys.modules
            assert not any(
                name == "retrieval_engine" or name.startswith("retrieval_engine.") for name in sys.modules
            )
            assert not any(
                name == "yaml" or name.startswith("yaml.") for name in sys.modules
            )

            parser = cli.build_parser()
            assert parser.parse_args(["settings"]).handler is cli.command_settings
            assert (
                parser.parse_args(["storage", "status"]).handler
                is cli.command_storage_status
            )

            settings_output = io.StringIO()
            with contextlib.redirect_stdout(settings_output):
                assert cli.main(["settings"]) == 0
            assert json.loads(settings_output.getvalue())["profile"] == "compatibility"

            help_output = io.StringIO()
            with contextlib.redirect_stdout(help_output):
                try:
                    cli.main(["--help"])
                except SystemExit as exc:
                    assert exc.code == 0
                else:
                    raise AssertionError("--help did not exit through argparse")
            assert "storage" in help_output.getvalue()
            assert "settings" in help_output.getvalue()

            storage_help = io.StringIO()
            with contextlib.redirect_stdout(storage_help):
                try:
                    cli.main(["storage", "--help"])
                except SystemExit as exc:
                    assert exc.code == 0
                else:
                    raise AssertionError("storage --help did not exit through argparse")
            assert "status" in storage_help.getvalue()

            assert "document_services.runtime.bookrag_bridge" not in sys.modules
            print("controller-boundary-ok")
            """
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(
            0,
            completed.returncode,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("controller-boundary-ok", completed.stdout)

    def test_controller_metadata_does_not_claim_pyyaml_as_a_dependency(self):
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as source:
            metadata = tomllib.load(source)

        dependencies = metadata["project"]["dependencies"]
        normalized = {
            dependency.split("[", 1)[0]
            .split("=", 1)[0]
            .split("<", 1)[0]
            .split(">", 1)[0]
            .strip()
            .lower()
            for dependency in dependencies
        }
        self.assertNotIn("pyyaml", normalized)
        self.assertNotIn("yaml", normalized)


if __name__ == "__main__":
    unittest.main()
