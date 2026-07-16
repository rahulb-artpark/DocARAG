from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from docarag import cli
from docarag.persistence import DocumentScope, SchemaStatus, StageEventStatus
from docarag.settings import RuntimeSettings


class _Health:
    reachable = True
    ok = True
    endpoint = "postgresql://127.0.0.1:5432/docarag"
    error_code = None
    schema = SchemaStatus(
        present=True, current=True, schema_sha256="a" * 64, applied_sha256="a" * 64
    )

    def as_dict(self):
        return {
            "ok": True,
            "reachable": True,
            "endpoint": self.endpoint,
            "schema": self.schema.as_dict(),
        }


class _Ledger:
    endpoint = "postgresql://127.0.0.1:5432/docarag"

    def __init__(self):
        self.events = []
        self.document = None

    def health(self):
        return _Health()

    def ensure_schema(self):
        return SchemaStatus(
            present=True, current=True, schema_sha256="a" * 64, applied_sha256="a" * 64
        )

    def reset_schema(self):
        return SchemaStatus(
            present=True, current=True, schema_sha256="a" * 64, applied_sha256="a" * 64
        )

    def record_document(self, corpus_scope, document):
        self.document = document
        return DocumentScope(
            corpus_scope.tenant_id,
            corpus_scope.corpus_id,
            document.document_id,
            document.document_version_id,
        )

    def append_stage_event(self, event):
        self.events.append(event)


class StorageCLITests(unittest.TestCase):
    def test_project_environment_aligns_cli_and_compose_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "DOCARAG_DATABASE_URL=postgresql://local:test@127.0.0.1:55442/docarag\n"
                "DOCARAG_QDRANT_URL=http://127.0.0.1:6335\n"
                "UNRELATED_COMMAND=never-load-this\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DOCARAG_QDRANT_URL": "https://operator.example:7443"},
                clear=True,
            ):
                cli._load_project_environment(root)

                self.assertEqual(
                    "https://operator.example:7443",
                    os.environ["DOCARAG_QDRANT_URL"],
                )
                self.assertEqual(
                    "postgresql://local:test@127.0.0.1:55442/docarag",
                    os.environ["DOCARAG_DATABASE_URL"],
                )
                self.assertNotIn("UNRELATED_COMMAND", os.environ)

    def test_main_restores_project_environment_after_each_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_path = root / "docarag.toml"
            settings_path.write_text(
                "schema_version = 1\n"
                "[runtime]\n"
                f'project_root = "{root}"\n',
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "DOCARAG_QDRANT_URL=http://project-qdrant:6333\n",
                encoding="utf-8",
            )
            observed: list[str | None] = []

            def handler(arguments):
                cli._load(arguments)
                observed.append(os.environ.get("DOCARAG_QDRANT_URL"))
                return 0

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("docarag.cli.build_parser") as parser_factory,
            ):
                parser_factory.return_value.parse_args.return_value = (
                    argparse.Namespace(settings=str(settings_path), handler=handler)
                )
                self.assertEqual(0, cli.main([]))
                self.assertNotIn("DOCARAG_QDRANT_URL", os.environ)

            self.assertEqual(["http://project-qdrant:6333"], observed)

    def test_storage_subcommands_are_public(self):
        parser = cli.build_parser()
        for arguments, handler in (
            (("storage", "up"), cli.command_storage_up),
            (("storage", "down"), cli.command_storage_down),
            (("storage", "init"), cli.command_storage_init),
            (("storage", "reset"), cli.command_storage_reset),
            (("storage", "status"), cli.command_storage_status),
        ):
            parsed = parser.parse_args(arguments)
            self.assertIs(handler, parsed.handler)

    def test_storage_status_requires_both_databases(self):
        settings = RuntimeSettings(project_root=Path("/tmp/project")).resolved()
        ledger = _Ledger()
        arguments = argparse.Namespace(settings=None, json=False)
        output = io.StringIO()
        with (
            patch("docarag.cli._load", return_value=settings),
            patch("docarag.cli._postgres_ledger", return_value=ledger),
            patch(
                "docarag.cli._qdrant_health",
                return_value={
                    "ok": True,
                    "endpoint": "http://127.0.0.1:6333",
                },
            ),
            contextlib.redirect_stdout(output),
        ):
            result = cli.command_storage_status(arguments)
        self.assertEqual(0, result)
        self.assertIn("PostgreSQL", output.getvalue())
        self.assertIn("Qdrant", output.getvalue())

    def test_supported_index_records_postgres_dispatch_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            script = root / "Scripts" / "index_pdf.sh"
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            system_config = root / "system.yaml"
            system_config.write_text("vdb: {}\n", encoding="utf-8")
            document_path = root / "document.pdf"
            document_path.write_bytes(b"%PDF-1.7\nsource bytes")
            settings = RuntimeSettings(
                project_root=root,
                system_config=system_config,
                runs_dir=root / "runs",
                python_executable="/usr/bin/python3",
                tenant_id="tenant-a",
                corpus_id="corpus-a",
            ).resolved()
            ledger = _Ledger()
            arguments = argparse.Namespace(settings=None, document=str(document_path))
            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._require_postgres", return_value=ledger),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                result = cli.command_index(arguments)

        self.assertEqual(0, result)
        runner.assert_called_once()
        dispatched_environment = runner.call_args.kwargs["env"]
        self.assertEqual("tenant-a", dispatched_environment["DOCARAG_TENANT_ID"])
        self.assertEqual("corpus-a", dispatched_environment["DOCARAG_CORPUS_ID"])
        self.assertEqual(
            ledger.document.document_id,
            dispatched_environment["DOCARAG_DOCUMENT_ID"],
        )
        self.assertEqual(
            ledger.document.document_version_id,
            dispatched_environment["DOCARAG_DOCUMENT_VERSION_ID"],
        )
        self.assertIsNotNone(ledger.document)
        self.assertTrue(ledger.document.source_uri.startswith("file://"))
        self.assertEqual({}, dict(ledger.document.metadata))
        self.assertEqual(2, len(ledger.events))
        self.assertEqual(StageEventStatus.STARTED, ledger.events[0].status)
        self.assertEqual(StageEventStatus.SUCCEEDED, ledger.events[1].status)
        self.assertEqual(
            ledger.events[0].stage_run_id, ledger.events[1].stage_run_id
        )
        self.assertIn(
            "configuration_sha256", dict(ledger.events[0].payload)
        )

    def test_supported_index_binds_ready_routed_profile_into_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Scripts").mkdir()
            script = root / "Scripts/index_pdf.sh"
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            system_config = root / "system.yaml"
            system_config.write_text("vdb: {}\n", encoding="utf-8")
            document = root / "document.pdf"
            document.write_bytes(b"%PDF-1.7\nrouted")
            profile_path = root / "routed.toml"
            profile_path.write_text("profile_id = 'ready'\n", encoding="utf-8")
            settings = RuntimeSettings(
                project_root=root,
                system_config=system_config,
                runs_dir=root / "runs",
                tenant_id="tenant-a",
                corpus_id="corpus-a",
            ).resolved()
            profile = SimpleNamespace(
                source_path=profile_path.resolve(),
                profile_id="ready",
                source_sha256="a" * 64,
                configuration_sha256="b" * 64,
                configuration_issues=lambda: (),
                revision_binding=lambda: {
                    "schema_version": "docarag.routed-runtime-profile.v1",
                    "profile_id": "ready",
                    "source_sha256": "a" * 64,
                    "configuration_sha256": "b" * 64,
                },
            )
            ledger = _Ledger()
            arguments = argparse.Namespace(
                settings=None,
                document=str(document),
                routed_profile=str(profile_path),
            )
            with (
                patch("docarag.cli._load", return_value=settings),
                patch("docarag.cli._require_postgres", return_value=ledger),
                patch("docarag.cli._load_routed_profile", return_value=profile),
                patch("docarag.cli._run", return_value=0) as runner,
            ):
                self.assertEqual(0, cli.command_index(arguments))

        environment = runner.call_args.kwargs["env"]
        self.assertEqual(str(profile.source_path), environment["DOCARAG_ROUTED_PROFILE"])
        self.assertEqual("a" * 64, environment["DOCARAG_ROUTED_PROFILE_SOURCE_SHA256"])
        self.assertEqual(
            "b" * 64,
            environment["DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256"],
        )
        self.assertTrue(environment["DOCARAG_TRACE_ID"].startswith("trace_"))
        self.assertEqual("routed-index-dispatch", ledger.events[0].stage_name)
        self.assertEqual(
            "routed", dict(ledger.events[0].payload)["ingestion_mode"]
        )


if __name__ == "__main__":
    unittest.main()
