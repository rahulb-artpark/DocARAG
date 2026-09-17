from __future__ import annotations

from pathlib import Path
import os
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main


class MainConfigSnapshotTests(unittest.TestCase):
    @staticmethod
    def config(payload):
        return SimpleNamespace(model_dump=lambda: payload)

    def test_snapshot_is_owner_read_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.yaml"
            config = self.config({"model": "first", "api_key": "secret"})

            main.write_config_snapshot(path, config)
            first = path.read_bytes()
            self.assertEqual(0o400, stat.S_IMODE(path.stat().st_mode))

            main.write_config_snapshot(path, config)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(0o400, stat.S_IMODE(path.stat().st_mode))

    def test_snapshot_refuses_to_replace_different_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.yaml"
            main.write_config_snapshot(path, self.config({"model": "first"}))
            original = path.read_bytes()

            with self.assertRaises(RuntimeError):
                main.write_config_snapshot(path, self.config({"model": "second"}))

            self.assertEqual(original, path.read_bytes())

    def test_log_redaction_is_recursive_without_mutating_runtime_config(self):
        original = {
            "llm": {"api_key": "top-secret", "model": "answer"},
            "providers": [{"password": "hidden", "url": "local"}],
        }

        redacted = main.redact_config(original)

        self.assertEqual("***REDACTED***", redacted["llm"]["api_key"])
        self.assertEqual("***REDACTED***", redacted["providers"][0]["password"])
        self.assertEqual("answer", redacted["llm"]["model"])
        self.assertEqual("top-secret", original["llm"]["api_key"])

    def test_routed_provider_is_owned_only_by_parse_bearing_stage(self):
        profile = SimpleNamespace(
            profile_id="balanced-v1",
            source_sha256="a" * 64,
            configuration_sha256="b" * 64,
            config=object(),
            configuration_issues=lambda: (),
        )
        provider = object()
        runtime = SimpleNamespace(
            provider=provider,
            worker_identities=(),
            close=lambda: None,
        )
        environment = {
            "DOCARAG_ROUTED_PROFILE": "/immutable/profile.toml",
            "DOCARAG_ROUTED_PROFILE_ID": profile.profile_id,
            "DOCARAG_ROUTED_PROFILE_SOURCE_SHA256": profile.source_sha256,
            "DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256": (
                profile.configuration_sha256
            ),
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch(
                "document_services.runtime.routing_profile.load_routed_runtime_profile",
                return_value=profile,
            ) as loader,
            patch(
                "document_services.runtime.parser_composition.build_balanced_routed_runtime",
                return_value=runtime,
            ) as builder,
        ):
            with main.routed_parsing_provider("tree") as selected:
                self.assertIs(provider, selected)
            with main.routed_parsing_provider("graph") as selected:
                self.assertIsNone(selected)

        loader.assert_called_once_with("/immutable/profile.toml")
        builder.assert_called_once_with(profile.config)

    def test_routed_provider_rejects_revision_binding_mismatch_before_start(self):
        profile = SimpleNamespace(
            profile_id="balanced-v1",
            source_sha256="a" * 64,
            configuration_sha256="b" * 64,
            config=object(),
            configuration_issues=lambda: (),
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DOCARAG_ROUTED_PROFILE": "/immutable/profile.toml",
                    "DOCARAG_ROUTED_PROFILE_ID": profile.profile_id,
                    "DOCARAG_ROUTED_PROFILE_SOURCE_SHA256": "0" * 64,
                    "DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256": (
                        profile.configuration_sha256
                    ),
                },
                clear=False,
            ),
            patch(
                "document_services.runtime.routing_profile.load_routed_runtime_profile",
                return_value=profile,
            ),
            patch(
                "document_services.runtime.parser_composition.build_balanced_routed_runtime"
            ) as builder,
            self.assertRaisesRegex(RuntimeError, "source_sha256"),
        ):
            with main.routed_parsing_provider("tree"):
                pass
        builder.assert_not_called()

    def test_all_stage_consumes_parser_provider_once(self):
        config = SimpleNamespace(pdf_path="source.pdf", save_path="run")
        provider = object()
        with (
            patch("main.construct_GBC_index") as construct_gbc,
            patch("main.construct_vdb") as construct_vdb,
        ):
            main.build_index(config, stage="all", parsing_provider=provider)

        self.assertEqual(2, construct_gbc.call_count)
        self.assertIs(
            provider, construct_gbc.call_args_list[0].kwargs["parsing_provider"]
        )
        self.assertIsNone(
            construct_gbc.call_args_list[1].kwargs["parsing_provider"]
        )
        self.assertIsNone(construct_vdb.call_args.kwargs["parsing_provider"])


if __name__ == "__main__":
    unittest.main()
