from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from document_services.parsing.routing_presets import (
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
)
from document_services.runtime.routing_profile import (
    ROUTED_PROFILE_SCHEMA_VERSION,
    RoutedProfileReadError,
    RoutedProfileValidationError,
    load_routed_runtime_profile,
)
from document_services.domain import MinerUEffort


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = REPOSITORY_ROOT / "config" / "routed-balanced.example.toml"


class RoutedRuntimeProfileTests(unittest.TestCase):
    def _copy(self, root: Path, transform=lambda value: value) -> Path:
        target = root / "routed.toml"
        source = EXAMPLE_PROFILE.read_text(encoding="utf-8")
        target.write_text(transform(source), encoding="utf-8")
        return target

    def test_checked_in_example_loads_into_the_balanced_immutable_contract(self):
        profile = load_routed_runtime_profile(EXAMPLE_PROFILE)

        self.assertEqual("balanced-local-example", profile.profile_id)
        self.assertEqual(
            {DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER},
            {item.adapter_name for item in profile.config.endpoints},
        )
        self.assertEqual(16, profile.config.orchestration_policy.max_batch_pages)
        self.assertTrue(profile.config.parse_options.parse_formulas)
        self.assertEqual(
            MinerUEffort.MEDIUM, profile.config.parse_options.mineru_effort
        )
        self.assertEqual(
            hashlib.sha256(profile.source_bytes).hexdigest(),
            profile.source_sha256,
        )
        with self.assertRaises(FrozenInstanceError):
            profile.profile_id = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            profile.revision_binding()["profile_id"] = "changed"  # type: ignore[index]

    def test_exact_file_hash_and_normalized_configuration_hash_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = self._copy(root)
            first = load_routed_runtime_profile(first_path)
            first_path.write_text(
                first_path.read_text(encoding="utf-8") + "\n# operator note\n",
                encoding="utf-8",
            )
            second = load_routed_runtime_profile(first_path)

        self.assertNotEqual(first.source_sha256, second.source_sha256)
        self.assertEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertEqual(ROUTED_PROFILE_SCHEMA_VERSION, first.revision_binding()["schema_version"])

    def test_mineru_effort_is_a_static_readiness_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace('mineru_effort = "medium"\n', "", 1),
            )
            profile = load_routed_runtime_profile(target)

        issue = next(
            item
            for item in profile.configuration_issues()
            if item.code == "mineru_effort_required"
        )
        self.assertEqual("parse_options.mineru_effort", issue.field)

    def test_unknown_fields_fail_instead_of_being_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    "max_primary_latency_ms = 120000",
                    "max_primary_latency_ms = 120000\nmax_total_latncy_ms = 1",
                ),
            )
            with self.assertRaisesRegex(
                RoutedProfileValidationError, "max_total_latncy_ms"
            ):
                load_routed_runtime_profile(target)

    def test_literal_secret_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    'runtime_id = "docling-worker-local"',
                    'runtime_id = "docling-worker-local"\nbearer_token = "literal-secret"',
                ),
            )
            with self.assertRaisesRegex(
                RoutedProfileValidationError, "literal secret field"
            ):
                load_routed_runtime_profile(target)

    def test_arbitrary_parse_extra_cannot_smuggle_a_secret_into_cli_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    'timeout_seconds = 300.0',
                    'timeout_seconds = 300.0\n'
                    'extra = { headers = "Bearer routed-profile-secret" }',
                    1,
                ),
            )
            with self.assertRaisesRegex(
                RoutedProfileValidationError,
                r"\[parse_options\]\.extra is not allowed",
            ):
                load_routed_runtime_profile(target)

    def test_camel_case_secret_field_names_are_rejected_before_normalization(self):
        sentinel = "camel-case-secret-must-not-be-serialized"
        for field_name in (
            "accessToken",
            "apiKey",
            "APIKey",
            "clientSecret",
            "privateKey",
            "refreshToken",
        ):
            with self.subTest(field_name=field_name), tempfile.TemporaryDirectory() as directory:
                target = self._copy(
                    Path(directory),
                    lambda value, field_name=field_name: value.replace(
                        'timeout_seconds = 300.0',
                        'timeout_seconds = 300.0\n'
                        f'extra = {{ {field_name} = "{sentinel}" }}',
                        1,
                    ),
                )
                with self.assertRaisesRegex(
                    RoutedProfileValidationError, "literal secret field"
                ) as raised:
                    load_routed_runtime_profile(target)
                self.assertNotIn(sentinel, str(raised.exception))

    def test_even_innocuous_file_extra_requires_a_typed_profile_field(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    'timeout_seconds = 300.0',
                    'timeout_seconds = 300.0\nextra = { vendor_mode = "fast" }',
                    1,
                ),
            )
            with self.assertRaisesRegex(
                RoutedProfileValidationError,
                "typed profile field",
            ):
                load_routed_runtime_profile(target)

    def test_token_env_name_is_serialized_but_resolved_value_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    '# bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
                    'bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
                    1,
                ),
            )
            with patch.dict(
                os.environ,
                {"DOCARAG_DOCLING_TOKEN": "must-never-enter-the-profile"},
            ):
                profile = load_routed_runtime_profile(target)

        rendered = json.dumps(profile.to_dict(), sort_keys=True)
        self.assertIn("DOCARAG_DOCLING_TOKEN", rendered)
        self.assertNotIn("must-never-enter-the-profile", rendered)
        self.assertNotIn("must-never-enter-the-profile", profile.source_bytes.decode())

    def test_invalid_token_environment_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(
                Path(directory),
                lambda value: value.replace(
                    '# bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
                    'bearer_token_env = "not-a-valid-env-name"',
                    1,
                ),
            )
            with self.assertRaisesRegex(
                RoutedProfileValidationError, "environment-variable name"
            ):
                load_routed_runtime_profile(target)

    def test_file_read_is_bounded_before_toml_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "large.toml"
            target.write_bytes(b"x" * 33)
            with self.assertRaisesRegex(RoutedProfileReadError, "32-byte limit"):
                load_routed_runtime_profile(target, max_bytes=32)


if __name__ == "__main__":
    unittest.main()
