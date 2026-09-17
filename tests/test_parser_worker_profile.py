from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from document_services import cli
from document_services.persistence.artifact_store import LocalContentAddressedArtifactStore
from document_services.domain.base import canonical_json_bytes
from document_services.runtime.parser_worker import (
    ParserWorkerCompositionError,
    ParserWorkerProfileValidationError,
    ParserWorkerVendor,
    WorkerArtifactCallbacks,
    WorkerArtifactProfile,
    _decode_image_data_uri,
    _publish_docling_export,
    build_parser_worker_runtime,
    load_parser_worker_profile,
)
from document_services.parser_clients.mineru import MAX_MINERU_TASKS_PER_PARSE_RESULT


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROFILE_ROOT = REPOSITORY_ROOT / "document_services" / "profiles"
EXAMPLES = {
    ParserWorkerVendor.DOCLING: PROFILE_ROOT / "docling.example.toml",
    ParserWorkerVendor.MINERU: PROFILE_ROOT / "mineru.example.toml",
    ParserWorkerVendor.PADDLEOCR_VL: PROFILE_ROOT / "paddleocr-vl.example.toml",
}


def _profile_for_temporary_roots(
    temporary: Path,
    vendor: ParserWorkerVendor,
    *,
    service_secret_env: str | None = None,
    server_secret_env: str | None = None,
):
    source_root = temporary / "sources"
    source_root.mkdir()
    artifact_root = temporary / "artifacts"
    text = EXAMPLES[vendor].read_text(encoding="utf-8")
    text = text.replace(
        "/srv/docarag/source-snapshots", source_root.as_posix()
    )
    original_artifact_root = {
        ParserWorkerVendor.DOCLING: "/srv/docarag/parser-artifacts/docling",
        ParserWorkerVendor.MINERU: "/srv/docarag/parser-artifacts/mineru",
        ParserWorkerVendor.PADDLEOCR_VL: "/srv/docarag/parser-artifacts/paddleocr-vl",
    }[vendor]
    text = text.replace(original_artifact_root, artifact_root.as_posix())
    if service_secret_env is not None:
        marker = {
            ParserWorkerVendor.DOCLING: (
                '# api_key_env = "DOCARAG_DOCLING_SERVICE_API_KEY"',
                f'api_key_env = "{service_secret_env}"',
            ),
            ParserWorkerVendor.MINERU: (
                '# bearer_token_env = "DOCARAG_MINERU_SERVICE_TOKEN"',
                f'bearer_token_env = "{service_secret_env}"',
            ),
            ParserWorkerVendor.PADDLEOCR_VL: (
                '# bearer_token_env = "DOCARAG_PADDLE_SERVICE_TOKEN"',
                f'bearer_token_env = "{service_secret_env}"',
            ),
        }[vendor]
        text = text.replace(*marker)
    if server_secret_env is not None:
        marker = {
            ParserWorkerVendor.DOCLING: "DOCARAG_DOCLING_WORKER_TOKEN",
            ParserWorkerVendor.MINERU: "DOCARAG_MINERU_WORKER_TOKEN",
            ParserWorkerVendor.PADDLEOCR_VL: "DOCARAG_PADDLE_WORKER_TOKEN",
        }[vendor]
        text = text.replace(
            f'# bearer_token_env = "{marker}"',
            f'bearer_token_env = "{server_secret_env}"',
        )
    path = temporary / f"{vendor.value}.toml"
    path.write_text(text, encoding="utf-8")
    return load_parser_worker_profile(path)


class _FakeStore:
    def __init__(self) -> None:
        self.closed = False

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        expected_sha256: str | None = None,
    ):
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError("digest mismatch")
        return SimpleNamespace(
            uri=f"cas://sha256/{digest}",
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
        )

    def close(self) -> None:
        self.closed = True


class _FakeServer:
    def __init__(self, adapter, config) -> None:
        self.adapter = adapter
        self.config = config
        self.manifest = adapter.manifest()
        self.base_url = "http://127.0.0.1:8199"
        self.served = False
        self.closed = False

    def serve_forever(self) -> None:
        self.served = True

    def close(self) -> None:
        self.closed = True


class ParserWorkerProfileTests(unittest.TestCase):
    def test_packaged_examples_are_strict_parseable_and_uncertified(self):
        for expected_vendor, path in EXAMPLES.items():
            with self.subTest(vendor=expected_vendor.value):
                profile = load_parser_worker_profile(path)
                self.assertEqual(expected_vendor, profile.vendor)
                self.assertEqual("uncertified", profile.qualification)
                self.assertEqual(64, len(profile.configuration_sha256))
                self.assertEqual(64, len(profile.source.binding_sha256))
                self.assertEqual(64, len(profile.artifacts.binding_sha256))

    def test_mineru_middle_artifact_bounds_are_operator_configurable(self):
        profile = load_parser_worker_profile(
            EXAMPLES[ParserWorkerVendor.MINERU]
        )

        safe = profile.safe_dict()

        resolved = safe["service"]["resolved_configuration"]
        self.assertEqual(128 * 1024 * 1024, resolved["max_middle_json_bytes"])
        self.assertEqual(128, resolved["max_middle_json_depth"])
        self.assertEqual(2_000_000, resolved["max_middle_json_nodes"])
        self.assertEqual(256, resolved["max_tasks_per_request"])

    def test_mineru_profile_rejects_task_count_above_receipt_capacity(self):
        original = EXAMPLES[ParserWorkerVendor.MINERU].read_text(encoding="utf-8")
        invalid = original.replace(
            "max_tasks_per_request = 256",
            f"max_tasks_per_request = {MAX_MINERU_TASKS_PER_PARSE_RESULT + 1}",
            1,
        )
        self.assertNotEqual(original, invalid)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mineru-too-many-tasks.toml"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(
                ParserWorkerProfileValidationError,
                f"between 1 and {MAX_MINERU_TASKS_PER_PARSE_RESULT}",
            ):
                load_parser_worker_profile(path)

    def test_paddle_response_shape_bounds_are_operator_configurable(self):
        profile = load_parser_worker_profile(
            EXAMPLES[ParserWorkerVendor.PADDLEOCR_VL]
        )

        resolved = profile.safe_dict()["service"]["bounds"]
        self.assertEqual(128, resolved["max_response_json_depth"])
        self.assertEqual(2_000_000, resolved["max_response_json_nodes"])

    def test_paddle_profile_store_can_retain_the_raw_response_bound(self):
        profile = load_parser_worker_profile(
            EXAMPLES[ParserWorkerVendor.PADDLEOCR_VL]
        )
        self.assertGreaterEqual(
            profile.artifacts.max_artifact_bytes,
            profile.service.max_response_bytes,
        )
        self.assertGreaterEqual(
            profile.service.max_total_artifact_bytes,
            profile.service.max_response_bytes,
        )

        original = EXAMPLES[ParserWorkerVendor.PADDLEOCR_VL].read_text(
            encoding="utf-8"
        )
        invalid = original.replace(
            "max_artifact_bytes = 134217728",
            "max_artifact_bytes = 67108864",
            1,
        )
        self.assertNotEqual(original, invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paddle-raw-too-large.toml"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(
                ParserWorkerProfileValidationError,
                "raw responses are retained",
            ):
                load_parser_worker_profile(path)

    def test_literal_secrets_unknown_fields_and_mutable_labels_fail_closed(self):
        original = EXAMPLES[ParserWorkerVendor.DOCLING].read_text(encoding="utf-8")
        cases = {
            "literal secret": original.replace(
                "qualification = \"uncertified\"",
                'qualification = "uncertified"\npassword = "do-not-print-this"',
            ),
            "unknown": original.replace(
                "qualification = \"uncertified\"",
                'qualification = "uncertified"\ntypo_field = true',
            ),
            "mutable": original.replace(
                'parser_version = "docling-runtime-unattested-v1"',
                'parser_version = "latest"',
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, text in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / f"{name}.toml"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ParserWorkerProfileValidationError) as raised:
                        load_parser_worker_profile(path)
                    self.assertNotIn("do-not-print-this", str(raised.exception))

    def test_source_and_docling_base64_cross_bounds_are_rejected(self):
        original = EXAMPLES[ParserWorkerVendor.DOCLING].read_text(encoding="utf-8")
        cases = (
            original.replace(
                "max_source_bytes = 33554432",
                "max_source_bytes = 33554433",
                1,
            ),
            original.replace(
                "max_request_bytes = 50331648",
                "max_request_bytes = 33554432",
                1,
            ),
            original.replace(
                "max_artifact_bytes = 268435456",
                "max_artifact_bytes = 134217728",
                1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(cases):
                path = Path(directory) / f"invalid-{index}.toml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ParserWorkerProfileValidationError):
                    load_parser_worker_profile(path)

    def test_secret_values_are_resolved_only_at_composition_and_not_in_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = _profile_for_temporary_roots(
                Path(directory),
                ParserWorkerVendor.DOCLING,
                service_secret_env="SERVICE_SECRET",
                server_secret_env="WORKER_SECRET",
            )
            safe = json.dumps(profile.safe_dict(), sort_keys=True)
            self.assertNotIn("secret-service-value", safe)
            self.assertNotIn("secret-worker-value", safe)

            stores: list[_FakeStore] = []

            def artifacts(_profile, _vendor):
                store = _FakeStore()
                stores.append(store)
                return WorkerArtifactCallbacks(owner=store)

            runtime = build_parser_worker_runtime(
                profile,
                environment={
                    "SERVICE_SECRET": "secret-service-value",
                    "WORKER_SECRET": "secret-worker-value",
                },
                artifact_factory=artifacts,
                server_factory=_FakeServer,
            )
            try:
                summary = json.dumps(dict(runtime.summary()), sort_keys=True)
                self.assertNotIn("secret-service-value", summary)
                self.assertNotIn("secret-worker-value", summary)
                self.assertIn("same-host-or-identically-mounted", summary)
                self.assertEqual(
                    "secret-worker-value", runtime.server.config.bearer_token
                )
            finally:
                runtime.close()
            self.assertTrue(stores[0].closed)

    def test_missing_named_secret_fails_without_constructing_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _profile_for_temporary_roots(
                root,
                ParserWorkerVendor.DOCLING,
                service_secret_env="MISSING_SERVICE_SECRET",
            )
            with self.assertRaisesRegex(
                ParserWorkerCompositionError, "MISSING_SERVICE_SECRET"
            ):
                build_parser_worker_runtime(
                    profile,
                    environment={},
                    server_factory=_FakeServer,
                )
            self.assertFalse(profile.artifacts.root.exists())

    def test_each_vendor_composes_exact_manifest_without_contacting_service(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for vendor in ParserWorkerVendor:
                with self.subTest(vendor=vendor.value):
                    vendor_root = temporary / vendor.value
                    vendor_root.mkdir()
                    profile = _profile_for_temporary_roots(vendor_root, vendor)
                    store = _FakeStore()

                    def artifacts(_profile, active_vendor):
                        self.assertEqual(vendor, active_vendor)
                        return WorkerArtifactCallbacks(
                            mineru_publisher=(
                                (lambda value: value)
                                if vendor is ParserWorkerVendor.MINERU
                                else None
                            ),
                            paddle_sink=(
                                (lambda value: value)
                                if vendor is ParserWorkerVendor.PADDLEOCR_VL
                                else None
                            ),
                            owner=store,
                        )

                    runtime = build_parser_worker_runtime(
                        profile,
                        environment={},
                        artifact_factory=artifacts,
                        page_renderer_factory=lambda _render, _service: (
                            lambda _source, _page: None
                        ),
                        server_factory=_FakeServer,
                    )
                    self.assertEqual(
                        profile.parser_version, runtime.server.manifest.parser_version
                    )
                    self.assertEqual(
                        profile.model_version, runtime.server.manifest.model_version
                    )
                    runtime.close()
                    self.assertTrue(store.closed)


class DoclingCASPublicationTests(unittest.TestCase):
    def test_raw_export_and_image_are_published_with_resolvable_lineage(self):
        export = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "image": {
                        "uri": "data:image/png;base64,aW1hZ2UtYnl0ZXM="
                    },
                }
            ],
            "tables": [],
            "pictures": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cas"
            store = LocalContentAddressedArtifactStore(root)
            profile = WorkerArtifactProfile(
                root=root,
                max_artifact_bytes=1024 * 1024,
                max_total_bytes_per_request=2 * 1024 * 1024,
                max_artifacts_per_request=4,
                chunk_size=4096,
            )
            output = _publish_docling_export(
                export, store, profile, externalize_images=True
            )

            raw = canonical_json_bytes(export)
            raw_digest = hashlib.sha256(raw).hexdigest()
            image_digest = hashlib.sha256(b"image-bytes").hexdigest()
            item = output["texts"][0]
            self.assertEqual(
                f"cas://sha256/{raw_digest}#/texts/0",
                item["raw_artifact_uri"],
            )
            self.assertEqual(
                f"cas://sha256/{image_digest}", item["image"]["uri"]
            )
            self.assertEqual(raw, (root / raw_digest[:2] / raw_digest).read_bytes())
            self.assertEqual(
                b"image-bytes",
                (root / image_digest[:2] / image_digest).read_bytes(),
            )
            self.assertNotIn("raw_artifact_uri", export["texts"][0])

    def test_percent_decoding_and_store_identity_fail_closed(self):
        with self.assertRaisesRegex(
            ParserWorkerCompositionError, "max_artifact_bytes"
        ):
            _decode_image_data_uri("data:image/png,%41%42", 1)

        class BadStore:
            def put_bytes(self, data, **_kwargs):
                digest = hashlib.sha256(data).hexdigest()
                return SimpleNamespace(
                    uri=f"cas://sha256/{digest}",
                    sha256=digest,
                    size_bytes=len(data) + 1,
                )

        profile = WorkerArtifactProfile(
            root=Path("/tmp/docarag-bad-store"),
            max_artifact_bytes=4096,
            max_total_bytes_per_request=8192,
            max_artifacts_per_request=2,
            chunk_size=1024,
        )
        export = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "texts": [],
            "tables": [],
            "pictures": [],
        }
        with self.assertRaisesRegex(ParserWorkerCompositionError, "CAS identity"):
            _publish_docling_export(
                export, BadStore(), profile, externalize_images=False
            )


class ParserWorkerCLITests(unittest.TestCase):
    def test_validate_is_read_only_and_json_is_secret_free(self):
        profile_path = EXAMPLES[ParserWorkerVendor.DOCLING]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(
                ["parser-worker", "validate", str(profile_path), "--json"]
            )
        self.assertEqual(0, code)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["validation"]["valid"])
        self.assertFalse(payload["validation"]["secrets_resolved"])
        self.assertFalse(payload["validation"]["filesystem_roots_created"])
        self.assertFalse(payload["validation"]["service_contacted"])

    def test_serve_prints_identity_and_closes_runtime(self):
        class Runtime:
            closed = False
            served = False

            def summary(self):
                return {"profile_id": "test", "qualification": "uncertified"}

            def serve_forever(self):
                self.served = True

            def close(self):
                self.closed = True

        runtime = Runtime()
        stdout = io.StringIO()
        with (
            patch(
                "document_services.runtime.parser_worker.build_parser_worker_runtime",
                return_value=runtime,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(
                [
                    "parser-worker",
                    "serve",
                    str(EXAMPLES[ParserWorkerVendor.DOCLING]),
                ]
            )
        self.assertEqual(0, code)
        self.assertTrue(runtime.served)
        self.assertTrue(runtime.closed)
        self.assertEqual("test", json.loads(stdout.getvalue())["profile_id"])


if __name__ == "__main__":
    unittest.main()
