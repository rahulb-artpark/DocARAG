from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest

from document_services.operations.diagnostics import (
    CommandOutput,
    DiagnosticProbes,
    HttpResponse,
    ParserWorkerDiagnosticSpec,
    PathProbeResult,
    Status,
    doctor_for_routed_profile,
    doctor_for_settings,
)
from document_services.domain.evidence import BlockType
from document_services.domain.parsing import (
    AcceleratorKind,
    ParserCapabilityManifest,
    ParserModality,
)
from document_services.runtime.routing_profile import load_routed_runtime_profile
from document_services.runtime.compatibility_manifest import (
    CompatibilityManifestError,
    DEFAULT_MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    PARSER_CONTRACT_SCHEMA_VERSION,
    PARSER_WORKER_PROTOCOL_VERSION,
    RuntimeCompatibilityManifest,
    VerificationLevel,
    load_runtime_compatibility_manifest,
    validate_worker_capability_payload,
    validate_worker_health_payload,
    worker_endpoint_url,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_ROUTED_PROFILE = (
    _REPOSITORY_ROOT / "config" / "routed-balanced.example.toml"
)


def _raw_default_manifest() -> dict[str, object]:
    with DEFAULT_MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _parser_manifest(
    *,
    adapter_name: str = "docling-export",
    adapter_version: str = "1.1.0",
    parser_name: str = "docling",
) -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        parser_name=parser_name,
        parser_version=f"{parser_name}-fixture-1",
        model_version=f"{parser_name}-model-fixture-1",
        supported_mime_types=("application/pdf",),
        modalities=(ParserModality.NATIVE_TEXT, ParserModality.LAYOUT),
        block_types=(BlockType.PARAGRAPH, BlockType.TITLE),
        accelerator=AcceleratorKind.ANY,
        supports_page_selection=True,
        supports_region_selection=False,
        supports_batch=True,
        max_pages_per_request=16,
        deterministic=True,
        metadata={"fixture": True},
    )


_MANIFEST_SHA = _parser_manifest().manifest_sha256


def _capability_payload(
    *,
    adapter_name: str = "docling-export",
    adapter_version: str = "1.1.0",
    parser_name: str = "docling",
    schema_version: str = PARSER_CONTRACT_SCHEMA_VERSION,
    manifest_sha256: str | None = None,
    capability_id: str | None = None,
) -> dict[str, object]:
    manifest = _parser_manifest(
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        parser_name=parser_name,
    ).to_dict()
    manifest["schema_version"] = schema_version
    if manifest_sha256 is not None:
        manifest["manifest_sha256"] = manifest_sha256
    if capability_id is not None:
        manifest["capability_id"] = capability_id
    return {
        "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
        "manifest": manifest,
    }


def _health_payload(
    *,
    adapter_name: str = "docling-export",
    adapter_version: str = "1.1.0",
    parser_name: str = "docling",
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    digest = manifest_sha256 or _parser_manifest(
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        parser_name=parser_name,
    ).manifest_sha256
    return {
        "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
        "state": "ready",
        "accepting_requests": True,
        "in_flight": 0,
        "max_in_flight": 1,
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "manifest_sha256": digest,
    }


def _probes(http_get=None) -> DiagnosticProbes:
    commands = {
        "bash": "/bin/bash",
        "python3": "/usr/bin/python3",
        "flock": "/usr/bin/flock",
        "setsid": "/usr/bin/setsid",
        "nohup": "/usr/bin/nohup",
        "/runtime/python": "/runtime/python",
    }
    return DiagnosticProbes(
        python_version=lambda: (3, 12, 4),
        which=lambda executable: commands.get(executable),
        run_command=lambda command, timeout: CommandOutput(0, "ok\n", ""),
        http_get=http_get or (lambda url, headers, timeout: HttpResponse(200, b"{}")),
        writable_path=lambda path: PathProbeResult(True, str(path), path.exists()),
        monotonic=lambda: 1.0,
        timestamp=lambda: "2026-07-15T00:00:00+00:00",
    )


def _settings(
    root: Path,
    *,
    manifest: Path | None = None,
    parser_workers=(),
) -> SimpleNamespace:
    (root / "Scripts").mkdir(exist_ok=True)
    (root / "Scripts" / "index_pdf.sh").write_text("", encoding="utf-8")
    (root / "main.py").write_text("", encoding="utf-8")
    config = root / "system.yaml"
    config.write_text("llm:\n  backend: local\n", encoding="utf-8")
    values = {
        "project_root": root,
        "system_config": config,
        "runs_dir": root / "runs",
        "conda_environment": "unused",
        "python_executable": "/runtime/python",
        "parser_workers": parser_workers,
    }
    if manifest is not None:
        values["runtime_compatibility_manifest"] = manifest
    return SimpleNamespace(**values)


def _qualified_routed_profile(
    path: Path, *, token_env: str | None = None
) -> Path:
    value = _EXAMPLE_ROUTED_PROFILE.read_text(encoding="utf-8")
    value = value.replace(
        'expected_adapter_version = "replace-with-qualified-version"',
        'expected_adapter_version = "1.1.0"',
        1,
    )
    value = value.replace(
        'expected_adapter_version = "replace-with-qualified-version"',
        'expected_adapter_version = "1.1.0"',
        1,
    )
    value = value.replace(
        'expected_adapter_version = "replace-with-qualified-version"',
        'expected_adapter_version = "1.1.0"',
    )
    for adapter_name, adapter_version, parser_name in (
        ("docling-export", "1.1.0", "docling"),
        ("mineru-normalized", "1.1.0", "mineru"),
        ("paddleocr-vl-page", "1.1.0", "paddleocr-vl"),
    ):
        digest = _parser_manifest(
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            parser_name=parser_name,
        ).manifest_sha256
        value = value.replace("0" * 64, digest, 1)
    if token_env is not None:
        value = value.replace(
            '# bearer_token_env = "DOCARAG_DOCLING_TOKEN"',
            f'bearer_token_env = "{token_env}"',
            1,
        )
    path.write_text(value, encoding="utf-8")
    return path


class RuntimeCompatibilityManifestTests(unittest.TestCase):
    def test_checked_in_manifest_is_truthful_and_dependency_light(self):
        manifest = load_runtime_compatibility_manifest()

        self.assertEqual(MANIFEST_SCHEMA_VERSION, manifest.schema_version)
        self.assertEqual(4, len(manifest.runtimes))
        self.assertEqual(
            {
                "docling-export",
                "mineru-normalized",
                "paddleocr-vl-page",
            },
            set(manifest.by_adapter_name),
        )
        for runtime in manifest.runtimes:
            with self.subTest(runtime=runtime.runtime_id):
                self.assertEqual(VerificationLevel.FIXTURE_ONLY, runtime.verification_level)
                self.assertFalse(runtime.is_production_verified)
        for runtime in manifest.parser_workers:
            with self.subTest(worker=runtime.runtime_id):
                self.assertFalse(runtime.minimum_vram_certified)
                self.assertTrue(runtime.uncertified_artifacts)
                self.assertTrue(
                    all(
                        not item.certified and not item.allowed_versions
                        for item in runtime.all_artifact_constraints
                    )
                )

        code = (
            "import json,sys; import document_services.runtime.compatibility_manifest; "
            "roots=('torch','paddle','paddleocr','docling','mineru','transformers','vllm'); "
            "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in roots)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("[]", completed.stdout.strip())

    def test_loader_accepts_equivalent_json(self):
        original = load_runtime_compatibility_manifest()
        raw = original.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compatibility.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = load_runtime_compatibility_manifest(path)

        self.assertEqual(
            original.source_sha256,
            loaded.source_sha256,
        )
        self.assertEqual(
            original.source_sha256,
            RuntimeCompatibilityManifest.from_dict(original.to_dict()).source_sha256,
        )

    def test_unknown_fields_and_schema_drift_are_rejected(self):
        raw = _raw_default_manifest()
        raw["future_guess"] = True
        with self.assertRaisesRegex(CompatibilityManifestError, "future_guess"):
            RuntimeCompatibilityManifest.from_dict(raw)

        raw = _raw_default_manifest()
        raw["schema_version"] = "docarag.runtime-compatibility.v99"
        with self.assertRaisesRegex(CompatibilityManifestError, "unsupported"):
            RuntimeCompatibilityManifest.from_dict(raw)

    def test_uncertified_constraint_cannot_smuggle_an_allowed_version(self):
        raw = _raw_default_manifest()
        worker = raw["runtimes"][1]  # type: ignore[index]
        constraint = worker["package_version_constraints"][0]  # type: ignore[index]
        constraint["allowed_versions"] = ["2.112.0"]  # type: ignore[index]

        with self.assertRaisesRegex(CompatibilityManifestError, "implying support"):
            RuntimeCompatibilityManifest.from_dict(raw)

    def test_worker_requires_all_artifact_categories_and_safe_locations(self):
        raw = _raw_default_manifest()
        worker = raw["runtimes"][1]  # type: ignore[index]
        worker["image_version_constraints"] = []  # type: ignore[index]
        with self.assertRaisesRegex(CompatibilityManifestError, "package, model, and image"):
            RuntimeCompatibilityManifest.from_dict(raw)

        raw = _raw_default_manifest()
        worker = raw["runtimes"][1]  # type: ignore[index]
        worker["health_location"] = "https://attacker.invalid/health"  # type: ignore[index]
        with self.assertRaisesRegex(CompatibilityManifestError, "absolute HTTP path"):
            RuntimeCompatibilityManifest.from_dict(raw)

    def test_capability_validation_detects_identity_drift_but_keeps_fixture_warning(self):
        expectation = load_runtime_compatibility_manifest().by_runtime_id[
            "docling-worker"
        ]

        compatible = validate_worker_capability_payload(
            expectation,
            _capability_payload(),
            expected_manifest_sha256=_MANIFEST_SHA,
        )
        malformed = validate_worker_capability_payload(
            expectation,
            _capability_payload(
                adapter_name="mineru-normalized",
                schema_version="docarag.parser-contract.v2",
            ),
        )
        incompatible = validate_worker_capability_payload(
            expectation,
            _capability_payload(
                adapter_name="mineru-normalized",
                parser_name="mineru",
            ),
        )

        self.assertTrue(compatible.compatible)
        self.assertTrue(any("fixture-only" in item for item in compatible.warnings))
        self.assertFalse(malformed.compatible)
        self.assertTrue(any("schema" in item for item in malformed.errors))
        self.assertFalse(incompatible.compatible)
        self.assertTrue(any("adapter" in item for item in incompatible.errors))

    def test_health_requires_ready_identity_and_endpoint_resolution_is_origin_bound(self):
        expectation = load_runtime_compatibility_manifest().by_runtime_id[
            "docling-worker"
        ]
        self.assertTrue(
            validate_worker_health_payload(expectation, _health_payload()).compatible
        )
        stopped = _health_payload()
        stopped["state"] = "stopping"
        stopped["accepting_requests"] = False
        self.assertFalse(
            validate_worker_health_payload(expectation, stopped).compatible
        )
        self.assertEqual(
            "https://worker.example/v1/capabilities",
            worker_endpoint_url(
                "https://worker.example", "/v1/capabilities"
            ),
        )
        for unsafe_base_url in (
            "file:///tmp/worker",
            "https://user:secret@worker.example",
            "https://worker.example/some/base",
            "https://worker.example?token=secret",
            "https://worker.example#fragment",
            "https://worker.example:invalid",
        ):
            with self.subTest(base_url=unsafe_base_url):
                with self.assertRaises(CompatibilityManifestError):
                    worker_endpoint_url(unsafe_base_url, "/v1/health")


class RuntimeCompatibilityDoctorTests(unittest.TestCase):
    def test_routed_example_is_reported_as_template_and_never_contacted(self):
        calls: list[str] = []
        profile = load_routed_runtime_profile(_EXAMPLE_ROUTED_PROFILE)
        with tempfile.TemporaryDirectory() as directory:
            report = doctor_for_routed_profile(
                _settings(Path(directory)),
                profile,
                check_endpoints=False,
                probes=_probes(
                    lambda url, headers, timeout: calls.append(url)
                    or HttpResponse(200, b"{}")
                ),
            )

        self.assertEqual([], calls)
        worker_checks = [
            item
            for item in report.checks
            if item.check_id.startswith("routed_profile.worker_config.")
        ]
        self.assertEqual(3, len(worker_checks))
        self.assertTrue(all(item.status is Status.RED for item in worker_checks))
        integration = next(
            item
            for item in report.checks
            if item.check_id == "routed_profile.indexing_integration"
        )
        self.assertEqual(Status.GREEN, integration.status)
        self.assertTrue(integration.details["routed_indexing_enabled"])

    def test_routed_offline_doctor_maps_instance_ids_by_adapter(self):
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_routed_runtime_profile(
                _qualified_routed_profile(root / "routed.toml")
            )
            report = doctor_for_routed_profile(
                _settings(root),
                profile,
                check_endpoints=False,
                probes=_probes(
                    lambda url, headers, timeout: calls.append(url)
                    or HttpResponse(200, b"{}")
                ),
            )

        self.assertEqual([], calls)
        for endpoint in profile.config.endpoints:
            configured = next(
                item
                for item in report.checks
                if item.check_id
                == f"routed_profile.worker_config.{endpoint.runtime_id}"
            )
            worker = next(
                item
                for item in report.checks
                if item.check_id
                == f"runtime_compatibility.worker.{endpoint.runtime_id}"
            )
            self.assertEqual(Status.GREEN, configured.status)
            self.assertEqual(Status.AMBER, worker.status)
            self.assertIn("offline", worker.message)

    def test_routed_offline_doctor_requires_bearer_environment_without_reading_it(self):
        calls: list[str] = []
        secret = "offline-doctor-must-not-serialize-this-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_routed_runtime_profile(
                _qualified_routed_profile(
                    root / "routed.toml", token_env="DOCARAG_DOCLING_TOKEN"
                )
            )
            missing = doctor_for_routed_profile(
                _settings(root),
                profile,
                check_endpoints=False,
                environment={},
                probes=_probes(
                    lambda url, headers, timeout: calls.append(url)
                    or HttpResponse(200, b"{}")
                ),
            )
            present = doctor_for_routed_profile(
                _settings(root),
                profile,
                check_endpoints=False,
                environment={"DOCARAG_DOCLING_TOKEN": secret},
                probes=_probes(
                    lambda url, headers, timeout: calls.append(url)
                    or HttpResponse(200, b"{}")
                ),
            )

        self.assertEqual([], calls)
        missing_check = next(
            item
            for item in missing.checks
            if item.check_id == "routed_profile.worker_config.docling-worker-local"
        )
        present_check = next(
            item
            for item in present.checks
            if item.check_id == "routed_profile.worker_config.docling-worker-local"
        )
        self.assertEqual(Status.RED, missing_check.status)
        self.assertIn(
            "bearer_token_environment_missing", missing_check.details["issue_codes"]
        )
        self.assertFalse(missing_check.details["bearer_token_present"])
        self.assertEqual(Status.GREEN, present_check.status)
        self.assertTrue(present_check.details["bearer_token_present"])
        self.assertNotIn(secret, present.to_json())

    def test_routed_online_doctor_checks_each_exact_worker_pin(self):
        identities = {
            "8101": ("docling-export", "1.1.0", "docling"),
            "8102": ("mineru-normalized", "1.1.0", "mineru"),
            "8103": ("paddleocr-vl-page", "1.1.0", "paddleocr-vl"),
        }
        calls: list[str] = []

        def http_get(url, headers, timeout):
            calls.append(url)
            adapter, version, parser = next(
                value for port, value in identities.items() if f":{port}/" in url
            )
            if url.endswith("/health"):
                payload = _health_payload(
                    adapter_name=adapter,
                    adapter_version=version,
                    parser_name=parser,
                )
            else:
                payload = _capability_payload(
                    adapter_name=adapter,
                    adapter_version=version,
                    parser_name=parser,
                )
            return HttpResponse(200, json.dumps(payload).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_routed_runtime_profile(
                _qualified_routed_profile(root / "routed.toml")
            )
            report = doctor_for_routed_profile(
                _settings(root), profile, probes=_probes(http_get)
            )

        self.assertEqual(6, len(calls))
        for endpoint in profile.config.endpoints:
            worker = next(
                item
                for item in report.checks
                if item.check_id
                == f"runtime_compatibility.worker.{endpoint.runtime_id}"
            )
            self.assertEqual(Status.AMBER, worker.status)
            self.assertIn("uncertified", worker.message)
            self.assertEqual(
                endpoint.expected_adapter_version,
                worker.details["configured_adapter_version"],
            )

    def test_missing_manifest_is_amber_for_legacy_profile_but_red_with_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-runtime-compatibility.toml"
            legacy = doctor_for_settings(
                _settings(root, manifest=missing),
                check_endpoints=False,
                probes=_probes(),
            )
            check = next(
                item
                for item in legacy.checks
                if item.check_id == "runtime_compatibility.manifest"
            )
            self.assertEqual(Status.AMBER, check.status)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = doctor_for_settings(
                _settings(
                    root,
                    manifest=root / "missing-runtime-compatibility.toml",
                    parser_workers={
                        "docling-worker": "http://127.0.0.1:8101"
                    },
                ),
                check_endpoints=False,
                probes=_probes(),
            )
            check = next(
                item
                for item in configured.checks
                if item.check_id == "runtime_compatibility.manifest"
            )
            self.assertEqual(Status.RED, check.status)

    def test_default_manifest_is_loaded_from_installed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(root), check_endpoints=False, probes=_probes()
            )

        check = next(
            item
            for item in report.checks
            if item.check_id == "runtime_compatibility.manifest"
        )
        self.assertEqual(Status.GREEN, check.status)
        self.assertEqual(str(DEFAULT_MANIFEST_PATH), check.details["path"])

    def test_invalid_manifest_is_red_with_remediation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "invalid.toml"
            manifest.write_text("schema_version = 'wrong'\n", encoding="utf-8")

            report = doctor_for_settings(
                _settings(root, manifest=manifest),
                check_endpoints=False,
                probes=_probes(),
            )

            check = next(
                item
                for item in report.checks
                if item.check_id == "runtime_compatibility.manifest"
            )
            self.assertEqual(Status.RED, check.status)
            self.assertIn("Replace", check.remediation)

    def test_offline_worker_is_not_contacted_and_remains_amber(self):
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers=(
                        ParserWorkerDiagnosticSpec(
                            "docling", "docling-worker", "http://127.0.0.1:8101"
                        ),
                    ),
                ),
                check_endpoints=False,
                probes=_probes(
                    lambda url, headers, timeout: calls.append(url)
                    or HttpResponse(200, b"{}")
                ),
            )

        worker = next(
            item
            for item in report.checks
            if item.check_id == "runtime_compatibility.worker.docling"
        )
        self.assertEqual([], calls)
        self.assertEqual(Status.AMBER, worker.status)
        self.assertIn("offline", worker.message)

    def test_ready_compatible_worker_is_amber_until_real_model_certified(self):
        calls: list[tuple[str, float, str | None]] = []

        def http_get(url, headers, timeout):
            calls.append((url, timeout, headers.get("Authorization")))
            payload = _health_payload() if url.endswith("/health") else _capability_payload()
            return HttpResponse(200, json.dumps(payload).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                            "timeout_seconds": 2.5,
                            "bearer_token": "secret-token",
                            "expected_manifest_sha256": _MANIFEST_SHA,
                        }
                    },
                ),
                probes=_probes(http_get),
            )

        worker = next(
            item
            for item in report.checks
            if item.check_id == "runtime_compatibility.worker.docling"
        )
        self.assertEqual(Status.AMBER, worker.status)
        self.assertIn("uncertified", worker.message)
        self.assertEqual(
            [
                "http://127.0.0.1:8101/v1/health",
                "http://127.0.0.1:8101/v1/capabilities",
            ],
            [item[0] for item in calls],
        )
        self.assertTrue(all(item[1] == 2.5 for item in calls))
        self.assertTrue(all(item[2] == "Bearer secret-token" for item in calls))
        self.assertNotIn("secret-token", report.to_json())

    def test_doctor_rejects_incomplete_or_tampered_capability_identity(self):
        def report_for(capability_payload):
            def http_get(url, headers, timeout):
                payload = (
                    _health_payload()
                    if url.endswith("/health")
                    else capability_payload
                )
                return HttpResponse(200, json.dumps(payload).encode("utf-8"))

            root = Path(directory)
            return doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                            "expected_manifest_sha256": _MANIFEST_SHA,
                        }
                    },
                ),
                probes=_probes(http_get),
            )

        fixtures: dict[str, dict[str, object]] = {}
        incomplete = _capability_payload()
        incomplete["manifest"].pop("model_version")  # type: ignore[union-attr]
        fixtures["missing required field"] = incomplete
        digest_drift = _capability_payload()
        digest_drift["manifest"]["metadata"] = {"fixture": False}  # type: ignore[index]
        fixtures["recomputed digest"] = digest_drift
        fixtures["capability identity"] = _capability_payload(
            capability_id="cap_tampered"
        )

        with tempfile.TemporaryDirectory() as directory:
            for label, payload in fixtures.items():
                with self.subTest(label=label):
                    report = report_for(payload)
                    worker = next(
                        item
                        for item in report.checks
                        if item.check_id == "runtime_compatibility.worker.docling"
                    )
                    self.assertEqual(Status.RED, worker.status)
                    self.assertTrue(
                        any(
                            "capability manifest is invalid" in error
                            for error in worker.details["errors"]
                        )
                    )

    def test_worker_probe_exception_cannot_echo_bearer_secret(self):
        secret = "probe-must-redact-this-token"

        def fails(url, headers, timeout):
            raise RuntimeError(f"transport rejected {headers['Authorization']}")

        with tempfile.TemporaryDirectory() as directory:
            report = doctor_for_settings(
                _settings(
                    Path(directory),
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                            "bearer_token": secret,
                        }
                    },
                ),
                probes=_probes(fails),
            )

        rendered = report.to_json()
        self.assertNotIn(secret, rendered)
        self.assertIn("<redacted>", rendered)

    def test_incompatible_or_racing_worker_is_red(self):
        def http_get(url, headers, timeout):
            if url.endswith("/health"):
                return HttpResponse(
                    200, json.dumps(_health_payload()).encode("utf-8")
                )
            return HttpResponse(
                200,
                json.dumps(
                    _capability_payload(
                        adapter_version="9.0.0"
                    )
                ).encode("utf-8"),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                        }
                    },
                ),
                probes=_probes(http_get),
            )

        worker = next(
            item
            for item in report.checks
            if item.check_id == "runtime_compatibility.worker.docling"
        )
        self.assertEqual(Status.RED, worker.status)
        self.assertTrue(
            any("adapter version" in item for item in worker.details["errors"])
        )
        self.assertTrue(
            any("different manifest" in item for item in worker.details["errors"])
        )

    def test_response_size_and_worker_config_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                            "timeout_seconds": 999,
                        }
                    },
                ),
                probes=_probes(),
            )
            invalid = next(
                item
                for item in report.checks
                if item.check_id == "runtime_compatibility.worker_config.0"
            )
            self.assertEqual(Status.RED, invalid.status)

        def oversized(url, headers, timeout):
            return HttpResponse(200, b"{" + b" " * (64 * 1024 + 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = doctor_for_settings(
                _settings(
                    root,
                    manifest=DEFAULT_MANIFEST_PATH,
                    parser_workers={
                        "docling": {
                            "runtime_id": "docling-worker",
                            "base_url": "http://127.0.0.1:8101",
                        }
                    },
                ),
                probes=_probes(oversized),
            )
            worker = next(
                item
                for item in report.checks
                if item.check_id == "runtime_compatibility.worker.docling"
            )
            self.assertEqual(Status.RED, worker.status)
            self.assertIn("exceeds", worker.details["error"])


if __name__ == "__main__":
    unittest.main()
