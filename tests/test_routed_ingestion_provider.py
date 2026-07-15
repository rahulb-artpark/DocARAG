from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from docarag.application import (
    DocumentExtractionRequest,
    ExtractDocumentUseCase,
    RoutedDocumentExtractionBackend,
)
from docarag.canonical_artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
)
from docarag.composition import (
    BookRAGHandoffNotReadyError,
    RoutedBookRAGFacade,
    RoutedBookRAGResult,
)
from docarag.domain import (
    AcceleratorKind,
    BlockType,
    BoundingBox,
    CoordinateSpace,
    DocumentRef,
    ErrorClassification,
    IssueSeverity,
    PageRef,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserBlockObservation,
    ParserCapabilityManifest,
    ParserModality,
    ParserProvenance,
    Representation,
    RepresentationKind,
    SourceGeometry,
)
from docarag.domain.base import content_sha256
from docarag.presets import (
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
)
from docarag.routed_ingestion import (
    CANONICAL_ARTIFACT_FILE,
    LAST_ATTEMPT_AUDIT_FILE,
    ROUTED_AUDIT_FILE,
    ROUTED_INGESTION_SCHEMA_VERSION,
    RoutedCanonicalIngestionProvider,
    RoutedIngestionArtifactError,
    RoutedIngestionArtifactSecurityError,
    RoutedIngestionConfigurationError,
    RoutedIngestionPageLimitError,
    RoutedIngestionPolicy,
    RoutedIngestionSourceBindingError,
)
from docarag.routing import PageFeatures
from docarag.source_material import SOURCE_MATERIAL_BINDING_KEY


class _FakeProbe:
    def __init__(self, page_count: int = 1) -> None:
        self.page_count = page_count
        self.calls: list[tuple[DocumentRef, Path | None]] = []
        self.closed = 0

    def probe(
        self,
        document: DocumentRef,
        *,
        source_path=None,
        page_indices=None,
        page_ranges=None,
    ):
        self.calls.append(
            (document, None if source_path is None else Path(source_path))
        )
        try:
            for page_index in range(self.page_count):
                page = PageRef(
                    document.document_version_id,
                    page_index,
                    width=612,
                    height=792,
                    native_stats={
                        "native_character_count": 500,
                        "native_word_count": 80,
                        "text_coverage_ratio": 0.45,
                        "raster_image_coverage_ratio": 0.05,
                        "replacement_character_ratio": 0.0,
                        "control_character_ratio": 0.0,
                        "duplicated_text_ratio": 0.0,
                    },
                )
                yield PageFeatures.from_page_ref(
                    page,
                    document.mime_type,
                    metadata={"probe_revision": "fixture-v1"},
                )
        finally:
            self.closed += 1


class _FakeAdapter:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.requests: list[ParseRequest] = []
        self._manifest = ParserCapabilityManifest(
            adapter_name=name,
            adapter_version="1.0.0",
            parser_name=name,
            parser_version="fixture-1",
            model_version="fixture-model",
            supported_mime_types=("application/pdf",),
            modalities=tuple(ParserModality),
            block_types=tuple(BlockType),
            accelerator=AcceleratorKind.CPU,
            supports_page_selection=True,
            supports_batch=True,
            deterministic=True,
        )

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request: ParseRequest) -> ParseResult:
        self.requests.append(request)
        if self.fail:
            return ParseResult(
                request_id=request.request_id,
                adapter_name=self.name,
                adapter_version=self._manifest.adapter_version,
                status=ParseStatus.FAILED,
                issues=(
                    ParseIssue(
                        ParseErrorCode.PARSER_FAILURE,
                        ErrorClassification.RETRYABLE,
                        IssueSeverity.ERROR,
                        f"{self.name} fixture failed",
                    ),
                ),
            )
        observations = tuple(
            ParserBlockObservation(
                document_version_id=request.document.document_version_id,
                page_id=page.page_id,
                page_index=page.page_index,
                provenance=ParserProvenance(
                    parser_name=self.name,
                    parser_version=self._manifest.parser_version,
                    model_version=self._manifest.model_version,
                    config_sha256="a" * 64,
                    parser_native_id=f"paragraph-{page.page_index}",
                    raw_artifact_uri=(
                        f"artifact://{self.name}/page-{page.page_index}.json"
                    ),
                    transformations=("routed_ingestion_fixture.v1",),
                ),
                block_type=BlockType.PARAGRAPH,
                geometry=SourceGeometry(
                    CoordinateSpace.PAGE_POINTS,
                    bbox=BoundingBox(20, 30, 500, 90),
                    page_width=612,
                    page_height=792,
                ),
                reading_order=0,
                representations=(
                    Representation(
                        RepresentationKind.TEXT,
                        f"Canonical page {page.page_index}",
                        "text/plain",
                        confidence=0.99,
                    ),
                ),
                confidence=0.99,
            )
            for page in request.pages
        )
        return ParseResult(
            request_id=request.request_id,
            adapter_name=self.name,
            adapter_version=self._manifest.adapter_version,
            status=ParseStatus.SUCCEEDED,
            observations=observations,
        )


def _facade(*, fail: bool = False):
    adapters = (
        _FakeAdapter(DOCLING_ADAPTER, fail=fail),
        _FakeAdapter(MINERU_ADAPTER, fail=fail),
        _FakeAdapter(PADDLEOCR_VL_ADAPTER, fail=fail),
    )
    return RoutedBookRAGFacade.routed_default(adapters=adapters), adapters


class RoutedCanonicalIngestionProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_bytes = b"%PDF-1.7 routed ingestion fixture\n"
        self.source_path = self.root / "source.pdf"
        self.source_path.write_bytes(self.source_bytes)
        self.source_sha256 = hashlib.sha256(self.source_bytes).hexdigest()
        self.cfg = SimpleNamespace(
            source=SimpleNamespace(
                display_name="source.pdf",
                original_uri="file:///original/source.pdf",
                sha256=self.source_sha256,
                snapshot_path=str(self.source_path),
                api_key="must-never-be-serialized",
            ),
            pdf_path=str(self.source_path),
            save_path=str(self.root / "run"),
            unrelated_secret="also-must-never-be-serialized",
        )

    def provider(self, *, page_count: int = 1, fail: bool = False, policy=None):
        probe = _FakeProbe(page_count)
        facade, adapters = _facade(fail=fail)
        provider = RoutedCanonicalIngestionProvider(
            probe,
            facade,
            policy=policy,
        )
        return provider, probe, adapters

    @property
    def audit_path(self) -> Path:
        return Path(self.cfg.save_path) / ROUTED_AUDIT_FILE

    @property
    def canonical_path(self) -> Path:
        return Path(self.cfg.save_path) / CANONICAL_ARTIFACT_FILE

    @property
    def last_attempt_path(self) -> Path:
        return Path(self.cfg.save_path) / LAST_ATTEMPT_AUDIT_FILE

    def test_success_publishes_immutable_core_compatible_payload_and_full_audit(self):
        provider, probe, adapters = self.provider(page_count=2)

        payload = provider.ingest(self.cfg)

        self.assertEqual(2, payload.document.page_count)
        self.assertEqual("file:///original/source.pdf", payload.document.source_uri)
        material_binding = payload.document.metadata[SOURCE_MATERIAL_BINDING_KEY]
        self.assertEqual(self.source_path.resolve().as_uri(), material_binding["uri"])
        self.assertEqual(self.source_sha256, material_binding["sha256"])
        self.assertEqual(2, len(payload.canonical_blocks))
        self.assertEqual(self.canonical_path.resolve(), payload.canonical_artifact_path)
        self.assertEqual(1, len(probe.calls))
        self.assertEqual(1, probe.closed)
        self.assertGreaterEqual(sum(len(adapter.requests) for adapter in adapters), 1)
        with self.assertRaises(FrozenInstanceError):
            payload.document = payload.document  # type: ignore[misc]

        audit = RoutedBookRAGResult.from_json(self.audit_path.read_bytes())
        self.assertTrue(audit.is_ready_for_bookrag_handoff)
        canonical = json.loads(self.canonical_path.read_text(encoding="utf-8"))
        self.assertEqual(
            CANONICAL_EVIDENCE_SCHEMA_VERSION,
            canonical["artifact_schema_version"],
        )
        self.assertEqual(
            CANONICAL_CONTRACT_SCHEMA_VERSION,
            canonical["contract_schema_version"],
        )
        self.assertEqual(
            ROUTED_INGESTION_SCHEMA_VERSION,
            canonical["producer_schema_version"],
        )
        self.assertEqual(2, canonical["block_count"])
        self.assertEqual(audit.projection_id, canonical["projection_id"])
        serialized = self.audit_path.read_text() + self.canonical_path.read_text()
        self.assertNotIn("must-never-be-serialized", serialized)

    def test_application_use_case_executes_the_real_routed_provider(self):
        provider, probe, adapters = self.provider(page_count=2)
        request = DocumentExtractionRequest(
            source_path=self.source_path.resolve(),
            output_root=Path(self.cfg.save_path).resolve(),
            source_sha256=self.source_sha256,
            source_uri=self.cfg.source.original_uri,
            display_name=self.cfg.source.display_name,
        )

        result = ExtractDocumentUseCase(
            RoutedDocumentExtractionBackend(provider)
        ).execute(request)

        self.assertEqual(2, result.canonical_block_count)
        self.assertEqual(self.source_sha256, result.document.content_sha256)
        self.assertEqual(self.canonical_path.resolve().as_uri(), result.canonical_artifact_uri)
        self.assertEqual(
            hashlib.sha256(self.canonical_path.read_bytes()).hexdigest(),
            result.canonical_artifact_sha256,
        )
        self.assertEqual(self.audit_path.resolve().as_uri(), result.audit_artifact_uri)
        self.assertEqual(
            hashlib.sha256(self.audit_path.read_bytes()).hexdigest(),
            result.audit_artifact_sha256,
        )
        self.assertEqual(self.audit_path.stat().st_size, result.audit_artifact_size_bytes)
        self.assertEqual(
            content_sha256(json.loads(self.audit_path.read_text(encoding="utf-8"))),
            result.routed_audit_sha256,
        )
        self.assertEqual(1, len(probe.calls))
        self.assertGreaterEqual(sum(len(adapter.requests) for adapter in adapters), 1)

    def test_ready_artifacts_are_reused_without_probe_or_parser_execution(self):
        first, _, _ = self.provider(page_count=2)
        expected = first.ingest(self.cfg)
        second, probe, adapters = self.provider(page_count=2)

        restored = second.ingest(self.cfg, reforce=False)

        self.assertEqual(expected.document, restored.document)
        self.assertEqual(expected.canonical_blocks, restored.canonical_blocks)
        self.assertEqual([], probe.calls)
        self.assertTrue(all(not adapter.requests for adapter in adapters))

    def test_forced_identical_run_is_byte_deterministic(self):
        provider, _, _ = self.provider(page_count=2)
        provider.ingest(self.cfg, reforce=True)
        first_audit = self.audit_path.read_bytes()
        first_canonical = self.canonical_path.read_bytes()

        provider.ingest(self.cfg, reforce=True)

        self.assertEqual(first_audit, self.audit_path.read_bytes())
        self.assertEqual(first_canonical, self.canonical_path.read_bytes())

    def test_blocked_handoff_persists_audit_and_never_publishes_canonical(self):
        provider, probe, _ = self.provider(fail=True)

        with self.assertRaises(BookRAGHandoffNotReadyError):
            provider.ingest(self.cfg)

        self.assertEqual(1, len(probe.calls))
        self.assertTrue(self.audit_path.is_file())
        self.assertFalse(self.canonical_path.exists())
        audit = RoutedBookRAGResult.from_json(self.audit_path.read_bytes())
        self.assertFalse(audit.is_ready_for_bookrag_handoff)

    def test_failed_forced_run_preserves_previous_ready_pair_and_audits_attempt(self):
        success, _, _ = self.provider()
        success.ingest(self.cfg)
        ready_audit_bytes = self.audit_path.read_bytes()
        ready_canonical_bytes = self.canonical_path.read_bytes()
        failure, _, _ = self.provider(fail=True)

        with self.assertRaises(BookRAGHandoffNotReadyError):
            failure.ingest(self.cfg, reforce=True)

        self.assertEqual(ready_audit_bytes, self.audit_path.read_bytes())
        self.assertEqual(ready_canonical_bytes, self.canonical_path.read_bytes())
        self.assertTrue(
            RoutedBookRAGResult.from_json(
                self.audit_path.read_bytes()
            ).is_ready_for_bookrag_handoff
        )
        attempt = RoutedBookRAGResult.from_json(self.last_attempt_path.read_bytes())
        self.assertFalse(attempt.is_ready_for_bookrag_handoff)

    def test_staged_pair_replacement_rolls_back_if_second_promotion_fails(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        ready_audit_bytes = self.audit_path.read_bytes()
        ready_canonical_bytes = self.canonical_path.read_bytes()
        real_replace = __import__("os").replace

        def replace_with_failure(source, destination):
            if (
                Path(destination) == self.canonical_path
                and Path(source).name.endswith(".stage")
            ):
                raise OSError("simulated second promotion failure")
            return real_replace(source, destination)

        with mock.patch(
            "docarag.routed_ingestion.os.replace",
            side_effect=replace_with_failure,
        ):
            with self.assertRaisesRegex(
                RoutedIngestionArtifactError, "pair replacement failed"
            ):
                provider.ingest(self.cfg, reforce=True)

        self.assertEqual(ready_audit_bytes, self.audit_path.read_bytes())
        self.assertEqual(ready_canonical_bytes, self.canonical_path.read_bytes())

    def test_live_source_mismatch_prevents_artifact_reuse(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        self.source_path.write_bytes(self.source_bytes + b"changed")
        second, probe, adapters = self.provider()

        with self.assertRaisesRegex(
            RoutedIngestionSourceBindingError, "SHA-256"
        ):
            second.ingest(self.cfg)

        self.assertEqual([], probe.calls)
        self.assertTrue(all(not adapter.requests for adapter in adapters))

    def test_changed_logical_source_identity_prevents_artifact_reuse(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        self.cfg.source.original_uri = "file:///another/source.pdf"
        second, _, _ = self.provider()

        with self.assertRaisesRegex(
            RoutedIngestionArtifactError, "another source document"
        ):
            second.ingest(self.cfg)

    def test_corrupt_audit_and_canonical_bindings_fail_closed(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        original_audit = self.audit_path.read_bytes()
        original_canonical = self.canonical_path.read_bytes()
        cases = (
            ("audit projection", self.audit_path, "projection_id", "bad"),
            (
                "canonical schema",
                self.canonical_path,
                "artifact_schema_version",
                "future.schema.v999",
            ),
            (
                "canonical digest",
                self.canonical_path,
                "canonical_blocks_sha256",
                "0" * 64,
            ),
            ("canonical count", self.canonical_path, "block_count", 999),
        )
        for label, path, field_name, value in cases:
            with self.subTest(label=label):
                self.audit_path.write_bytes(original_audit)
                self.canonical_path.write_bytes(original_canonical)
                document = json.loads(path.read_text(encoding="utf-8"))
                document[field_name] = value
                path.write_text(json.dumps(document), encoding="utf-8")
                second, probe, adapters = self.provider()
                with self.assertRaises(RoutedIngestionArtifactError):
                    second.ingest(self.cfg)
                self.assertEqual([], probe.calls)
                self.assertTrue(all(not adapter.requests for adapter in adapters))

    def test_symlinked_artifact_and_output_root_are_rejected(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        outside = self.root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        self.canonical_path.unlink()
        self.canonical_path.symlink_to(outside)
        second, _, _ = self.provider()
        with self.assertRaises(RoutedIngestionArtifactSecurityError):
            second.ingest(self.cfg)

        alternate_cfg = SimpleNamespace(**vars(self.cfg))
        linked_root = self.root / "linked-run"
        linked_root.symlink_to(Path(self.cfg.save_path), target_is_directory=True)
        alternate_cfg.save_path = str(linked_root)
        third, _, _ = self.provider()
        with self.assertRaises(RoutedIngestionArtifactSecurityError):
            third.ingest(alternate_cfg)

    def test_page_feature_materialization_is_bounded_and_closes_probe(self):
        policy = RoutedIngestionPolicy(max_page_features=2)
        provider, probe, adapters = self.provider(page_count=3, policy=policy)

        with self.assertRaisesRegex(
            RoutedIngestionPageLimitError, "max_page_features"
        ):
            provider.ingest(self.cfg)

        self.assertEqual(1, probe.closed)
        self.assertTrue(all(not adapter.requests for adapter in adapters))
        self.assertFalse(self.audit_path.exists())
        self.assertFalse(self.canonical_path.exists())

    def test_authoritative_source_sha256_is_required_before_any_probe(self):
        self.cfg.source.sha256 = None
        provider, probe, adapters = self.provider()

        with self.assertRaisesRegex(
            RoutedIngestionConfigurationError, "sha256"
        ):
            provider.ingest(self.cfg)

        self.assertEqual([], probe.calls)
        self.assertTrue(all(not adapter.requests for adapter in adapters))

    def test_ready_audit_without_canonical_and_canonical_without_audit_are_rejected(self):
        provider, _, _ = self.provider()
        provider.ingest(self.cfg)
        canonical_bytes = self.canonical_path.read_bytes()
        self.canonical_path.unlink()
        second, _, _ = self.provider()
        with self.assertRaisesRegex(RoutedIngestionArtifactError, "missing"):
            second.ingest(self.cfg)

        self.audit_path.unlink()
        self.canonical_path.write_bytes(canonical_bytes)
        third, _, _ = self.provider()
        with self.assertRaisesRegex(RoutedIngestionArtifactError, "without"):
            third.ingest(self.cfg)


if __name__ == "__main__":
    unittest.main()
