from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest

from docarag.application import (
    CanonicalExtractionPayload,
    DocumentExtractionArtifactError,
    DocumentExtractionBindingError,
    DocumentExtractionPolicy,
    DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
    DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
    DocumentExtractionRequest,
    DocumentExtractionResult,
    ExtractDocumentUseCase,
    RoutedDocumentExtractionBackend,
)
from docarag.canonical_artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_ARTIFACT_ROLE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
)
from docarag.composition import RoutedBookRAGFacade
from docarag.domain import (
    AcceleratorKind,
    BlockType,
    BoundingBox,
    CanonicalBlock,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserBlockObservation,
    ParserCapabilityManifest,
    ParserModality,
    ParserProvenance,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
)
from docarag.domain.base import (
    ContractValidationError,
    canonical_json_bytes,
    content_sha256,
)
from docarag.presets import DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER
from docarag.routed_ingestion import RoutedBookRAGAuditValidator
from docarag.routing import PageFeatures


SOURCE_BYTES = b"%PDF-1.7 application extraction fixture\n"
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()
CONFIGURATION_SHA256 = "c" * 64


def _request(root: Path, *, force: bool = False) -> DocumentExtractionRequest:
    source = root / "source.pdf"
    source.write_bytes(SOURCE_BYTES)
    return DocumentExtractionRequest(
        source_path=source,
        output_root=root / "canonical-run",
        source_sha256=SOURCE_SHA256,
        source_uri="urn:test:application-source",
        display_name="source.pdf",
        force=force,
    )


def _document(request: DocumentExtractionRequest) -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=request.source_sha256,
        mime_type=request.media_type,
        source_uri=request.source_uri,
        display_name=request.display_name,
        page_count=1,
    )


def _block(
    document: DocumentRef,
    *,
    quality_status: QualityStatus = QualityStatus.ACCEPTED,
) -> CanonicalBlock:
    page = PageRef(document.document_version_id, 0)
    return CanonicalBlock(
        canonical_key="paragraph-0",
        revision=1,
        document_version_id=document.document_version_id,
        page_id=page.page_id,
        page_index=0,
        block_type=BlockType.PARAGRAPH,
        representations=(
            Representation(
                RepresentationKind.TEXT,
                "Canonical application text",
                "text/plain",
            ),
        ),
        source_observation_ids=("obs_application_fixture",),
        reconciliation_method="application-fixture.v1",
        quality_status=quality_status,
    )


class _AuditFixtureAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self._manifest = ParserCapabilityManifest(
            adapter_name=name,
            adapter_version="1.0.0",
            parser_name=name,
            parser_version="fixture-1",
            model_version="fixture-model",
            supported_mime_types=("application/pdf",),
            modalities=(ParserModality.NATIVE_TEXT, ParserModality.LAYOUT),
            block_types=(BlockType.PARAGRAPH,),
            accelerator=AcceleratorKind.CPU,
            supports_page_selection=True,
            supports_batch=True,
            deterministic=True,
        )

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request: ParseRequest) -> ParseResult:
        observations = tuple(
            ParserBlockObservation(
                document_version_id=request.document.document_version_id,
                page_id=page.page_id,
                page_index=page.page_index,
                provenance=ParserProvenance(
                    parser_name=self.name,
                    parser_version="fixture-1",
                    model_version="fixture-model",
                    config_sha256="a" * 64,
                    parser_native_id=f"paragraph-{page.page_index}",
                    raw_artifact_uri=(
                        f"artifact://{self.name}/page-{page.page_index}.json"
                    ),
                    transformations=("application_audit_fixture.v1",),
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
                        "Canonical application text",
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
            adapter_version="1.0.0",
            status=ParseStatus.SUCCEEDED,
            observations=observations,
        )


def _routed_audit(document: DocumentRef):
    page = PageRef(
        document.document_version_id,
        0,
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
    feature = PageFeatures.from_page_ref(page, document.mime_type)
    request = ParseRequest(
        document=document,
        pages=(page,),
        run_id="run-application-audit-fixture",
        trace_id="trace-application-audit-fixture",
        idempotency_key="application-audit-fixture-v1",
    )
    facade = RoutedBookRAGFacade.routed_default(
        adapters=tuple(
            _AuditFixtureAdapter(name)
            for name in (DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER)
        )
    )
    return facade.run(request, (feature,))


def _artifact_payload(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, ...],
    audit: dict[str, object],
) -> dict[str, object]:
    serialized = [block.to_dict() for block in blocks]
    request = audit["request"]
    parsing = audit["parsing"]
    assert isinstance(request, dict) and isinstance(parsing, dict)
    return {
        "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
        "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
        "artifact_role": CANONICAL_EVIDENCE_ARTIFACT_ROLE,
        "producer_schema_version": "docarag.routed-canonical-ingestion.v1",
        "configuration_sha256": CONFIGURATION_SHA256,
        "audit_sha256": content_sha256(audit),
        "parse_request_id": request["request_id"],
        "projection_id": audit["projection_id"],
        "routed_parsing_result_id": parsing["result_id"],
        "document_version_id": document.document_version_id,
        "source_sha256": document.content_sha256,
        "document": document.to_dict(),
        "block_count": len(blocks),
        "canonical_blocks_sha256": content_sha256(serialized),
        "canonical_blocks": serialized,
    }


def _write_artifact(path: Path, payload: object) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


class _Backend:
    def __init__(
        self,
        *,
        document_factory=_document,
        block_factory=_block,
        artifact_path_factory=None,
    ) -> None:
        self.configuration_sha256 = CONFIGURATION_SHA256
        self.audit_validator = RoutedBookRAGAuditValidator()
        self.document_factory = document_factory
        self.block_factory = block_factory
        self.artifact_path_factory = artifact_path_factory
        self.requests: list[DocumentExtractionRequest] = []

    def extract(self, request: DocumentExtractionRequest) -> CanonicalExtractionPayload:
        self.requests.append(request)
        document = self.document_factory(request)
        routed = _routed_audit(document)
        block = (
            routed.parsing.canonical_blocks[0]
            if self.block_factory is _block
            else self.block_factory(document)
        )
        audit_payload = routed.to_dict()
        if self.artifact_path_factory is None:
            artifact = Path(request.output_root) / "canonical_blocks.json"
        else:
            artifact = Path(self.artifact_path_factory(request))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        audit = artifact.parent / "routed_parsing_audit.json"
        _write_artifact(audit, audit_payload)
        _write_artifact(
            artifact,
            _artifact_payload(document, (block,), audit_payload),
        )
        return CanonicalExtractionPayload(
            document=document,
            canonical_blocks=(block,),
            canonical_artifact_path=artifact,
            audit_artifact_path=audit,
            canonical_blocks_sha256=content_sha256([block.to_dict()]),
        )


class DocumentExtractionContractTests(unittest.TestCase):
    def test_dependency_light_application_import_does_not_load_composition(self):
        script = """
import json, sys
import docarag.application.extract_document
print(json.dumps('docarag.composition' in sys.modules))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("false", completed.stdout.strip())

    def test_request_is_immutable_deterministic_bounded_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary), force=True)

            restored = DocumentExtractionRequest.from_json(request.to_json())

            self.assertEqual(request, restored)
            self.assertEqual(
                "docarag.application.document-extraction.v1",
                DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
            )
            self.assertEqual(
                DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
                request.to_dict()["schema_version"],
            )
            response_version_request = request.to_dict()
            response_version_request["schema_version"] = (
                DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION
            )
            with self.assertRaisesRegex(
                ContractValidationError,
                "document extraction request schema_version",
            ):
                DocumentExtractionRequest.from_dict(response_version_request)
            self.assertTrue(request.source_path.startswith(os.sep))
            self.assertTrue(request.output_root.startswith(os.sep))
            with self.assertRaises(FrozenInstanceError):
                request.force = False  # type: ignore[misc]
            damaged = request.to_dict()
            damaged["request_id"] = "document-extraction-request_" + "0" * 32
            with self.assertRaisesRegex(ContractValidationError, "request_id mismatch"):
                DocumentExtractionRequest.from_dict(damaged)
            with self.assertRaisesRegex(ContractValidationError, "serialized limit"):
                DocumentExtractionRequest.from_json("{" + " " * (128 * 1024) + "}")

    def test_request_rejects_ambiguous_paths_secrets_and_unsupported_media(self):
        common = dict(
            source_path="relative.pdf",
            output_root="/tmp/output",
            source_sha256=SOURCE_SHA256,
            source_uri="urn:test:source",
            display_name="source.pdf",
        )
        with self.assertRaisesRegex(ContractValidationError, "source_path must be absolute"):
            DocumentExtractionRequest(**common)
        common["source_path"] = "/tmp/source.pdf"
        common["source_uri"] = "https://example.test/doc.pdf?access_token=secret"
        with self.assertRaisesRegex(ContractValidationError, "secret query"):
            DocumentExtractionRequest(**common)
        common["source_uri"] = "urn:test:source"
        common["media_type"] = "image/png"
        with self.assertRaisesRegex(ContractValidationError, "application/pdf only"):
            DocumentExtractionRequest(**common)


class ExtractDocumentUseCaseTests(unittest.TestCase):
    def test_audit_replay_validator_is_an_explicit_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))
            backend = _Backend()
            validator = backend.audit_validator
            del backend.audit_validator

            with self.assertRaisesRegex(TypeError, "audit_validator"):
                ExtractDocumentUseCase(backend)

            result = ExtractDocumentUseCase(
                backend,
                audit_validator=validator,
            ).execute(request)
            self.assertEqual(1, result.canonical_block_count)

    def test_success_returns_compact_exact_receipt_without_block_duplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            backend = _Backend()
            use_case = ExtractDocumentUseCase(backend)

            result = use_case.execute(request)

            self.assertEqual([request], backend.requests)
            artifact = Path(request.output_root) / "canonical_blocks.json"
            self.assertEqual(artifact.resolve().as_uri(), result.canonical_artifact_uri)
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                result.canonical_artifact_sha256,
            )
            self.assertEqual(artifact.stat().st_size, result.canonical_artifact_size_bytes)
            audit = Path(request.output_root) / "routed_parsing_audit.json"
            self.assertEqual(audit.resolve().as_uri(), result.audit_artifact_uri)
            self.assertEqual(
                hashlib.sha256(audit.read_bytes()).hexdigest(),
                result.audit_artifact_sha256,
            )
            self.assertEqual(audit.stat().st_size, result.audit_artifact_size_bytes)
            self.assertEqual(
                content_sha256(json.loads(audit.read_text(encoding="utf-8"))),
                result.routed_audit_sha256,
            )
            self.assertEqual(1, result.canonical_block_count)
            self.assertEqual(
                "docarag.application.document-extraction.v2",
                DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
            )
            self.assertEqual(
                DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
                result.to_dict()["schema_version"],
            )
            self.assertNotIn("canonical_blocks", result.to_dict())
            self.assertEqual(result, DocumentExtractionResult.from_json(result.to_json()))

    def test_rejects_source_mismatch_review_content_and_configuration_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)

            def wrong_document(value):
                return DocumentRef.identified(
                    content_sha256="f" * 64,
                    mime_type=value.media_type,
                    source_uri=value.source_uri,
                    display_name=value.display_name,
                    page_count=1,
                )

            with self.assertRaisesRegex(
                DocumentExtractionBindingError, "document identity"
            ):
                ExtractDocumentUseCase(
                    _Backend(document_factory=wrong_document)
                ).execute(request)

            def review_block(document):
                return _block(document, quality_status=QualityStatus.NEEDS_REVIEW)

            with self.assertRaisesRegex(
                DocumentExtractionBindingError, "only accepted"
            ):
                ExtractDocumentUseCase(
                    _Backend(block_factory=review_block)
                ).execute(request)

            backend = _Backend()
            use_case = ExtractDocumentUseCase(backend)
            backend.configuration_sha256 = "d" * 64
            with self.assertRaisesRegex(
                DocumentExtractionBindingError, "configuration identity changed"
            ):
                use_case.execute(request)

    def test_rejects_a_backend_semantic_digest_that_does_not_match_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))

            class WrongDigestBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    return replace(payload, canonical_blocks_sha256="f" * 64)

            with self.assertRaisesRegex(
                DocumentExtractionBindingError,
                "canonical_blocks_sha256 differs",
            ):
                ExtractDocumentUseCase(WrongDigestBackend()).execute(request)

    def test_rejects_an_artifact_whose_blocks_differ_from_the_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))

            class SplitBrainBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    artifact = Path(payload.canonical_artifact_path)
                    envelope = json.loads(artifact.read_text(encoding="utf-8"))
                    envelope["canonical_blocks"][0]["representations"][0][
                        "content"
                    ] = "unrelated artifact content"
                    envelope["canonical_blocks_sha256"] = content_sha256(
                        envelope["canonical_blocks"]
                    )
                    _write_artifact(artifact, envelope)
                    return payload

            with self.assertRaisesRegex(
                DocumentExtractionBindingError,
                "canonical artifact canonical_blocks_sha256 differs",
            ):
                ExtractDocumentUseCase(SplitBrainBackend()).execute(request)

    def test_artifact_must_be_bounded_regular_and_beneath_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            Path(request.output_root).mkdir(parents=True)
            outside = root / "outside.json"
            with self.assertRaisesRegex(
                DocumentExtractionArtifactError, "beneath output_root"
            ):
                ExtractDocumentUseCase(
                    _Backend(artifact_path_factory=lambda _: outside)
                ).execute(request)

            real_directory = root / "real-artifact-directory"
            real_directory.mkdir()
            linked_directory = Path(request.output_root) / "linked"
            linked_directory.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaisesRegex(
                DocumentExtractionArtifactError,
                "without symlinks|opened safely",
            ):
                ExtractDocumentUseCase(
                    _Backend(
                        artifact_path_factory=lambda _: (
                            linked_directory / "canonical_blocks.json"
                        )
                    )
                ).execute(request)

            policy = DocumentExtractionPolicy(max_canonical_artifact_bytes=1024)

            class LargeBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    Path(payload.canonical_artifact_path).write_bytes(b"x" * 1025)
                    return payload

            with self.assertRaisesRegex(
                DocumentExtractionArtifactError, "exceeds"
            ):
                ExtractDocumentUseCase(LargeBackend(), policy=policy).execute(request)

    def test_rejects_missing_or_content_mismatched_routed_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))

            class MissingAuditBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    Path(payload.audit_artifact_path).unlink()
                    return payload

            with self.assertRaisesRegex(
                DocumentExtractionArtifactError, "routed audit artifact"
            ):
                ExtractDocumentUseCase(MissingAuditBackend()).execute(request)

            class MismatchedAuditBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    audit = json.loads(
                        Path(payload.audit_artifact_path).read_text(encoding="utf-8")
                    )
                    audit["projection_name"] = "unrelated-projection"
                    _write_artifact(Path(payload.audit_artifact_path), audit)
                    return payload

            with self.assertRaisesRegex(
                DocumentExtractionBindingError, "audit_sha256"
            ):
                ExtractDocumentUseCase(MismatchedAuditBackend()).execute(request)

    def test_rejects_nested_audit_tampering_even_when_its_digest_is_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))

            class ArbitraryAuditBackend(_Backend):
                def extract(self, value):
                    payload = super().extract(value)
                    tampered = json.loads(
                        Path(payload.audit_artifact_path).read_text(encoding="utf-8")
                    )
                    tampered["request"]["forged_nested_trace"] = True
                    _write_artifact(Path(payload.audit_artifact_path), tampered)
                    canonical_path = Path(payload.canonical_artifact_path)
                    canonical = json.loads(
                        canonical_path.read_text(encoding="utf-8")
                    )
                    canonical["audit_sha256"] = content_sha256(tampered)
                    _write_artifact(canonical_path, canonical)
                    return payload

            with self.assertRaisesRegex(
                DocumentExtractionArtifactError, "routed audit artifact is not replayable"
            ):
                ExtractDocumentUseCase(ArbitraryAuditBackend()).execute(request)

    def test_routed_adapter_translates_request_and_force_without_core_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary), force=True)
            document = _document(request)
            block = _block(document)
            artifact = Path(request.output_root) / "canonical_blocks.json"
            audit = Path(request.output_root) / "routed_parsing_audit.json"
            calls = []

            class Provider:
                configuration_sha256 = CONFIGURATION_SHA256

                def ingest(self, cfg, *, reforce=False):
                    calls.append((cfg, reforce))
                    return SimpleNamespace(
                        document=document,
                        canonical_blocks=(block,),
                        canonical_artifact_path=artifact,
                        audit_artifact_path=audit,
                        canonical_blocks_sha256=content_sha256([block.to_dict()]),
                    )

            payload = RoutedDocumentExtractionBackend(
                Provider(),
                audit_validator=RoutedBookRAGAuditValidator(),
            ).extract(request)

            self.assertEqual(document, payload.document)
            self.assertEqual(1, len(calls))
            cfg, force = calls[0]
            self.assertTrue(force)
            self.assertEqual(request.source_path, cfg.pdf_path)
            self.assertEqual(request.output_root, cfg.save_path)
            self.assertEqual(request.source_sha256, cfg.source.sha256)
            self.assertEqual(request.source_uri, cfg.source.original_uri)
            self.assertEqual(request.source_path, cfg.source.snapshot_path)
            self.assertEqual(audit, Path(payload.audit_artifact_path))


if __name__ == "__main__":
    unittest.main()
