from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest

from docarag.bookrag_projection import BookRAGImageMaterialPolicy
from docarag.composition import (
    BookRAGHandoffNotReadyError,
    BookRAGHandoffReadiness,
    CompositionBindingError,
    RoutedBookRAGFacade,
    RoutedBookRAGResult,
)
from docarag.domain import (
    AcceleratorKind,
    BlockType,
    BoundingBox,
    ContractValidationError,
    CoordinateSpace,
    DocumentRef,
    ErrorClassification,
    IssueSeverity,
    PageRef,
    ParseErrorCode,
    ParseIssue,
    ParseOptions,
    ParseRegion,
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
from docarag.parsing_orchestrator import RoutedParsingStatus
from docarag.ports import ParserRegistry
from docarag.presets import (
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
    balanced_pdf_routing_policy,
)
from docarag.routing import PageFeatures


def document(page_count: int = 1) -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=hashlib.sha256(b"composition source").hexdigest(),
        mime_type="application/pdf",
        source_uri="file:///fixtures/composition.pdf",
        document_id="composition-fixture",
        display_name="composition.pdf",
        page_count=page_count,
    )


def stats(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "native_character_count": 500,
        "native_word_count": 80,
        "text_coverage_ratio": 0.45,
        "raster_image_coverage_ratio": 0.05,
        "replacement_character_ratio": 0.0,
        "control_character_ratio": 0.0,
        "duplicated_text_ratio": 0.0,
    }
    values.update(overrides)
    return values


def page_features(source: DocumentRef) -> tuple[PageFeatures, ...]:
    return tuple(
        PageFeatures.from_page_ref(
            PageRef(
                source.document_version_id,
                page_index,
                width=612,
                height=792,
                native_stats=stats(),
            ),
            source.mime_type,
        )
        for page_index in range(source.page_count or 0)
    )


def request_for(
    source: DocumentRef,
    features: tuple[PageFeatures, ...],
    *,
    options: ParseOptions | None = None,
) -> ParseRequest:
    return ParseRequest(
        document=source,
        pages=tuple(feature.page for feature in features),
        options=options or ParseOptions(languages=("en",)),
        run_id="run-composition",
        trace_id="trace-composition",
        idempotency_key="composition-fixture-v1",
    )


class FakeBalancedAdapter:
    def __init__(self, name: str, *, mode: str = "success", text: str = "Hello RAG") -> None:
        self.name = name
        self.mode = mode
        self.text = text
        self.requests = []
        self._manifest = ParserCapabilityManifest(
            adapter_name=name,
            adapter_version="1.0.0",
            parser_name=name,
            parser_version="fixture-1",
            model_version="fixture-model",
            supported_mime_types=("application/pdf",),
            modalities=(
                ParserModality.NATIVE_TEXT,
                ParserModality.OCR_TEXT,
                ParserModality.LAYOUT,
            ),
            block_types=(BlockType.PARAGRAPH,),
            accelerator=AcceleratorKind.CPU,
            supports_page_selection=True,
            supports_batch=True,
            deterministic=True,
        )

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request: ParseRequest) -> ParseResult:
        self.requests.append(request)
        if self.mode == "failure":
            return ParseResult(
                request_id=request.request_id,
                adapter_name=self.name,
                adapter_version="1.0.0",
                status=ParseStatus.FAILED,
                issues=(
                    ParseIssue(
                        ParseErrorCode.PARSER_FAILURE,
                        ErrorClassification.RETRYABLE,
                        IssueSeverity.ERROR,
                        f"{self.name} fixture failure",
                    ),
                ),
            )
        confidence = 0.10 if self.mode == "low_quality" else 0.99
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
                    raw_artifact_uri=f"artifact://{self.name}/page-{page.page_index}.json",
                    transformations=("composition_fake.v1",),
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
                        self.text,
                        "text/plain",
                        confidence=confidence,
                    ),
                ),
                confidence=confidence,
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


def adapters(
    *,
    docling_mode: str = "success",
    mineru_mode: str = "success",
) -> tuple[FakeBalancedAdapter, FakeBalancedAdapter, FakeBalancedAdapter]:
    return (
        FakeBalancedAdapter(DOCLING_ADAPTER, mode=docling_mode),
        FakeBalancedAdapter(MINERU_ADAPTER, mode=mineru_mode),
        FakeBalancedAdapter(PADDLEOCR_VL_ADAPTER),
    )


class RoutedBookRAGFacadeTests(unittest.TestCase):
    def test_routed_default_projects_traceable_rows_without_eager_secondary(self):
        source = document()
        features = page_features(source)
        parser_adapters = adapters()
        facade = RoutedBookRAGFacade.routed_default(adapters=parser_adapters)

        result = facade.run(request_for(source, features), features)

        self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.parsing.status)
        self.assertEqual(1, len(parser_adapters[0].requests))
        self.assertEqual(0, len(parser_adapters[1].requests))
        self.assertEqual(0, len(parser_adapters[2].requests))
        rows = result.to_bookrag_pdf_list()
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("Hello RAG", row["text"])
        self.assertEqual(source.content_sha256, row["source_sha256"])
        self.assertEqual(
            result.parsing.canonical_blocks[0].revision_id,
            row["canonical_revision_id"],
        )
        self.assertEqual(
            BookRAGHandoffReadiness.READY,
            result.bookrag_handoff_readiness,
        )
        self.assertTrue(result.is_ready_for_bookrag_handoff)
        self.assertFalse(row["legacy_inspection_only"])
        inspection_rows = result.to_bookrag_pdf_list(
            image_material_policy=(
                BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
            )
        )
        self.assertTrue(inspection_rows[0]["legacy_inspection_only"])
        self.assertEqual(
            BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS.value,
            inspection_rows[0]["image_material_policy"],
        )
        rows[0]["text"] = "mutated copy"
        self.assertEqual("Hello RAG", result.to_bookrag_pdf_list()[0]["text"])

    def test_routed_factory_accepts_registry_or_adapters_but_not_hidden_defaults(self):
        parser_adapters = adapters()
        registry = ParserRegistry(parser_adapters)
        facade = RoutedBookRAGFacade.routed_default(registry=registry)
        self.assertIs(registry, facade.registry)
        self.assertEqual(
            balanced_pdf_routing_policy(), facade.routing_policy
        )

        with self.assertRaisesRegex(CompositionBindingError, "exactly one"):
            RoutedBookRAGFacade.routed_default()
        with self.assertRaisesRegex(CompositionBindingError, "exactly one"):
            RoutedBookRAGFacade.routed_default(
                registry=registry,
                adapters=parser_adapters,
            )
        with self.assertRaisesRegex(
            CompositionBindingError, PADDLEOCR_VL_ADAPTER
        ):
            RoutedBookRAGFacade.routed_default(adapters=parser_adapters[:2])

        with self.assertWarns(DeprecationWarning):
            legacy = RoutedBookRAGFacade.balanced(registry=registry)
        self.assertEqual(facade.routing_policy, legacy.routing_policy)

    def test_outer_request_page_and_region_bindings_fail_before_parser_calls(self):
        source = document()
        features = page_features(source)
        parser_adapters = adapters()
        facade = RoutedBookRAGFacade.routed_default(adapters=parser_adapters)
        different_feature = PageFeatures.from_page_ref(
            PageRef(
                source.document_version_id,
                0,
                width=612,
                height=792,
                native_stats=stats(native_character_count=999),
            ),
            source.mime_type,
        )
        with self.assertRaisesRegex(CompositionBindingError, "exactly match"):
            facade.run(request_for(source, features), (different_feature,))

        multipage_source = document(page_count=2)
        multipage_features = page_features(multipage_source)
        with self.assertRaisesRegex(CompositionBindingError, "exactly match"):
            facade.run(
                request_for(multipage_source, multipage_features),
                tuple(reversed(multipage_features)),
            )

        region_request = ParseRequest(
            document=source,
            pages=(features[0].page,),
            regions=(
                ParseRegion(
                    0,
                    SourceGeometry(
                        CoordinateSpace.NORMALIZED,
                        bbox=BoundingBox(0.1, 0.1, 0.5, 0.5),
                        page_width=1,
                        page_height=1,
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(CompositionBindingError, "region"):
            facade.run(region_request, features)
        self.assertTrue(all(not adapter.requests for adapter in parser_adapters))

    def test_run_pages_convenience_still_constructs_explicit_outer_request(self):
        source = document(page_count=2)
        features = page_features(source)
        facade = RoutedBookRAGFacade.routed_default(adapters=adapters())
        options = ParseOptions(languages=("en", "de"))

        result = facade.run_pages(
            source,
            features,
            options=options,
            run_id="run-convenience",
            trace_id="trace-convenience",
            idempotency_key="convenience-v1",
        )

        self.assertEqual(options, result.request.options)
        self.assertEqual("run-convenience", result.request.run_id)
        self.assertEqual("trace-convenience", result.request.trace_id)
        self.assertEqual(2, len(result.to_bookrag_pdf_list()))

    def test_failed_primary_and_fallback_block_handoff_but_keep_failed_audit(self):
        source = document()
        features = page_features(source)
        parser_adapters = adapters(docling_mode="failure", mineru_mode="failure")
        result = RoutedBookRAGFacade.routed_default(adapters=parser_adapters).run(
            request_for(source, features), features
        )

        self.assertEqual(RoutedParsingStatus.FAILED, result.parsing.status)
        self.assertEqual(2, len(result.parsing.invocations))
        self.assertEqual(
            BookRAGHandoffReadiness.BLOCKED_FAILED,
            result.bookrag_handoff_readiness,
        )
        self.assertFalse(result.is_ready_for_bookrag_handoff)
        with self.assertRaisesRegex(BookRAGHandoffNotReadyError, "not ready"):
            result.to_bookrag_pdf_list()
        self.assertEqual([], result.to_audit_partial_bookrag_pdf_list())

    def test_review_output_fails_closed_and_requires_named_audit_export(self):
        source = document()
        features = page_features(source)
        parser_adapters = adapters(
            docling_mode="low_quality",
            mineru_mode="low_quality",
        )

        result = RoutedBookRAGFacade.routed_default(adapters=parser_adapters).run(
            request_for(source, features), features
        )

        self.assertEqual(RoutedParsingStatus.PARTIAL, result.parsing.status)
        self.assertEqual(
            BookRAGHandoffReadiness.BLOCKED_REVIEW,
            result.bookrag_handoff_readiness,
        )
        self.assertEqual((features[0].page.page_id,), result.parsing.review_page_ids)
        self.assertNotIn("bookrag_rows", result.to_dict())
        with self.assertRaisesRegex(
            BookRAGHandoffNotReadyError,
            "to_audit_partial_bookrag_pdf_list",
        ):
            result.to_bookrag_pdf_list()
        audit_rows = result.to_audit_partial_bookrag_pdf_list()
        self.assertGreaterEqual(len(audit_rows), 1)
        self.assertEqual(source.content_sha256, audit_rows[0]["source_sha256"])

    def test_prior_blocks_reach_reconciliation_before_projection(self):
        source = document()
        features = page_features(source)
        parser_adapters = adapters()
        facade = RoutedBookRAGFacade.routed_default(adapters=parser_adapters)
        first = facade.run(request_for(source, features), features)
        prior = first.parsing.canonical_blocks[0]
        parser_adapters[0].text = "Changed content"

        second = facade.run(
            request_for(source, features),
            features,
            prior_blocks=(prior,),
        )
        current = second.parsing.canonical_blocks[0]

        self.assertEqual(prior.canonical_key, current.canonical_key)
        self.assertEqual(prior.revision + 1, current.revision)
        self.assertEqual(
            current.revision_id,
            second.to_bookrag_pdf_list()[0]["canonical_revision_id"],
        )


class RoutedBookRAGResultContractTests(unittest.TestCase):
    def test_result_json_is_deterministic_round_trippable_and_tamper_evident(self):
        source = document()
        features = page_features(source)
        request = request_for(source, features)
        first = RoutedBookRAGFacade.routed_default(adapters=adapters()).run(
            request, features
        )
        second = RoutedBookRAGFacade.routed_default(adapters=adapters()).run(
            request, features
        )

        self.assertEqual(first.projection_id, second.projection_id)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first, RoutedBookRAGResult.from_json(first.to_json()))

        tampered_id = first.to_dict()
        tampered_id["projection_id"] = "bookrag-projection_tampered"
        with self.assertRaisesRegex(ContractValidationError, "projection_id"):
            RoutedBookRAGResult.from_dict(tampered_id)

        tampered_readiness = first.to_dict()
        tampered_readiness["bookrag_handoff_readiness"] = "blocked_review"
        with self.assertRaisesRegex(ContractValidationError, "readiness mismatch"):
            RoutedBookRAGResult.from_dict(tampered_readiness)

        injected_rows = first.to_dict()
        injected_rows["bookrag_rows"] = []
        with self.assertRaisesRegex(ContractValidationError, "unknown field"):
            RoutedBookRAGResult.from_dict(injected_rows)

    def test_import_does_not_load_vendor_model_or_bookrag_core_modules(self):
        script = """
import json, sys
import docarag.composition
forbidden = ('Core', 'torch', 'mineru', 'paddle', 'paddleocr', 'docling', 'transformers', 'vllm')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("[]", completed.stdout.strip())


if __name__ == "__main__":
    unittest.main()
