from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest

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
from docarag.parsing_orchestrator import (
    InvocationRole,
    InvocationTrigger,
    OrchestrationContractError,
    OrchestrationReasonCode,
    PageParseQuality,
    PageParsingAudit,
    PageResolution,
    ParseInvocationAudit,
    ParsingOrchestrationPolicy,
    ParsingOrchestrator,
    RoutedParsingResult,
    RoutedParsingStatus,
)
from docarag.ports import ParserRegistry
from docarag.routing import (
    AdapterRoutingProfile,
    DeterministicPageRouter,
    PageClass,
    PageFeatures,
    PageRoutingPolicy,
    SecondaryRouteRole,
)


def document(page_count: int = 1) -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=hashlib.sha256(b"orchestration fixture").hexdigest(),
        mime_type="application/pdf",
        source_uri="file:///fixtures/orchestration.pdf",
        document_id="orchestration-fixture",
        display_name="orchestration.pdf",
        page_count=page_count,
    )


def core_stats(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "native_character_count": 500,
        "native_word_count": 80,
        "text_coverage_ratio": 0.45,
        "raster_image_coverage_ratio": 0.05,
        "replacement_character_ratio": 0.0,
        "control_character_ratio": 0.0,
        "duplicated_text_ratio": 0.01,
    }
    values.update(overrides)
    return values


def features_for(
    source: DocumentRef,
    *,
    stats: dict[str, object] | None = None,
) -> tuple[PageFeatures, ...]:
    return tuple(
        PageFeatures.from_page_ref(
            PageRef(
                source.document_version_id,
                index,
                width=612,
                height=792,
                native_stats=stats or core_stats(),
            ),
            source.mime_type,
        )
        for index in range(source.page_count or 0)
    )


def adapter_manifest(
    adapter_name: str,
    *,
    supports_batch: bool = False,
    max_pages_per_request: int | None = None,
) -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name=adapter_name,
        adapter_version="1.0.0",
        parser_name=adapter_name,
        parser_version="2.0.0",
        model_version="fixture-v1",
        supported_mime_types=("application/pdf",),
        modalities=(
            ParserModality.NATIVE_TEXT,
            ParserModality.OCR_TEXT,
            ParserModality.LAYOUT,
        ),
        block_types=(BlockType.PARAGRAPH,),
        accelerator=AcceleratorKind.CPU,
        supports_page_selection=True,
        supports_batch=supports_batch,
        max_pages_per_request=max_pages_per_request,
        deterministic=True,
    )


class FakeAdapter:
    def __init__(
        self,
        name: str,
        *,
        mode: str = "success",
        supports_batch: bool = False,
        max_pages_per_request: int | None = None,
        text: str = "source-bound fixture text",
    ) -> None:
        self._manifest = adapter_manifest(
            name,
            supports_batch=supports_batch,
            max_pages_per_request=max_pages_per_request,
        )
        self.mode = mode
        self.text = text
        self.requests = []

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request):
        self.requests.append(request)
        if self.mode == "raise":
            raise RuntimeError(f"{self._manifest.adapter_name} worker unavailable")
        if self.mode == "failure":
            return ParseResult(
                request_id=request.request_id,
                adapter_name=self._manifest.adapter_name,
                adapter_version=self._manifest.adapter_version,
                status=ParseStatus.FAILED,
                issues=(
                    ParseIssue(
                        ParseErrorCode.PARSER_FAILURE,
                        ErrorClassification.RETRYABLE,
                        IssueSeverity.ERROR,
                        "fixture parser failure",
                    ),
                ),
                duration_ms=7,
            )
        if self.mode == "missing":
            return ParseResult(
                request_id=request.request_id,
                adapter_name=self._manifest.adapter_name,
                adapter_version=self._manifest.adapter_version,
                status=ParseStatus.SUCCEEDED,
                duration_ms=7,
            )
        confidence = 0.10 if self.mode == "low" else 0.99
        observations = tuple(
            ParserBlockObservation(
                document_version_id=request.document.document_version_id,
                page_id=page.page_id,
                page_index=page.page_index,
                provenance=ParserProvenance(
                    parser_name=self._manifest.parser_name,
                    parser_version=self._manifest.parser_version,
                    model_version=self._manifest.model_version,
                    config_sha256="a" * 64,
                    parser_native_id=f"block-{page.page_index}",
                    raw_artifact_uri=(
                        f"artifact://{self._manifest.adapter_name}/"
                        f"page-{page.page_index}.json"
                    ),
                    transformations=("fake_adapter.v1",),
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
        result_request_id = (
            "parse_wrong_binding" if self.mode == "bad_binding" else request.request_id
        )
        return ParseResult(
            request_id=result_request_id,
            adapter_name=self._manifest.adapter_name,
            adapter_version=self._manifest.adapter_version,
            status=ParseStatus.SUCCEEDED,
            observations=observations,
            duration_ms=7,
        )


def routing_policy() -> PageRoutingPolicy:
    return PageRoutingPolicy(
        policy_name="orchestration-fixture",
        policy_version="1.0.0",
        adapter_profiles=(
            AdapterRoutingProfile(
                "docling",
                native_text_rank=0,
                technical_rank=0,
                visual_rank=1,
                ambiguous_rank=1,
                estimated_latency_ms=10,
                estimated_cost_microusd=1,
            ),
            AdapterRoutingProfile(
                "mineru",
                native_text_rank=1,
                technical_rank=1,
                visual_rank=0,
                ambiguous_rank=0,
                estimated_latency_ms=20,
                estimated_cost_microusd=2,
            ),
        ),
    )


def service(
    docling: FakeAdapter,
    mineru: FakeAdapter,
    *,
    policy: ParsingOrchestrationPolicy | None = None,
) -> ParsingOrchestrator:
    registry = ParserRegistry((docling, mineru))
    return ParsingOrchestrator(
        registry,
        DeterministicPageRouter(registry, routing_policy()),
        policy=policy,
    )


class ParsingOrchestratorExecutionTests(unittest.TestCase):
    def test_primary_success_does_not_eagerly_invoke_fallback(self):
        source = document()
        docling = FakeAdapter("docling")
        mineru = FakeAdapter("mineru")

        result = service(docling, mineru).parse(source, features_for(source))

        self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.status)
        self.assertEqual(1, len(docling.requests))
        self.assertEqual(0, len(mineru.requests))
        self.assertEqual(1, len(result.invocations))
        self.assertEqual(InvocationRole.PRIMARY, result.invocations[0].role)
        page = result.pages[0]
        self.assertEqual(PageParseQuality.ACCEPTED, page.primary_quality)
        self.assertIsNone(page.secondary_quality)
        self.assertEqual(PageResolution.RESOLVED, page.resolution)
        self.assertIn(
            OrchestrationReasonCode.SECONDARY_NOT_REQUIRED, page.reason_codes
        )
        self.assertEqual(1, len(result.canonical_blocks))

    def test_fallback_runs_only_for_failed_low_quality_or_missing_primary(self):
        cases = (
            ("failure", PageParseQuality.FAILED, InvocationTrigger.PRIMARY_FAILED),
            (
                "low",
                PageParseQuality.LOW_QUALITY,
                InvocationTrigger.PRIMARY_LOW_QUALITY,
            ),
            (
                "missing",
                PageParseQuality.MISSING_OUTPUT,
                InvocationTrigger.PRIMARY_MISSING_OUTPUT,
            ),
        )
        for mode, expected_quality, expected_trigger in cases:
            with self.subTest(mode=mode):
                source = document()
                docling = FakeAdapter("docling", mode=mode)
                mineru = FakeAdapter("mineru")

                result = service(docling, mineru).parse(
                    source, features_for(source)
                )

                self.assertEqual(1, len(docling.requests))
                self.assertEqual(1, len(mineru.requests))
                self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.status)
                self.assertEqual(expected_quality, result.pages[0].primary_quality)
                self.assertEqual(
                    PageParseQuality.ACCEPTED,
                    result.pages[0].secondary_quality,
                )
                fallback = result.invocations[1]
                self.assertEqual(InvocationRole.FALLBACK, fallback.role)
                self.assertEqual((expected_trigger,), fallback.triggers)

    def test_ambiguous_page_runs_adjudicator_and_fuses_both_sources(self):
        source = document()
        ambiguous = features_for(
            source,
            stats=core_stats(ocr_native_disagreement_ratio=0.90),
        )
        docling = FakeAdapter("docling")
        mineru = FakeAdapter("mineru")

        result = service(docling, mineru).parse(source, ambiguous)

        page = result.pages[0]
        self.assertEqual(PageClass.AMBIGUOUS, page.route.page_class)
        self.assertEqual(SecondaryRouteRole.ADJUDICATOR, page.route.secondary_role)
        self.assertEqual(1, len(docling.requests))
        self.assertEqual(1, len(mineru.requests))
        self.assertEqual(InvocationRole.ADJUDICATOR, result.invocations[1].role)
        self.assertEqual(
            (InvocationTrigger.AMBIGUOUS_ADJUDICATION,),
            result.invocations[1].triggers,
        )
        self.assertEqual(PageResolution.RESOLVED, page.resolution)
        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(
            2, len(result.canonical_blocks[0].source_observation_ids)
        )

    def test_batch_capability_is_bounded_by_manifest_and_split_deterministically(self):
        source = document(page_count=5)
        docling = FakeAdapter(
            "docling",
            supports_batch=True,
            max_pages_per_request=2,
        )
        mineru = FakeAdapter("mineru", supports_batch=True)
        orchestrator = service(
            docling,
            mineru,
            policy=ParsingOrchestrationPolicy(max_batch_pages=3),
        )

        result = orchestrator.parse(source, features_for(source))

        self.assertEqual([(0, 1), (2, 3), (4,)], [
            tuple(page.page_index for page in request.pages)
            for request in docling.requests
        ])
        self.assertEqual(0, len(mineru.requests))
        self.assertEqual(3, len(result.invocations))
        self.assertEqual([0, 1, 2], [item.sequence for item in result.invocations])
        self.assertTrue(
            all(len(item.request.pages) <= 2 for item in result.invocations)
        )
        self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.status)

    def test_sparse_routes_are_never_overparsed_as_one_vendor_page_range(self):
        source = document(page_count=5)
        docling = FakeAdapter("docling", supports_batch=True)
        mineru = FakeAdapter("mineru", supports_batch=True)
        orchestrator = service(
            docling,
            mineru,
            policy=ParsingOrchestrationPolicy(max_batch_pages=5),
        )
        features = tuple(
            PageFeatures.from_page_ref(
                PageRef(
                    source.document_version_id,
                    index,
                    width=612,
                    height=792,
                    native_stats=(
                        core_stats()
                        if index % 2 == 0
                        else core_stats(
                            native_character_count=0,
                            native_word_count=0,
                            text_coverage_ratio=0.0,
                            raster_image_coverage_ratio=0.95,
                            is_scanned_page=True,
                        )
                    ),
                ),
                source.mime_type,
            )
            for index in range(5)
        )

        result = orchestrator.parse(source, features)

        self.assertEqual(
            [(0,), (2,), (4,)],
            [
                tuple(page.page_index for page in request.pages)
                for request in docling.requests
            ],
        )
        self.assertEqual(
            [(1,), (3,)],
            [
                tuple(page.page_index for page in request.pages)
                for request in mineru.requests
            ],
        )
        self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.status)

    def test_all_parser_failures_return_unresolved_audit_instead_of_empty_success(self):
        source = document()
        docling = FakeAdapter("docling", mode="failure")
        mineru = FakeAdapter("mineru", mode="raise")

        result = service(docling, mineru).parse(source, features_for(source))

        self.assertEqual(RoutedParsingStatus.FAILED, result.status)
        self.assertIsNone(result.reconciliation)
        self.assertEqual((features_for(source)[0].page.page_id,), result.unresolved_page_ids)
        self.assertEqual((), result.review_page_ids)
        self.assertEqual(result.unresolved_page_ids, result.manual_review_page_ids)
        self.assertEqual(PageResolution.UNRESOLVED, result.pages[0].resolution)
        self.assertIn(
            OrchestrationReasonCode.NO_CANONICAL_OUTPUT,
            result.pages[0].reason_codes,
        )
        self.assertIsNone(result.invocations[1].result)
        self.assertEqual("RuntimeError", result.invocations[1].error_type)

    def test_low_quality_from_both_routes_returns_traceable_review_page(self):
        source = document()
        feature = features_for(source)[0]

        result = service(
            FakeAdapter("docling", mode="low"),
            FakeAdapter("mineru", mode="low"),
        ).parse(source, (feature,))

        self.assertEqual(RoutedParsingStatus.PARTIAL, result.status)
        self.assertEqual((), result.unresolved_page_ids)
        self.assertEqual((feature.page.page_id,), result.review_page_ids)
        self.assertEqual(PageResolution.NEEDS_REVIEW, result.pages[0].resolution)
        self.assertIn(
            OrchestrationReasonCode.CANONICAL_DEGRADED,
            result.pages[0].reason_codes,
        )
        self.assertTrue(result.pages[0].canonical_revision_ids)

    def test_registry_binding_violation_is_discarded_and_triggers_fallback(self):
        source = document()
        docling = FakeAdapter("docling", mode="bad_binding")
        mineru = FakeAdapter("mineru")

        result = service(docling, mineru).parse(source, features_for(source))

        self.assertIsNone(result.invocations[0].result)
        self.assertEqual(
            "ContractValidationError", result.invocations[0].error_type
        )
        self.assertEqual(PageParseQuality.FAILED, result.pages[0].primary_quality)
        self.assertEqual(
            PageParseQuality.ACCEPTED, result.pages[0].secondary_quality
        )
        self.assertEqual(RoutedParsingStatus.SUCCEEDED, result.status)

    def test_prior_canonical_revision_keeps_identity_and_increments_on_change(self):
        source = document()
        first = service(
            FakeAdapter("docling", text="first revision"),
            FakeAdapter("mineru"),
        ).parse(source, features_for(source))
        prior = first.canonical_blocks[0]

        second = service(
            FakeAdapter("docling", text="changed revision"),
            FakeAdapter("mineru"),
        ).parse(source, features_for(source), prior_blocks=(prior,))
        current = second.canonical_blocks[0]

        self.assertEqual(prior.canonical_key, current.canonical_key)
        self.assertEqual(prior.revision + 1, current.revision)
        self.assertEqual((prior.revision_id,), second.prior_revision_ids)

    def test_source_mismatch_fails_before_adapter_execution(self):
        source = document()
        another = DocumentRef.identified(
            content_sha256="b" * 64,
            mime_type="application/pdf",
            source_uri="file:///fixtures/other.pdf",
            document_id="other",
            page_count=1,
        )
        docling = FakeAdapter("docling")
        mineru = FakeAdapter("mineru")

        with self.assertRaisesRegex(
            OrchestrationContractError, "another document"
        ):
            service(docling, mineru).parse(source, features_for(another))
        self.assertEqual([], docling.requests)
        self.assertEqual([], mineru.requests)


class ParsingOrchestrationAuditContractTests(unittest.TestCase):
    def test_all_audits_round_trip_and_result_is_deterministic(self):
        source = document(page_count=2)
        first = service(
            FakeAdapter("docling", supports_batch=True),
            FakeAdapter("mineru"),
        ).parse(source, features_for(source))
        second = service(
            FakeAdapter("docling", supports_batch=True),
            FakeAdapter("mineru"),
        ).parse(source, features_for(source))

        self.assertEqual(first.result_id, second.result_id)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first, RoutedParsingResult.from_json(first.to_json()))
        self.assertEqual(
            first.orchestration_policy,
            ParsingOrchestrationPolicy.from_json(
                first.orchestration_policy.to_json()
            ),
        )
        self.assertEqual(
            first.invocations[0],
            ParseInvocationAudit.from_json(first.invocations[0].to_json()),
        )
        self.assertEqual(
            first.pages[0],
            PageParsingAudit.from_json(first.pages[0].to_json()),
        )

        tampered = first.to_dict()
        tampered["result_id"] = "routed-parsing-result_tampered"
        with self.assertRaisesRegex(ContractValidationError, "result_id"):
            RoutedParsingResult.from_dict(tampered)

    def test_import_does_not_load_vendor_model_or_bookrag_core_modules(self):
        script = """
import json, sys
import docarag.parsing_orchestrator
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
