from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest

from docarag.adapters.mineru_compat import MinerUNormalizedAdapter
from docarag.domain import (
    AcceleratorKind,
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    CanonicalBlock,
    ContractValidationError,
    CoordinateSpace,
    DocumentRef,
    ErrorClassification,
    IssueSeverity,
    MinerUEffort,
    PageRef,
    ParseErrorCode,
    ParseIssue,
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserArtifactAvailability,
    ParserArtifactReceipt,
    ParserBlockObservation,
    ParserCapabilityManifest,
    ParserExecutionError,
    ParserModality,
    ParserProvenance,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
    UnsupportedSchemaVersionError,
    stable_id,
)
from docarag.ports import ParserRegistry


def request_fixture() -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=hashlib.sha256(b"contract fixture").hexdigest(),
        mime_type="application/pdf",
        source_uri="file:///fixtures/contract.pdf",
        document_id="document-contract-fixture",
        display_name="contract.pdf",
        page_count=2,
        metadata={"tenant": "test"},
    )
    return ParseRequest(
        document=document,
        pages=(
            PageRef(document.document_version_id, 1, width=612, height=792),
            PageRef(document.document_version_id, 0, width=612, height=792),
        ),
        options=ParseOptions(languages=("en",), extra={"quality": "balanced"}),
        run_id="run-contract",
        trace_id="trace-contract",
        idempotency_key="fixture-v1",
    )


def fake_manifest() -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name="fake",
        adapter_version="1.2.3",
        parser_name="fake-parser",
        parser_version="9.8.7",
        model_version="fixture-1",
        supported_mime_types=("application/pdf",),
        modalities=(ParserModality.NATIVE_TEXT, ParserModality.LAYOUT),
        block_types=(BlockType.PARAGRAPH,),
        accelerator=AcceleratorKind.CPU,
        supports_page_selection=True,
        deterministic=True,
    )


def fake_observation(request: ParseRequest) -> ParserBlockObservation:
    page = request.pages[0]
    return ParserBlockObservation(
        document_version_id=request.document.document_version_id,
        page_id=page.page_id,
        page_index=page.page_index,
        provenance=ParserProvenance(
            parser_name="fake-parser",
            parser_version="9.8.7",
            model_version="fixture-1",
            config_sha256="a" * 64,
            parser_native_id="native-7",
            raw_artifact_uri="artifact://fixture/raw.json#block-7",
            transformations=("fake_to_observation.v1",),
        ),
        block_type=BlockType.PARAGRAPH,
        geometry=SourceGeometry(
            coordinate_space=CoordinateSpace.PAGE_POINTS,
            bbox=BoundingBox(10, 20, 300, 60),
            page_width=612,
            page_height=792,
        ),
        reading_order=2,
        representations=(
            Representation(
                kind=RepresentationKind.TEXT,
                content="Traceable fixture text",
                media_type="text/plain",
                language="en",
                confidence=0.99,
            ),
        ),
        assets=(
            AssetRef(
                kind=AssetKind.REGION_CROP,
                uri="artifact://fixture/page-2/block-7.png",
                media_type="image/png",
                sha256="b" * 64,
            ),
        ),
        confidence=0.99,
    )


class FakeAdapter:
    def manifest(self) -> ParserCapabilityManifest:
        return fake_manifest()

    def parse(self, request: ParseRequest) -> ParseResult:
        return ParseResult(
            request_id=request.request_id,
            adapter_name="fake",
            adapter_version="1.2.3",
            status=ParseStatus.SUCCEEDED,
            observations=(fake_observation(request),),
            metrics={"fixture": True},
        )


class ParserDomainContractTests(unittest.TestCase):
    def test_result_scoped_parser_artifacts_are_bound_and_hash_neutral_when_empty(self):
        request = request_fixture()
        emitted = ParserArtifactReceipt(
            artifact_role="parser_middle_json",
            availability=ParserArtifactAvailability.EMITTED,
            artifact_schema="vendor.middle-json.v1",
            producer_task_id="task-1",
            document_version_id=request.document.document_version_id,
            source_sha256=request.document.content_sha256,
            uri="cas://sha256/" + "d" * 64,
            sha256="d" * 64,
            size_bytes=17,
            media_type="application/json",
            metadata={"backend": "fixture"},
        )
        result = ParseResult(
            request_id=request.request_id,
            adapter_name="fake",
            adapter_version="1.2.3",
            status=ParseStatus.SUCCEEDED,
            artifacts=(emitted,),
        )

        self.assertEqual(result, ParseResult.from_json(result.to_json()))
        self.assertIn("artifacts", result.to_dict())
        without = ParseResult(
            request_id=request.request_id,
            adapter_name="fake",
            adapter_version="1.2.3",
            status=ParseStatus.SUCCEEDED,
        )
        self.assertNotIn("artifacts", without.to_dict())
        self.assertNotEqual(without.result_sha256, result.result_sha256)
        with self.assertRaises(TypeError):
            result.artifacts[0].metadata["backend"] = "changed"
        with self.assertRaisesRegex(ContractValidationError, "not-emitted"):
            ParserArtifactReceipt(
                artifact_role="parser_middle_json",
                availability=ParserArtifactAvailability.NOT_EMITTED,
                artifact_schema="vendor.middle-json.v1",
                producer_task_id="task-2",
                document_version_id=request.document.document_version_id,
                source_sha256=request.document.content_sha256,
                uri="cas://unexpected",
            )

    def test_result_rejects_duplicate_semantic_parser_artifact_receipts(self):
        request = request_fixture()
        emitted = ParserArtifactReceipt(
            artifact_role="parser_middle_json",
            availability=ParserArtifactAvailability.EMITTED,
            artifact_schema="vendor.middle-json.v1",
            producer_task_id="task-1",
            document_version_id=request.document.document_version_id,
            source_sha256=request.document.content_sha256,
            uri="cas://sha256/" + "d" * 64,
            sha256="d" * 64,
            size_bytes=17,
            media_type="application/json",
        )
        conflicts = (
            ParserArtifactReceipt(
                artifact_role="parser_middle_json",
                availability=ParserArtifactAvailability.NOT_EMITTED,
                artifact_schema="vendor.middle-json.v1",
                producer_task_id="task-1",
                document_version_id=request.document.document_version_id,
                source_sha256=request.document.content_sha256,
                media_type="application/json",
            ),
            ParserArtifactReceipt(
                artifact_role="parser_middle_json",
                availability=ParserArtifactAvailability.EMITTED,
                artifact_schema="vendor.middle-json.v1",
                producer_task_id="task-1",
                document_version_id=request.document.document_version_id,
                source_sha256=request.document.content_sha256,
                uri="cas://sha256/" + "e" * 64,
                sha256="e" * 64,
                size_bytes=19,
                media_type="application/json",
            ),
        )

        for conflict in conflicts:
            with self.subTest(availability=conflict.availability.value), self.assertRaisesRegex(
                ContractValidationError,
                "duplicate semantic parser artifact receipts",
            ):
                ParseResult(
                    request_id="request-1",
                    adapter_name="fake",
                    adapter_version="1.2.3",
                    status=ParseStatus.SUCCEEDED,
                    artifacts=(emitted, conflict),
                )

    def test_artifact_key_allows_distinct_native_artifacts_but_not_contradictions(
        self,
    ):
        request = request_fixture()

        def receipt(key: str, availability=ParserArtifactAvailability.EMITTED):
            emitted = availability is ParserArtifactAvailability.EMITTED
            return ParserArtifactReceipt(
                artifact_role="parser_page_result",
                availability=availability,
                artifact_schema="vendor.page-result.v1",
                producer_task_id="task-1",
                document_version_id=request.document.document_version_id,
                source_sha256=request.document.content_sha256,
                artifact_key=key,
                uri=("cas://sha256/" + ("d" if key == "page-0.json" else "e") * 64)
                if emitted
                else None,
                sha256=(("d" if key == "page-0.json" else "e") * 64)
                if emitted
                else None,
                size_bytes=17 if emitted else None,
                media_type="application/json",
            )

        first = receipt("page-0.json")
        second = receipt("page-1.json")
        result = ParseResult(
            request_id=request.request_id,
            adapter_name="fake",
            adapter_version="1.2.3",
            status=ParseStatus.SUCCEEDED,
            artifacts=(first, second),
        )
        self.assertEqual(
            {"page-0.json", "page-1.json"},
            {item.artifact_key for item in result.artifacts},
        )

        with self.assertRaisesRegex(
            ContractValidationError,
            "duplicate semantic parser artifact receipts",
        ):
            ParseResult(
                request_id=request.request_id,
                adapter_name="fake",
                adapter_version="1.2.3",
                status=ParseStatus.SUCCEEDED,
                artifacts=(
                    first,
                    receipt(
                        "page-0.json",
                        ParserArtifactAvailability.NOT_EMITTED,
                    ),
                ),
            )
        with self.assertRaisesRegex(ContractValidationError, "artifact_key"):
            receipt("x" * 1025)

        failure = ParserExecutionError(
            ParseIssue(
                ParseErrorCode.PARSER_FAILURE,
                ErrorClassification.RETRYABLE,
                IssueSeverity.ERROR,
                "later task failed",
            ),
            artifacts=(first,),
        )
        self.assertEqual((first,), failure.artifacts)

    def test_keyless_pre_release_receipt_identity_is_strictly_rejected(self):
        request = request_fixture()
        receipt = ParserArtifactReceipt(
            artifact_role="parser_middle_json",
            availability=ParserArtifactAvailability.EMITTED,
            artifact_schema="vendor.middle-json.v1",
            producer_task_id="task-1",
            document_version_id=request.document.document_version_id,
            source_sha256=request.document.content_sha256,
            uri="cas://sha256/" + "d" * 64,
            sha256="d" * 64,
            size_bytes=17,
            media_type="application/json",
            metadata={"backend": "fixture"},
        )
        keyless = receipt.to_dict()
        del keyless["artifact_key"]
        keyless["receipt_id"] = stable_id(
            "parser-artifact",
            receipt.artifact_role,
            receipt.availability.value,
            receipt.artifact_schema,
            receipt.producer_task_id,
            receipt.document_version_id,
            receipt.source_sha256,
            receipt.uri,
            receipt.sha256,
            receipt.size_bytes,
            receipt.media_type,
            receipt.metadata,
        )

        with self.assertRaisesRegex(
            ContractValidationError,
            "receipt_id mismatch",
        ):
            ParserArtifactReceipt.from_dict(keyless)

    def test_mineru_effort_is_typed_and_changes_request_identity(self):
        default = request_fixture()
        medium_options = ParseOptions(languages=("en",), mineru_effort="medium")
        medium = ParseRequest(
            document=default.document,
            pages=default.pages,
            options=medium_options,
            run_id=default.run_id,
            trace_id=default.trace_id,
            idempotency_key=default.idempotency_key,
        )

        self.assertEqual(MinerUEffort.MEDIUM, medium.options.mineru_effort)
        self.assertNotEqual(default.request_id, medium.request_id)
        self.assertEqual(medium, ParseRequest.from_json(medium.to_json()))
        with self.assertRaisesRegex(ContractValidationError, "mineru_effort"):
            ParseOptions(mineru_effort="maximum")  # type: ignore[arg-type]

    def test_request_result_observation_and_canonical_block_round_trip(self):
        request = request_fixture()
        result = FakeAdapter().parse(request)
        observation = result.observations[0]
        canonical = CanonicalBlock.from_observation(observation)

        self.assertEqual(request, ParseRequest.from_json(request.to_json()))
        self.assertEqual(result, ParseResult.from_json(result.to_json()))
        self.assertEqual(
            observation, ParserBlockObservation.from_json(observation.to_json())
        )
        self.assertEqual(canonical, CanonicalBlock.from_json(canonical.to_json()))
        self.assertEqual(QualityStatus.ACCEPTED, canonical.quality_status)
        self.assertEqual((observation.observation_id,), canonical.source_observation_ids)

        nested_dtos = (
            request.document,
            request.pages[0],
            request.options,
            observation.geometry,
            observation.provenance,
            observation.representations[0],
            observation.assets[0],
            fake_manifest(),
            ParseIssue(
                ParseErrorCode.TIMEOUT,
                ErrorClassification.RETRYABLE,
                IssueSeverity.WARNING,
                "fixture warning",
            ),
        )
        for dto in nested_dtos:
            with self.subTest(dto=type(dto).__name__):
                self.assertEqual(dto, type(dto).from_json(dto.to_json()))

    def test_stable_ids_and_hashes_do_not_depend_on_mapping_order(self):
        left = Representation(
            RepresentationKind.STRUCTURED_DATA,
            {"b": [2, 3], "a": 1},
            "application/json",
        )
        right = Representation(
            RepresentationKind.STRUCTURED_DATA,
            {"a": 1, "b": [2, 3]},
            "application/json",
        )
        self.assertEqual(left.representation_id, right.representation_id)
        self.assertEqual(left.content_sha256, right.content_sha256)
        first = request_fixture()
        second = request_fixture()
        self.assertEqual(first.request_id, second.request_id)

        observation = fake_observation(first)
        revision_one = CanonicalBlock.from_observation(observation, revision=1)
        revision_two = CanonicalBlock.from_observation(observation, revision=2)
        self.assertEqual(revision_one.block_id, revision_two.block_id)
        self.assertEqual(revision_one.content_sha256, revision_two.content_sha256)
        self.assertNotEqual(revision_one.revision_id, revision_two.revision_id)

        fast = ParseResult(
            first.request_id,
            "fake",
            "1.2.3",
            ParseStatus.SUCCEEDED,
            observations=(observation,),
            duration_ms=1,
            metrics={"gpu_peak_mib": 1},
        )
        slow = ParseResult(
            first.request_id,
            "fake",
            "1.2.3",
            ParseStatus.SUCCEEDED,
            observations=(observation,),
            duration_ms=500,
            metrics={"gpu_peak_mib": 200},
        )
        self.assertEqual(fast.result_sha256, slow.result_sha256)

    def test_deserialization_rejects_wrong_schema_and_tampered_hash(self):
        request = request_fixture()
        wrong_schema = request.to_dict()
        wrong_schema["schema_version"] = "docarag.parser-contract.v999"
        with self.assertRaises(UnsupportedSchemaVersionError):
            ParseRequest.from_dict(wrong_schema)

        observation = fake_observation(request)
        tampered = observation.to_dict()
        tampered["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractValidationError, "content_sha256"):
            ParserBlockObservation.from_dict(tampered)

    def test_geometry_rejects_ambiguous_or_out_of_bounds_coordinates(self):
        with self.assertRaisesRegex(ContractValidationError, "bbox or polygon"):
            SourceGeometry(CoordinateSpace.PAGE_PIXELS)
        with self.assertRaisesRegex(ContractValidationError, "outside"):
            SourceGeometry(
                CoordinateSpace.PAGE_POINTS,
                bbox=BoundingBox(0, 0, 700, 10),
                page_width=612,
                page_height=792,
            )


class ParserRegistryConformanceTests(unittest.TestCase):
    def test_fake_adapter_registers_discovers_and_parses(self):
        request = request_fixture()
        registry = ParserRegistry((FakeAdapter(),))

        self.assertEqual(("fake",), tuple(m.adapter_name for m in registry.manifests()))
        matches = registry.matching(
            mime_type="application/pdf",
            modalities=frozenset({ParserModality.NATIVE_TEXT}),
            block_types=frozenset({BlockType.PARAGRAPH}),
        )
        self.assertEqual(1, len(matches))
        result = registry.parse("fake", request)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual(request.request_id, result.request_id)

    def test_registry_rejects_result_with_wrong_adapter_binding(self):
        class BadAdapter(FakeAdapter):
            def parse(self, request: ParseRequest) -> ParseResult:
                return ParseResult(
                    request_id=request.request_id,
                    adapter_name="not-fake",
                    adapter_version="1.2.3",
                    status=ParseStatus.SUCCEEDED,
                )

        with self.assertRaisesRegex(ContractValidationError, "name differs"):
            ParserRegistry((BadAdapter(),)).parse("fake", request_fixture())

    def test_registry_rejects_artifact_receipts_for_another_source(self):
        request = request_fixture()

        class ArtifactAdapter(FakeAdapter):
            def __init__(self, *, document_version_id: str, source_sha256: str):
                self.document_version_id = document_version_id
                self.source_sha256 = source_sha256

            def parse(self, received: ParseRequest) -> ParseResult:
                receipt = ParserArtifactReceipt(
                    artifact_role="raw_result",
                    availability=ParserArtifactAvailability.EMITTED,
                    artifact_schema="vendor.raw-result.v1",
                    producer_task_id="task-1",
                    document_version_id=self.document_version_id,
                    source_sha256=self.source_sha256,
                    artifact_key="result.json",
                    uri="cas://sha256/" + "d" * 64,
                    sha256="d" * 64,
                    size_bytes=17,
                    media_type="application/json",
                )
                return ParseResult(
                    request_id=received.request_id,
                    adapter_name="fake",
                    adapter_version="1.2.3",
                    status=ParseStatus.SUCCEEDED,
                    artifacts=(receipt,),
                )

        cases = (
            (
                "another-document-version",
                request.document.content_sha256,
                "different document version",
            ),
            (
                request.document.document_version_id,
                "f" * 64,
                "different source digest",
            ),
        )
        for document_version_id, source_sha256, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ContractValidationError,
                message,
            ):
                ParserRegistry(
                    (
                        ArtifactAdapter(
                            document_version_id=document_version_id,
                            source_sha256=source_sha256,
                        ),
                    )
                ).parse("fake", request)


class MinerUNormalizedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.request = request_fixture()
        self.blocks = [
            {
                "type": "text",
                "text": "A stable title",
                "text_level": 1,
                "page_idx": 0,
                "middle_json": {
                    "type": "title",
                    "bbox": [10, 20, 210, 50],
                    "index": 0,
                    "lines": [{"spans": [{"score": 0.98}]}],
                },
            },
            {
                "type": "table",
                "table_body": "<table><tr><td>A</td></tr></table>",
                "table_caption": ["Table 1"],
                "table_footnote": ["Source: fixture"],
                "img_path": "/artifacts/table.png",
                "page_idx": 0,
                "middle_json": {
                    "type": "table",
                    "bbox": [10, 80, 400, 250],
                    "index": 2,
                    "blocks": [{"lines": [{"spans": [{"score": 0.91}]}]}],
                },
            },
            {
                "type": "equation",
                "text": "$$E=mc^2$$",
                "text_format": "latex",
                "img_path": "/artifacts/equation.jpg",
                "page_idx": 1,
                "middle_json": {
                    "type": "interline_equation",
                    "bbox": [40, 300, 180, 340],
                    "index": 4.5,
                    "lines": [{"spans": [{"score": 0.88}]}],
                },
            },
            {
                "type": "image",
                "image_caption": [],
                "image_footnote": [],
                "img_path": "/artifacts/figure.webp",
                "page_idx": 1,
                "middle_json": {
                    "type": "image",
                    "bbox": [200, 300, 500, 600],
                    "index": 5,
                },
            },
        ]

    def adapter(self, blocks=None) -> MinerUNormalizedAdapter:
        values = self.blocks if blocks is None else blocks
        return MinerUNormalizedAdapter(
            lambda _request: values,
            parser_version="2.1.0-compat",
            model_version="pipeline-model-fixture",
            config={"backend": "pipeline", "formula": True, "table": True},
            raw_artifact_uri="artifact://run/mineru/merged_content.json",
        )

    def test_translates_current_bookrag_types_and_preserves_multimodal_evidence(self):
        adapter = self.adapter()
        manifest = adapter.manifest()
        result = ParserRegistry((adapter,)).parse(
            "mineru-normalized", self.request
        )
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual(4, len(result.observations))
        self.assertEqual("1.1.0", result.adapter_version)
        self.assertEqual("mineru", manifest.parser_name)
        self.assertEqual("2.1.0-compat", manifest.parser_version)

        by_type = {item.block_type: item for item in result.observations}
        self.assertEqual(
            RepresentationKind.TEXT, by_type[BlockType.TITLE].representations[0].kind
        )
        table_kinds = {item.kind for item in by_type[BlockType.TABLE].representations}
        self.assertEqual(
            {RepresentationKind.HTML, RepresentationKind.CAPTION, RepresentationKind.FOOTNOTE},
            table_kinds,
        )
        self.assertEqual(AssetKind.REGION_CROP, by_type[BlockType.TABLE].assets[0].kind)
        self.assertEqual(
            RepresentationKind.LATEX,
            by_type[BlockType.EQUATION].representations[0].kind,
        )
        self.assertTrue(by_type[BlockType.EQUATION].assets)
        self.assertEqual("figure", by_type[BlockType.FIGURE].semantic_subtype)
        self.assertEqual(CoordinateSpace.PARSER_PAGE, by_type[BlockType.FIGURE].geometry.coordinate_space)
        for observation in result.observations:
            self.assertEqual("mineru", observation.provenance.parser_name)
            self.assertEqual("2.1.0-compat", observation.provenance.parser_version)
            self.assertEqual(
                "artifact://run/mineru/merged_content.json",
                observation.provenance.raw_artifact_uri,
            )

    def test_observation_ids_and_hashes_survive_distinct_block_reordering(self):
        forward = self.adapter().parse(self.request)
        reverse = self.adapter(list(reversed(self.blocks))).parse(self.request)
        forward_identity = {
            (item.observation_id, item.content_sha256) for item in forward.observations
        }
        reverse_identity = {
            (item.observation_id, item.content_sha256) for item in reverse.observations
        }
        self.assertEqual(forward_identity, reverse_identity)

    def test_invalid_block_returns_typed_partial_result_without_fake_content(self):
        bad = {"type": "text", "text": "wrong page", "page_idx": 99, "middle_json": {}}
        result = self.adapter(self.blocks[:1] + [bad]).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertEqual(1, len(result.observations))
        self.assertEqual(1, len(result.issues))
        issue = result.issues[0]
        self.assertEqual(ParseErrorCode.INVALID_NORMALIZED_BLOCK, issue.code)
        self.assertEqual(ErrorClassification.PERMANENT, issue.classification)
        self.assertEqual(IssueSeverity.ERROR, issue.severity)

    def test_loader_failure_is_retryable_typed_failure(self):
        def fail(_request):
            raise RuntimeError("worker disconnected")

        adapter = MinerUNormalizedAdapter(
            fail,
            parser_version="2.1.0-compat",
            model_version="pipeline-model-fixture",
            config={"backend": "pipeline"},
        )
        result = adapter.parse(self.request)
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertFalse(result.observations)
        self.assertEqual(ErrorClassification.RETRYABLE, result.issues[0].classification)

    def test_loader_can_preserve_permanent_service_input_failure(self):
        issue = ParseIssue(
            ParseErrorCode.INVALID_REQUEST,
            ErrorClassification.PERMANENT,
            IssueSeverity.ERROR,
            "MinerU effort does not match the worker",
        )

        def fail(_request):
            raise ParserExecutionError(issue)

        result = MinerUNormalizedAdapter(
            fail,
            parser_version="3.4.4",
            model_version="mineru-fixture",
            config={"backend": "hybrid-engine", "effort": "medium"},
        ).parse(self.request)
        self.assertEqual((issue,), result.issues)

    def test_contract_imports_do_not_load_ml_or_vendor_packages(self):
        script = """
import json, sys
import docarag.domain, docarag.ports, docarag.adapters.mineru_compat
forbidden = ('torch', 'mineru', 'paddle', 'paddleocr', 'docling', 'transformers', 'vllm')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual([], json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
