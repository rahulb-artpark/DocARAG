from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from docarag.domain import (
    BlockType,
    DocumentRef,
    ErrorClassification,
    PageRef,
    ParseErrorCode,
    ParseOptions,
    ParseRequest,
    ParseStatus,
    ParserArtifactAvailability,
    ParserExecutionError,
)
from docarag.ports import ParserRegistry
from docarag.services.paddleocr_vl import (
    OFFICIAL_LAYOUT_PARSING_PATH,
    PADDLE_MARKDOWN_IMAGE_ARTIFACT_SCHEMA,
    PADDLE_RAW_PAGE_ARTIFACT_SCHEMA,
    PaddleOCRVLArtifactRole,
    PaddleOCRVLCompletePipelineLoader,
    PaddleOCRVLEndpointKind,
    PaddleOCRVLHTTPResponse,
    PaddleOCRVLPageMaterial,
    PaddleOCRVLRemoteServiceError,
    PaddleOCRVLServiceConfig,
    PaddleOCRVLServiceContractError,
    PaddleOCRVLServiceLimitError,
    PaddleOCRVLServiceTimeoutError,
    RawVLMEndpointRejectedError,
    StoredPaddleOCRVLArtifact,
    build_paddleocr_vl_service_adapter,
)
from docarag.source_material import VerifiedSourceMaterial


PDF_BYTES = b"%PDF-1.7\nverified immutable snapshot\n%%EOF\n"
PDF_SHA256 = hashlib.sha256(PDF_BYTES).hexdigest()
PAGE_BYTES = b"\x89PNG\r\n\x1a\nrendered-page"
PAGE_SHA256 = hashlib.sha256(PAGE_BYTES).hexdigest()
LOG_ID = "b8efaf43-a527-4d66-bfc2-629eb701aa17"
IMAGE_NAME = "imgs/img_in_image_box_300_100_700_500.jpg"
IMAGE_BYTES = b"fixture image crop"


def request_fixture(
    *,
    page_count: int = 1,
    pages: tuple[int, ...] = (0,),
    parse_images: bool = True,
) -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=PDF_SHA256,
        mime_type="application/pdf",
        source_uri="s3://citation-only/private/report.pdf",
        document_id="service-pdf",
        display_name="report.pdf",
        page_count=page_count,
    )
    return ParseRequest(
        document=document,
        pages=tuple(
            PageRef(document.document_version_id, index, width=1000, height=800)
            for index in pages
        ),
        options=ParseOptions(
            languages=("en",),
            parse_images=parse_images,
            timeout_seconds=30,
        ),
        run_id="run-service",
        trace_id="trace-service",
        idempotency_key="idempotent-service-call",
    )


def source_fixture(request: ParseRequest) -> VerifiedSourceMaterial:
    return VerifiedSourceMaterial(
        document_version_id=request.document.document_version_id,
        logical_source_uri=request.document.source_uri,
        material_uri="file:///immutable/report.pdf",
        media_type="application/pdf",
        sha256=PDF_SHA256,
        size_bytes=len(PDF_BYTES),
        path=Path("/immutable/report.pdf"),
        data=PDF_BYTES,
    )


def page_fixture(request: ParseRequest, page: PageRef) -> PaddleOCRVLPageMaterial:
    return PaddleOCRVLPageMaterial(
        document_version_id=request.document.document_version_id,
        source_sha256=request.document.content_sha256,
        page_index=page.page_index,
        media_type="image/png",
        width=1000,
        height=800,
        data=PAGE_BYTES,
        image_sha256=PAGE_SHA256,
        render_identity="fixture-renderer.v1",
    )


def service_page_fixture() -> dict[str, object]:
    text = {
        "block_label": "text",
        "block_content": "Verified OCR text",
        "block_bbox": [20, 30, 280, 80],
        "block_polygon_points": [[20, 30], [280, 30], [280, 80], [20, 80]],
        "block_id": 1,
        "block_order": 1,
        "group_id": 1,
    }
    image = {
        "block_label": "image",
        "block_content": "",
        "block_bbox": [300, 100, 700, 500],
        "block_polygon_points": [
            [300, 100],
            [700, 100],
            [700, 500],
            [300, 500],
        ],
        "block_id": 2,
        "block_order": 2,
        "group_id": 2,
    }
    return {
        "prunedResult": {
            "model_settings": {
                "use_doc_preprocessor": True,
                "use_layout_detection": True,
                "use_chart_recognition": True,
            },
            "layout_det_res": {
                "boxes": [
                    {
                        "label": "text",
                        "score": 0.98,
                        "coordinate": text["block_bbox"],
                    },
                    {
                        "label": "image",
                        "score": 0.91,
                        "coordinate": image["block_bbox"],
                    },
                ]
            },
            "parsing_res_list": [text, image],
        },
        "markdown": {
            "text": "Verified OCR text\n\n![](imgs/img_in_image_box_300_100_700_500.jpg)",
            "images": {
                IMAGE_NAME: base64.b64encode(IMAGE_BYTES).decode("ascii")
            },
        },
        "outputImages": {},
        "inputImage": None,
        "exports": None,
    }


def response_fixture(*, page: object | None = None) -> PaddleOCRVLHTTPResponse:
    payload = {
        "logId": LOG_ID,
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": [
                service_page_fixture() if page is None else page
            ],
            "dataInfo": {"fileType": 1},
        },
    }
    return PaddleOCRVLHTTPResponse(
        status=200,
        headers={"content-type": "application/json; charset=utf-8"},
        body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def config_fixture(**overrides: object) -> PaddleOCRVLServiceConfig:
    values: dict[str, object] = {
        "endpoint_kind": PaddleOCRVLEndpointKind.COMPLETE_PIPELINE,
        "base_url": "http://127.0.0.1:8080",
        "parser_version": "paddleocr-unqualified-v1.6-worker",
        "max_pages_per_request": 4,
        "max_source_bytes": 1024 * 1024,
        "max_page_bytes": 1024 * 1024,
        "max_total_page_bytes": 2 * 1024 * 1024,
        "max_request_bytes": 2 * 1024 * 1024,
        "max_response_bytes": 2 * 1024 * 1024,
        "max_artifact_bytes": 1024 * 1024,
        "max_total_artifact_bytes": 2 * 1024 * 1024,
    }
    values.update(overrides)
    return PaddleOCRVLServiceConfig(**values)  # type: ignore[arg-type]


class FixtureRuntime:
    def __init__(self, request: ParseRequest) -> None:
        self.request = request
        self.requests = []
        self.artifacts = []
        self.response = response_fixture()

    def source_loader(self, request: ParseRequest) -> VerifiedSourceMaterial:
        self.assert_same_request(request)
        return source_fixture(request)

    def page_renderer(
        self, source: VerifiedSourceMaterial, page: PageRef
    ) -> PaddleOCRVLPageMaterial:
        if source.sha256 != PDF_SHA256:
            raise AssertionError("wrong source")
        return page_fixture(self.request, page)

    def sink(self, artifact):
        self.artifacts.append(artifact)
        return StoredPaddleOCRVLArtifact(
            uri=(
                f"artifact://paddle/{artifact.document_version_id}/"
                f"{artifact.page_index}/{artifact.role.value}/{artifact.name}"
            ),
            sha256=artifact.sha256,
            media_type=artifact.media_type,
        )

    def exchange(self, request):
        self.requests.append(request)
        return self.response

    def assert_same_request(self, request: ParseRequest) -> None:
        if request.request_id != self.request.request_id:
            raise AssertionError("wrong request")

    def adapter(self, *, config: PaddleOCRVLServiceConfig | None = None):
        return build_paddleocr_vl_service_adapter(
            config or config_fixture(),
            source_loader=self.source_loader,
            page_renderer=self.page_renderer,
            artifact_sink=self.sink,
            http_exchange=self.exchange,
        )

    def loader(self, *, config: PaddleOCRVLServiceConfig | None = None):
        return PaddleOCRVLCompletePipelineLoader(
            config or config_fixture(),
            source_loader=self.source_loader,
            page_renderer=self.page_renderer,
            artifact_sink=self.sink,
            http_exchange=self.exchange,
        )


class PaddleOCRVLServiceHappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = request_fixture()
        self.runtime = FixtureRuntime(self.request)

    def test_complete_service_result_translates_through_existing_adapter(self):
        adapter = self.runtime.adapter()
        result = ParserRegistry((adapter,)).parse(adapter.manifest().adapter_name, self.request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual({BlockType.PARAGRAPH, BlockType.FIGURE}, {o.block_type for o in result.observations})
        figure = next(o for o in result.observations if o.block_type is BlockType.FIGURE)
        self.assertEqual(hashlib.sha256(IMAGE_BYTES).hexdigest(), figure.assets[0].sha256)
        self.assertTrue(figure.provenance.raw_artifact_uri.startswith("artifact://paddle/"))
        self.assertEqual(2, len(self.runtime.artifacts))
        self.assertEqual(
            {PaddleOCRVLArtifactRole.MARKDOWN_IMAGE, PaddleOCRVLArtifactRole.RAW_PAGE_RESULT},
            {artifact.role for artifact in self.runtime.artifacts},
        )
        raw = next(
            artifact
            for artifact in self.runtime.artifacts
            if artifact.role is PaddleOCRVLArtifactRole.RAW_PAGE_RESULT
        )
        self.assertEqual(self.runtime.response.body, raw.data)
        self.assertEqual(2, len(result.artifacts))
        receipt = next(
            item for item in result.artifacts
            if item.artifact_role == "paddleocr_vl_raw_page_result"
        )
        self.assertEqual(
            "paddleocr_vl_raw_page_result", receipt.artifact_role
        )
        self.assertEqual(ParserArtifactAvailability.EMITTED, receipt.availability)
        self.assertEqual(PADDLE_RAW_PAGE_ARTIFACT_SCHEMA, receipt.artifact_schema)
        self.assertEqual(raw.sha256, receipt.sha256)
        self.assertEqual(len(raw.data), receipt.size_bytes)
        self.assertEqual(
            figure.provenance.raw_artifact_uri,
            receipt.uri,
        )
        self.assertTrue(receipt.metadata["exact_http_response_body"])
        crop_receipt = next(
            item for item in result.artifacts
            if item.artifact_role == "paddleocr_vl_markdown_image"
        )
        self.assertEqual(
            PADDLE_MARKDOWN_IMAGE_ARTIFACT_SCHEMA,
            crop_receipt.artifact_schema,
        )
        self.assertEqual(IMAGE_NAME, crop_receipt.artifact_key)
        self.assertEqual(
            hashlib.sha256(IMAGE_BYTES).hexdigest(),
            crop_receipt.sha256,
        )
        self.assertEqual(len(IMAGE_BYTES), crop_receipt.size_bytes)
        retained = json.loads(raw.data)
        self.assertEqual(
            "Verified OCR text\n\n![](imgs/img_in_image_box_300_100_700_500.jpg)",
            retained["result"]["layoutParsingResults"][0]["markdown"]["text"],
        )

    def test_table_and_formula_markdown_crops_attach_to_their_blocks(self):
        page = service_page_fixture()
        table = {
            "block_label": "table",
            "block_content": "<table><tr><td>A</td></tr></table>",
            "content_format": "html",
            "block_bbox": [10, 100, 500, 300],
            "block_id": 3,
            "block_order": 3,
            "group_id": 3,
        }
        formula = {
            "block_label": "formula",
            "block_content": "E=mc^2",
            "content_format": "latex",
            "block_bbox": [100, 400, 400, 460],
            "block_id": 4,
            "block_order": 4,
            "group_id": 4,
        }
        page["prunedResult"]["parsing_res_list"].extend((table, formula))
        page["prunedResult"]["layout_det_res"]["boxes"].extend(
            (
                {"label": "table", "score": 0.9, "coordinate": table["block_bbox"]},
                {
                    "label": "formula",
                    "score": 0.89,
                    "coordinate": formula["block_bbox"],
                },
            )
        )
        table_bytes = b"verified table crop"
        formula_bytes = b"verified formula crop"
        page["markdown"]["images"].update(
            {
                "imgs/img_in_table_box_10_100_500_300.jpg": base64.b64encode(
                    table_bytes
                ).decode("ascii"),
                "imgs/img_in_formula_box_100_400_400_460.jpg": base64.b64encode(
                    formula_bytes
                ).decode("ascii"),
            }
        )
        self.runtime.response = response_fixture(page=page)

        result = self.runtime.adapter().parse(self.request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        by_type = {item.block_type: item for item in result.observations}
        self.assertEqual(
            hashlib.sha256(table_bytes).hexdigest(),
            by_type[BlockType.TABLE].assets[0].sha256,
        )
        self.assertEqual(
            hashlib.sha256(formula_bytes).hexdigest(),
            by_type[BlockType.EQUATION].assets[0].sha256,
        )

    def test_request_is_page_image_only_and_never_uses_logical_source_uri(self):
        pages = self.runtime.loader().load(self.request)

        self.assertEqual(1, len(self.runtime.requests))
        outbound = self.runtime.requests[0]
        self.assertTrue(outbound.url.endswith(OFFICIAL_LAYOUT_PARSING_PATH))
        body = json.loads(outbound.body)
        self.assertEqual(1, body["fileType"])
        self.assertEqual(PAGE_BYTES, base64.b64decode(body["file"], validate=True))
        self.assertTrue(body["useLayoutDetection"])
        self.assertFalse(body["visualize"])
        self.assertNotIn(self.request.document.source_uri, outbound.body.decode("utf-8"))
        self.assertEqual(self.request.request_id, outbound.headers["X-DocARAG-Parse-Request-ID"])
        self.assertTrue(
            pages[0]["model_settings"]["docarag_inference_options"][
                "useOcrForImageBlock"
            ]
        )

    def test_effective_options_match_exact_outbound_request(self):
        request = request_fixture(parse_images=False)
        runtime = FixtureRuntime(request)
        page = service_page_fixture()
        page["markdown"]["images"] = {}
        runtime.response = response_fixture(page=page)

        pages = runtime.loader().load(request)

        outbound = json.loads(runtime.requests[0].body)
        outbound_options = {
            key: value
            for key, value in outbound.items()
            if key not in {"file", "fileType"}
        }
        self.assertFalse(outbound_options["returnMarkdownImages"])
        self.assertEqual(
            outbound_options,
            pages[0]["model_settings"]["docarag_inference_options"],
        )

    def test_empty_page_retains_exact_raw_response_receipt(self):
        page = service_page_fixture()
        page["prunedResult"]["parsing_res_list"] = []
        page["prunedResult"]["layout_det_res"]["boxes"] = []
        page["markdown"] = {"text": "", "images": {}}
        self.runtime.response = response_fixture(page=page)

        result = self.runtime.adapter().parse(self.request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual((), result.observations)
        self.assertEqual(1, len(result.artifacts))
        self.assertEqual(
            hashlib.sha256(self.runtime.response.body).hexdigest(),
            result.artifacts[0].sha256,
        )
        self.assertEqual(
            len(self.runtime.response.body), result.artifacts[0].size_bytes
        )

    def test_manifest_is_bounded_complete_and_truthfully_uncertified(self):
        manifest = self.runtime.adapter().manifest()

        self.assertTrue(manifest.supports_page_selection)
        self.assertTrue(manifest.supports_batch)
        self.assertEqual(4, manifest.max_pages_per_request)
        self.assertTrue(manifest.metadata["complete_pipeline"])
        self.assertFalse(manifest.metadata["raw_vlm_endpoint_accepted"])
        self.assertEqual("configured-unattested", manifest.metadata["remote_identity_assurance"])
        self.assertEqual("uncertified", manifest.metadata["qualification"])
        self.assertEqual("PaddleOCR-VL-1.6-0.9B", manifest.model_version)


class PaddleOCRVLServiceAdversarialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = request_fixture()
        self.runtime = FixtureRuntime(self.request)

    def test_raw_vllm_configuration_is_explicitly_rejected(self):
        with self.assertRaises(RawVLMEndpointRejectedError):
            config_fixture(endpoint_kind=PaddleOCRVLEndpointKind.VLM_INFERENCE)
        with self.assertRaises(RawVLMEndpointRejectedError):
            config_fixture(base_url="http://127.0.0.1:8080/v1")

    def test_identity_and_transport_configuration_fail_closed(self):
        with self.assertRaises(ValueError):
            config_fixture(pipeline_name="PaddleOCR-VL")
        with self.assertRaises(ValueError):
            config_fixture(recognition_model_name="some-model")
        with self.assertRaises(ValueError):
            config_fixture(base_url="http://parser.example.com:8080")
        with self.assertRaises(ValueError):
            config_fixture(max_pages_per_request=0)
        with self.assertRaisesRegex(
            ValueError, "max_response_bytes must not exceed max_total_artifact_bytes"
        ):
            config_fixture(
                max_response_bytes=2 * 1024 * 1024,
                max_total_artifact_bytes=1024 * 1024,
            )
        with self.assertRaisesRegex(
            ValueError,
            "exceed ParseResult artifact receipt capacity",
        ):
            config_fixture(
                max_pages_per_request=16,
                max_artifacts_per_page=256,
            )

    def test_openai_response_is_permanent_contract_failure(self):
        self.runtime.response = PaddleOCRVLHTTPResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {"object": "chat.completion", "choices": [], "model": "vlm"}
            ).encode(),
        )
        result = self.runtime.adapter().parse(self.request)

        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
        self.assertEqual(ErrorClassification.PERMANENT, result.issues[0].classification)
        self.assertEqual("RawVLMEndpointRejectedError", result.issues[0].details["exception_type"])

    def test_response_must_prove_all_complete_pipeline_stages(self):
        page = service_page_fixture()
        page["prunedResult"]["model_settings"]["use_layout_detection"] = False
        self.runtime.response = response_fixture(page=page)
        with self.assertRaises(PaddleOCRVLServiceContractError):
            self.runtime.loader().load(self.request)

        page = service_page_fixture()
        del page["prunedResult"]["layout_det_res"]
        self.runtime.response = response_fixture(page=page)
        with self.assertRaises(PaddleOCRVLServiceContractError):
            self.runtime.loader().load(self.request)

    def test_page_cardinality_and_service_envelope_are_strict(self):
        payload = json.loads(response_fixture().body)
        payload["result"]["layoutParsingResults"].append(service_page_fixture())
        self.runtime.response = PaddleOCRVLHTTPResponse(200, {}, json.dumps(payload).encode())
        with self.assertRaises(PaddleOCRVLServiceContractError):
            self.runtime.loader().load(self.request)

        payload = json.loads(response_fixture().body)
        payload["unexpected"] = True
        self.runtime.response = PaddleOCRVLHTTPResponse(200, {}, json.dumps(payload).encode())
        with self.assertRaises(PaddleOCRVLServiceContractError):
            self.runtime.loader().load(self.request)

    def test_source_and_render_bindings_are_reverified(self):
        bad_source = source_fixture(self.request)
        bad_source = VerifiedSourceMaterial(
            document_version_id="another-version",
            logical_source_uri=bad_source.logical_source_uri,
            material_uri=bad_source.material_uri,
            media_type=bad_source.media_type,
            sha256=bad_source.sha256,
            size_bytes=bad_source.size_bytes,
            path=bad_source.path,
            data=bad_source.data,
        )
        loader = PaddleOCRVLCompletePipelineLoader(
            config_fixture(),
            source_loader=lambda _request: bad_source,
            page_renderer=self.runtime.page_renderer,
            artifact_sink=self.runtime.sink,
            http_exchange=self.runtime.exchange,
        )
        with self.assertRaises(PaddleOCRVLServiceContractError):
            loader.load(self.request)

        loader = PaddleOCRVLCompletePipelineLoader(
            config_fixture(),
            source_loader=self.runtime.source_loader,
            page_renderer=lambda source, page: PaddleOCRVLPageMaterial(
                document_version_id=self.request.document.document_version_id,
                source_sha256=self.request.document.content_sha256,
                page_index=page.page_index + 1,
                media_type="image/png",
                width=1000,
                height=800,
                data=PAGE_BYTES,
            ),
            artifact_sink=self.runtime.sink,
            http_exchange=self.runtime.exchange,
        )
        with self.assertRaises(PaddleOCRVLServiceContractError):
            loader.load(self.request)

    def test_request_response_and_artifact_limits_apply_to_injected_exchange(self):
        with self.assertRaises(PaddleOCRVLServiceLimitError):
            self.runtime.loader(config=config_fixture(max_request_bytes=32)).load(self.request)

        response = response_fixture()
        self.runtime.response = response
        with self.assertRaises(PaddleOCRVLServiceLimitError):
            self.runtime.loader(
                config=config_fixture(max_response_bytes=len(response.body) - 1)
            ).load(self.request)

        with self.assertRaises(PaddleOCRVLServiceLimitError):
            self.runtime.loader(
                config=config_fixture(max_artifact_bytes=len(IMAGE_BYTES) - 1)
            ).load(self.request)

    def test_total_artifact_budget_is_shared_across_all_requested_pages(self):
        request = request_fixture(page_count=2, pages=(0, 1))
        runtime = FixtureRuntime(request)
        per_page_bytes = len(runtime.response.body) + len(IMAGE_BYTES)
        config = config_fixture(
            max_artifact_bytes=per_page_bytes,
            max_total_artifact_bytes=per_page_bytes,
            max_response_bytes=per_page_bytes,
        )

        with self.assertRaises(ParserExecutionError) as caught:
            runtime.loader(config=config).load(request)
        self.assertEqual(2, len(caught.exception.artifacts))
        self.assertEqual(
            ParseErrorCode.RESOURCE_EXHAUSTED,
            caught.exception.issue.code,
        )

        # Page zero was complete, but page one is rejected before any of its
        # artifacts are published.  A per-page-reset counter would store four.
        self.assertEqual(2, len(runtime.artifacts))

    def test_second_page_failure_retains_first_page_raw_artifact_receipt(self):
        request = request_fixture(page_count=2, pages=(0, 1))
        runtime = FixtureRuntime(request)
        responses = [
            response_fixture(),
            PaddleOCRVLHTTPResponse(
                503,
                {"content-type": "application/json"},
                json.dumps({"errorMsg": "busy"}).encode(),
            ),
        ]

        def exchange(_request):
            return responses.pop(0)

        runtime.exchange = exchange
        result = runtime.adapter().parse(request)

        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(2, len(result.artifacts))
        receipt = next(
            item for item in result.artifacts
            if item.artifact_role == "paddleocr_vl_raw_page_result"
        )
        self.assertEqual("paddleocr_vl_raw_page_result", receipt.artifact_role)
        self.assertEqual(0, receipt.metadata["page_index"])
        self.assertEqual("page-response.json", receipt.artifact_key)
        crop_receipt = next(
            item for item in result.artifacts
            if item.artifact_role == "paddleocr_vl_markdown_image"
        )
        self.assertEqual(IMAGE_NAME, crop_receipt.artifact_key)
        self.assertEqual(2, len(runtime.artifacts))
        self.assertEqual(
            ErrorClassification.RETRYABLE,
            result.issues[0].classification,
        )

    def test_missing_and_unmatched_markdown_crops_fail_before_storage(self):
        page = service_page_fixture()
        page["markdown"]["images"] = {}
        self.runtime.response = response_fixture(page=page)
        with self.assertRaisesRegex(
            PaddleOCRVLServiceContractError, "missing expected block crops"
        ):
            self.runtime.loader().load(self.request)
        self.assertEqual([], self.runtime.artifacts)

        page = service_page_fixture()
        page["markdown"]["images"]["imgs/unmatched.jpg"] = base64.b64encode(
            b"unmatched"
        ).decode("ascii")
        self.runtime.response = response_fixture(page=page)
        with self.assertRaisesRegex(
            PaddleOCRVLServiceContractError, "no unique owning block"
        ):
            self.runtime.loader().load(self.request)
        self.assertEqual([], self.runtime.artifacts)

    def test_crop_association_rejects_normalized_name_and_block_id_collisions(self):
        page = service_page_fixture()
        page["markdown"]["images"][IMAGE_NAME.replace("/", "\\")] = (
            base64.b64encode(IMAGE_BYTES).decode("ascii")
        )
        self.runtime.response = response_fixture(page=page)
        with self.assertRaisesRegex(
            PaddleOCRVLServiceContractError, "normalize to the same safe path"
        ):
            self.runtime.loader().load(self.request)
        self.assertEqual([], self.runtime.artifacts)

        page = service_page_fixture()
        duplicate = {
            "block_label": "image",
            "block_content": "",
            "block_bbox": [10, 600, 100, 700],
            "block_id": 2,
            "block_order": 3,
            "group_id": 3,
        }
        page["prunedResult"]["parsing_res_list"].append(duplicate)
        duplicate_name = "imgs/img_in_image_box_10_600_100_700.jpg"
        page["markdown"]["images"][duplicate_name] = base64.b64encode(
            b"second crop"
        ).decode("ascii")
        self.runtime.response = response_fixture(page=page)
        with self.assertRaisesRegex(
            PaddleOCRVLServiceContractError, "block IDs collide"
        ):
            self.runtime.loader().load(self.request)
        self.assertEqual([], self.runtime.artifacts)

    def test_response_json_rejects_duplicate_keys_and_bounded_shape(self):
        response = response_fixture()
        self.runtime.response = PaddleOCRVLHTTPResponse(
            response.status,
            response.headers,
            response.body.replace(
                b'"errorCode":0',
                b'"errorCode":0,"errorCode":0',
                1,
            ),
        )
        with self.assertRaisesRegex(
            PaddleOCRVLServiceContractError, "duplicate JSON object key"
        ):
            self.runtime.loader().load(self.request)

        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(token=token):
                response = response_fixture()
                self.runtime.response = PaddleOCRVLHTTPResponse(
                    response.status,
                    response.headers,
                    response.body.replace(b'"errorCode":0', b'"errorCode":' + token, 1),
                )
                with self.assertRaisesRegex(
                    PaddleOCRVLServiceContractError, "non-finite JSON number"
                ):
                    self.runtime.loader().load(self.request)

        self.runtime.response = response_fixture()
        with self.assertRaises(PaddleOCRVLServiceLimitError):
            self.runtime.loader(
                config=config_fixture(max_response_json_depth=2)
            ).load(self.request)
        with self.assertRaises(PaddleOCRVLServiceLimitError):
            self.runtime.loader(
                config=config_fixture(max_response_json_nodes=10)
            ).load(self.request)
        identity = config_fixture(
            max_response_json_depth=17,
            max_response_json_nodes=101,
        ).identity_dict()
        self.assertEqual(
            {
                "max_bytes": 2 * 1024 * 1024,
                "max_depth": 17,
                "max_nodes": 101,
            },
            identity["response_json_bounds"],
        )
        self.assertEqual(
            2 * 1024 * 1024,
            identity["artifact_bounds"]["max_total_artifact_bytes"],
        )
        self.assertEqual(
            2 * 1024 * 1024,
            identity["artifact_bounds"]["max_raw_response_artifact_bytes"],
        )

    def test_url_artifacts_and_sink_identity_mismatch_are_rejected(self):
        page = service_page_fixture()
        page["markdown"]["images"][IMAGE_NAME] = "https://storage.example/crop.jpg"
        self.runtime.response = response_fixture(page=page)
        with self.assertRaises(PaddleOCRVLServiceContractError):
            self.runtime.loader().load(self.request)

        def bad_sink(artifact):
            return StoredPaddleOCRVLArtifact(
                uri="artifact://bad",
                sha256="0" * 64,
                media_type=artifact.media_type,
            )

        loader = PaddleOCRVLCompletePipelineLoader(
            config_fixture(),
            source_loader=self.runtime.source_loader,
            page_renderer=self.runtime.page_renderer,
            artifact_sink=bad_sink,
            http_exchange=lambda _request: response_fixture(),
        )
        with self.assertRaises(PaddleOCRVLServiceContractError):
            loader.load(self.request)

    def test_timeout_and_remote_5xx_remain_retryable(self):
        timeout_adapter = build_paddleocr_vl_service_adapter(
            config_fixture(),
            source_loader=self.runtime.source_loader,
            page_renderer=self.runtime.page_renderer,
            artifact_sink=self.runtime.sink,
            http_exchange=lambda _request: (_ for _ in ()).throw(
                PaddleOCRVLServiceTimeoutError("timeout")
            ),
        )
        timeout_result = timeout_adapter.parse(self.request)
        self.assertEqual(ParseErrorCode.TIMEOUT, timeout_result.issues[0].code)
        self.assertEqual(ErrorClassification.RETRYABLE, timeout_result.issues[0].classification)

        self.runtime.response = PaddleOCRVLHTTPResponse(
            503,
            {"content-type": "application/json"},
            json.dumps({"errorMsg": "busy"}).encode(),
        )
        failed = self.runtime.adapter().parse(self.request)
        self.assertEqual(ParseErrorCode.PARSER_FAILURE, failed.issues[0].code)
        self.assertEqual(ErrorClassification.RETRYABLE, failed.issues[0].classification)
        with self.assertRaises(PaddleOCRVLRemoteServiceError):
            self.runtime.loader().load(self.request)

    def test_import_boundary_loads_no_vendor_or_model_runtime(self):
        script = r"""
import builtins, json, sys
blocked = ('paddle', 'paddleocr', 'torch', 'transformers', 'vllm', 'requests')
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import docarag.services.paddleocr_vl
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in blocked)))
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
