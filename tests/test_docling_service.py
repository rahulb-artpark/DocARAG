from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from docarag.adapters.docling import DoclingExportAdapter
from docarag.domain import (
    DocumentRef,
    ErrorClassification,
    PageRef,
    ParseRequest,
    ParseStatus,
)
from docarag.services.docling import (
    DoclingHTTPResponse,
    DoclingServiceConfig,
    DoclingServiceConfigurationError,
    DoclingServiceContractError,
    DoclingServiceExportLoader,
    DoclingServiceRemoteError,
    DoclingServiceSourceError,
    DoclingServiceTimeoutError,
)
from docarag.source_material import VerifiedSourceMaterial


SOURCE = b"%PDF-1.7\nverified docling service fixture\n%%EOF\n"
SOURCE_SHA256 = hashlib.sha256(SOURCE).hexdigest()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeExchange:
    def __init__(self, responses: list[DoclingHTTPResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        headers,
        body,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> DoclingHTTPResponse:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def json_response(
    payload: object,
    *,
    status: int = 200,
    content_type: str = "application/json",
) -> DoclingHTTPResponse:
    return DoclingHTTPResponse(
        status=status,
        headers={"Content-Type": content_type},
        body=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
    )


def request_fixture(*, pages: tuple[int, ...] = (0, 1)) -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=SOURCE_SHA256,
        mime_type="application/pdf",
        source_uri="s3://logical-citation-bucket/source.pdf",
        display_name="source.pdf",
        page_count=max(pages) + 1,
    )
    return ParseRequest(
        document=document,
        pages=tuple(
            PageRef(document.document_version_id, page, width=612, height=792)
            for page in pages
        ),
        run_id="run-docling-service",
        trace_id="trace-docling-service",
        idempotency_key="docling-service-test",
    )


def material(request: ParseRequest, *, data: bytes = SOURCE) -> VerifiedSourceMaterial:
    return VerifiedSourceMaterial(
        document_version_id=request.document.document_version_id,
        logical_source_uri=request.document.source_uri,
        material_uri="file:///verified/source.pdf",
        media_type="application/pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        path=Path("/verified/source.pdf"),
        data=data,
    )


def export_fixture(*, pages: tuple[int, ...] = (1, 2)) -> dict[str, object]:
    texts = []
    children = []
    page_map = {}
    for index, page_no in enumerate(pages):
        reference = f"#/texts/{index}"
        children.append({"$ref": reference})
        texts.append(
            {
                "self_ref": reference,
                "parent": {"$ref": "#/body"},
                "content_layer": "body",
                "label": "text",
                "text": f"page {page_no}",
                "orig": f"page {page_no}",
                "prov": [
                    {
                        "page_no": page_no,
                        "bbox": {
                            "l": 10,
                            "t": 100,
                            "r": 120,
                            "b": 80,
                            "coord_origin": "BOTTOMLEFT",
                        },
                        "charspan": [0, 6],
                    }
                ],
            }
        )
        page_map[str(page_no)] = {
            "page_no": page_no,
            "size": {"width": 612, "height": 792},
        }
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "source",
        "origin": {"mimetype": "application/pdf", "filename": "source.pdf"},
        "pages": page_map,
        "body": {"self_ref": "#/body", "children": children},
        "groups": [],
        "texts": texts,
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "form_items": [],
        "field_regions": [],
        "field_items": [],
    }


def task(task_id: str, status: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": "convert",
        "task_status": status,
        "task_position": None,
        "task_meta": None,
    }


def result_fixture(export: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "document": {
            "filename": "source.pdf",
            "md_content": None,
            "json_content": export or export_fixture(),
            "html_content": None,
            "text_content": None,
            "doctags_content": None,
        },
        "status": "success",
        "processing_time": 1.25,
        "timings": {},
        "errors": [],
    }


def success_exchange() -> FakeExchange:
    return FakeExchange(
        [
            json_response(task("task-1", "pending")),
            json_response(task("task-1", "started")),
            json_response(task("task-1", "success")),
            json_response(result_fixture()),
        ]
    )


def loader_for(exchange: FakeExchange, **config_overrides: object):
    request = request_fixture()
    values = {"base_url": "http://127.0.0.1:5001"}
    values.update(config_overrides)
    clock = Clock()
    loader = DoclingServiceExportLoader(
        DoclingServiceConfig(**values),  # type: ignore[arg-type]
        source_loader=lambda active: material(active),
        exchange=exchange,
        monotonic=clock,
        sleeper=clock.sleep,
    )
    return request, loader, clock


class DoclingServiceHappyPathTests(unittest.TestCase):
    def test_async_flow_is_bounded_source_bound_and_adapter_compatible(self):
        exchange = success_exchange()
        request, loader, _clock = loader_for(exchange)

        export = loader.load(request)

        self.assertEqual(export["source_sha256"], SOURCE_SHA256)
        self.assertEqual(
            [call["path"] for call in exchange.calls],
            [
                "/v1/convert/source/async",
                "/v1/status/poll/task-1",
                "/v1/status/poll/task-1",
                "/v1/result/task-1",
            ],
        )
        submitted = json.loads(exchange.calls[0]["body"])
        self.assertEqual(submitted["options"]["page_range"], [1, 2])
        self.assertEqual(submitted["options"]["to_formats"], ["json"])
        self.assertEqual(submitted["target"], {"kind": "inbody"})
        decoded = submitted["file_sources"][0]["base64_string"]
        import base64

        self.assertEqual(base64.b64decode(decoded), SOURCE)

        adapter_exchange = success_exchange()
        _request, adapter_loader, _clock = loader_for(adapter_exchange)
        adapter = DoclingExportAdapter(
            adapter_loader,
            parser_version="qualified-parser-fixture",
            model_version="qualified-model-fixture",
            config=adapter_loader.safe_config,
        )
        parsed = adapter.parse(request)
        self.assertEqual(parsed.status, ParseStatus.SUCCEEDED)
        self.assertEqual(len(parsed.observations), 2)

    def test_safe_config_never_serializes_api_key(self):
        config = DoclingServiceConfig(
            base_url="https://docling.example", api_key="super-secret"
        )
        serialized = json.dumps(config.safe_dict())
        self.assertNotIn("super-secret", serialized)
        self.assertTrue(config.safe_dict()["api_key_configured"])
        self.assertFalse(config.safe_dict()["service_release_qualified"])


class DoclingServiceFailureTests(unittest.TestCase):
    def test_source_and_sparse_page_fail_before_network(self):
        for request, source_loader in (
            (
                request_fixture(),
                lambda active: material(active, data=b"another source"),
            ),
            (request_fixture(pages=(0, 2)), lambda active: material(active)),
        ):
            with self.subTest(pages=tuple(page.page_index for page in request.pages)):
                exchange = FakeExchange([])
                loader = DoclingServiceExportLoader(
                    DoclingServiceConfig(base_url="http://127.0.0.1:5001"),
                    source_loader=source_loader,
                    exchange=exchange,
                )
                with self.assertRaises(DoclingServiceSourceError):
                    loader.load(request)
                self.assertEqual(exchange.calls, [])

    def test_adapter_preserves_permanent_source_failure(self):
        request = request_fixture()
        loader = DoclingServiceExportLoader(
            DoclingServiceConfig(base_url="http://127.0.0.1:5001"),
            source_loader=lambda active: material(active, data=b"wrong"),
            exchange=FakeExchange([]),
        )
        result = DoclingExportAdapter(
            loader,
            parser_version="parser",
            model_version="model",
            config=loader.safe_config,
        ).parse(request)
        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertEqual(result.issues[0].classification, ErrorClassification.PERMANENT)
        self.assertEqual(result.issues[0].code.value, "invalid_request")

    def test_task_identity_schema_and_json_are_fail_closed(self):
        malformed = DoclingHTTPResponse(
            200,
            {"Content-Type": "application/json"},
            b'{"task_id":"one","task_id":"two","task_status":"pending"}',
        )
        cases = (
            [json_response({**task("task-1", "pending"), "new_field": 1})],
            [
                json_response(task("task-1", "pending")),
                json_response(task("task-2", "started")),
            ],
            [malformed],
        )
        for responses in cases:
            with self.subTest(responses=len(responses)):
                request, loader, _clock = loader_for(FakeExchange(responses))
                with self.assertRaises(DoclingServiceContractError):
                    loader.load(request)

    def test_poll_attempt_bound_maps_to_retryable_timeout(self):
        responses = [
            json_response(task("task-1", "pending")),
            json_response(task("task-1", "pending")),
        ]
        exchange = FakeExchange(list(responses))
        request, loader, _clock = loader_for(
            exchange, max_poll_attempts=1, operation_timeout_seconds=20
        )
        with self.assertRaises(DoclingServiceTimeoutError):
            loader.load(request)
        request, loader, _clock = loader_for(
            FakeExchange(list(responses)),
            max_poll_attempts=1,
            operation_timeout_seconds=20,
        )
        result = DoclingExportAdapter(
            loader,
            parser_version="parser",
            model_version="model",
            config=loader.safe_config,
        ).parse(request)
        self.assertEqual(result.issues[0].classification, ErrorClassification.RETRYABLE)
        self.assertEqual(result.issues[0].code.value, "timeout")

    def test_http_retryability_and_content_type_are_explicit(self):
        for status, classification in (
            (503, ErrorClassification.RETRYABLE),
            (401, ErrorClassification.PERMANENT),
        ):
            with self.subTest(status=status):
                request, loader, _clock = loader_for(
                    FakeExchange([json_response({}, status=status)])
                )
                result = DoclingExportAdapter(
                    loader,
                    parser_version="parser",
                    model_version="model",
                    config=loader.safe_config,
                ).parse(request)
                self.assertEqual(result.issues[0].classification, classification)
        request, loader, _clock = loader_for(
            FakeExchange([json_response({}, content_type="text/html")])
        )
        with self.assertRaises(DoclingServiceContractError):
            loader.load(request)

    def test_result_status_document_version_pages_and_new_structures_fail_closed(self):
        variants = []
        partial = result_fixture()
        partial["status"] = "partial_success"
        variants.append((partial, DoclingServiceRemoteError))
        wrong_version = export_fixture()
        wrong_version["version"] = "1.11.0"
        variants.append((result_fixture(wrong_version), DoclingServiceContractError))
        wrong_pages = export_fixture(pages=(1,))
        variants.append((result_fixture(wrong_pages), DoclingServiceContractError))
        unsupported = export_fixture()
        unsupported["key_value_items"] = [{"self_ref": "#/key_value_items/0"}]
        variants.append((result_fixture(unsupported), DoclingServiceContractError))
        for result, error in variants:
            with self.subTest(error=error.__name__):
                exchange = FakeExchange(
                    [
                        json_response(task("task-1", "success")),
                        json_response(result),
                    ]
                )
                request, loader, _clock = loader_for(exchange)
                with self.assertRaises(error):
                    loader.load(request)

    def test_request_and_response_byte_limits_apply_to_injected_exchange(self):
        request, loader, _clock = loader_for(
            FakeExchange([]), max_request_bytes=64
        )
        with self.assertRaises(DoclingServiceSourceError):
            loader.load(request)

        oversized = DoclingHTTPResponse(
            200,
            {"Content-Type": "application/json"},
            b"x" * 129,
        )
        request, loader, _clock = loader_for(
            FakeExchange([oversized]), max_response_bytes=128
        )
        with self.assertRaises(DoclingServiceContractError):
            loader.load(request)


class DoclingServiceConfigurationTests(unittest.TestCase):
    def test_url_schema_and_secret_validation(self):
        for values in (
            {"base_url": "http://docling.internal:5001"},
            {"base_url": "https://docling.example/prefix"},
            {
                "base_url": "https://docling.example",
                "expected_document_schema_version": "latest",
            },
            {"base_url": "https://docling.example", "api_key": "bad\nkey"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(DoclingServiceConfigurationError):
                    DoclingServiceConfig(**values)

    def test_controller_import_does_not_load_docling_vendor_packages(self):
        script = (
            "import sys; import docarag.services.docling; "
            "bad=[n for n in sys.modules if n == 'docling' or "
            "n.startswith(('docling.', 'torch', 'transformers', 'paddle'))]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
