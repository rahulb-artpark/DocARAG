from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from urllib.parse import quote
import zipfile

from docarag.adapters.mineru_compat import MinerUNormalizedAdapter
from docarag.domain import (
    BlockType,
    DocumentRef,
    ErrorClassification,
    MinerUEffort,
    PageRef,
    ParseErrorCode,
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserArtifactAvailability,
)
from docarag.ports import ParserRegistry
from docarag.services.mineru import (
    MAX_MINERU_TASKS_PER_PARSE_RESULT,
    MINERU_RESULT_ARCHIVE_ARTIFACT_SCHEMA,
    MinerUAPIClient,
    MinerUBackend,
    MinerUHTTPResponse,
    MinerUProtocolError,
    MinerUServiceConfig,
    MinerUServiceLoader,
    PublishedMinerUArchive,
)
from docarag.source_material import VerifiedSourceMaterial


PDF_BYTES = b"%PDF-1.7\nverified mineru fixture\n%%EOF"
PDF_SHA = hashlib.sha256(PDF_BYTES).hexdigest()
FIGURE_BYTES = b"image bytes"
CHART_BYTES = b"chart bytes"
DEBUG_BYTES = b"unreferenced but identity-verified"


def middle_fixture(*, pages=(0, 1)):
    return {
        "_backend": "hybrid",
        "_version_name": "3.4.4",
        "pdf_info": [
            {
                "page_idx": page_index,
                "page_size": [1000, 1000],
                "para_blocks": [],
            }
            for page_index in pages
        ],
    }


def request_fixture(*, effort: MinerUEffort | None = MinerUEffort.MEDIUM) -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=PDF_SHA,
        mime_type="application/pdf",
        source_uri="s3://citation-only/private.pdf",
        display_name="doc.pdf",
        page_count=2,
    )
    return ParseRequest(
        document=document,
        pages=(
            PageRef(document.document_version_id, 0, width=612, height=792),
            PageRef(document.document_version_id, 1, width=612, height=792),
        ),
        options=ParseOptions(
            languages=("en",),
            mineru_effort=effort,
            timeout_seconds=30,
        ),
        run_id="run-mineru-service",
        trace_id="trace-mineru-service",
    )


def source_fixture(request: ParseRequest, *, data: bytes | None = PDF_BYTES):
    return VerifiedSourceMaterial(
        document_version_id=request.document.document_version_id,
        logical_source_uri=request.document.source_uri,
        material_uri="file:///verified/snapshot.pdf",
        media_type="application/pdf",
        sha256=PDF_SHA,
        size_bytes=len(PDF_BYTES),
        path=Path("/verified/snapshot.pdf"),
        data=data,
    )


def json_response(status: int, payload: object) -> MinerUHTTPResponse:
    return MinerUHTTPResponse(
        status,
        {"content-type": "application/json"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def result_zip(
    *,
    unsafe: bool = False,
    two_lists: bool = False,
    emit_middle: bool = True,
    middle_payload: object | bytes | None = None,
    two_middles: bool = False,
    empty_blocks: bool = False,
) -> bytes:
    blocks = [
        {
            "type": "title",
            "text": "Technical report",
            "text_level": 1,
            "bbox": [10, 20, 800, 80],
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_body": "<table><tr><td>A</td></tr></table>",
            "table_caption": ["Table 1"],
            "bbox": [10, 100, 800, 400],
            "page_idx": 0,
        },
        {
            "type": "equation",
            "text": "E=mc^2",
            "text_format": "latex",
            "bbox": [20, 420, 400, 500],
            "page_idx": 0,
        },
        {
            "type": "image",
            "sub_type": "diagram",
            "img_path": "images/figure.png",
            "image_caption": ["System diagram"],
            "bbox": [20, 20, 900, 500],
            "page_idx": 1,
        },
        {
            "type": "chart",
            "content": "| Q | Value |\n|---|---|\n| Q1 | 3 |",
            "img_path": "images/chart.png",
            "chart_caption": ["Revenue"],
            "bbox": [20, 520, 900, 900],
            "page_idx": 1,
        },
        {
            "type": "code",
            "sub_type": "algorithm",
            "code_body": "return answer",
            "bbox": [20, 910, 500, 990],
            "page_idx": 1,
        },
    ]
    if empty_blocks:
        blocks = []
    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc/doc_content_list.json", json.dumps(blocks))
        if emit_middle:
            payload = middle_fixture() if middle_payload is None else middle_payload
            archive.writestr(
                "doc/doc_middle.json",
                payload if isinstance(payload, bytes) else json.dumps(payload),
            )
        if two_middles:
            archive.writestr("doc/other_middle.json", json.dumps(middle_fixture()))
        archive.writestr("doc/images/figure.png", FIGURE_BYTES)
        archive.writestr("doc/images/chart.png", CHART_BYTES)
        archive.writestr("doc/debug/trace.txt", DEBUG_BYTES)
        if two_lists:
            archive.writestr("other/other_content_list.json", "[]")
        if unsafe:
            archive.writestr("../escape.txt", "bad")
    return target.getvalue()


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def request(self, method, url, *, headers, body, timeout_seconds, max_response_bytes):
        body_bytes = b"" if body is None else b"".join(bytes(chunk) for chunk in body)
        if body is not None:
            assert len(body_bytes) == body.content_length
        self.requests.append((method, url, dict(headers), body_bytes, timeout_seconds))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def responses_for_success(archive: bytes | None = None):
    archive = result_zip() if archive is None else archive
    return [
        json_response(
            200,
            {
                "status": "healthy",
                "protocol_version": 2,
                "processing_window_size": 64,
                "max_concurrent_requests": 3,
            },
        ),
        json_response(
            202,
            {
                "task_id": "task-1",
                "status_url": "https://mineru.example/tasks/task-1",
                "result_url": "https://mineru.example/tasks/task-1/result",
                "file_names": ["doc.pdf"],
                "queued_ahead": 2,
            },
        ),
        json_response(200, {"task_id": "task-1", "status": "pending", "queued_ahead": 1}),
        json_response(200, {"task_id": "task-1", "status": "completed"}),
        MinerUHTTPResponse(200, {"content-type": "application/zip"}, archive),
    ]


def publisher(archive):
    return PublishedMinerUArchive(
        archive_uri=f"artifact://mineru/{archive.archive_sha256}/result.zip",
        archive_sha256=archive.archive_sha256,
        member_uris={
            member: f"artifact://mineru/{archive.archive_sha256}/{quote(member)}"
            for member in archive.members
        },
    )


def service_adapter(
    request: ParseRequest,
    transport: FakeTransport,
    *,
    artifact_publisher=publisher,
    config_overrides=None,
):
    config_values = dict(
        base_url="https://mineru.example",
        backend=MinerUBackend.HYBRID_ENGINE,
        effort=MinerUEffort.MEDIUM,
        poll_interval_seconds=0.001,
    )
    config_values.update(config_overrides or {})
    config = MinerUServiceConfig(**config_values)
    client = MinerUAPIClient(
        config,
        transport=transport,
        owns_transport=False,
        sleeper=lambda _seconds: None,
    )
    loader = MinerUServiceLoader(
        client,
        source_loader=lambda received: source_fixture(received),
        artifact_publisher=artifact_publisher,
    )
    adapter = MinerUNormalizedAdapter(
        loader,
        parser_version="3.x-unqualified",
        model_version="mineru-runtime-unqualified",
        config=config.identity(),
    )
    return config, loader, adapter


class MinerUServiceHappyPathTests(unittest.TestCase):
    def test_loaded_blocks_preserve_tuple_like_len_and_index_access(self):
        request = request_fixture()
        _config, loader, _adapter = service_adapter(
            request,
            FakeTransport(responses_for_success()),
        )

        loaded = loader.load(request)

        self.assertEqual(len(loaded.blocks), len(loaded))
        self.assertIs(loaded.blocks[0], loaded[0])
        self.assertEqual(loaded.blocks[:2], loaded[:2])

    def test_task_api_translates_structured_output_and_published_assets(self):
        request = request_fixture()
        transport = FakeTransport(responses_for_success())
        captured = []

        def capturing_publisher(archive):
            captured.append(archive)
            return publisher(archive)

        config, loader, adapter = service_adapter(
            request, transport, artifact_publisher=capturing_publisher
        )

        result = ParserRegistry((adapter,)).parse("mineru-normalized", request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual(6, len(result.observations))
        by_type = {item.block_type: item for item in result.observations}
        self.assertIn(BlockType.TITLE, by_type)
        self.assertIn(BlockType.TABLE, by_type)
        self.assertIn(BlockType.EQUATION, by_type)
        self.assertIn(BlockType.DIAGRAM, by_type)
        self.assertIn(BlockType.CHART, by_type)
        self.assertIn(BlockType.CODE, by_type)
        self.assertEqual(1000, by_type[BlockType.TITLE].geometry.page_width)
        self.assertTrue(by_type[BlockType.DIAGRAM].assets[0].uri.startswith("artifact://mineru/"))
        self.assertEqual(
            hashlib.sha256(FIGURE_BYTES).hexdigest(),
            by_type[BlockType.DIAGRAM].assets[0].sha256,
        )
        self.assertEqual(
            "image/png", by_type[BlockType.DIAGRAM].assets[0].media_type
        )
        self.assertEqual(
            hashlib.sha256(CHART_BYTES).hexdigest(),
            by_type[BlockType.CHART].assets[0].sha256,
        )
        service = by_type[BlockType.DIAGRAM].metadata["mineru_service"]
        self.assertEqual("medium", service["effort"])
        self.assertTrue(service["archive_uri"].startswith("artifact://mineru/"))
        self.assertEqual(config.revision_sha256, service["service_revision_sha256"])
        self.assertEqual("process_local_non_durable", service["vendor_task_state"])
        archive = captured[0]
        self.assertEqual(
            hashlib.sha256(DEBUG_BYTES).hexdigest(),
            archive.member_sha256s["doc/debug/trace.txt"],
        )
        self.assertEqual(
            archive.member_sha256s["doc/doc_content_list.json"],
            service["content_list_sha256"],
        )
        self.assertEqual(2, len(result.artifacts))
        middle = next(
            item for item in result.artifacts
            if item.artifact_role == "mineru_middle_json"
        )
        primary = next(
            item for item in result.artifacts
            if item.artifact_role == "mineru_result_archive"
        )
        self.assertEqual(
            service["middle_artifact_receipt_id"], middle.receipt_id
        )
        self.assertEqual(ParserArtifactAvailability.EMITTED, middle.availability)
        self.assertEqual("mineru.middle-json", middle.artifact_schema)
        self.assertEqual("hybrid", middle.metadata["backend"])
        self.assertEqual("3.4.4", middle.metadata["version"])
        self.assertEqual(
            archive.member_sha256s["doc/doc_middle.json"],
            middle.sha256,
        )
        self.assertTrue(middle.uri.endswith("doc/doc_middle.json"))
        self.assertGreater(middle.size_bytes, 0)
        self.assertEqual(
            MINERU_RESULT_ARCHIVE_ARTIFACT_SCHEMA,
            primary.artifact_schema,
        )
        self.assertEqual(archive.archive_sha256, primary.sha256)
        self.assertEqual(len(archive.data), primary.size_bytes)
        self.assertEqual(0, primary.metadata["first_page"])
        self.assertEqual(1, primary.metadata["last_page"])
        self.assertEqual("result.zip", primary.artifact_key)
        round_tripped = ParseResult.from_json(result.to_json())
        self.assertEqual(result.result_sha256, round_tripped.result_sha256)
        round_trip_receipt = next(
            item for item in round_tripped.artifacts
            if item.artifact_role == "mineru_middle_json"
        )
        self.assertEqual(middle, round_trip_receipt)
        with self.assertRaises(TypeError):
            round_trip_receipt.metadata["version"] = "mutated"
        with self.assertRaises(TypeError):
            archive.member_sha256s["doc/debug/trace.txt"] = "0" * 64

        submit = transport.requests[1]
        self.assertEqual(("POST", "https://mineru.example/tasks"), submit[:2])
        self.assertIn(b'name="effort"\r\n\r\nmedium', submit[3])
        self.assertIn(b'name="return_content_list"\r\n\r\ntrue', submit[3])
        self.assertIn(b'name="start_page_id"\r\n\r\n0', submit[3])
        self.assertNotIn(b"s3://citation-only/private.pdf", submit[3])
        self.assertEqual(loader.config_sha256, config.revision_sha256)

    def test_missing_middle_artifact_is_an_explicit_non_emission_receipt(self):
        request = request_fixture()
        transport = FakeTransport(
            responses_for_success(result_zip(emit_middle=False))
        )
        _config, _loader, adapter = service_adapter(request, transport)

        result = adapter.parse(request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual(2, len(result.artifacts))
        receipt = next(
            item for item in result.artifacts
            if item.artifact_role == "mineru_middle_json"
        )
        self.assertEqual(
            ParserArtifactAvailability.NOT_EMITTED, receipt.availability
        )
        self.assertIsNone(receipt.uri)
        self.assertIsNone(receipt.sha256)
        self.assertIsNone(receipt.size_bytes)
        self.assertEqual(
            receipt.receipt_id,
            result.observations[0].metadata["mineru_service"][
                "middle_artifact_receipt_id"
            ],
        )

    def test_empty_content_list_keeps_result_scoped_middle_receipt(self):
        request = request_fixture()
        transport = FakeTransport(
            responses_for_success(result_zip(empty_blocks=True))
        )
        _config, loader, adapter = service_adapter(request, transport)

        loaded = loader.load(request)

        self.assertEqual((), loaded.blocks)
        self.assertEqual(2, len(loaded.artifact_receipts))
        middle = next(
            item for item in loaded.artifact_receipts
            if item.artifact_role == "mineru_middle_json"
        )
        self.assertEqual(
            ParserArtifactAvailability.EMITTED,
            middle.availability,
        )
        self.assertEqual(0, len(loaded))
        self.assertEqual((), loaded[:])

        # A fresh transport proves the normal adapter/result path also retains
        # the receipt without manufacturing an observation.
        _config, _loader, adapter = service_adapter(
            request,
            FakeTransport(responses_for_success(result_zip(empty_blocks=True))),
        )
        result = adapter.parse(request)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual((), result.observations)
        self.assertEqual(2, len(result.artifacts))
        result_middle = next(
            item for item in result.artifacts
            if item.artifact_role == "mineru_middle_json"
        )
        self.assertEqual(
            middle.sha256, result_middle.sha256
        )

    def test_effort_changes_request_and_service_revision_identity(self):
        medium = request_fixture(effort=MinerUEffort.MEDIUM)
        high = request_fixture(effort=MinerUEffort.HIGH)
        medium_config = MinerUServiceConfig(
            "https://mineru.example", MinerUBackend.HYBRID_ENGINE, MinerUEffort.MEDIUM
        )
        high_config = MinerUServiceConfig(
            "https://mineru.example", MinerUBackend.HYBRID_ENGINE, MinerUEffort.HIGH
        )
        self.assertNotEqual(medium.request_id, high.request_id)
        self.assertNotEqual(medium_config.revision_sha256, high_config.revision_sha256)

    def test_middle_resource_bounds_are_validated_and_part_of_identity(self):
        base = MinerUServiceConfig(
            "https://mineru.example", MinerUBackend.HYBRID_ENGINE, MinerUEffort.MEDIUM
        )
        bounded = MinerUServiceConfig(
            "https://mineru.example",
            MinerUBackend.HYBRID_ENGINE,
            MinerUEffort.MEDIUM,
            max_middle_json_bytes=1234,
            max_middle_json_depth=17,
            max_middle_json_nodes=4567,
        )
        self.assertEqual(1234, bounded.identity()["bounds"]["max_middle_json_bytes"])
        self.assertEqual(17, bounded.identity()["bounds"]["max_middle_json_depth"])
        self.assertEqual(4567, bounded.identity()["bounds"]["max_middle_json_nodes"])
        self.assertNotEqual(base.revision_sha256, bounded.revision_sha256)
        for field_name in (
            "max_middle_json_bytes",
            "max_middle_json_depth",
            "max_middle_json_nodes",
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                MinerUServiceConfig(
                    "https://mineru.example",
                    MinerUBackend.HYBRID_ENGINE,
                    MinerUEffort.MEDIUM,
                    **{field_name: 0},
                )

    def test_task_count_cannot_exceed_parse_result_artifact_receipt_capacity(self):
        with self.assertRaisesRegex(
            ValueError,
            f"between 1 and {MAX_MINERU_TASKS_PER_PARSE_RESULT}",
        ):
            MinerUServiceConfig(
                "https://mineru.example",
                MinerUBackend.HYBRID_ENGINE,
                MinerUEffort.MEDIUM,
                max_tasks_per_request=MAX_MINERU_TASKS_PER_PARSE_RESULT + 1,
            )

        bounded = MinerUServiceConfig(
            "https://mineru.example",
            MinerUBackend.HYBRID_ENGINE,
            MinerUEffort.MEDIUM,
            max_tasks_per_request=MAX_MINERU_TASKS_PER_PARSE_RESULT,
        )
        self.assertEqual(
            MAX_MINERU_TASKS_PER_PARSE_RESULT,
            bounded.max_tasks_per_request,
        )


class MinerUServiceFailureTests(unittest.TestCase):
    def test_second_task_failure_retains_first_task_published_receipts(self):
        request = request_fixture()
        first_archive = result_zip(
            empty_blocks=True,
            middle_payload=middle_fixture(pages=(0,)),
        )
        responses = responses_for_success(first_archive)
        responses.append(json_response(503, {"detail": "temporarily unavailable"}))
        _config, _loader, adapter = service_adapter(
            request,
            FakeTransport(responses),
            config_overrides={"max_pages_per_task": 1},
        )

        result = adapter.parse(request)

        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(2, len(result.artifacts))
        self.assertEqual(
            {"task-1"},
            {receipt.producer_task_id for receipt in result.artifacts},
        )
        primary = next(
            receipt for receipt in result.artifacts
            if receipt.artifact_role == "mineru_result_archive"
        )
        self.assertEqual(0, primary.metadata["first_page"])
        self.assertEqual(0, primary.metadata["last_page"])
        self.assertEqual(
            ErrorClassification.RETRYABLE,
            result.issues[0].classification,
        )

    def test_missing_effort_becomes_permanent_contract_issue(self):
        request = request_fixture(effort=None)
        transport = FakeTransport([])
        _config, _loader, adapter = service_adapter(request, transport)
        result = adapter.parse(request)
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
        self.assertEqual(ErrorClassification.PERMANENT, result.issues[0].classification)
        self.assertFalse(transport.requests)

    def test_lost_process_local_task_becomes_retryable_dependency_issue(self):
        request = request_fixture()
        responses = responses_for_success()[:2] + [json_response(404, {"detail": "gone"})]
        transport = FakeTransport(responses)
        _config, _loader, adapter = service_adapter(request, transport)
        result = adapter.parse(request)
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(ParseErrorCode.DEPENDENCY_UNAVAILABLE, result.issues[0].code)
        self.assertEqual(ErrorClassification.RETRYABLE, result.issues[0].classification)
        self.assertEqual(
            "process_local_non_durable", result.issues[0].details["vendor_task_state"]
        )

    def test_protocol_drift_and_task_identity_mismatch_fail_permanently(self):
        request = request_fixture()
        bad_health = FakeTransport([
            json_response(200, {
                "status": "healthy", "protocol_version": 999,
                "processing_window_size": 64, "max_concurrent_requests": 3,
            })
        ])
        _config, _loader, adapter = service_adapter(request, bad_health)
        result = adapter.parse(request)
        self.assertEqual(ErrorClassification.PERMANENT, result.issues[0].classification)

        wrong_origin = responses_for_success()
        payload = {
            "task_id": "task-1",
            "status_url": "https://attacker.example/tasks/task-1",
            "result_url": "https://mineru.example/tasks/task-1/result",
            "file_names": ["doc.pdf"],
        }
        wrong_origin[1] = json_response(202, payload)
        transport = FakeTransport(wrong_origin)
        _config, _loader, adapter = service_adapter(request, transport)
        result = adapter.parse(request)
        self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)

    def test_unsafe_or_ambiguous_result_archive_is_rejected(self):
        request = request_fixture()
        for archive in (result_zip(unsafe=True), result_zip(two_lists=True)):
            with self.subTest(size=len(archive)):
                transport = FakeTransport(responses_for_success(archive))
                _config, _loader, adapter = service_adapter(request, transport)
                result = adapter.parse(request)
                self.assertEqual(ParseStatus.FAILED, result.status)
                self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
                self.assertEqual(ErrorClassification.PERMANENT, result.issues[0].classification)

    def test_multiple_malformed_or_page_mismatched_middle_artifacts_fail_closed(self):
        request = request_fixture()
        archives = (
            result_zip(two_middles=True),
            result_zip(middle_payload=b"{"),
            result_zip(middle_payload=[]),
            result_zip(middle_payload=middle_fixture(pages=(0, 7))),
        )
        for archive in archives:
            with self.subTest(archive_sha256=hashlib.sha256(archive).hexdigest()):
                transport = FakeTransport(responses_for_success(archive))
                _config, _loader, adapter = service_adapter(request, transport)
                result = adapter.parse(request)
                self.assertEqual(ParseStatus.FAILED, result.status)
                self.assertEqual(
                    ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code
                )
                self.assertEqual(
                    ErrorClassification.PERMANENT,
                    result.issues[0].classification,
                )

    def test_middle_byte_depth_and_node_limits_fail_as_resource_exhaustion(self):
        request = request_fixture()
        cases = (
            {"max_middle_json_bytes": 16},
            {"max_middle_json_depth": 3},
            {"max_middle_json_nodes": 10},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                transport = FakeTransport(responses_for_success())
                _config, _loader, adapter = service_adapter(
                    request, transport, config_overrides=overrides
                )
                result = adapter.parse(request)
                self.assertEqual(ParseStatus.FAILED, result.status)
                self.assertEqual(
                    ParseErrorCode.RESOURCE_EXHAUSTED, result.issues[0].code
                )
                self.assertEqual(
                    ErrorClassification.PERMANENT,
                    result.issues[0].classification,
                )

    def test_publisher_digest_and_referenced_member_coverage_are_verified(self):
        request = request_fixture()

        def wrong_digest(archive):
            published = publisher(archive)
            return PublishedMinerUArchive(
                archive_uri=published.archive_uri,
                archive_sha256="0" * 64,
                member_uris=published.member_uris,
            )

        def missing_referenced_asset(archive):
            published = publisher(archive)
            return PublishedMinerUArchive(
                archive_uri=published.archive_uri,
                archive_sha256=archive.archive_sha256,
                member_uris={
                    member: uri
                    for member, uri in published.member_uris.items()
                    if member != "doc/images/figure.png"
                },
            )

        def unknown_member(archive):
            published = publisher(archive)
            uris = dict(published.member_uris)
            uris["doc/not-in-archive.png"] = "artifact://mineru/not-in-archive"
            return PublishedMinerUArchive(
                archive_uri=published.archive_uri,
                archive_sha256=archive.archive_sha256,
                member_uris=uris,
            )

        def missing_middle_artifact(archive):
            published = publisher(archive)
            return PublishedMinerUArchive(
                archive_uri=published.archive_uri,
                archive_sha256=archive.archive_sha256,
                member_uris={
                    member: uri
                    for member, uri in published.member_uris.items()
                    if member != "doc/doc_middle.json"
                },
            )

        for bad_publisher in (
            wrong_digest,
            missing_referenced_asset,
            unknown_member,
            missing_middle_artifact,
        ):
            with self.subTest(publisher=bad_publisher.__name__):
                transport = FakeTransport(responses_for_success())
                _config, _loader, adapter = service_adapter(
                    request, transport, artifact_publisher=bad_publisher
                )
                result = adapter.parse(request)
                self.assertEqual(ParseStatus.FAILED, result.status)
                self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
                self.assertEqual(
                    ErrorClassification.PERMANENT,
                    result.issues[0].classification,
                )

    def test_verified_bytes_are_required_and_never_inferred_from_source_uri(self):
        request = request_fixture()
        config = MinerUServiceConfig(
            "https://mineru.example", MinerUBackend.HYBRID_ENGINE, MinerUEffort.MEDIUM
        )
        client = MinerUAPIClient(config, transport=FakeTransport([]), owns_transport=False)
        loader = MinerUServiceLoader(
            client,
            source_loader=lambda received: source_fixture(received, data=None),
            artifact_publisher=publisher,
        )
        adapter = MinerUNormalizedAdapter(
            loader,
            parser_version="3.x-unqualified",
            model_version="unqualified",
            config=config.identity(),
        )
        result = adapter.parse(request)
        self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
        self.assertEqual(ErrorClassification.PERMANENT, result.issues[0].classification)

    def test_remote_plain_http_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            MinerUServiceConfig(
                "http://mineru.example", MinerUBackend.PIPELINE, MinerUEffort.MEDIUM
            )
        local = MinerUServiceConfig(
            "http://127.0.0.1:8002", MinerUBackend.PIPELINE, MinerUEffort.MEDIUM
        )
        self.assertEqual("http://127.0.0.1:8002", local.base_url)

    def test_module_imports_no_vendor_or_ml_runtime(self):
        script = """
import json, sys
import docarag.services.mineru
forbidden = ('mineru', 'torch', 'transformers', 'vllm', 'httpx')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], text=True, capture_output=True, check=True
        )
        self.assertEqual([], json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
