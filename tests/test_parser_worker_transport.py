from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

from document_services.domain import (
    AcceleratorKind,
    BlockType,
    DocumentRef,
    PageRef,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserCapabilityManifest,
    ParserModality,
)
from document_services.ports import ParserRegistry
from document_services.worker_transport import (
    HTTPParserAdapter,
    HTTPParserClientConfig,
    PARSER_WORKER_PROTOCOL_VERSION,
    ParserWorkerHTTPServer,
    ParserWorkerServerConfig,
    RemoteWorkerError,
    TransportClosedError,
    TransportConfigurationError,
    TransportConnectionError,
    TransportRequestTooLargeError,
    TransportResponseTooLargeError,
    TransportTimeoutError,
    WorkerIdentityError,
    WorkerState,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_ROOT = REPOSITORY_ROOT / "document_services" / "worker_transport"


def manifest_fixture() -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name="reference-http-parser",
        adapter_version="1.2.3",
        parser_name="reference-parser",
        parser_version="4.5.6",
        model_version="test-model-v1",
        supported_mime_types=("application/pdf",),
        modalities=(ParserModality.NATIVE_TEXT,),
        block_types=(BlockType.PARAGRAPH,),
        accelerator=AcceleratorKind.CPU,
        supports_page_selection=True,
        supports_batch=True,
        max_pages_per_request=8,
        deterministic=True,
    )


def request_fixture(*, padding: int = 0) -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256="a" * 64,
        mime_type="application/pdf",
        source_uri="artifact://source/document.pdf",
        display_name="document.pdf" + ("x" * padding),
        page_count=1,
    )
    return ParseRequest(
        document=document,
        pages=(
            PageRef(
                document_version_id=document.document_version_id,
                page_index=0,
                width=612,
                height=792,
            ),
        ),
        run_id="run-transport-test",
        trace_id="trace-transport-test",
        idempotency_key="transport-test-page-0",
    )


class ReferenceAdapter:
    def __init__(self) -> None:
        self.value = manifest_fixture()
        self.mode = "success"
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def manifest(self) -> ParserCapabilityManifest:
        return self.value

    def parse(self, request: ParseRequest) -> ParseResult:
        self.calls += 1
        if self.mode == "raise-secret":
            raise RuntimeError("vendor secret: token=do-not-disclose")
        if self.mode == "block":
            self.started.set()
            if not self.release.wait(2):
                raise RuntimeError("test did not release parser")
        if self.mode == "wrong-request":
            request_id = "parse_bound_to_someone_else"
        else:
            request_id = request.request_id
        metrics = {"padding": "z" * 5000} if self.mode == "large-result" else {}
        return ParseResult(
            request_id=request_id,
            adapter_name=self.value.adapter_name,
            adapter_version=self.value.adapter_version,
            status=ParseStatus.SUCCEEDED,
            metrics=metrics,
        )


def client_for(
    worker: ParserWorkerHTTPServer,
    **overrides: object,
) -> HTTPParserAdapter:
    values: dict[str, object] = {
        "base_url": worker.base_url,
        "connect_timeout_seconds": 1,
        "operation_timeout_seconds": 1,
        "expected_adapter_name": worker.manifest.adapter_name,
        "expected_adapter_version": worker.manifest.adapter_version,
        "expected_manifest_sha256": worker.manifest.manifest_sha256,
    }
    values.update(overrides)
    return HTTPParserAdapter(HTTPParserClientConfig(**values))  # type: ignore[arg-type]


class ParserWorkerHappyPathTests(unittest.TestCase):
    def test_remote_adapter_conforms_to_registry_and_round_trips_contracts(self):
        adapter = ReferenceAdapter()
        with ParserWorkerHTTPServer(adapter) as worker:
            with client_for(worker) as client:
                health = client.health(require_ready=True)
                self.assertEqual(health.state, WorkerState.READY)
                self.assertTrue(health.accepting_requests)
                self.assertEqual(health.adapter_name, adapter.value.adapter_name)
                self.assertEqual(client.manifest(), adapter.value)

                request = request_fixture()
                result = ParserRegistry((client,)).parse(
                    adapter.value.adapter_name, request
                )

        self.assertEqual(result.request_id, request.request_id)
        self.assertEqual(result.status, ParseStatus.SUCCEEDED)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(worker.state, WorkerState.CLOSED)

    def test_context_managers_close_client_and_worker_idempotently(self):
        worker = ParserWorkerHTTPServer(ReferenceAdapter()).start()
        client = client_for(worker)
        client.close()
        client.close()
        with self.assertRaises(TransportClosedError):
            client.health()
        worker.close()
        worker.close()
        self.assertEqual(worker.state, WorkerState.CLOSED)

    def test_capabilities_are_immutable_even_if_injected_adapter_mutates(self):
        adapter = ReferenceAdapter()
        worker = ParserWorkerHTTPServer(adapter)
        original = worker.manifest
        adapter.value = ParserCapabilityManifest(
            adapter_name="mutated-after-construction",
            adapter_version="9",
            parser_name="mutated",
            parser_version="9",
            model_version="9",
            supported_mime_types=("application/pdf",),
            modalities=(ParserModality.OCR_TEXT,),
            block_types=(BlockType.PARAGRAPH,),
        )
        with worker:
            client = client_for(worker)
            self.assertEqual(client.manifest(), original)


class ParserWorkerFailureTests(unittest.TestCase):
    def test_server_rejects_result_bound_to_another_request(self):
        adapter = ReferenceAdapter()
        adapter.mode = "wrong-request"
        with ParserWorkerHTTPServer(adapter) as worker:
            with self.assertRaises(RemoteWorkerError) as caught:
                client_for(worker).parse(request_fixture())
        self.assertEqual(caught.exception.status, 502)
        self.assertEqual(caught.exception.code, "contract_violation")
        self.assertFalse(caught.exception.retryable)

    def test_unexpected_vendor_failure_is_sanitized(self):
        adapter = ReferenceAdapter()
        adapter.mode = "raise-secret"
        with ParserWorkerHTTPServer(adapter) as worker:
            with self.assertRaises(RemoteWorkerError) as caught:
                client_for(worker).parse(request_fixture())
        self.assertEqual(caught.exception.code, "parser_failure")
        self.assertNotIn("do-not-disclose", str(caught.exception))
        self.assertNotIn("token", str(caught.exception))

    def test_operation_deadline_is_enforced_without_automatic_retry(self):
        adapter = ReferenceAdapter()
        adapter.mode = "block"
        with ParserWorkerHTTPServer(adapter) as worker:
            client = client_for(worker, operation_timeout_seconds=0.03)
            client.manifest()
            with self.assertRaises(TransportTimeoutError):
                client.parse(request_fixture())
            self.assertEqual(adapter.calls, 1)
            adapter.release.set()
            time.sleep(0.03)

    def test_admission_limit_returns_typed_retryable_busy_error(self):
        adapter = ReferenceAdapter()
        adapter.mode = "block"
        with ParserWorkerHTTPServer(
            adapter, ParserWorkerServerConfig(max_in_flight=1)
        ) as worker:
            client = client_for(worker)
            client.manifest()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.parse, request_fixture())
                self.assertTrue(adapter.started.wait(1))
                health = client.health(require_ready=True)
                self.assertEqual(health.in_flight, 1)
                with self.assertRaises(RemoteWorkerError) as caught:
                    client.parse(request_fixture())
                self.assertEqual(caught.exception.status, 429)
                self.assertEqual(caught.exception.code, "worker_busy")
                self.assertTrue(caught.exception.retryable)
                adapter.release.set()
                self.assertEqual(future.result().status, ParseStatus.SUCCEEDED)

    def test_local_request_limit_fails_before_worker_execution(self):
        adapter = ReferenceAdapter()
        with ParserWorkerHTTPServer(adapter) as worker:
            client = client_for(worker, max_request_bytes=100)
            client.manifest()
            with self.assertRaises(TransportRequestTooLargeError):
                client.parse(request_fixture())
        self.assertEqual(adapter.calls, 0)

    def test_server_request_limit_is_enforced(self):
        adapter = ReferenceAdapter()
        with ParserWorkerHTTPServer(
            adapter, ParserWorkerServerConfig(max_request_bytes=1024)
        ) as worker:
            client = client_for(worker)
            client.manifest()
            with self.assertRaises(RemoteWorkerError) as caught:
                client.parse(request_fixture(padding=3000))
        self.assertEqual(caught.exception.status, 413)
        self.assertEqual(caught.exception.code, "request_too_large")
        self.assertEqual(adapter.calls, 0)

    def test_client_response_limit_is_enforced_before_json_decode(self):
        with ParserWorkerHTTPServer(ReferenceAdapter()) as worker:
            client = client_for(worker, max_response_bytes=100)
            with self.assertRaises(TransportResponseTooLargeError):
                client.health()

    def test_server_response_limit_prevents_unbounded_result_transfer(self):
        adapter = ReferenceAdapter()
        adapter.mode = "large-result"
        with ParserWorkerHTTPServer(
            adapter, ParserWorkerServerConfig(max_response_bytes=1024)
        ) as worker:
            with self.assertRaises(RemoteWorkerError) as caught:
                client_for(worker).parse(request_fixture())
        self.assertEqual(caught.exception.status, 507)
        self.assertEqual(caught.exception.code, "response_too_large")

    def test_closed_worker_is_not_silently_restarted(self):
        worker = ParserWorkerHTTPServer(ReferenceAdapter()).start()
        client = client_for(worker)
        client.health()
        worker.close()
        with self.assertRaises(TransportConnectionError):
            client.health()


class ParserWorkerSecurityTests(unittest.TestCase):
    def test_authentication_applies_to_health_capability_and_parse(self):
        adapter = ReferenceAdapter()
        config = ParserWorkerServerConfig(bearer_token="worker-secret")
        with ParserWorkerHTTPServer(adapter, config) as worker:
            unauthenticated = client_for(worker)
            for operation in (
                unauthenticated.health,
                unauthenticated.manifest,
                lambda: unauthenticated.parse(request_fixture()),
            ):
                with self.subTest(operation=operation):
                    with self.assertRaises(RemoteWorkerError) as caught:
                        operation()
                    self.assertEqual(caught.exception.status, 401)
                    self.assertNotIn("worker-secret", str(caught.exception))

            authenticated = client_for(worker, bearer_token="worker-secret")
            self.assertEqual(authenticated.parse(request_fixture()).status, ParseStatus.SUCCEEDED)

    def test_remote_plain_http_and_non_loopback_bind_require_explicit_opt_in(self):
        with self.assertRaises(TransportConfigurationError):
            HTTPParserClientConfig(base_url="http://parser.internal:8080")
        config = HTTPParserClientConfig(
            base_url="http://parser.internal:8080", allow_insecure_http=True
        )
        self.assertTrue(config.allow_insecure_http)
        with self.assertRaises(TransportConfigurationError):
            ParserWorkerServerConfig(host="0.0.0.0")
        with self.assertRaises(TransportConfigurationError):
            ParserWorkerServerConfig(
                host="0.0.0.0", allow_non_loopback_bind=True
            )

    def test_url_userinfo_paths_queries_and_non_http_schemes_are_rejected(self):
        invalid = (
            "file:///tmp/worker.sock",
            "http://user:secret@localhost:9000",
            "http://localhost:9000/api",
            "http://localhost:9000?target=other",
            "http://localhost:9000/#fragment",
        )
        for base_url in invalid:
            with self.subTest(base_url=base_url):
                with self.assertRaises(TransportConfigurationError):
                    HTTPParserClientConfig(base_url=base_url)

    def test_identity_pinning_rejects_wrong_worker(self):
        with ParserWorkerHTTPServer(ReferenceAdapter()) as worker:
            client = client_for(worker, expected_adapter_name="different-parser")
            with self.assertRaises(WorkerIdentityError):
                client.manifest()

    def test_wrong_protocol_version_and_unknown_fields_are_rejected(self):
        with ParserWorkerHTTPServer(ReferenceAdapter()) as worker:
            connection = http.client.HTTPConnection(*worker.address, timeout=1)
            payload = json.dumps(
                {
                    "protocol_version": "future-protocol.v99",
                    "manifest_sha256": worker.manifest.manifest_sha256,
                    "request": request_fixture().to_dict(),
                    "unexpected": "must-not-be-ignored",
                }
            ).encode()
            connection.request(
                "POST",
                "/v1/parse",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = json.loads(response.read())
            connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["protocol_version"], PARSER_WORKER_PROTOCOL_VERSION)
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_query_string_cannot_select_or_modify_an_endpoint(self):
        with ParserWorkerHTTPServer(ReferenceAdapter()) as worker:
            connection = http.client.HTTPConnection(*worker.address, timeout=1)
            connection.request("GET", "/v1/health?adapter=other")
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
        self.assertEqual(response.status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")


class ParserWorkerImportBoundaryTests(unittest.TestCase):
    def test_controller_transport_import_does_not_load_vendor_or_bookrag_runtimes(self):
        script = """
import json
import sys
import document_services.worker_transport

forbidden = (
    "retrieval_engine", "docling", "mineru", "paddle", "paddleocr", "psycopg",
    "qdrant_client", "spacy", "torch", "transformers", "vllm"
)
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
print(json.dumps(loaded))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_transport_has_no_shell_execution_or_hidden_parser_registry(self):
        forbidden_calls = {"system", "popen", "run", "Popen", "call", "check_call"}
        for path in sorted(TRANSPORT_ROOT.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            with self.subTest(module=path.name):
                call_names = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                }
                direct_calls = {
                    node.func.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertFalse((call_names | direct_calls) & forbidden_calls)
                self.assertNotIn("ParserRegistry", source)
                imports = {
                    alias.name.split(".", 1)[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                self.assertFalse(
                    imports
                    & {
                        "retrieval_engine",
                        "docling",
                        "mineru",
                        "paddle",
                        "paddleocr",
                        "torch",
                        "transformers",
                        "vllm",
                    }
                )


if __name__ == "__main__":
    unittest.main()
