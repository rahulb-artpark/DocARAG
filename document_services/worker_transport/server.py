"""Explicit single-adapter HTTP server for isolated vendor environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import socket
import threading
import uuid

from document_services.domain.base import ContractValidationError, dump_json
from document_services.domain.parsing import (
    ErrorClassification,
    ParseRequest,
    ParseResult,
    ParserCapabilityManifest,
    ParserExecutionError,
)
from document_services.ports.parser import ParserAdapter, validate_result_binding

from document_services.worker_transport.errors import (
    TransportClosedError,
    TransportConfigurationError,
)
from document_services.worker_transport.protocol import (
    CAPABILITIES_PATH,
    HEALTH_PATH,
    PARSE_PATH,
    PARSER_WORKER_PROTOCOL_VERSION,
    WorkerHealth,
    WorkerState,
    reject_unknown_fields,
    require_protocol_version,
)


_MIB = 1024 * 1024
_MAX_CONFIGURED_BODY_BYTES = 1024 * _MIB


def _byte_limit(value: object, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1024 <= value <= _MAX_CONFIGURED_BODY_BYTES
    ):
        raise TransportConfigurationError(
            f"{field_name} must be between 1024 and {_MAX_CONFIGURED_BODY_BYTES} bytes"
        )
    return value


def _positive_timeout(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise TransportConfigurationError(f"{field_name} must be finite and positive")
    return float(value)


def _loopback_bind(host: str) -> bool:
    return host.lower() == "localhost" or host in {"127.0.0.1", "::1"}


@dataclass(frozen=True, slots=True)
class ParserWorkerServerConfig:
    """Worker admission and network policy.

    The built-in server intentionally has no TLS termination.  Bind it to
    loopback (the default), a private process namespace, or place it behind an
    authenticated TLS proxy.  A non-loopback bind requires an explicit opt-in
    and bearer token.
    """

    host: str = "127.0.0.1"
    port: int = 0
    max_request_bytes: int = 16 * _MIB
    max_response_bytes: int = 64 * _MIB
    max_in_flight: int = 1
    request_read_timeout_seconds: float = 30.0
    bearer_token: str | None = field(default=None, repr=False)
    allow_non_loopback_bind: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise TransportConfigurationError("host must be a non-empty string")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 0 <= self.port <= 65535
        ):
            raise TransportConfigurationError("port must be between 0 and 65535")
        object.__setattr__(
            self,
            "max_request_bytes",
            _byte_limit(self.max_request_bytes, "max_request_bytes"),
        )
        object.__setattr__(
            self,
            "max_response_bytes",
            _byte_limit(self.max_response_bytes, "max_response_bytes"),
        )
        if (
            isinstance(self.max_in_flight, bool)
            or not isinstance(self.max_in_flight, int)
            or not 1 <= self.max_in_flight <= 1024
        ):
            raise TransportConfigurationError("max_in_flight must be between 1 and 1024")
        object.__setattr__(
            self,
            "request_read_timeout_seconds",
            _positive_timeout(
                self.request_read_timeout_seconds, "request_read_timeout_seconds"
            ),
        )
        if not isinstance(self.allow_non_loopback_bind, bool):
            raise TransportConfigurationError("allow_non_loopback_bind must be a boolean")
        if self.bearer_token is not None and (
            not isinstance(self.bearer_token, str) or not self.bearer_token
        ):
            raise TransportConfigurationError("bearer_token must be non-empty when set")
        if not _loopback_bind(self.host):
            if not self.allow_non_loopback_bind:
                raise TransportConfigurationError(
                    "non-loopback worker bind requires allow_non_loopback_bind=True"
                )
            if self.bearer_token is None:
                raise TransportConfigurationError(
                    "non-loopback worker bind requires a bearer_token"
                )


class _RequestRejected(Exception):
    def __init__(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = request_id


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False
    request_queue_size = 32


class ParserWorkerHTTPServer:
    """Lifecycle owner for one explicitly injected parser adapter.

    A worker serves one immutable capability identity for its whole lifetime.
    Replacing a model or adapter means starting a new worker.  This makes
    routing caches and provenance deterministic and avoids a hidden registry.
    """

    def __init__(
        self,
        adapter: ParserAdapter,
        config: ParserWorkerServerConfig | None = None,
    ) -> None:
        if not isinstance(adapter, ParserAdapter):
            raise TypeError("adapter must implement ParserAdapter")
        manifest = adapter.manifest()
        if not isinstance(manifest, ParserCapabilityManifest):
            raise ContractValidationError(
                "adapter.manifest() must return ParserCapabilityManifest"
            )
        self._adapter = adapter
        self._manifest = manifest
        self._config = config or ParserWorkerServerConfig()
        if not isinstance(self._config, ParserWorkerServerConfig):
            raise TypeError("config must be ParserWorkerServerConfig")
        self._worker_instance_id = f"worker_{uuid.uuid4().hex}"
        self._state = WorkerState.STARTING
        self._in_flight = 0
        self._state_lock = threading.RLock()
        self._admission = threading.BoundedSemaphore(self._config.max_in_flight)
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._serve_thread: threading.Thread | None = None
        self._httpd = _BoundedThreadingHTTPServer(
            (self._config.host, self._config.port), _ParserWorkerRequestHandler
        )
        self._httpd.worker_owner = self  # type: ignore[attr-defined]

    @property
    def config(self) -> ParserWorkerServerConfig:
        return self._config

    @property
    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1" if host == "0.0.0.0" else "::1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{port}"

    @property
    def state(self) -> WorkerState:
        with self._state_lock:
            return self._state

    def health(self) -> WorkerHealth:
        with self._state_lock:
            state = self._state
            in_flight = self._in_flight
        return WorkerHealth(
            worker_instance_id=self._worker_instance_id,
            state=state,
            accepting_requests=state == WorkerState.READY,
            in_flight=in_flight,
            max_in_flight=self._config.max_in_flight,
            adapter_name=self._manifest.adapter_name,
            adapter_version=self._manifest.adapter_version,
            manifest_sha256=self._manifest.manifest_sha256,
        )

    def start(self, *, startup_timeout_seconds: float = 5.0) -> "ParserWorkerHTTPServer":
        """Start a persistent worker thread, primarily for embedding and tests."""

        _positive_timeout(startup_timeout_seconds, "startup_timeout_seconds")
        with self._state_lock:
            if self._state == WorkerState.CLOSED:
                raise TransportClosedError("parser-worker server is closed")
            if self._serve_thread is not None:
                raise RuntimeError("parser-worker server has already been started")
            thread = threading.Thread(
                target=self.serve_forever,
                name=f"docarag-parser-worker-{self.address[1]}",
                daemon=True,
            )
            self._serve_thread = thread
            thread.start()
        if not self._ready.wait(float(startup_timeout_seconds)):
            self.close()
            raise TimeoutError("parser-worker server did not become ready")
        return self

    def serve_forever(self) -> None:
        """Serve until ``close`` is called from another thread or process signal."""

        with self._state_lock:
            if self._state == WorkerState.CLOSED:
                raise TransportClosedError("parser-worker server is closed")
            if self._state == WorkerState.READY:
                raise RuntimeError("parser-worker server is already running")
            self._state = WorkerState.READY
            self._ready.set()
        try:
            self._httpd.serve_forever(poll_interval=0.1)
        finally:
            with self._state_lock:
                if self._state != WorkerState.CLOSED:
                    self._state = WorkerState.STOPPING
            self._ready.clear()

    def close(self, *, join_timeout_seconds: float = 5.0) -> None:
        """Stop accepting work and release the listening socket idempotently.

        Python cannot cancel a vendor call already executing in a thread.  The
        intended hard-cancellation boundary is the worker *process*; request
        threads are daemonized so socket cleanup itself remains bounded.
        """

        _positive_timeout(join_timeout_seconds, "join_timeout_seconds")
        with self._state_lock:
            if self._state == WorkerState.CLOSED:
                return
            was_running = self._state == WorkerState.READY
            self._state = WorkerState.STOPPING
            thread = self._serve_thread
        if was_running:
            if thread is threading.current_thread():
                raise RuntimeError("close must be called outside the serve_forever thread")
            self._httpd.shutdown()
        self._httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(float(join_timeout_seconds))
        with self._state_lock:
            self._state = WorkerState.CLOSED
        self._ready.clear()
        self._closed.set()

    def __enter__(self) -> "ParserWorkerHTTPServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _try_enter_parse(self) -> bool:
        with self._state_lock:
            if self._state != WorkerState.READY:
                return False
        if not self._admission.acquire(blocking=False):
            return False
        with self._state_lock:
            if self._state != WorkerState.READY:
                self._admission.release()
                return False
            self._in_flight += 1
        return True

    def _leave_parse(self) -> None:
        with self._state_lock:
            self._in_flight -= 1
        self._admission.release()


class _ParserWorkerRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DocARAGParserWorker"
    sys_version = ""

    @property
    def owner(self) -> ParserWorkerHTTPServer:
        return self.server.worker_owner  # type: ignore[attr-defined, no-any-return]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.owner.config.request_read_timeout_seconds)

    def log_message(self, _format: str, *_args: object) -> None:
        # The embedding vendor environment owns structured/redacted logging.
        return

    def _authenticated(self) -> bool:
        expected = self.owner.config.bearer_token
        if expected is None:
            return True
        actual = self.headers.get("Authorization", "")
        return hmac.compare_digest(actual, f"Bearer {expected}")

    def _send_payload(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = dump_json(payload).encode("utf-8")
        if len(body) > self.owner.config.max_response_bytes:
            status = HTTPStatus.INSUFFICIENT_STORAGE
            body = dump_json(
                {
                    "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
                    "error": {
                        "code": "response_too_large",
                        "message": "serialized parser response exceeds worker limit",
                        "retryable": False,
                        "request_id": None,
                    },
                }
            ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
        self.close_connection = True

    def _send_rejection(self, rejection: _RequestRejected) -> None:
        self._send_payload(
            rejection.status,
            {
                "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
                "error": {
                    "code": rejection.code,
                    "message": rejection.message,
                    "retryable": rejection.retryable,
                    "request_id": rejection.request_id,
                },
            },
        )

    def _authorize(self) -> None:
        if not self._authenticated():
            raise _RequestRejected(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "worker authentication failed",
            )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            self._authorize()
            if self.path == HEALTH_PATH:
                self._send_payload(HTTPStatus.OK, self.owner.health().to_dict())
                return
            if self.path == CAPABILITIES_PATH:
                self._send_payload(
                    HTTPStatus.OK,
                    {
                        "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
                        "manifest": self.owner.manifest.to_dict(),
                    },
                )
                return
            raise _RequestRejected(
                HTTPStatus.NOT_FOUND, "not_found", "worker endpoint does not exist"
            )
        except _RequestRejected as rejection:
            self._send_rejection(rejection)

    def _read_json_body(self) -> Mapping[str, object]:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding is not None:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Transfer-Encoding is not supported",
            )
        content_lengths = self.headers.get_all("Content-Length", [])
        if len(content_lengths) != 1:
            raise _RequestRejected(
                HTTPStatus.LENGTH_REQUIRED,
                "invalid_request",
                "exactly one Content-Length header is required",
            )
        try:
            size = int(content_lengths[0], 10)
        except ValueError as exc:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length is invalid",
            ) from exc
        if size < 0:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length is invalid",
            )
        if size > self.owner.config.max_request_bytes:
            raise _RequestRejected(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "serialized parser request exceeds worker limit",
            )
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise _RequestRejected(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "invalid_request",
                "Content-Type must be application/json",
            )
        try:
            body = self.rfile.read(size)
        except socket.timeout as exc:
            raise _RequestRejected(
                HTTPStatus.REQUEST_TIMEOUT,
                "request_timeout",
                "request body was not received before the deadline",
                retryable=True,
            ) from exc
        if len(body) != size:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request body length does not match Content-Length",
            )
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST, "invalid_request", "request body is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request body must be a JSON object",
            )
        return payload

    @staticmethod
    def _validate_supported(
        manifest: ParserCapabilityManifest, request: ParseRequest
    ) -> None:
        if request.document.mime_type not in manifest.supported_mime_types:
            raise ContractValidationError("worker does not support the requested MIME type")
        if request.regions and not manifest.supports_region_selection:
            raise ContractValidationError("worker does not support region selection")
        if not manifest.supports_page_selection and request.document.page_count != len(
            request.pages
        ):
            raise ContractValidationError("worker requires the complete document")
        if (
            manifest.max_pages_per_request is not None
            and len(request.pages) > manifest.max_pages_per_request
        ):
            raise ContractValidationError("request exceeds worker page limit")

    def _parse(self, payload: Mapping[str, object]) -> None:
        request: ParseRequest | None = None
        try:
            require_protocol_version(payload)
            reject_unknown_fields(
                payload,
                frozenset({"protocol_version", "manifest_sha256", "request"}),
                envelope="parse request",
            )
            if payload.get("manifest_sha256") != self.owner.manifest.manifest_sha256:
                raise ContractValidationError(
                    "requested worker capability identity is unavailable"
                )
            request_payload = payload.get("request")
            if not isinstance(request_payload, dict):
                raise ContractValidationError("parse request.request must be an object")
            request = ParseRequest.from_dict(request_payload)
            self._validate_supported(self.owner.manifest, request)
        except ContractValidationError as exc:
            raise _RequestRejected(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "parse request violates the worker contract",
                request_id=request.request_id if request is not None else None,
            ) from exc

        if not self.owner._try_enter_parse():
            health = self.owner.health()
            code = "worker_busy" if health.state == WorkerState.READY else "worker_not_ready"
            raise _RequestRejected(
                HTTPStatus.TOO_MANY_REQUESTS
                if health.state == WorkerState.READY
                else HTTPStatus.SERVICE_UNAVAILABLE,
                code,
                "parser worker is not accepting another request",
                retryable=True,
                request_id=request.request_id,
            )
        try:
            try:
                result = self.owner._adapter.parse(request)
            except ParserExecutionError as exc:
                issue = exc.issue
                raise _RequestRejected(
                    HTTPStatus.SERVICE_UNAVAILABLE
                    if issue.classification == ErrorClassification.RETRYABLE
                    else HTTPStatus.UNPROCESSABLE_ENTITY,
                    issue.code.value,
                    issue.message,
                    retryable=issue.classification == ErrorClassification.RETRYABLE,
                    request_id=request.request_id,
                ) from exc
            except Exception as exc:
                # Do not send vendor exception strings, paths, prompts, or credentials.
                raise _RequestRejected(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "parser_failure",
                    "parser worker failed while executing the request",
                    retryable=False,
                    request_id=request.request_id,
                ) from exc
            try:
                if not isinstance(result, ParseResult):
                    raise ContractValidationError(
                        "adapter.parse() must return ParseResult"
                    )
                validate_result_binding(self.owner.manifest, request, result)
            except ContractValidationError as exc:
                raise _RequestRejected(
                    HTTPStatus.BAD_GATEWAY,
                    "contract_violation",
                    "parser result violates its declared source or capability binding",
                    retryable=False,
                    request_id=request.request_id,
                ) from exc
            self._send_payload(
                HTTPStatus.OK,
                {
                    "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
                    "manifest_sha256": self.owner.manifest.manifest_sha256,
                    "request_id": request.request_id,
                    "result": result.to_dict(),
                },
            )
        finally:
            self.owner._leave_parse()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            self._authorize()
            if self.path != PARSE_PATH:
                raise _RequestRejected(
                    HTTPStatus.NOT_FOUND, "not_found", "worker endpoint does not exist"
                )
            payload = self._read_json_body()
            self._parse(payload)
        except _RequestRejected as rejection:
            self._send_rejection(rejection)

    def _method_not_allowed(self) -> None:
        try:
            self._authorize()
            raise _RequestRejected(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "HTTP method is not supported by this worker endpoint",
            )
        except _RequestRejected as rejection:
            self._send_rejection(rejection)

    do_HEAD = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_OPTIONS = _method_not_allowed

