"""Standard-library HTTP client implementing the ``ParserAdapter`` port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import http.client
import ipaddress
import json
import math
import socket
import ssl
import threading
import time
from types import MappingProxyType
from urllib.parse import SplitResult, urlsplit

from docarag.domain.base import ContractValidationError, dump_json
from docarag.domain.parsing import ParseRequest, ParseResult, ParserCapabilityManifest
from docarag.ports.parser import validate_result_binding

from .errors import (
    ParserTransportError,
    RemoteWorkerError,
    TransportClosedError,
    TransportConfigurationError,
    TransportConnectionError,
    TransportContractError,
    TransportProtocolError,
    TransportRequestTooLargeError,
    TransportResponseTooLargeError,
    TransportTimeoutError,
    WorkerIdentityError,
    WorkerNotReadyError,
)
from .protocol import (
    CAPABILITIES_PATH,
    PARSE_PATH,
    PARSER_WORKER_PROTOCOL_VERSION,
    HEALTH_PATH,
    WorkerHealth,
    WorkerState,
    reject_unknown_fields,
    require_protocol_version,
)


_MIB = 1024 * 1024
_MAX_CONFIGURED_BODY_BYTES = 1024 * _MIB


def _positive_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise TransportConfigurationError(f"{field_name} must be finite and positive")
    return float(value)


def _bounded_byte_limit(value: object, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_CONFIGURED_BODY_BYTES
    ):
        raise TransportConfigurationError(
            f"{field_name} must be between 1 and {_MAX_CONFIGURED_BODY_BYTES} bytes"
        )
    return value


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_base_url(base_url: object, *, allow_insecure_http: bool) -> SplitResult:
    if not isinstance(base_url, str) or not base_url.strip():
        raise TransportConfigurationError("base_url must be a non-empty string")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise TransportConfigurationError("base_url scheme must be http or https")
    if parsed.hostname is None:
        raise TransportConfigurationError("base_url must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise TransportConfigurationError("base_url must not contain user information")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise TransportConfigurationError(
            "base_url must not contain a path, query string, or fragment"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise TransportConfigurationError("base_url contains an invalid port") from exc
    if (
        parsed.scheme == "http"
        and not allow_insecure_http
        and not _is_loopback(parsed.hostname)
    ):
        raise TransportConfigurationError(
            "plain HTTP is allowed only for loopback workers; use HTTPS or set "
            "allow_insecure_http explicitly"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class HTTPParserClientConfig:
    """Explicit connection, identity, deadline, and memory policy."""

    base_url: str
    connect_timeout_seconds: float = 5.0
    operation_timeout_seconds: float = 300.0
    max_request_bytes: int = 16 * _MIB
    max_response_bytes: int = 64 * _MIB
    bearer_token: str | None = field(default=None, repr=False)
    expected_adapter_name: str | None = None
    expected_adapter_version: str | None = None
    expected_manifest_sha256: str | None = None
    allow_insecure_http: bool = False
    tls_ca_file: str | None = None
    _endpoint: SplitResult = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.allow_insecure_http, bool):
            raise TransportConfigurationError("allow_insecure_http must be a boolean")
        endpoint = _validate_base_url(
            self.base_url, allow_insecure_http=self.allow_insecure_http
        )
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            _positive_number(self.connect_timeout_seconds, "connect_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "operation_timeout_seconds",
            _positive_number(self.operation_timeout_seconds, "operation_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "max_request_bytes",
            _bounded_byte_limit(self.max_request_bytes, "max_request_bytes"),
        )
        object.__setattr__(
            self,
            "max_response_bytes",
            _bounded_byte_limit(self.max_response_bytes, "max_response_bytes"),
        )
        if self.bearer_token is not None and (
            not isinstance(self.bearer_token, str) or not self.bearer_token
        ):
            raise TransportConfigurationError("bearer_token must be non-empty when set")
        for field_name in ("expected_adapter_name", "expected_adapter_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TransportConfigurationError(f"{field_name} must be non-empty when set")
        if self.expected_manifest_sha256 is not None and (
            len(self.expected_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_manifest_sha256
            )
        ):
            raise TransportConfigurationError(
                "expected_manifest_sha256 must be a lowercase SHA-256 digest"
            )
        if self.tls_ca_file is not None and (
            not isinstance(self.tls_ca_file, str) or not self.tls_ca_file.strip()
        ):
            raise TransportConfigurationError("tls_ca_file must be non-empty when set")
        if self.tls_ca_file is not None and endpoint.scheme != "https":
            raise TransportConfigurationError("tls_ca_file requires an HTTPS base_url")
        object.__setattr__(self, "_endpoint", endpoint)


class HTTPParserAdapter:
    """A remote parser that conforms to the in-process ``ParserAdapter`` port.

    Each call uses a fresh HTTP connection.  This avoids hidden session state,
    remains thread-safe, and still amortizes expensive model startup because
    the worker process is persistent.  Calls are deliberately not retried:
    retrying an uncertain parse completion is an orchestration/idempotency
    decision, not a transport decision.
    """

    def __init__(self, config: HTTPParserClientConfig) -> None:
        if not isinstance(config, HTTPParserClientConfig):
            raise TypeError("config must be HTTPParserClientConfig")
        self._config = config
        self._lock = threading.RLock()
        self._closed = False
        self._manifest: ParserCapabilityManifest | None = None
        self._worker_instance_id: str | None = None

    @property
    def config(self) -> HTTPParserClientConfig:
        return self._config

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Prevent future calls; in-flight calls retain their own connection."""

        with self._lock:
            self._closed = True
            self._manifest = None

    def __enter__(self) -> "HTTPParserAdapter":
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise TransportClosedError("parser-worker client is closed")

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        endpoint = self._config._endpoint
        host = endpoint.hostname
        assert host is not None
        if endpoint.scheme == "https":
            context = ssl.create_default_context(cafile=self._config.tls_ca_file)
            return http.client.HTTPSConnection(
                host,
                endpoint.port or 443,
                timeout=timeout,
                context=context,
            )
        return http.client.HTTPConnection(host, endpoint.port or 80, timeout=timeout)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportTimeoutError("parser-worker operation timed out")
        return remaining

    def _decode_response(
        self,
        response: http.client.HTTPResponse,
        *,
        deadline: float,
    ) -> Mapping[str, object]:
        content_encoding = response.getheader("Content-Encoding")
        if content_encoding not in {None, "", "identity"}:
            raise TransportProtocolError("compressed worker responses are not supported")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise TransportProtocolError("worker response Content-Type must be application/json")
        content_lengths = response.headers.get_all("Content-Length", [])
        if len(content_lengths) > 1:
            raise TransportProtocolError("worker sent multiple Content-Length headers")
        if content_lengths:
            try:
                declared_size = int(content_lengths[0], 10)
            except ValueError as exc:
                raise TransportProtocolError("worker sent an invalid Content-Length") from exc
            if declared_size < 0:
                raise TransportProtocolError("worker sent a negative Content-Length")
            if declared_size > self._config.max_response_bytes:
                raise TransportResponseTooLargeError(
                    "worker response exceeds max_response_bytes"
                )
        if response.fp is not None and getattr(response.fp, "raw", None) is not None:
            raw_socket = getattr(response.fp.raw, "_sock", None)
            if raw_socket is not None:
                raw_socket.settimeout(self._remaining(deadline))
        body = response.read(self._config.max_response_bytes + 1)
        if len(body) > self._config.max_response_bytes:
            raise TransportResponseTooLargeError(
                "worker response exceeds max_response_bytes"
            )
        if content_lengths and len(body) != declared_size:
            raise TransportProtocolError("worker response body length does not match Content-Length")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportProtocolError("worker returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise TransportProtocolError("worker response must be a JSON object")
        return MappingProxyType(payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        self._ensure_open()
        body = None if payload is None else dump_json(payload).encode("utf-8")
        if body is not None and len(body) > self._config.max_request_bytes:
            raise TransportRequestTooLargeError(
                "serialized parser request exceeds max_request_bytes"
            )
        headers = {
            "Accept": "application/json",
            "Connection": "close",
            "User-Agent": "DocARAG-parser-controller/1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        if self._config.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        deadline = time.monotonic() + self._config.operation_timeout_seconds
        connect_timeout = min(
            self._config.connect_timeout_seconds, self._remaining(deadline)
        )
        connection = self._connection(connect_timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            if connection.sock is not None:
                connection.sock.settimeout(self._remaining(deadline))
            response = connection.getresponse()
            decoded = self._decode_response(response, deadline=deadline)
            if not 200 <= response.status < 300:
                self._raise_remote_error(response.status, decoded)
            return decoded
        except ParserTransportError:
            raise
        except (socket.timeout, TimeoutError) as exc:
            raise TransportTimeoutError("parser-worker operation timed out") from exc
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            raise TransportConnectionError("could not communicate with parser worker") from exc
        finally:
            connection.close()

    @staticmethod
    def _raise_remote_error(status: int, payload: Mapping[str, object]) -> None:
        try:
            require_protocol_version(payload)
            reject_unknown_fields(
                payload,
                frozenset({"protocol_version", "error"}),
                envelope="error response",
            )
            error = payload.get("error")
            if not isinstance(error, dict):
                raise ContractValidationError("error response.error must be an object")
            reject_unknown_fields(
                error,
                frozenset({"code", "message", "retryable", "request_id"}),
                envelope="remote error",
            )
            code = error.get("code")
            message = error.get("message")
            retryable = error.get("retryable")
            request_id = error.get("request_id")
            if not isinstance(code, str) or not code:
                raise ContractValidationError("remote error.code must be non-empty")
            if not isinstance(message, str) or not message:
                raise ContractValidationError("remote error.message must be non-empty")
            if not isinstance(retryable, bool):
                raise ContractValidationError("remote error.retryable must be a boolean")
            if request_id is not None and (
                not isinstance(request_id, str) or not request_id
            ):
                raise ContractValidationError("remote error.request_id is invalid")
        except ContractValidationError as exc:
            raise TransportProtocolError(
                f"worker returned malformed HTTP {status} error envelope"
            ) from exc
        raise RemoteWorkerError(
            status=status,
            code=code,
            message=message,
            retryable=retryable,
            request_id=request_id,
        )

    def _validate_identity(self, manifest: ParserCapabilityManifest) -> None:
        pins = (
            ("adapter_name", self._config.expected_adapter_name, manifest.adapter_name),
            (
                "adapter_version",
                self._config.expected_adapter_version,
                manifest.adapter_version,
            ),
            (
                "manifest_sha256",
                self._config.expected_manifest_sha256,
                manifest.manifest_sha256,
            ),
        )
        for field_name, expected, actual in pins:
            if expected is not None and expected != actual:
                raise WorkerIdentityError(
                    f"parser worker {field_name} does not match the configured identity"
                )

    def refresh_manifest(self) -> ParserCapabilityManifest:
        payload = self._request_json("GET", CAPABILITIES_PATH)
        try:
            require_protocol_version(payload)
            reject_unknown_fields(
                payload,
                frozenset({"protocol_version", "manifest"}),
                envelope="capabilities response",
            )
            manifest_payload = payload.get("manifest")
            if not isinstance(manifest_payload, dict):
                raise ContractValidationError("capabilities manifest must be an object")
            manifest = ParserCapabilityManifest.from_dict(manifest_payload)
            self._validate_identity(manifest)
        except ContractValidationError as exc:
            raise TransportContractError("worker returned an invalid capability manifest") from exc
        with self._lock:
            if self._closed:
                raise TransportClosedError("parser-worker client is closed")
            self._manifest = manifest
        return manifest

    def manifest(self) -> ParserCapabilityManifest:
        with self._lock:
            manifest = self._manifest
        return manifest if manifest is not None else self.refresh_manifest()

    def health(self, *, require_ready: bool = False) -> WorkerHealth:
        payload = self._request_json("GET", HEALTH_PATH)
        try:
            health = WorkerHealth.from_dict(payload)
        except ContractValidationError as exc:
            raise TransportContractError("worker returned an invalid health response") from exc
        manifest = self._manifest
        if manifest is not None and (
            health.adapter_name != manifest.adapter_name
            or health.adapter_version != manifest.adapter_version
            or health.manifest_sha256 != manifest.manifest_sha256
        ):
            raise WorkerIdentityError("worker health identity differs from cached capabilities")
        self._worker_instance_id = health.worker_instance_id
        if require_ready and health.state != WorkerState.READY:
            raise WorkerNotReadyError(
                f"parser worker is {health.state.value} and is not accepting requests"
            )
        return health

    def parse(self, request: ParseRequest) -> ParseResult:
        if not isinstance(request, ParseRequest):
            raise TypeError("request must be ParseRequest")
        manifest = self.manifest()
        payload = self._request_json(
            "POST",
            PARSE_PATH,
            {
                "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
                "manifest_sha256": manifest.manifest_sha256,
                "request": request.to_dict(),
            },
        )
        try:
            require_protocol_version(payload)
            reject_unknown_fields(
                payload,
                frozenset(
                    {
                        "protocol_version",
                        "manifest_sha256",
                        "request_id",
                        "result",
                    }
                ),
                envelope="parse response",
            )
            if payload.get("manifest_sha256") != manifest.manifest_sha256:
                raise ContractValidationError("worker manifest changed during parse")
            if payload.get("request_id") != request.request_id:
                raise ContractValidationError("parse response is bound to another request")
            result_payload = payload.get("result")
            if not isinstance(result_payload, dict):
                raise ContractValidationError("parse response.result must be an object")
            result = ParseResult.from_dict(result_payload)
            validate_result_binding(manifest, request, result)
        except ContractValidationError as exc:
            raise TransportContractError(
                "worker returned an invalid or incorrectly bound ParseResult"
            ) from exc
        return result

