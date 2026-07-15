"""Strict client/loader for the supported MinerU 3.x task API.

The implementation intentionally imports neither MinerU nor an HTTP SDK.  It
targets the common ``mineru-api`` / ``mineru-router`` protocol-2 surface:
``GET /health``, ``POST /tasks``, ``GET /tasks/{id}``, and
``GET /tasks/{id}/result``.  Vendor task state is process-local and expiring,
so a task id is trace information only; it is never treated as a durable
checkpoint or silently resumed after a 404.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import http.client
from io import BytesIO
import ipaddress
import json
import math
import mimetypes
from pathlib import PurePosixPath
import re
import ssl
import stat
from time import monotonic, sleep
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from urllib.parse import quote, urlsplit, urlunsplit
import zipfile

from docarag.domain.base import content_sha256, stable_id
from docarag.domain.parsing import (
    ErrorClassification,
    IssueSeverity,
    MAX_PARSER_ARTIFACT_RECEIPTS,
    MinerUEffort,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParserArtifactAvailability,
    ParserArtifactReceipt,
    ParserExecutionError,
)
from docarag.source_material import SourceMaterialError, VerifiedSourceMaterial


MINERU_API_PROTOCOL_VERSION = 2
MINERU_SERVICE_SCHEMA_VERSION = "docarag.mineru-service.v1"
MINERU_MIDDLE_RECEIPT_SCHEMA_VERSION = "docarag.mineru-middle-receipt.v1"
MINERU_MIDDLE_ARTIFACT_SCHEMA = "mineru.middle-json"
MINERU_RESULT_ARCHIVE_RECEIPT_SCHEMA_VERSION = (
    "docarag.mineru-result-archive-receipt.v1"
)
MINERU_RESULT_ARCHIVE_ARTIFACT_SCHEMA = "mineru.result-archive"
# Every completed task has one primary archive receipt and one explicit middle
# artifact outcome.  Keep the configured task ceiling inside ParseResult's
# receipt capacity rather than discovering the mismatch after expensive work.
MAX_MINERU_TASKS_PER_PARSE_RESULT = MAX_PARSER_ARTIFACT_RECEIPTS // 2
_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_TERMINAL_FAILURES = frozenset({"failed", "error", "cancelled", "canceled"})


class MinerUBackend(StrEnum):
    PIPELINE = "pipeline"
    VLM_ENGINE = "vlm-engine"
    HYBRID_ENGINE = "hybrid-engine"
    VLM_HTTP_CLIENT = "vlm-http-client"
    HYBRID_HTTP_CLIENT = "hybrid-http-client"


class MinerUServiceError(RuntimeError):
    retryable = False


class MinerUConfigurationError(MinerUServiceError, ValueError):
    pass


class MinerUSourceBindingError(MinerUServiceError):
    pass


class MinerUProtocolError(MinerUServiceError):
    pass


class MinerULimitError(MinerUServiceError):
    pass


class MinerUTransportError(MinerUServiceError):
    retryable = True


class MinerUTimeoutError(MinerUTransportError, TimeoutError):
    pass


class MinerUTaskFailedError(MinerUServiceError):
    pass


class MinerUTaskStateLostError(MinerUTransportError):
    """The process-local vendor task disappeared and must be resubmitted."""


def _positive_int(value: object, field_name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise MinerUConfigurationError(
            f"{field_name} must be an integer between 1 and {maximum}"
        )
    return value


def _positive_float(value: object, field_name: str, *, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= maximum
    ):
        raise MinerUConfigurationError(
            f"{field_name} must be finite, positive, and at most {maximum:g}"
        )
    return float(value)


def _loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _normalize_base_url(value: str, *, allow_insecure_http: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MinerUConfigurationError("base_url must be a non-empty URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MinerUConfigurationError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MinerUConfigurationError(
            "base_url must not contain credentials, a query, or a fragment"
        )
    if parsed.path not in {"", "/"}:
        raise MinerUConfigurationError("base_url path must be empty")
    if parsed.scheme == "http" and not (
        allow_insecure_http or _loopback(parsed.hostname)
    ):
        raise MinerUConfigurationError(
            "plain HTTP is allowed only for loopback unless explicitly enabled"
        )
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    authority = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), authority, "", "", ""))


@dataclass(frozen=True, slots=True)
class MinerUServiceConfig:
    """Secret-free semantic and resource contract for one MinerU endpoint."""

    base_url: str
    backend: MinerUBackend
    effort: MinerUEffort
    expected_protocol_version: int = MINERU_API_PROTOCOL_VERSION
    allow_insecure_http: bool = False
    connect_timeout_seconds: float = 10.0
    control_timeout_seconds: float = 60.0
    upload_timeout_seconds: float = 300.0
    task_timeout_seconds: float = 3600.0
    result_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 1.0
    max_poll_attempts: int = 7200
    max_source_bytes: int = 512 * 1024 * 1024
    max_control_response_bytes: int = 256 * 1024
    max_result_archive_bytes: int = 512 * 1024 * 1024
    max_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_content_list_bytes: int = 256 * 1024 * 1024
    # Middle JSON is decoded eagerly today, so keep its independent default
    # below the legacy content-list cap to bound controller RSS amplification.
    max_middle_json_bytes: int = 128 * 1024 * 1024
    max_middle_json_depth: int = 128
    max_middle_json_nodes: int = 2_000_000
    max_archive_members: int = 100_000
    max_blocks: int = 2_000_000
    max_pages_per_task: int = 256
    max_tasks_per_request: int = 256
    max_queued_ahead: int = 1_000_000
    max_compression_ratio: float = 1000.0
    revision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            backend = MinerUBackend(self.backend)
            effort = MinerUEffort(self.effort)
        except (TypeError, ValueError) as exc:
            raise MinerUConfigurationError("backend or effort is unsupported") from exc
        for name, maximum in (
            ("expected_protocol_version", 1000),
            ("max_poll_attempts", 10_000_000),
            ("max_source_bytes", 4 * 1024**3),
            ("max_control_response_bytes", 16 * 1024**2),
            ("max_result_archive_bytes", 4 * 1024**3),
            ("max_uncompressed_bytes", 16 * 1024**3),
            ("max_content_list_bytes", 4 * 1024**3),
            ("max_middle_json_bytes", 4 * 1024**3),
            ("max_middle_json_depth", 4096),
            ("max_middle_json_nodes", 50_000_000),
            ("max_archive_members", 10_000_000),
            ("max_blocks", 10_000_000),
            ("max_pages_per_task", 100_000),
            ("max_tasks_per_request", MAX_MINERU_TASKS_PER_PARSE_RESULT),
            ("max_queued_ahead", 100_000_000),
        ):
            _positive_int(getattr(self, name), name, maximum=maximum)
        for name, maximum in (
            ("connect_timeout_seconds", 3600),
            ("control_timeout_seconds", 3600),
            ("upload_timeout_seconds", 86400),
            ("task_timeout_seconds", 7 * 86400),
            ("result_timeout_seconds", 86400),
            ("poll_interval_seconds", 3600),
            ("max_compression_ratio", 1_000_000),
        ):
            _positive_float(getattr(self, name), name, maximum=maximum)
        base_url = _normalize_base_url(
            self.base_url, allow_insecure_http=self.allow_insecure_http
        )
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "revision_sha256", content_sha256(self.identity()))

    def identity(self) -> dict[str, object]:
        return {
            "schema_version": MINERU_SERVICE_SCHEMA_VERSION,
            "base_url": self.base_url,
            "backend": self.backend.value,
            "effort": self.effort.value,
            "expected_protocol_version": self.expected_protocol_version,
            "bounds": {
                name: getattr(self, name)
                for name in (
                    "control_timeout_seconds",
                    "upload_timeout_seconds",
                    "task_timeout_seconds",
                    "result_timeout_seconds",
                    "poll_interval_seconds",
                    "max_poll_attempts",
                    "max_source_bytes",
                    "max_result_archive_bytes",
                    "max_uncompressed_bytes",
                    "max_content_list_bytes",
                    "max_middle_json_bytes",
                    "max_middle_json_depth",
                    "max_middle_json_nodes",
                    "max_archive_members",
                    "max_blocks",
                    "max_pages_per_task",
                    "max_tasks_per_request",
                )
            },
            "runtime_certification": "uncertified",
        }


@dataclass(frozen=True, slots=True)
class MinerUHTTPRequestBody:
    content_length: int
    chunks: tuple[bytes | memoryview, ...]

    def __iter__(self) -> Iterable[bytes | memoryview]:
        return iter(self.chunks)


@dataclass(frozen=True, slots=True)
class MinerUHTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("response status must be an integer")
        object.__setattr__(
            self,
            "headers",
            MappingProxyType({str(k).lower(): str(v) for k, v in self.headers.items()}),
        )
        if not isinstance(self.body, bytes):
            raise TypeError("response body must be bytes")


@runtime_checkable
class MinerUHTTPTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: MinerUHTTPRequestBody | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> MinerUHTTPResponse: ...

    def close(self) -> None: ...


class StdlibMinerUHTTPTransport:
    """One-request-per-connection transport with a wall-clock deadline."""

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self._ssl_context = ssl_context
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: MinerUHTTPRequestBody | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> MinerUHTTPResponse:
        if self._closed:
            raise MinerUTransportError("MinerU HTTP transport is closed")
        parsed = urlsplit(url)
        deadline = monotonic() + timeout_seconds
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        kwargs: dict[str, object] = {
            "host": parsed.hostname,
            "port": parsed.port,
            "timeout": timeout_seconds,
        }
        if parsed.scheme == "https":
            kwargs["context"] = self._ssl_context
        connection = connection_type(**kwargs)  # type: ignore[arg-type]
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.putrequest(method, path, skip_accept_encoding=True)
            for key, value in headers.items():
                connection.putheader(key, value)
            if body is not None:
                connection.putheader("Content-Length", str(body.content_length))
            connection.endheaders()
            if body is not None:
                for chunk in body:
                    if monotonic() >= deadline:
                        raise MinerUTimeoutError("MinerU HTTP request deadline expired")
                    connection.send(chunk)
            if monotonic() >= deadline:
                raise MinerUTimeoutError("MinerU HTTP request deadline expired")
            response = connection.getresponse()
            chunks: list[bytes] = []
            total = 0
            while True:
                if monotonic() >= deadline:
                    raise MinerUTimeoutError("MinerU HTTP response deadline expired")
                chunk = response.read(min(64 * 1024, max_response_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_response_bytes:
                    raise MinerULimitError("MinerU response exceeds configured byte limit")
                chunks.append(chunk)
            return MinerUHTTPResponse(
                response.status, {key: value for key, value in response.getheaders()}, b"".join(chunks)
            )
        except MinerUServiceError:
            raise
        except TimeoutError as exc:
            raise MinerUTimeoutError("MinerU HTTP operation timed out") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise MinerUTransportError("MinerU HTTP operation failed") from exc
        finally:
            connection.close()


@dataclass(frozen=True, slots=True)
class MinerUHealth:
    protocol_version: int
    processing_window_size: int
    max_concurrent_requests: int


@dataclass(frozen=True, slots=True)
class MinerUTask:
    operation_id: str
    task_id: str
    status_url: str
    result_url: str
    file_name: str
    first_page: int
    last_page: int
    queued_ahead: int | None


@dataclass(frozen=True, slots=True)
class MinerUArchive:
    operation_id: str
    task_id: str
    archive_sha256: str
    data: bytes
    members: tuple[str, ...]
    member_sha256s: Mapping[str, str]

    def __post_init__(self) -> None:
        if hashlib.sha256(self.data).hexdigest() != self.archive_sha256:
            raise MinerUProtocolError(
                "MinerU archive digest differs from downloaded archive bytes"
            )
        member_sha256s: dict[str, str] = {}
        for member, digest in self.member_sha256s.items():
            if not isinstance(member, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise MinerUProtocolError("MinerU archive member digest is invalid")
            member_sha256s[member] = digest
        if tuple(member_sha256s) != self.members:
            raise MinerUProtocolError(
                "MinerU archive member names and digest mapping differ"
            )
        object.__setattr__(
            self, "member_sha256s", MappingProxyType(member_sha256s)
        )


@dataclass(frozen=True, slots=True)
class PublishedMinerUArchive:
    archive_uri: str
    archive_sha256: str
    member_uris: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.archive_uri, str) or not self.archive_uri:
            raise MinerUProtocolError("publisher returned an empty archive URI")
        if not re.fullmatch(r"[0-9a-f]{64}", self.archive_sha256):
            raise MinerUProtocolError("publisher returned an invalid archive digest")
        normalized: dict[str, str] = {}
        for member, uri in self.member_uris.items():
            if not isinstance(member, str) or not isinstance(uri, str) or not uri:
                raise MinerUProtocolError("publisher returned an invalid member URI")
            normalized[member] = uri
        object.__setattr__(self, "member_uris", MappingProxyType(normalized))


MinerUSourceLoader = Callable[[ParseRequest], VerifiedSourceMaterial]
MinerUArtifactPublisher = Callable[[MinerUArchive], PublishedMinerUArchive]


@dataclass(frozen=True, slots=True)
class MinerULoadedBlocks:
    operation_id: str
    service_revision_sha256: str
    effort: MinerUEffort
    blocks: tuple[Mapping[str, object], ...]
    task_ids: tuple[str, ...]
    archive_sha256s: tuple[str, ...]
    artifact_receipts: tuple[ParserArtifactReceipt, ...] = ()

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | tuple[Mapping[str, object], ...]:
        return self.blocks[index]


def _strict_json(raw: bytes, *, label: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise MinerUProtocolError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MinerUProtocolError(f"{label} contains non-finite JSON")
            ),
        )
    except UnicodeDecodeError as exc:
        raise MinerUProtocolError(f"{label} is not UTF-8") from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise MinerUProtocolError(f"{label} is not valid JSON") from exc


def _preflight_json_depth(
    raw: bytes, *, label: str, maximum_depth: int
) -> None:
    """Bound container nesting before the stdlib decoder allocates a tree.

    This is deliberately a lexical preflight, not an alternate JSON parser.
    The strict decoder below remains authoritative for syntax.  Tracking string
    and escape state prevents braces in text from affecting the depth bound.
    """

    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ or {
            depth += 1
            if depth > maximum_depth:
                raise MinerULimitError(
                    f"{label} exceeds configured nesting-depth limit"
                )
        elif byte in (0x5D, 0x7D):  # ] or }
            depth -= 1
            if depth < 0:
                # The JSON decoder will provide the authoritative syntax
                # rejection, but never let an invalid close evade accounting.
                depth = 0


def _validate_json_structure(
    value: object,
    *,
    label: str,
    maximum_depth: int,
    maximum_nodes: int,
) -> None:
    """Validate a decoded JSON tree iteratively under explicit resource caps."""

    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > maximum_nodes:
            raise MinerULimitError(f"{label} exceeds configured JSON-node limit")
        if depth > maximum_depth:
            raise MinerULimitError(
                f"{label} exceeds configured nesting-depth limit"
            )
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise MinerUProtocolError(f"{label} contains a non-string object key")
            # Count keys as allocated JSON nodes as well as their values.
            nodes += len(current)
            if nodes > maximum_nodes:
                raise MinerULimitError(
                    f"{label} exceeds configured JSON-node limit"
                )
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
        elif current is None or isinstance(current, (str, bool, int)):
            continue
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise MinerUProtocolError(f"{label} contains non-finite JSON")
        else:
            raise MinerUProtocolError(
                f"{label} contains a non-JSON {type(current).__name__} value"
            )


def _json_object(response: MinerUHTTPResponse, *, label: str, maximum: int) -> Mapping[str, object]:
    if len(response.body) > maximum:
        raise MinerULimitError(f"{label} exceeds configured byte limit")
    value = _strict_json(response.body, label=label)
    if not isinstance(value, Mapping):
        raise MinerUProtocolError(f"{label} must be a JSON object")
    return value


def _safe_upload_name(display_name: str) -> str:
    value = _SAFE_NAME.sub("_", display_name.rsplit("/", 1)[-1]).strip("._")
    if not value:
        value = "document.pdf"
    if not value.lower().endswith(".pdf"):
        value += ".pdf"
    return value[:240]


def _multipart(
    fields: Sequence[tuple[str, str]], *, file_name: str, media_type: str, data: bytes, seed: str
) -> tuple[str, MinerUHTTPRequestBody]:
    boundary = "docarag-" + hashlib.sha256((seed + file_name).encode()).hexdigest()
    marker = boundary.encode()
    if marker in data:
        boundary += "-source"
    chunks: list[bytes | memoryview] = []
    for name, value in fields:
        chunks.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
                f"\r\n\r\n{value}\r\n"
            ).encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{file_name}\"\r\nContent-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(memoryview(data))
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return boundary, MinerUHTTPRequestBody(sum(len(item) for item in chunks), tuple(chunks))


def _page_ranges(page_indices: Sequence[int], maximum: int) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = previous = page_indices[0]
    for page in page_indices[1:]:
        if page != previous + 1 or page - start + 1 > maximum:
            ranges.append((start, previous))
            start = page
        previous = page
    ranges.append((start, previous))
    return tuple(ranges)


class MinerUAPIClient:
    """No-retry client for MinerU's process-local asynchronous task surface."""

    def __init__(
        self,
        config: MinerUServiceConfig,
        *,
        transport: MinerUHTTPTransport | None = None,
        bearer_token: str | None = None,
        owns_transport: bool | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if not isinstance(config, MinerUServiceConfig):
            raise TypeError("config must be MinerUServiceConfig")
        if bearer_token is not None and (not isinstance(bearer_token, str) or not bearer_token):
            raise MinerUConfigurationError("bearer_token must be non-empty when supplied")
        if transport is not None and not isinstance(transport, MinerUHTTPTransport):
            raise TypeError("transport must implement MinerUHTTPTransport")
        self.config = config
        self._transport = transport or StdlibMinerUHTTPTransport()
        self._owns_transport = (transport is None) if owns_transport is None else owns_transport
        self._bearer_token = bearer_token
        self._clock = clock
        self._sleep = sleeper
        self._closed = False

    def close(self) -> None:
        if not self._closed and self._owns_transport:
            self._transport.close()
        self._closed = True

    def __enter__(self) -> "MinerUAPIClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _headers(self, *, accept: str) -> dict[str, str]:
        if self._closed:
            raise MinerUTransportError("MinerU API client is closed")
        result = {"Accept": accept, "User-Agent": "DocARAG-MinerU/1"}
        if self._bearer_token is not None:
            result["Authorization"] = "Bearer " + self._bearer_token
        return result

    def _request(
        self,
        method: str,
        url: str,
        *,
        accept: str,
        body: MinerUHTTPRequestBody | None,
        timeout: float,
        maximum: int,
        headers: Mapping[str, str] = MappingProxyType({}),
    ) -> MinerUHTTPResponse:
        merged = self._headers(accept=accept)
        merged.update(headers)
        response = self._transport.request(
            method, url, headers=merged, body=body,
            timeout_seconds=timeout, max_response_bytes=maximum,
        )
        if len(response.body) > maximum:
            raise MinerULimitError("MinerU transport returned an oversized response")
        return response

    def health(self) -> MinerUHealth:
        response = self._request(
            "GET", self.config.base_url + "/health", accept="application/json",
            body=None, timeout=self.config.control_timeout_seconds,
            maximum=self.config.max_control_response_bytes,
        )
        if response.status != 200:
            raise MinerUTransportError(f"MinerU health returned HTTP {response.status}")
        payload = _json_object(
            response, label="MinerU health response", maximum=self.config.max_control_response_bytes
        )
        if payload.get("status") != "healthy":
            raise MinerUTransportError("MinerU service is not healthy")
        protocol = payload.get("protocol_version")
        concurrent = payload.get("max_concurrent_requests")
        window = payload.get("processing_window_size")
        if protocol != self.config.expected_protocol_version:
            raise MinerUProtocolError(
                "MinerU API protocol identity differs from configured protocol"
            )
        for value, name in ((concurrent, "max_concurrent_requests"), (window, "processing_window_size")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MinerUProtocolError(f"MinerU health has invalid {name}")
        return MinerUHealth(protocol, window, concurrent)

    def _task_url(self, value: object, task_id: str, *, result: bool) -> str:
        if not isinstance(value, str) or not value:
            raise MinerUProtocolError("MinerU task response has an invalid task URL")
        parsed = urlsplit(value)
        if not parsed.scheme:
            if not value.startswith("/"):
                raise MinerUProtocolError("MinerU task URL must be absolute-path or absolute URL")
            value = self.config.base_url + value
            parsed = urlsplit(value)
        base = urlsplit(self.config.base_url)
        if (parsed.scheme, parsed.hostname, parsed.port) != (base.scheme, base.hostname, base.port):
            raise MinerUProtocolError("MinerU task URL escaped the configured service origin")
        expected = f"/tasks/{quote(task_id, safe='')}" + ("/result" if result else "")
        if parsed.path != expected or parsed.query or parsed.fragment:
            raise MinerUProtocolError("MinerU task URL is not bound to the returned task id")
        return value

    def submit(
        self,
        request: ParseRequest,
        source: VerifiedSourceMaterial,
        *,
        first_page: int,
        last_page: int,
        operation_id: str,
    ) -> MinerUTask:
        file_name = _safe_upload_name(request.document.display_name or "document.pdf")
        languages = request.options.languages or ("ch",)
        method = {"auto": "auto", "force": "ocr", "disabled": "txt"}[
            request.options.ocr_mode.value
        ]
        fields: list[tuple[str, str]] = [("lang_list", language) for language in languages]
        fields.extend(
            (
                ("backend", self.config.backend.value),
                ("effort", self.config.effort.value),
                ("parse_method", method),
                ("formula_enable", str(request.options.parse_formulas).lower()),
                ("table_enable", str(request.options.parse_tables).lower()),
                ("image_analysis", str(request.options.parse_images).lower()),
                ("return_md", "false"),
                ("return_middle_json", "true"),
                ("return_model_output", "false"),
                ("return_content_list", "true"),
                ("return_images", str(request.options.parse_images).lower()),
                ("response_format_zip", "true"),
                ("return_original_file", "false"),
                ("client_side_output_generation", "false"),
                ("start_page_id", str(first_page)),
                ("end_page_id", str(last_page)),
            )
        )
        assert source.data is not None
        boundary, body = _multipart(
            fields, file_name=file_name, media_type=source.media_type,
            data=source.data, seed=operation_id,
        )
        response = self._request(
            "POST", self.config.base_url + "/tasks", accept="application/json",
            body=body, timeout=self.config.upload_timeout_seconds,
            maximum=self.config.max_control_response_bytes,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        if response.status != 202:
            raise MinerUTransportError(f"MinerU task submission returned HTTP {response.status}")
        payload = _json_object(
            response, label="MinerU task submission", maximum=self.config.max_control_response_bytes
        )
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
            raise MinerUProtocolError("MinerU task id is invalid")
        file_names = payload.get("file_names")
        if file_names != [file_name]:
            raise MinerUProtocolError("MinerU task response is not bound to the uploaded file")
        queued = payload.get("queued_ahead")
        if queued is not None and (
            isinstance(queued, bool) or not isinstance(queued, int)
            or not 0 <= queued <= self.config.max_queued_ahead
        ):
            raise MinerUProtocolError("MinerU queued_ahead is invalid")
        return MinerUTask(
            operation_id, task_id,
            self._task_url(payload.get("status_url"), task_id, result=False),
            self._task_url(payload.get("result_url"), task_id, result=True),
            file_name, first_page, last_page, queued,
        )

    def wait(self, task: MinerUTask, *, timeout_seconds: float | None = None) -> None:
        timeout = self.config.task_timeout_seconds
        if timeout_seconds is not None:
            timeout = min(timeout, timeout_seconds)
        deadline = self._clock() + timeout
        for attempt in range(self.config.max_poll_attempts):
            if self._clock() >= deadline:
                break
            response = self._request(
                "GET", task.status_url, accept="application/json", body=None,
                timeout=min(self.config.control_timeout_seconds, max(0.001, deadline - self._clock())),
                maximum=self.config.max_control_response_bytes,
            )
            if response.status == 404:
                raise MinerUTaskStateLostError(
                    "MinerU task state disappeared; task state is process-local/non-durable and the durable source must be resubmitted"
                )
            if response.status != 200:
                raise MinerUTransportError(f"MinerU task status returned HTTP {response.status}")
            payload = _json_object(
                response, label="MinerU task status", maximum=self.config.max_control_response_bytes
            )
            returned_id = payload.get("task_id")
            if returned_id is not None and returned_id != task.task_id:
                raise MinerUProtocolError("MinerU status belongs to a different task")
            status_value = payload.get("status")
            if status_value == "completed":
                return
            if status_value in _TERMINAL_FAILURES:
                raise MinerUTaskFailedError(
                    f"MinerU task reached terminal status {status_value!r}"
                )
            if status_value not in {"pending", "processing"}:
                raise MinerUProtocolError("MinerU task returned an unknown status")
            queued = payload.get("queued_ahead")
            if queued is not None and (
                isinstance(queued, bool) or not isinstance(queued, int)
                or not 0 <= queued <= self.config.max_queued_ahead
            ):
                raise MinerUProtocolError("MinerU status queued_ahead is invalid")
            if attempt + 1 < self.config.max_poll_attempts:
                remaining = deadline - self._clock()
                if remaining > 0:
                    self._sleep(min(self.config.poll_interval_seconds, remaining))
        raise MinerUTimeoutError(
            "MinerU task deadline expired; the process-local task id is trace-only, not a durable checkpoint"
        )

    def result(self, task: MinerUTask) -> MinerUArchive:
        response = self._request(
            "GET", task.result_url, accept="application/zip", body=None,
            timeout=self.config.result_timeout_seconds,
            maximum=self.config.max_result_archive_bytes,
        )
        if response.status == 404:
            raise MinerUTaskStateLostError(
                "MinerU result expired or its process-local task state was lost; resubmit the durable source"
            )
        if response.status != 200:
            raise MinerUTransportError(f"MinerU task result returned HTTP {response.status}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/zip":
            raise MinerUProtocolError("MinerU result is not application/zip")
        member_sha256s = _validate_archive(response.body, self.config)
        return MinerUArchive(
            operation_id=task.operation_id,
            task_id=task.task_id,
            archive_sha256=hashlib.sha256(response.body).hexdigest(),
            data=response.body,
            members=tuple(member_sha256s),
            member_sha256s=member_sha256s,
        )


def _validate_member_name(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise MinerUProtocolError("MinerU ZIP contains an unsafe member name")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise MinerUProtocolError("MinerU ZIP contains a path-traversal member")
    return path


def _validate_archive(
    data: bytes, config: MinerUServiceConfig
) -> Mapping[str, str]:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except (OSError, zipfile.BadZipFile) as exc:
        raise MinerUProtocolError("MinerU result is not a valid ZIP archive") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > config.max_archive_members:
            raise MinerULimitError("MinerU ZIP has too many members")
        seen: set[str] = set()
        declared_total = 0
        file_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            _validate_member_name(info.filename)
            if info.filename in seen:
                raise MinerUProtocolError("MinerU ZIP contains duplicate member names")
            seen.add(info.filename)
            if info.flag_bits & 0x1:
                raise MinerUProtocolError("MinerU ZIP contains an encrypted member")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise MinerUProtocolError("MinerU ZIP contains a symbolic-link member")
            if info.is_dir():
                continue
            declared_total += info.file_size
            if declared_total > config.max_uncompressed_bytes:
                raise MinerULimitError("MinerU ZIP exceeds uncompressed byte limit")
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > config.max_compression_ratio:
                raise MinerULimitError("MinerU ZIP member exceeds compression-ratio limit")
            file_infos.append(info)
        if not file_infos:
            raise MinerUProtocolError("MinerU ZIP contains no files")
        # Hash the actual decompressed bytes.  Central-directory sizes and CRCs
        # are useful preflight metadata, but are not evidence identities.
        actual_total = 0
        member_sha256s: dict[str, str] = {}
        try:
            for info in file_infos:
                member_size = 0
                digest = hashlib.sha256()
                with archive.open(info, "r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        member_size += len(chunk)
                        actual_total += len(chunk)
                        if actual_total > config.max_uncompressed_bytes:
                            raise MinerULimitError(
                                "MinerU ZIP exceeds uncompressed byte limit"
                            )
                        if member_size > info.file_size:
                            raise MinerUProtocolError(
                                "MinerU ZIP member exceeded its declared size"
                            )
                        digest.update(chunk)
                if member_size != info.file_size:
                    raise MinerUProtocolError(
                        "MinerU ZIP member size differs after decompression"
                    )
                member_sha256s[info.filename] = digest.hexdigest()
        except MinerUServiceError:
            raise
        except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
            raise MinerUProtocolError(
                "MinerU ZIP member failed bounded integrity verification"
            ) from exc
        return MappingProxyType(member_sha256s)


def _read_zip_member(
    data: bytes, member: str, maximum: int, *, label: str
) -> bytes:
    with zipfile.ZipFile(BytesIO(data)) as archive, archive.open(member) as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise MinerULimitError(f"{label} exceeds configured byte limit")
    return raw


def _resolve_member(content_member: str, relative: str, members: set[str]) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise MinerUProtocolError("MinerU content list has an invalid asset path")
    candidate = PurePosixPath(content_member).parent / PurePosixPath(relative)
    normalized: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                raise MinerUProtocolError("MinerU asset path escapes its result root")
            normalized.pop()
        else:
            normalized.append(part)
    name = "/".join(normalized)
    if name not in members:
        raise MinerUProtocolError("MinerU content list references a missing asset")
    return name


def _middle_artifact_receipt(
    *,
    content_member: str,
    archive: MinerUArchive,
    publication: PublishedMinerUArchive,
    allowed_pages: set[int],
    config: MinerUServiceConfig,
    producer_task_id: str,
    document_version_id: str,
    source_sha256: str,
) -> ParserArtifactReceipt:
    """Validate a sibling middle artifact and return identity metadata only.

    The rich payload is intentionally not joined to legacy content-list blocks:
    no vendor-stable block binding exists at this boundary.  Its CAS URI and
    byte digest make a later, explicitly qualified reconciler possible without
    silently inventing positional links here.
    """

    content_path = PurePosixPath(content_member)
    expected_name = (
        content_path.name[: -len("_content_list.json")] + "_middle.json"
    )
    expected_member = str(content_path.parent / expected_name)
    sibling_candidates = sorted(
        name
        for name in archive.members
        if PurePosixPath(name).parent == content_path.parent
        and PurePosixPath(name).name.endswith("_middle.json")
    )
    if not sibling_candidates:
        return ParserArtifactReceipt(
            artifact_role="mineru_middle_json",
            availability=ParserArtifactAvailability.NOT_EMITTED,
            artifact_schema=MINERU_MIDDLE_ARTIFACT_SCHEMA,
            producer_task_id=producer_task_id,
            document_version_id=document_version_id,
            source_sha256=source_sha256,
            artifact_key=expected_member,
            media_type="application/json",
            metadata={
                "receipt_schema_version": MINERU_MIDDLE_RECEIPT_SCHEMA_VERSION,
                "content_list_member": content_member,
                "configured_backend": config.backend.value,
            },
        )
    if len(sibling_candidates) != 1:
        raise MinerUProtocolError(
            "MinerU result contains multiple sibling middle JSON artifacts"
        )

    middle_member = sibling_candidates[0]
    if middle_member != expected_member:
        raise MinerUProtocolError(
            "MinerU middle JSON basename does not match the selected content list"
        )
    try:
        member_uri = publication.member_uris[middle_member]
    except KeyError as exc:
        raise MinerUProtocolError(
            "publisher omitted the MinerU middle JSON artifact"
        ) from exc

    raw = _read_zip_member(
        archive.data,
        middle_member,
        config.max_middle_json_bytes,
        label="MinerU middle JSON",
    )
    _preflight_json_depth(
        raw,
        label="MinerU middle JSON",
        maximum_depth=config.max_middle_json_depth,
    )
    payload = _strict_json(raw, label="MinerU middle JSON")
    _validate_json_structure(
        payload,
        label="MinerU middle JSON",
        maximum_depth=config.max_middle_json_depth,
        maximum_nodes=config.max_middle_json_nodes,
    )
    if not isinstance(payload, Mapping):
        raise MinerUProtocolError("MinerU middle JSON must be a JSON object")
    pages = payload.get("pdf_info")
    if not isinstance(pages, list):
        raise MinerUProtocolError(
            "MinerU middle JSON must contain a pdf_info array"
        )
    if len(pages) != len(allowed_pages):
        raise MinerUProtocolError(
            "MinerU middle JSON page count differs from the submitted page range"
        )
    observed_pages: set[int] = set()
    for page in pages:
        if not isinstance(page, Mapping):
            raise MinerUProtocolError(
                "MinerU middle JSON pdf_info entries must be objects"
            )
        page_index = page.get("page_idx")
        if (
            isinstance(page_index, bool)
            or not isinstance(page_index, int)
            or page_index not in allowed_pages
            or page_index in observed_pages
        ):
            raise MinerUProtocolError(
                "MinerU middle JSON pages are not uniquely bound to the submitted range"
            )
        observed_pages.add(page_index)
    if observed_pages != allowed_pages:
        raise MinerUProtocolError(
            "MinerU middle JSON pages differ from the submitted page range"
        )

    declared_backend = payload.get("_backend")
    declared_version = payload.get("_version_name")
    for value, field_name in (
        (declared_backend, "_backend"),
        (declared_version, "_version_name"),
    ):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or len(value) > 256
        ):
            raise MinerUProtocolError(
                f"MinerU middle JSON {field_name} must be a short non-empty string"
            )

    return ParserArtifactReceipt(
        artifact_role="mineru_middle_json",
        availability=ParserArtifactAvailability.EMITTED,
        artifact_schema=MINERU_MIDDLE_ARTIFACT_SCHEMA,
        producer_task_id=producer_task_id,
        document_version_id=document_version_id,
        source_sha256=source_sha256,
        artifact_key=middle_member,
        uri=member_uri,
        sha256=archive.member_sha256s[middle_member],
        size_bytes=len(raw),
        media_type="application/json",
        metadata={
            "receipt_schema_version": MINERU_MIDDLE_RECEIPT_SCHEMA_VERSION,
            "content_list_member": content_member,
            "member_name": middle_member,
            "backend": declared_backend,
            "version": declared_version,
        },
    )


def _result_archive_receipt(
    *,
    request: ParseRequest,
    task: MinerUTask,
    archive: MinerUArchive,
    publication: PublishedMinerUArchive,
    first_page: int,
    last_page: int,
) -> ParserArtifactReceipt:
    """Bind the exact published primary task archive at result scope."""

    return ParserArtifactReceipt(
        artifact_role="mineru_result_archive",
        availability=ParserArtifactAvailability.EMITTED,
        artifact_schema=MINERU_RESULT_ARCHIVE_ARTIFACT_SCHEMA,
        producer_task_id=task.task_id,
        document_version_id=request.document.document_version_id,
        source_sha256=request.document.content_sha256,
        artifact_key="result.zip",
        uri=publication.archive_uri,
        sha256=archive.archive_sha256,
        size_bytes=len(archive.data),
        media_type="application/zip",
        metadata={
            "receipt_schema_version": (
                MINERU_RESULT_ARCHIVE_RECEIPT_SCHEMA_VERSION
            ),
            "first_page": first_page,
            "last_page": last_page,
            "page_count": last_page - first_page + 1,
        },
    )


def _normalize_block(
    value: object,
    *,
    position: int,
    allowed_pages: set[int],
    content_member: str,
    publication: PublishedMinerUArchive,
    task: MinerUTask,
    service_revision: str,
    effort: MinerUEffort,
    archive_sha256: str,
    member_sha256s: Mapping[str, str],
    middle_artifact_receipt_id: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MinerUProtocolError("MinerU content-list block must be a JSON object")
    block = dict(value)
    page = block.get("page_idx")
    if isinstance(page, bool) or not isinstance(page, int) or page not in allowed_pages:
        raise MinerUProtocolError("MinerU block page is outside the submitted page range")
    bbox = block.get("bbox")
    middle = block.get("middle_json", {})
    if not isinstance(middle, Mapping):
        raise MinerUProtocolError("MinerU block middle_json must be an object")
    middle_copy = dict(middle)
    if bbox is not None:
        if (
            not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes, bytearray))
            or len(bbox) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or not 0 <= float(item) <= 1000
                for item in bbox
            )
        ):
            raise MinerUProtocolError("MinerU content-list bbox must be within 0..1000")
        x0, y0, x1, y1 = (float(item) for item in bbox)
        if x1 < x0 or y1 < y0:
            raise MinerUProtocolError("MinerU content-list bbox has inverted bounds")
        middle_copy.setdefault("bbox", list(bbox))
        middle_copy.setdefault("coordinate_space", "parser_page")
        middle_copy.setdefault("page_width", 1000)
        middle_copy.setdefault("page_height", 1000)
    middle_copy.setdefault("index", position)
    block["middle_json"] = middle_copy
    subtype = block.get("sub_type")
    if isinstance(subtype, str) and subtype:
        block.setdefault("semantic_subtype", subtype)
    raw_type = str(block.get("type", "")).lower()
    if raw_type == "code" and "text" not in block:
        block["text"] = block.get("code_body")
    if raw_type == "list" and "text" not in block:
        block["text"] = block.get("list_items")
    if raw_type == "chart" and "text" not in block:
        block["text"] = block.get("content")
    image_path = block.get("img_path")
    if image_path is not None:
        asset_member = _resolve_member(
            content_member, image_path, set(member_sha256s)
        )
        media_type, _encoding = mimetypes.guess_type(asset_member)
        if media_type is not None:
            # CAS URIs intentionally have no filename extension. Preserve the
            # type inferred from the verified archive member before replacing
            # the vendor-relative path.
            block["img_media_type"] = media_type
        try:
            block["img_path"] = publication.member_uris[asset_member]
        except KeyError as exc:
            raise MinerUProtocolError("publisher omitted a referenced MinerU asset") from exc
        # This exact digest comes from decompressed ZIP bytes, never from the
        # vendor content list or publication callback.
        block["img_sha256"] = member_sha256s[asset_member]
    try:
        raw_uri = publication.member_uris[content_member]
    except KeyError as exc:
        raise MinerUProtocolError("publisher omitted the MinerU content-list artifact") from exc
    block["raw_artifact_uri"] = raw_uri
    block["_mineru_service"] = MappingProxyType(
        {
            "schema_version": MINERU_SERVICE_SCHEMA_VERSION,
            "operation_id": task.operation_id,
            "task_id": task.task_id,
            "archive_uri": publication.archive_uri,
            "archive_sha256": archive_sha256,
            "service_revision_sha256": service_revision,
            "effort": effort.value,
            "content_list_member": content_member,
            "content_list_sha256": member_sha256s[content_member],
            "middle_artifact_receipt_id": middle_artifact_receipt_id,
            "vendor_task_state": "process_local_non_durable",
        }
    )
    return MappingProxyType(block)


class MinerUServiceLoader:
    """Resolve source bytes, execute bounded tasks, publish, and normalize output."""

    def __init__(
        self,
        client: MinerUAPIClient,
        *,
        source_loader: MinerUSourceLoader,
        artifact_publisher: MinerUArtifactPublisher,
    ) -> None:
        if not isinstance(client, MinerUAPIClient):
            raise TypeError("client must be MinerUAPIClient")
        if not callable(source_loader) or not callable(artifact_publisher):
            raise TypeError("source_loader and artifact_publisher must be callable")
        self.client = client
        self._source_loader = source_loader
        self._publisher = artifact_publisher

    @property
    def config_sha256(self) -> str:
        return self.client.config.revision_sha256

    def __call__(self, request: ParseRequest) -> MinerULoadedBlocks:
        try:
            return self.load(request)
        except ParserExecutionError:
            raise
        except Exception as exc:
            raise ParserExecutionError(self._execution_issue(exc)) from exc

    @staticmethod
    def _execution_issue(exc: Exception) -> ParseIssue:
        if isinstance(exc, MinerUTimeoutError):
            code = ParseErrorCode.TIMEOUT
            classification = ErrorClassification.RETRYABLE
        elif isinstance(exc, MinerUTaskStateLostError):
            code = ParseErrorCode.DEPENDENCY_UNAVAILABLE
            classification = ErrorClassification.RETRYABLE
        elif isinstance(exc, MinerUTransportError):
            code = ParseErrorCode.DEPENDENCY_UNAVAILABLE
            classification = ErrorClassification.RETRYABLE
        elif isinstance(exc, MinerUTaskFailedError):
            code = ParseErrorCode.PARSER_FAILURE
            classification = ErrorClassification.PERMANENT
        elif isinstance(exc, MinerULimitError):
            code = ParseErrorCode.RESOURCE_EXHAUSTED
            classification = ErrorClassification.PERMANENT
        elif isinstance(
            exc,
            (
                MinerUConfigurationError,
                MinerUSourceBindingError,
                MinerUProtocolError,
                SourceMaterialError,
                TypeError,
                ValueError,
            ),
        ):
            code = ParseErrorCode.CONTRACT_VIOLATION
            classification = ErrorClassification.PERMANENT
        else:
            code = ParseErrorCode.PARSER_FAILURE
            classification = ErrorClassification.RETRYABLE
        return ParseIssue(
            code=code,
            classification=classification,
            severity=IssueSeverity.ERROR,
            message="MinerU service execution failed",
            details={
                "exception_type": type(exc).__name__,
                "service": "mineru-api-or-router",
                "vendor_task_state": "process_local_non_durable",
            },
        )

    def load(self, request: ParseRequest) -> MinerULoadedBlocks:
        if not isinstance(request, ParseRequest):
            raise TypeError("request must be ParseRequest")
        effort = request.options.mineru_effort
        if effort is None or effort != self.client.config.effort:
            raise MinerUConfigurationError(
                "ParseRequest mineru_effort must explicitly equal the MinerU service effort"
            )
        source = self._source_loader(request)
        self._validate_source(request, source)
        health = self.client.health()
        ranges = _page_ranges(
            [page.page_index for page in request.pages],
            min(self.client.config.max_pages_per_task, health.processing_window_size),
        )
        if len(ranges) > self.client.config.max_tasks_per_request:
            raise MinerULimitError("ParseRequest requires too many MinerU tasks")
        operation_id = stable_id(
            "mineruop", request.request_id, self.client.config.revision_sha256,
            effort.value, ranges,
        )
        blocks: list[Mapping[str, object]] = []
        task_ids: list[str] = []
        archive_hashes: list[str] = []
        artifact_receipts: list[ParserArtifactReceipt] = []
        request_timeout = request.options.timeout_seconds
        for first_page, last_page in ranges:
            try:
                task_operation = stable_id(
                    "minerutask", operation_id, first_page, last_page
                )
                task = self.client.submit(
                    request,
                    source,
                    first_page=first_page,
                    last_page=last_page,
                    operation_id=task_operation,
                )
                self.client.wait(task, timeout_seconds=request_timeout)
                archive = self.client.result(task)
                publication = self._publisher(archive)
                if publication.archive_sha256 != archive.archive_sha256:
                    raise MinerUProtocolError(
                        "publisher archive digest differs from downloaded bytes"
                    )
                unknown = set(publication.member_uris).difference(archive.members)
                if unknown:
                    raise MinerUProtocolError(
                        "publisher returned URIs for unknown ZIP members"
                    )
                # Publish the primary receipt before decoding vendor output so
                # a later contract/limit failure cannot orphan a trusted archive.
                artifact_receipts.append(
                    _result_archive_receipt(
                        request=request,
                        task=task,
                        archive=archive,
                        publication=publication,
                        first_page=first_page,
                        last_page=last_page,
                    )
                )
                task_blocks, middle_receipt = self._load_archive_blocks(
                    request,
                    task,
                    archive,
                    publication,
                    first_page,
                    last_page,
                )
                artifact_receipts.append(middle_receipt)
                if len(blocks) + len(task_blocks) > self.client.config.max_blocks:
                    raise MinerULimitError(
                        "MinerU output exceeds configured block limit"
                    )
                blocks.extend(task_blocks)
                task_ids.append(task.task_id)
                archive_hashes.append(archive.archive_sha256)
            except ParserExecutionError as exc:
                raise ParserExecutionError(
                    exc.issue,
                    artifacts=tuple(artifact_receipts) + exc.artifacts,
                ) from exc
            except Exception as exc:
                if artifact_receipts:
                    raise ParserExecutionError(
                        self._execution_issue(exc),
                        artifacts=tuple(artifact_receipts),
                    ) from exc
                raise
        return MinerULoadedBlocks(
            operation_id, self.client.config.revision_sha256, effort,
            tuple(blocks), tuple(task_ids), tuple(archive_hashes),
            tuple(artifact_receipts),
        )

    def _validate_source(self, request: ParseRequest, source: VerifiedSourceMaterial) -> None:
        if not isinstance(source, VerifiedSourceMaterial):
            raise MinerUSourceBindingError("source_loader must return VerifiedSourceMaterial")
        if source.data is None:
            raise MinerUSourceBindingError(
                "verified source bytes were not retained; resolve with include_bytes=True"
            )
        if (
            source.document_version_id != request.document.document_version_id
            or source.sha256 != request.document.content_sha256
            or source.media_type != request.document.mime_type
            or source.size_bytes != len(source.data)
            or hashlib.sha256(source.data).hexdigest() != source.sha256
        ):
            raise MinerUSourceBindingError("verified source material identity is inconsistent")
        if source.size_bytes > self.client.config.max_source_bytes:
            raise MinerULimitError("source PDF exceeds MinerU upload byte limit")
        if source.media_type != "application/pdf":
            raise MinerUSourceBindingError("MinerU routed loader currently accepts PDF only")

    def _load_archive_blocks(
        self,
        request: ParseRequest,
        task: MinerUTask,
        archive: MinerUArchive,
        publication: PublishedMinerUArchive,
        first_page: int,
        last_page: int,
    ) -> tuple[tuple[Mapping[str, object], ...], ParserArtifactReceipt]:
        candidates = [
            name for name in archive.members
            if name.endswith("_content_list.json")
            and not name.endswith("_content_list_v2.json")
        ]
        if len(candidates) != 1:
            raise MinerUProtocolError(
                "MinerU ZIP must contain exactly one legacy content_list.json for one uploaded file"
            )
        member = candidates[0]
        if member not in publication.member_uris:
            raise MinerUProtocolError(
                "publisher omitted the MinerU content-list artifact"
            )
        raw = _read_zip_member(
            archive.data,
            member,
            self.client.config.max_content_list_bytes,
            label="MinerU content list",
        )
        payload = _strict_json(raw, label="MinerU content list")
        if not isinstance(payload, list):
            raise MinerUProtocolError("MinerU content list must be a JSON array")
        if len(payload) > self.client.config.max_blocks:
            raise MinerULimitError("MinerU content list exceeds configured block limit")
        allowed = set(range(first_page, last_page + 1)).intersection(request.page_by_index)
        middle_artifact = _middle_artifact_receipt(
            content_member=member,
            archive=archive,
            publication=publication,
            allowed_pages=allowed,
            config=self.client.config,
            producer_task_id=task.task_id,
            document_version_id=request.document.document_version_id,
            source_sha256=request.document.content_sha256,
        )
        blocks = tuple(
            _normalize_block(
                value, position=index, allowed_pages=allowed, content_member=member,
                publication=publication, task=task,
                service_revision=self.client.config.revision_sha256,
                effort=self.client.config.effort,
                archive_sha256=archive.archive_sha256,
                member_sha256s=archive.member_sha256s,
                middle_artifact_receipt_id=middle_artifact.receipt_id,
            )
            for index, value in enumerate(payload)
        )
        return blocks, middle_artifact


__all__ = [
    "MINERU_API_PROTOCOL_VERSION",
    "MINERU_MIDDLE_ARTIFACT_SCHEMA",
    "MINERU_MIDDLE_RECEIPT_SCHEMA_VERSION",
    "MINERU_RESULT_ARCHIVE_ARTIFACT_SCHEMA",
    "MINERU_RESULT_ARCHIVE_RECEIPT_SCHEMA_VERSION",
    "MINERU_SERVICE_SCHEMA_VERSION",
    "MAX_MINERU_TASKS_PER_PARSE_RESULT",
    "MinerUAPIClient",
    "MinerUArchive",
    "MinerUArtifactPublisher",
    "MinerUBackend",
    "MinerUConfigurationError",
    "MinerUHTTPResponse",
    "MinerUHTTPRequestBody",
    "MinerUHTTPTransport",
    "MinerUHealth",
    "MinerULimitError",
    "MinerULoadedBlocks",
    "MinerUProtocolError",
    "MinerUServiceConfig",
    "MinerUServiceError",
    "MinerUServiceLoader",
    "MinerUSourceBindingError",
    "MinerUSourceLoader",
    "MinerUTask",
    "MinerUTaskFailedError",
    "MinerUTaskStateLostError",
    "MinerUTimeoutError",
    "MinerUTransportError",
    "PublishedMinerUArchive",
    "StdlibMinerUHTTPTransport",
]
