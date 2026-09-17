"""Bounded standard-library client for the supported Docling Serve REST API.

This module implements the asynchronous single-file flow documented by
Docling Serve: submit ``/v1/convert/source/async``, poll
``/v1/status/poll/{task_id}``, and fetch ``/v1/result/{task_id}``.  The source
is sent as the documented base64 ``file_sources`` JSON shape and the only
requested output is the lossless Docling JSON document.

The client is intentionally not a Docling SDK replacement.  It imports no
Docling, Pydantic, HTTP framework, GPU, or model package.  Its output is the
JSON-compatible mapping expected by :class:`DoclingExportAdapter`.  The exact
source bytes must be supplied through an injected verified-material loader;
``DocumentRef.source_uri`` is never opened or fetched.

API facts are bound to the official Docling REST documentation synchronized
from docling-serve v1.21.0.  That documentation does not expose a server-build
identity in conversion responses, so this module does not claim to qualify or
pin a particular container, parser model, or service release.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import time
from types import MappingProxyType
from typing import Protocol
from urllib.parse import SplitResult, urlsplit

from document_services.domain.base import canonical_json_bytes
from document_services.domain.parsing import (
    ErrorClassification,
    IssueSeverity,
    OcrMode,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParserExecutionError,
)
from document_services.evidence.sources import VerifiedSourceMaterial


DOCLING_SERVE_API_CONTRACT = "docling-serve.rest.v1.21.0-docs"
DOCLING_DOCUMENT_SCHEMA_NAME = "DoclingDocument"
DEFAULT_DOCLING_DOCUMENT_SCHEMA_VERSION = "1.10.0"

_ASYNC_CONVERT_PATH = "/v1/convert/source/async"
_POLL_PATH_PREFIX = "/v1/status/poll/"
_RESULT_PATH_PREFIX = "/v1/result/"
_MIB = 1024 * 1024
_MAX_BODY_LIMIT = 1024 * _MIB
_MAX_SOURCE_LIMIT = 512 * _MIB
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_SEMVER = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
)
_PRESET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TASK_FIELDS = frozenset(
    {"task_id", "task_status", "task_position", "task_meta", "task_type"}
)
_RESULT_FIELDS = frozenset(
    {"document", "status", "processing_time", "timings", "errors"}
)
_DOCUMENT_RESPONSE_FIELDS = frozenset(
    {
        "filename",
        "md_content",
        "json_content",
        "html_content",
        "text_content",
        "doctags_content",
        "doclang_content",
    }
)
_KNOWN_CONVERSION_STATUSES = frozenset(
    {"success", "partial_success", "skipped", "failure"}
)
_KNOWN_TASK_STATUSES = frozenset({"pending", "started", "success", "failure"})
_UNSUPPORTED_EXPORT_COLLECTIONS = (
    "key_value_items",
    "form_items",
    "field_regions",
    "field_items",
)


class DoclingServiceError(RuntimeError):
    """Base failure at the dependency-light Docling service boundary."""


class DoclingServiceConfigurationError(DoclingServiceError, ValueError):
    """A URL, limit, option, or injected dependency is invalid."""


class DoclingServiceSourceError(DoclingServiceError, ValueError):
    """The supplied bytes are missing, unsafe, or bound to another source."""


class DoclingServiceTransportError(DoclingServiceError):
    """A connection or HTTP framing failure occurred."""


class DoclingServiceTimeoutError(DoclingServiceTransportError, TimeoutError):
    """A per-request or whole-operation deadline was exhausted."""


class DoclingServiceRemoteError(DoclingServiceError):
    """The service returned a known terminal failure or non-success HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        retryable: bool = True,
    ) -> None:
        self.http_status = http_status
        self.retryable = retryable
        super().__init__(message)


class DoclingServiceContractError(DoclingServiceError, ValueError):
    """The service response drifted from the supported strict wire contract."""


def _positive_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise DoclingServiceConfigurationError(
            f"{field_name} must be a finite positive number"
        )
    return float(value)


def _bounded_integer(
    value: object, field_name: str, *, minimum: int, maximum: int
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DoclingServiceConfigurationError(
            f"{field_name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _endpoint(value: object, *, allow_insecure_http: bool) -> SplitResult:
    if not isinstance(value, str) or not value.strip():
        raise DoclingServiceConfigurationError("base_url must be non-empty")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise DoclingServiceConfigurationError(
            "base_url scheme must be http or https"
        )
    if parsed.hostname is None:
        raise DoclingServiceConfigurationError("base_url must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise DoclingServiceConfigurationError(
            "base_url must not contain user information"
        )
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise DoclingServiceConfigurationError(
            "base_url must not contain a path, query, or fragment"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise DoclingServiceConfigurationError(
            "base_url contains an invalid port"
        ) from exc
    if (
        parsed.scheme == "http"
        and not allow_insecure_http
        and not _is_loopback(parsed.hostname)
    ):
        raise DoclingServiceConfigurationError(
            "plain HTTP is allowed only for loopback services; use HTTPS or "
            "explicitly allow insecure HTTP"
        )
    return parsed


def _optional_preset(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _PRESET.fullmatch(value) is None:
        raise DoclingServiceConfigurationError(
            f"{field_name} must be a bounded preset identifier"
        )
    return value


@dataclass(frozen=True, slots=True)
class DoclingServiceConfig:
    """Immutable transport, conversion, schema, and resource policy.

    ``expected_document_schema_version`` is a wire-schema compatibility pin,
    not evidence that a service image/model combination has been qualified.
    """

    base_url: str
    api_key: str | None = field(default=None, repr=False)
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    operation_timeout_seconds: float = 1800.0
    poll_interval_seconds: float = 1.0
    max_poll_attempts: int = 3600
    max_source_bytes: int = 32 * _MIB
    max_request_bytes: int = 48 * _MIB
    max_response_bytes: int = 256 * _MIB
    max_pages_per_request: int = 64
    expected_document_schema_version: str = DEFAULT_DOCLING_DOCUMENT_SCHEMA_VERSION
    allow_insecure_http: bool = False
    tls_ca_file: str | None = None
    ocr_preset: str = "auto"
    table_mode: str = "accurate"
    pipeline: str = "standard"
    picture_description_preset: str | None = None
    enable_picture_classification: bool = False
    enable_chart_extraction: bool = False
    enable_code_enrichment: bool = False
    _endpoint: SplitResult = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in (
            "allow_insecure_http",
            "enable_picture_classification",
            "enable_chart_extraction",
            "enable_code_enrichment",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise DoclingServiceConfigurationError(
                    f"{field_name} must be a boolean"
                )
        endpoint = _endpoint(
            self.base_url, allow_insecure_http=self.allow_insecure_http
        )
        for field_name in (
            "connect_timeout_seconds",
            "request_timeout_seconds",
            "operation_timeout_seconds",
            "poll_interval_seconds",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_number(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "max_poll_attempts",
            _bounded_integer(
                self.max_poll_attempts,
                "max_poll_attempts",
                minimum=1,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "max_source_bytes",
            _bounded_integer(
                self.max_source_bytes,
                "max_source_bytes",
                minimum=1,
                maximum=_MAX_SOURCE_LIMIT,
            ),
        )
        for field_name in ("max_request_bytes", "max_response_bytes"):
            object.__setattr__(
                self,
                field_name,
                _bounded_integer(
                    getattr(self, field_name),
                    field_name,
                    minimum=1,
                    maximum=_MAX_BODY_LIMIT,
                ),
            )
        object.__setattr__(
            self,
            "max_pages_per_request",
            _bounded_integer(
                self.max_pages_per_request,
                "max_pages_per_request",
                minimum=1,
                maximum=4096,
            ),
        )
        if (
            not isinstance(self.expected_document_schema_version, str)
            or _SEMVER.fullmatch(self.expected_document_schema_version) is None
        ):
            raise DoclingServiceConfigurationError(
                "expected_document_schema_version must be an exact semantic version"
            )
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or "\r" in self.api_key
            or "\n" in self.api_key
        ):
            raise DoclingServiceConfigurationError(
                "api_key must be a non-empty single-line string or null"
            )
        if self.tls_ca_file is not None and (
            not isinstance(self.tls_ca_file, str) or not self.tls_ca_file.strip()
        ):
            raise DoclingServiceConfigurationError(
                "tls_ca_file must be a non-empty path or null"
            )
        if not isinstance(self.ocr_preset, str) or _PRESET.fullmatch(
            self.ocr_preset
        ) is None:
            raise DoclingServiceConfigurationError(
                "ocr_preset must be a bounded preset identifier"
            )
        if self.table_mode not in {"fast", "accurate"}:
            raise DoclingServiceConfigurationError(
                "table_mode must be 'fast' or 'accurate'"
            )
        if self.pipeline not in {"standard", "vlm"}:
            raise DoclingServiceConfigurationError(
                "pipeline must be 'standard' or 'vlm'"
            )
        object.__setattr__(
            self,
            "picture_description_preset",
            _optional_preset(
                self.picture_description_preset,
                "picture_description_preset",
            ),
        )
        object.__setattr__(self, "_endpoint", endpoint)

    @property
    def endpoint(self) -> SplitResult:
        return self._endpoint

    def safe_dict(self) -> dict[str, object]:
        """Return a secret-free configuration identity suitable for audit/hash."""

        return {
            "api_contract": DOCLING_SERVE_API_CONTRACT,
            "base_url": self.base_url,
            "api_key_configured": self.api_key is not None,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "operation_timeout_seconds": self.operation_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_poll_attempts": self.max_poll_attempts,
            "max_source_bytes": self.max_source_bytes,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_pages_per_request": self.max_pages_per_request,
            "expected_document_schema_name": DOCLING_DOCUMENT_SCHEMA_NAME,
            "expected_document_schema_version": self.expected_document_schema_version,
            "allow_insecure_http": self.allow_insecure_http,
            "tls_ca_file": self.tls_ca_file,
            "ocr_preset": self.ocr_preset,
            "table_mode": self.table_mode,
            "pipeline": self.pipeline,
            "picture_description_preset": self.picture_description_preset,
            "enable_picture_classification": self.enable_picture_classification,
            "enable_chart_extraction": self.enable_chart_extraction,
            "enable_code_enrichment": self.enable_code_enrichment,
            "service_release_qualified": False,
            "model_identity_verified_by_api": False,
        }


@dataclass(frozen=True, slots=True)
class DoclingHTTPResponse:
    """Small immutable response used by the injectable HTTP exchange seam."""

    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.status, bool)
            or not isinstance(self.status, int)
            or not 100 <= self.status <= 599
        ):
            raise DoclingServiceContractError("HTTP response status is invalid")
        if not isinstance(self.body, bytes):
            raise DoclingServiceContractError("HTTP response body must be bytes")
        if not isinstance(self.headers, Mapping):
            raise DoclingServiceContractError("HTTP response headers must be a mapping")
        normalized: dict[str, str] = {}
        for raw_name, raw_value in self.headers.items():
            if not isinstance(raw_name, str) or not isinstance(raw_value, str):
                raise DoclingServiceContractError(
                    "HTTP response headers must contain string pairs"
                )
            name = raw_name.lower()
            if name in normalized:
                raise DoclingServiceContractError(
                    f"HTTP response repeats header {name!r}"
                )
            normalized[name] = raw_value
        object.__setattr__(self, "headers", MappingProxyType(normalized))


class DoclingHTTPExchange(Protocol):
    """Injectable one-request HTTP seam; implementations must not retry."""

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> DoclingHTTPResponse:
        ...


SourceMaterialLoader = Callable[[ParseRequest], VerifiedSourceMaterial]


class _StdlibDoclingHTTPExchange:
    """Fresh-connection, no-retry HTTP implementation."""

    def __init__(self, config: DoclingServiceConfig) -> None:
        self._config = config

    def _connection(self, timeout_seconds: float) -> http.client.HTTPConnection:
        endpoint = self._config.endpoint
        connect_timeout = min(
            self._config.connect_timeout_seconds, timeout_seconds
        )
        if endpoint.scheme == "https":
            try:
                context = ssl.create_default_context(
                    cafile=self._config.tls_ca_file
                )
            except (OSError, ssl.SSLError) as exc:
                raise DoclingServiceConfigurationError(
                    "could not create the configured TLS context"
                ) from exc
            return http.client.HTTPSConnection(
                endpoint.hostname,
                endpoint.port or 443,
                timeout=connect_timeout,
                context=context,
            )
        return http.client.HTTPConnection(
            endpoint.hostname,
            endpoint.port or 80,
            timeout=connect_timeout,
        )

    @staticmethod
    def _headers(response: http.client.HTTPResponse) -> Mapping[str, str]:
        normalized: dict[str, str] = {}
        for raw_name, raw_value in response.getheaders():
            name = raw_name.lower()
            if name in normalized and name in {
                "content-length",
                "content-type",
                "content-encoding",
            }:
                raise DoclingServiceContractError(
                    f"HTTP response repeats security-sensitive header {name!r}"
                )
            normalized[name] = (
                raw_value
                if name not in normalized
                else f"{normalized[name]}, {raw_value}"
            )
        return normalized

    @staticmethod
    def _read_body(
        response: http.client.HTTPResponse, maximum: int
    ) -> bytes:
        declared = response.getheader("Content-Length")
        if declared is not None:
            try:
                length = int(declared, 10)
            except ValueError as exc:
                raise DoclingServiceContractError(
                    "Docling response Content-Length is invalid"
                ) from exc
            if length < 0:
                raise DoclingServiceContractError(
                    "Docling response Content-Length is negative"
                )
            if length > maximum:
                raise DoclingServiceContractError(
                    "Docling response exceeds max_response_bytes"
                )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(64 * 1024, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise DoclingServiceContractError(
                    "Docling response exceeds max_response_bytes"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> DoclingHTTPResponse:
        connection = self._connection(timeout_seconds)
        try:
            connection.connect()
            if connection.sock is None:
                raise DoclingServiceTransportError(
                    "Docling connection did not expose a connected socket"
                )
            connection.sock.settimeout(timeout_seconds)
            connection.request(method, path, body=body, headers=dict(headers))
            response = connection.getresponse()
            response_headers = self._headers(response)
            response_body = self._read_body(response, max_response_bytes)
            return DoclingHTTPResponse(
                status=response.status,
                headers=response_headers,
                body=response_body,
            )
        except (socket.timeout, TimeoutError) as exc:
            raise DoclingServiceTimeoutError(
                "Docling HTTP request exceeded its per-call timeout"
            ) from exc
        except DoclingServiceError:
            raise
        except (ConnectionError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise DoclingServiceTransportError(
                "Docling HTTP request failed"
            ) from exc
        finally:
            connection.close()


def _reject_unknown(
    payload: Mapping[str, object], allowed: frozenset[str], label: str
) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise DoclingServiceContractError(
            f"{label} has unknown field(s): {', '.join(unknown)}"
        )


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    def pairs(values: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise DoclingServiceContractError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise DoclingServiceContractError(
            f"{label} contains non-finite JSON number {value!r}"
        )

    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except DoclingServiceContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DoclingServiceContractError(
            f"{label} is not a valid bounded UTF-8 JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise DoclingServiceContractError(f"{label} must be a JSON object")
    return value


def _finite_non_negative(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise DoclingServiceContractError(
            f"{field_name} must be a finite non-negative number"
        )
    return float(value)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise DoclingServiceContractError(f"{field_name} must be a JSON object")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise DoclingServiceContractError(f"{field_name} must be a JSON array")
    return value


@dataclass(frozen=True, slots=True)
class _Task:
    task_id: str
    status: str


class DoclingServiceExportLoader:
    """Callable export loader for direct injection into ``DoclingExportAdapter``.

    ``source_loader`` must return one ``VerifiedSourceMaterial`` with immutable
    bytes retained.  A typical composition root passes::

        lambda request: resolver.resolve(request.document, include_bytes=True)

    No fallback to the logical/citation ``source_uri`` exists.
    """

    def __init__(
        self,
        config: DoclingServiceConfig,
        *,
        source_loader: SourceMaterialLoader,
        exchange: DoclingHTTPExchange | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, DoclingServiceConfig):
            raise TypeError("config must be DoclingServiceConfig")
        if not callable(source_loader):
            raise TypeError("source_loader must be callable")
        if exchange is not None and not callable(getattr(exchange, "request", None)):
            raise TypeError("exchange must implement request()")
        if not callable(monotonic) or not callable(sleeper):
            raise TypeError("monotonic and sleeper must be callable")
        self._config = config
        self._source_loader = source_loader
        self._exchange = exchange or _StdlibDoclingHTTPExchange(config)
        self._monotonic = monotonic
        self._sleeper = sleeper

    @property
    def config(self) -> DoclingServiceConfig:
        return self._config

    @property
    def safe_config(self) -> Mapping[str, object]:
        return MappingProxyType(self._config.safe_dict())

    def _operation_limit(self, request: ParseRequest) -> float:
        limit = self._config.operation_timeout_seconds
        if request.options.timeout_seconds is not None:
            limit = min(limit, float(request.options.timeout_seconds))
        return limit

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise DoclingServiceTimeoutError(
                "Docling conversion exceeded its total operation deadline"
            )
        return remaining

    def _source(self, request: ParseRequest, deadline: float) -> bytes:
        try:
            material = self._source_loader(request)
        except DoclingServiceError:
            raise
        except Exception as exc:
            raise DoclingServiceSourceError(
                "verified source loader failed "
                f"({type(exc).__name__}); logical source URI was not used"
            ) from exc
        self._remaining(deadline)
        if not isinstance(material, VerifiedSourceMaterial):
            raise DoclingServiceSourceError(
                "source_loader must return VerifiedSourceMaterial"
            )
        if material.document_version_id != request.document.document_version_id:
            raise DoclingServiceSourceError(
                "verified source belongs to another document version"
            )
        if material.logical_source_uri != request.document.source_uri:
            raise DoclingServiceSourceError(
                "verified source logical URI differs from the request"
            )
        if material.media_type != request.document.mime_type:
            raise DoclingServiceSourceError(
                "verified source media type differs from the request"
            )
        if not hmac.compare_digest(
            material.sha256, request.document.content_sha256
        ):
            raise DoclingServiceSourceError(
                "verified source digest differs from the request"
            )
        if not isinstance(material.data, bytes):
            raise DoclingServiceSourceError(
                "verified source loader must retain an immutable byte snapshot"
            )
        if not 1 <= len(material.data) <= self._config.max_source_bytes:
            raise DoclingServiceSourceError(
                "verified source size is outside max_source_bytes"
            )
        if material.size_bytes != len(material.data):
            raise DoclingServiceSourceError(
                "verified source size metadata differs from retained bytes"
            )
        digest = hashlib.sha256(material.data).hexdigest()
        if not hmac.compare_digest(digest, request.document.content_sha256):
            raise DoclingServiceSourceError(
                "retained source bytes differ from the document digest"
            )
        return material.data

    def _pages(self, request: ParseRequest) -> tuple[int, int, frozenset[int]]:
        if request.document.mime_type != "application/pdf":
            raise DoclingServiceSourceError(
                "Docling service loader currently accepts application/pdf only"
            )
        if request.regions:
            raise DoclingServiceSourceError(
                "Docling service loader does not support region selection"
            )
        if request.options.extra:
            raise DoclingServiceSourceError(
                "Docling service loader refuses unmapped ParseOptions.extra values"
            )
        indices = tuple(page.page_index for page in request.pages)
        if len(indices) > self._config.max_pages_per_request:
            raise DoclingServiceSourceError(
                "parse request exceeds max_pages_per_request"
            )
        first, last = indices[0], indices[-1]
        expected = tuple(range(first, last + 1))
        if indices != expected:
            raise DoclingServiceSourceError(
                "Docling page_range requires a contiguous page selection"
            )
        return first + 1, last + 1, frozenset(index + 1 for index in indices)

    @staticmethod
    def _filename(request: ParseRequest) -> str:
        candidate = request.document.display_name
        if (
            isinstance(candidate, str)
            and 1 <= len(candidate.encode("utf-8")) <= 255
            and candidate not in {".", ".."}
            and "/" not in candidate
            and "\\" not in candidate
            and "\x00" not in candidate
            and candidate.lower().endswith(".pdf")
        ):
            return candidate
        return "document.pdf"

    def _options(
        self,
        request: ParseRequest,
        *,
        first_page: int,
        last_page: int,
        document_timeout: float,
    ) -> dict[str, object]:
        languages = tuple(request.options.languages)
        if len(languages) > 16 or any(
            len(value.encode("utf-8")) > 64
            or any(ord(character) < 0x20 for character in value)
            for value in languages
        ):
            raise DoclingServiceSourceError(
                "OCR language selection exceeds service boundary limits"
            )
        do_ocr = request.options.ocr_mode is not OcrMode.DISABLED
        force_ocr = request.options.ocr_mode is OcrMode.FORCE
        parse_images = request.options.parse_images
        options: dict[str, object] = {
            "from_formats": ["pdf"],
            "to_formats": ["json"],
            "image_export_mode": "embedded" if parse_images else "placeholder",
            "do_ocr": do_ocr,
            "force_ocr": force_ocr,
            "ocr_preset": self._config.ocr_preset,
            "table_mode": self._config.table_mode,
            "pipeline": self._config.pipeline,
            "page_range": [first_page, last_page],
            "document_timeout": document_timeout,
            "abort_on_error": True,
            "do_table_structure": request.options.parse_tables,
            "include_images": parse_images,
            "include_page_images": False,
            "do_formula_enrichment": request.options.parse_formulas,
            "do_picture_classification": (
                parse_images and self._config.enable_picture_classification
            ),
            "do_chart_extraction": (
                parse_images and self._config.enable_chart_extraction
            ),
            "do_code_enrichment": self._config.enable_code_enrichment,
            "do_picture_description": (
                parse_images
                and self._config.picture_description_preset is not None
            ),
        }
        if languages:
            options["ocr_lang"] = list(languages)
        if parse_images and self._config.picture_description_preset is not None:
            options["picture_description_preset"] = (
                self._config.picture_description_preset
            )
        return options

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        remaining = self._remaining(deadline)
        body = None if payload is None else canonical_json_bytes(payload)
        if body is not None and len(body) > self._config.max_request_bytes:
            raise DoclingServiceSourceError(
                "Docling JSON request exceeds max_request_bytes"
            )
        headers: dict[str, str] = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._config.api_key is not None:
            headers["X-Api-Key"] = self._config.api_key
        timeout = min(self._config.request_timeout_seconds, remaining)
        try:
            response = self._exchange.request(
                method,
                path,
                headers=headers,
                body=body,
                timeout_seconds=timeout,
                max_response_bytes=self._config.max_response_bytes,
            )
        except DoclingServiceError:
            raise
        except Exception as exc:
            raise DoclingServiceTransportError(
                "injected Docling HTTP exchange failed"
            ) from exc
        self._remaining(deadline)
        if not isinstance(response, DoclingHTTPResponse):
            raise DoclingServiceContractError(
                "HTTP exchange must return DoclingHTTPResponse"
            )
        if len(response.body) > self._config.max_response_bytes:
            raise DoclingServiceContractError(
                "Docling response exceeds max_response_bytes"
            )
        if response.status != 200:
            raise DoclingServiceRemoteError(
                f"Docling service returned HTTP {response.status}",
                http_status=response.status,
                retryable=(
                    response.status in {408, 425, 429} or response.status >= 500
                ),
            )
        content_encoding = response.headers.get("content-encoding")
        if content_encoding is not None and content_encoding.lower() != "identity":
            raise DoclingServiceContractError(
                "compressed Docling responses are unsupported at this bounded boundary"
            )
        content_type = response.headers.get("content-type")
        if content_type is None or content_type.split(";", 1)[0].strip().lower() != (
            "application/json"
        ):
            raise DoclingServiceContractError(
                "Docling response Content-Type must be application/json"
            )
        return _json_object(response.body, "Docling response")

    def _task(
        self,
        payload: Mapping[str, object],
        *,
        expected_task_id: str | None = None,
        prior_status: str | None = None,
    ) -> _Task:
        _reject_unknown(payload, _TASK_FIELDS, "Docling task response")
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
            raise DoclingServiceContractError(
                "Docling task_id is missing or unsafe"
            )
        if expected_task_id is not None and not hmac.compare_digest(
            task_id, expected_task_id
        ):
            raise DoclingServiceContractError(
                "Docling poll response changed task identity"
            )
        status = payload.get("task_status")
        if not isinstance(status, str) or status not in _KNOWN_TASK_STATUSES:
            raise DoclingServiceContractError(
                "Docling task_status is unsupported"
            )
        task_type = payload.get("task_type")
        if task_type is not None and task_type != "convert":
            raise DoclingServiceContractError(
                "Docling task_type is not a conversion task"
            )
        position = payload.get("task_position")
        if position is not None and (
            isinstance(position, bool) or not isinstance(position, int) or position < 0
        ):
            raise DoclingServiceContractError(
                "Docling task_position must be null or a non-negative integer"
            )
        meta = payload.get("task_meta")
        if meta is not None:
            _mapping(meta, "Docling task_meta")
        if prior_status == "started" and status == "pending":
            raise DoclingServiceContractError(
                "Docling task status regressed from started to pending"
            )
        return _Task(task_id=task_id, status=status)

    def _wait(self, task: _Task, deadline: float) -> _Task:
        if task.status == "failure":
            raise DoclingServiceRemoteError("Docling conversion task failed")
        if task.status == "success":
            return task
        current = task
        for _attempt in range(self._config.max_poll_attempts):
            remaining = self._remaining(deadline)
            self._sleeper(min(self._config.poll_interval_seconds, remaining))
            payload = self._request_json(
                "GET",
                _POLL_PATH_PREFIX + current.task_id,
                deadline=deadline,
            )
            current = self._task(
                payload,
                expected_task_id=task.task_id,
                prior_status=current.status,
            )
            if current.status == "failure":
                raise DoclingServiceRemoteError("Docling conversion task failed")
            if current.status == "success":
                return current
        raise DoclingServiceTimeoutError(
            "Docling conversion exceeded max_poll_attempts"
        )

    def _export(
        self,
        payload: Mapping[str, object],
        *,
        request: ParseRequest,
        filename: str,
        expected_pages: frozenset[int],
    ) -> Mapping[str, object]:
        _reject_unknown(payload, _RESULT_FIELDS, "Docling conversion result")
        status = payload.get("status")
        if not isinstance(status, str) or status not in _KNOWN_CONVERSION_STATUSES:
            raise DoclingServiceContractError(
                "Docling conversion status is unsupported"
            )
        if status != "success":
            raise DoclingServiceRemoteError(
                f"Docling conversion finished with status {status!r}",
                retryable=False,
            )
        _finite_non_negative(payload.get("processing_time"), "processing_time")
        _mapping(payload.get("timings"), "timings")
        errors = _sequence(payload.get("errors"), "errors")
        if errors:
            raise DoclingServiceContractError(
                "Docling reported errors despite a success status"
            )
        document = _mapping(payload.get("document"), "document")
        _reject_unknown(
            document, _DOCUMENT_RESPONSE_FIELDS, "Docling document response"
        )
        response_filename = document.get("filename")
        if response_filename != filename:
            raise DoclingServiceContractError(
                "Docling response filename differs from the submitted source"
            )
        for field_name in (
            "md_content",
            "html_content",
            "text_content",
            "doctags_content",
            "doclang_content",
        ):
            unexpected = document.get(field_name)
            if unexpected is not None and unexpected != "":
                raise DoclingServiceContractError(
                    f"Docling populated unrequested {field_name}"
                )
        export = dict(_mapping(document.get("json_content"), "json_content"))
        if export.get("schema_name") != DOCLING_DOCUMENT_SCHEMA_NAME:
            raise DoclingServiceContractError(
                "DoclingDocument schema_name is unsupported"
            )
        if export.get("version") != self._config.expected_document_schema_version:
            raise DoclingServiceContractError(
                "DoclingDocument version differs from the configured exact schema"
            )
        origin = _mapping(export.get("origin"), "DoclingDocument.origin")
        if origin.get("mimetype") != request.document.mime_type:
            raise DoclingServiceContractError(
                "DoclingDocument origin MIME type differs from the source"
            )
        origin_filename = origin.get("filename")
        if origin_filename is not None and origin_filename != filename:
            raise DoclingServiceContractError(
                "DoclingDocument origin filename differs from the source"
            )
        pages = _mapping(export.get("pages"), "DoclingDocument.pages")
        observed_pages: set[int] = set()
        for key, raw_page in pages.items():
            page = _mapping(raw_page, f"DoclingDocument.pages[{key!r}]")
            page_no = page.get("page_no")
            if (
                isinstance(page_no, bool)
                or not isinstance(page_no, int)
                or page_no < 1
                or page_no in observed_pages
            ):
                raise DoclingServiceContractError(
                    "DoclingDocument pages contain an invalid or duplicate page_no"
                )
            observed_pages.add(page_no)
        if observed_pages != set(expected_pages):
            raise DoclingServiceContractError(
                "DoclingDocument pages differ from the exact requested page range"
            )
        for collection in _UNSUPPORTED_EXPORT_COLLECTIONS:
            value = export.get(collection, [])
            if _sequence(value, f"DoclingDocument.{collection}"):
                raise DoclingServiceContractError(
                    f"DoclingDocument contains unsupported non-empty {collection}"
                )
        declared_digest = export.get("source_sha256")
        if declared_digest is not None and declared_digest != (
            request.document.content_sha256
        ):
            raise DoclingServiceContractError(
                "DoclingDocument source digest differs from the request"
            )
        # The official DoclingDocument carries a truncated integer binary hash,
        # not the controller's SHA-256.  Add the digest only after verifying the
        # immutable submitted bytes so the existing adapter can bind it.
        export["source_sha256"] = request.document.content_sha256
        return export

    def load(self, request: ParseRequest) -> Mapping[str, object]:
        """Load one export while preserving typed service exceptions."""

        if not isinstance(request, ParseRequest):
            raise TypeError("request must be ParseRequest")
        operation_limit = self._operation_limit(request)
        deadline = self._monotonic() + operation_limit
        first_page, last_page, expected_pages = self._pages(request)
        source = self._source(request, deadline)
        filename = self._filename(request)
        payload = {
            "file_sources": [
                {
                    "base64_string": base64.b64encode(source).decode("ascii"),
                    "filename": filename,
                }
            ],
            "options": self._options(
                request,
                first_page=first_page,
                last_page=last_page,
                document_timeout=min(operation_limit, self._remaining(deadline)),
            ),
            "target": {"kind": "inbody"},
        }
        task_payload = self._request_json(
            "POST", _ASYNC_CONVERT_PATH, deadline=deadline, payload=payload
        )
        task = self._wait(self._task(task_payload), deadline)
        result_payload = self._request_json(
            "GET", _RESULT_PATH_PREFIX + task.task_id, deadline=deadline
        )
        return self._export(
            result_payload,
            request=request,
            filename=filename,
            expected_pages=expected_pages,
        )

    @staticmethod
    def _parser_execution_error(exc: DoclingServiceError) -> ParserExecutionError:
        if isinstance(exc, DoclingServiceSourceError):
            code = ParseErrorCode.INVALID_REQUEST
            classification = ErrorClassification.PERMANENT
            message = "Docling service source binding or request is invalid"
        elif isinstance(exc, DoclingServiceConfigurationError):
            code = ParseErrorCode.DEPENDENCY_UNAVAILABLE
            classification = ErrorClassification.PERMANENT
            message = "Docling service configuration is invalid"
        elif isinstance(exc, DoclingServiceContractError):
            code = ParseErrorCode.CONTRACT_VIOLATION
            classification = ErrorClassification.PERMANENT
            message = "Docling service response violates the supported contract"
        elif isinstance(exc, DoclingServiceTimeoutError):
            code = ParseErrorCode.TIMEOUT
            classification = ErrorClassification.RETRYABLE
            message = "Docling service operation timed out"
        elif isinstance(exc, DoclingServiceRemoteError):
            code = ParseErrorCode.PARSER_FAILURE
            classification = (
                ErrorClassification.RETRYABLE
                if exc.retryable
                else ErrorClassification.PERMANENT
            )
            message = "Docling service reported a conversion failure"
        else:
            code = ParseErrorCode.DEPENDENCY_UNAVAILABLE
            classification = ErrorClassification.RETRYABLE
            message = "Docling service transport is unavailable"
        details: dict[str, object] = {
            "api_contract": DOCLING_SERVE_API_CONTRACT,
            "exception_type": type(exc).__name__,
        }
        if isinstance(exc, DoclingServiceRemoteError) and exc.http_status is not None:
            details["http_status"] = exc.http_status
        return ParserExecutionError(
            ParseIssue(
                code=code,
                classification=classification,
                severity=IssueSeverity.ERROR,
                message=message,
                details=details,
            )
        )

    def __call__(self, request: ParseRequest) -> Mapping[str, object]:
        """Adapter-facing call that preserves permanent/retryable classification."""

        try:
            return self.load(request)
        except ParserExecutionError:
            raise
        except DoclingServiceError as exc:
            raise self._parser_execution_error(exc) from exc


__all__ = [
    "DEFAULT_DOCLING_DOCUMENT_SCHEMA_VERSION",
    "DOCLING_DOCUMENT_SCHEMA_NAME",
    "DOCLING_SERVE_API_CONTRACT",
    "DoclingHTTPExchange",
    "DoclingHTTPResponse",
    "DoclingServiceConfig",
    "DoclingServiceConfigurationError",
    "DoclingServiceContractError",
    "DoclingServiceError",
    "DoclingServiceExportLoader",
    "DoclingServiceRemoteError",
    "DoclingServiceSourceError",
    "DoclingServiceTimeoutError",
    "DoclingServiceTransportError",
    "SourceMaterialLoader",
]
