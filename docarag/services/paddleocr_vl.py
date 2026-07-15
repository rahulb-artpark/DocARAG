"""Bounded worker-side client for the complete PaddleOCR-VL 1.6 service.

This module is intentionally dependency-light.  It does not import PaddleOCR,
PaddlePaddle, vLLM, Transformers, Torch, a PDF renderer, or an HTTP SDK.  A
worker composition root supplies two source-bound callbacks:

* ``source_loader`` resolves the immutable PDF snapshot represented by a
  :class:`~docarag.domain.evidence.DocumentRef`; and
* ``page_renderer`` renders an explicitly requested page from those verified
  bytes into an immutable image.

The rendered image is sent to Paddle's *complete* pipeline service at
``POST /layout-parsing``.  It is never sent directly to the OpenAI-compatible
VLM recognition endpoint.  The response is reduced to Paddle's documented
``prunedResult`` page shape and translated by the existing normalization-only
adapter.

The official service response does not attest its package, pipeline, layout
model, or VLM model identity.  This client therefore enforces configured
identity and complete-response shape, but deliberately labels the deployment
uncertified until an operator qualifies and pins the worker image/configuration.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import http.client
import ipaddress
import json
import math
import mimetypes
import socket
import ssl
import time
from types import MappingProxyType
from urllib.parse import SplitResult, urlsplit
import uuid

from docarag.adapters.paddleocr_vl import (
    ADAPTER_NAME,
    ADAPTER_VERSION,
    PARSER_NAME,
    PaddleOCRVLPageAdapter,
)
from docarag.domain.base import (
    ContractValidationError,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
    stable_id,
)
from docarag.domain.evidence import BlockType, PageRef
from docarag.domain.parsing import (
    AcceleratorKind,
    ErrorClassification,
    IssueSeverity,
    MAX_PARSER_ARTIFACT_RECEIPTS,
    OcrMode,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParseResult,
    ParserArtifactAvailability,
    ParserArtifactReceipt,
    ParserExecutionError,
    ParserCapabilityManifest,
)
from docarag.source_material import SourceMaterialError, VerifiedSourceMaterial


OFFICIAL_LAYOUT_PARSING_PATH = "/layout-parsing"
OFFICIAL_SERVICE_CONTRACT_VERSION = "paddlex.paddleocr-vl.layout-parsing.v1"
SUPPORTED_PIPELINE_NAME = "PaddleOCR-VL-1.6"
SUPPORTED_PIPELINE_VERSION = "v1.6"
SUPPORTED_LAYOUT_MODEL = "PP-DocLayoutV3"
SUPPORTED_RECOGNITION_MODEL = "PaddleOCR-VL-1.6-0.9B"
PADDLE_RAW_PAGE_ARTIFACT_SCHEMA = (
    "paddlex.paddleocr-vl.layout-parsing.http-response.v1"
)
PADDLE_RAW_PAGE_RECEIPT_SCHEMA = "docarag.paddleocr-vl.raw-page-receipt.v1"
PADDLE_MARKDOWN_IMAGE_ARTIFACT_SCHEMA = (
    "paddlex.paddleocr-vl.markdown-image.v1"
)
PADDLE_MARKDOWN_IMAGE_RECEIPT_SCHEMA = (
    "docarag.paddleocr-vl.markdown-image-receipt.v1"
)

_MIB = 1024 * 1024
_MAX_BODY_BYTES = 1024 * _MIB
_MAX_PAGES = 256
_MAX_RESPONSE_JSON_DEPTH = 512
_MAX_RESPONSE_JSON_NODES = 5_000_000
_SUPPORTED_PAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/tiff"}
)
_COMPLETE_RESULT_FIELDS = frozenset(
    {"prunedResult", "markdown", "outputImages", "inputImage", "exports"}
)
# Markdown resources are not limited to figure-like regions.  Paddle may also
# return the exact table or formula crop referenced by a page result.  Keep the
# finite filename contract here so those bytes can be attached to the owning
# observation instead of becoming unreferenced artifacts.
_MARKDOWN_ASSET_LABELS = frozenset(
    {
        "image",
        "figure",
        "seal",
        "header_image",
        "footer_image",
        "chart",
        "plot",
        "diagram",
        "flowchart",
        "workflow",
        "sketch",
        "table",
        "display_formula",
        "formula",
        "equation",
        "inline_formula",
    }
)


class PaddleOCRVLServiceError(RuntimeError):
    """Base failure before normalized Paddle page results are available."""


class PaddleOCRVLServiceConfigurationError(PaddleOCRVLServiceError, ValueError):
    """The service, identity, security, or resource policy is invalid."""


class RawVLMEndpointRejectedError(PaddleOCRVLServiceConfigurationError):
    """A region-recognition endpoint was presented as the complete pipeline."""


class PaddleOCRVLServiceContractError(PaddleOCRVLServiceError):
    """Source material or the remote response violated the strict contract."""


class PaddleOCRVLServiceLimitError(PaddleOCRVLServiceError):
    """A configured page, request, response, or artifact bound was exceeded."""


class PaddleOCRVLServiceTimeoutError(PaddleOCRVLServiceError, TimeoutError):
    """The end-to-end parse deadline expired."""


class PaddleOCRVLServiceConnectionError(PaddleOCRVLServiceError):
    """The complete pipeline service could not be reached."""


class PaddleOCRVLRemoteServiceError(PaddleOCRVLServiceError):
    """The complete pipeline returned a bounded non-success response."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


class _DuplicateJSONKeyError(ValueError):
    """Internal marker raised by the strict JSON object hook."""


class _NonFiniteJSONValueError(ValueError):
    """Internal marker raised for JSON extensions forbidden by RFC 8259."""


class PaddleOCRVLEndpointKind(StrEnum):
    COMPLETE_PIPELINE = "complete-pipeline"
    VLM_INFERENCE = "vlm-inference"


def _positive_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise PaddleOCRVLServiceConfigurationError(
            f"{field_name} must be finite and positive"
        )
    return float(value)


def _bounded_positive_int(
    value: object, field_name: str, *, maximum: int
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise PaddleOCRVLServiceConfigurationError(
            f"{field_name} must be between 1 and {maximum}"
        )
    return value


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validated_endpoint(
    base_url: object, *, allow_insecure_http: bool
) -> SplitResult:
    if not isinstance(base_url, str) or not base_url.strip():
        raise PaddleOCRVLServiceConfigurationError(
            "base_url must be a non-empty string"
        )
    parsed = urlsplit(base_url)
    normalized_path = parsed.path.rstrip("/")
    if normalized_path == "/v1" or "/v1/" in parsed.path:
        raise RawVLMEndpointRejectedError(
            "a /v1 OpenAI-compatible endpoint is only PaddleOCR-VL region "
            "recognition; configure the complete pipeline service root that "
            "provides POST /layout-parsing"
        )
    if parsed.scheme not in {"http", "https"}:
        raise PaddleOCRVLServiceConfigurationError(
            "base_url scheme must be http or https"
        )
    if parsed.hostname is None:
        raise PaddleOCRVLServiceConfigurationError(
            "base_url must include a hostname"
        )
    if parsed.username is not None or parsed.password is not None:
        raise PaddleOCRVLServiceConfigurationError(
            "base_url must not contain user information"
        )
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise PaddleOCRVLServiceConfigurationError(
            "base_url must be the complete service root without a path, query, "
            "or fragment"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise PaddleOCRVLServiceConfigurationError(
            "base_url contains an invalid port"
        ) from exc
    if (
        parsed.scheme == "http"
        and not allow_insecure_http
        and not _is_loopback(parsed.hostname)
    ):
        raise PaddleOCRVLServiceConfigurationError(
            "plain HTTP is allowed only for loopback services; use HTTPS or "
            "explicitly allow insecure HTTP"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class PaddleOCRVLInferenceOptions:
    """Small, immutable subset of documented complete-pipeline options."""

    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_chart_recognition: bool = True
    use_seal_recognition: bool = True
    use_ocr_for_image_block: bool = True
    format_block_content: bool = True
    temperature: float = 0.0
    max_new_tokens: int = 16384
    return_markdown_images: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_chart_recognition",
            "use_seal_recognition",
            "use_ocr_for_image_block",
            "format_block_content",
            "return_markdown_images",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise PaddleOCRVLServiceConfigurationError(
                    f"{field_name} must be a boolean"
                )
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0 <= float(self.temperature) <= 2
        ):
            raise PaddleOCRVLServiceConfigurationError(
                "temperature must be finite and between 0 and 2"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        _bounded_positive_int(
            self.max_new_tokens, "max_new_tokens", maximum=262144
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useLayoutDetection": True,
            "useChartRecognition": self.use_chart_recognition,
            "useSealRecognition": self.use_seal_recognition,
            "useOcrForImageBlock": self.use_ocr_for_image_block,
            "formatBlockContent": self.format_block_content,
            "temperature": self.temperature,
            "maxNewTokens": self.max_new_tokens,
            "returnMarkdownImages": self.return_markdown_images,
            "restructurePages": False,
            "visualize": False,
        }


@dataclass(frozen=True, slots=True)
class PaddleOCRVLServiceConfig:
    """Pinned complete-pipeline identity, endpoint, deadline, and body policy."""

    endpoint_kind: PaddleOCRVLEndpointKind
    base_url: str
    parser_version: str
    pipeline_name: str = SUPPORTED_PIPELINE_NAME
    pipeline_version: str = SUPPORTED_PIPELINE_VERSION
    layout_model_name: str = SUPPORTED_LAYOUT_MODEL
    recognition_model_name: str = SUPPORTED_RECOGNITION_MODEL
    connect_timeout_seconds: float = 5.0
    operation_timeout_seconds: float = 300.0
    max_pages_per_request: int = 16
    max_source_bytes: int = 512 * _MIB
    max_page_bytes: int = 32 * _MIB
    max_total_page_bytes: int = 128 * _MIB
    max_request_bytes: int = 48 * _MIB
    max_response_bytes: int = 128 * _MIB
    max_response_json_depth: int = 128
    max_response_json_nodes: int = 2_000_000
    max_artifacts_per_page: int = 255
    # Per decoded Markdown image.  The exact raw response uses its independent
    # max_response_bytes bound and both consume the request-wide total below.
    max_artifact_bytes: int = 32 * _MIB
    # The exact HTTP response is itself one retained artifact, so the request-
    # wide budget must be at least the response bound before any crops are added.
    max_total_artifact_bytes: int = 256 * _MIB
    bearer_token: str | None = field(default=None, repr=False)
    allow_insecure_http: bool = False
    tls_ca_file: str | None = None
    _endpoint: SplitResult = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            endpoint_kind = PaddleOCRVLEndpointKind(self.endpoint_kind)
        except (TypeError, ValueError) as exc:
            raise PaddleOCRVLServiceConfigurationError(
                "endpoint_kind must be complete-pipeline or vlm-inference"
            ) from exc
        if endpoint_kind is not PaddleOCRVLEndpointKind.COMPLETE_PIPELINE:
            raise RawVLMEndpointRejectedError(
                "vlm-inference performs only region recognition and cannot be "
                "registered as a complete document parser"
            )
        object.__setattr__(self, "endpoint_kind", endpoint_kind)
        if not isinstance(self.allow_insecure_http, bool):
            raise PaddleOCRVLServiceConfigurationError(
                "allow_insecure_http must be a boolean"
            )
        endpoint = _validated_endpoint(
            self.base_url, allow_insecure_http=self.allow_insecure_http
        )
        object.__setattr__(self, "_endpoint", endpoint)
        for field_name in (
            "parser_version",
            "pipeline_name",
            "pipeline_version",
            "layout_model_name",
            "recognition_model_name",
        ):
            try:
                require_non_empty(getattr(self, field_name), field_name)
            except ContractValidationError as exc:
                raise PaddleOCRVLServiceConfigurationError(str(exc)) from exc
        expected = (
            ("pipeline_name", self.pipeline_name, SUPPORTED_PIPELINE_NAME),
            ("pipeline_version", self.pipeline_version, SUPPORTED_PIPELINE_VERSION),
            ("layout_model_name", self.layout_model_name, SUPPORTED_LAYOUT_MODEL),
            (
                "recognition_model_name",
                self.recognition_model_name,
                SUPPORTED_RECOGNITION_MODEL,
            ),
        )
        for field_name, actual, supported in expected:
            if actual != supported:
                raise PaddleOCRVLServiceConfigurationError(
                    f"{field_name} must be {supported!r} for this v1.6 contract"
                )
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            _positive_number(
                self.connect_timeout_seconds, "connect_timeout_seconds"
            ),
        )
        object.__setattr__(
            self,
            "operation_timeout_seconds",
            _positive_number(
                self.operation_timeout_seconds, "operation_timeout_seconds"
            ),
        )
        for field_name, maximum in (
            ("max_pages_per_request", _MAX_PAGES),
            ("max_source_bytes", _MAX_BODY_BYTES),
            ("max_page_bytes", _MAX_BODY_BYTES),
            ("max_total_page_bytes", _MAX_BODY_BYTES),
            ("max_request_bytes", _MAX_BODY_BYTES),
            ("max_response_bytes", _MAX_BODY_BYTES),
            ("max_response_json_depth", _MAX_RESPONSE_JSON_DEPTH),
            ("max_response_json_nodes", _MAX_RESPONSE_JSON_NODES),
            ("max_artifacts_per_page", 10000),
            ("max_artifact_bytes", _MAX_BODY_BYTES),
            ("max_total_artifact_bytes", _MAX_BODY_BYTES),
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_positive_int(
                    getattr(self, field_name), field_name, maximum=maximum
                ),
            )
        if self.max_page_bytes > self.max_total_page_bytes:
            raise PaddleOCRVLServiceConfigurationError(
                "max_page_bytes must not exceed max_total_page_bytes"
            )
        if self.max_artifact_bytes > self.max_total_artifact_bytes:
            raise PaddleOCRVLServiceConfigurationError(
                "max_artifact_bytes must not exceed max_total_artifact_bytes"
            )
        if self.max_response_bytes > self.max_total_artifact_bytes:
            raise PaddleOCRVLServiceConfigurationError(
                "max_response_bytes must not exceed max_total_artifact_bytes "
                "because every successful response is retained exactly"
            )
        if (
            self.max_pages_per_request * (self.max_artifacts_per_page + 1)
            > MAX_PARSER_ARTIFACT_RECEIPTS
        ):
            raise PaddleOCRVLServiceConfigurationError(
                "page/artifact bounds exceed ParseResult artifact receipt capacity"
            )
        if self.bearer_token is not None and (
            not isinstance(self.bearer_token, str) or not self.bearer_token
        ):
            raise PaddleOCRVLServiceConfigurationError(
                "bearer_token must be non-empty when set"
            )
        if self.tls_ca_file is not None and (
            not isinstance(self.tls_ca_file, str) or not self.tls_ca_file.strip()
        ):
            raise PaddleOCRVLServiceConfigurationError(
                "tls_ca_file must be non-empty when set"
            )
        if self.tls_ca_file is not None and endpoint.scheme != "https":
            raise PaddleOCRVLServiceConfigurationError(
                "tls_ca_file requires an HTTPS base_url"
            )

    @property
    def endpoint(self) -> SplitResult:
        return self._endpoint

    def identity_dict(self) -> dict[str, object]:
        """Return secret-free identity used in provenance and configuration hashes."""

        return {
            "service_contract_version": OFFICIAL_SERVICE_CONTRACT_VERSION,
            "endpoint_kind": self.endpoint_kind.value,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "parser_version": self.parser_version,
            "layout_model_name": self.layout_model_name,
            "recognition_model_name": self.recognition_model_name,
            "endpoint_path": OFFICIAL_LAYOUT_PARSING_PATH,
            "identity_assurance": "configured-unattested",
            "qualification": "uncertified",
            "response_json_bounds": {
                "max_bytes": self.max_response_bytes,
                "max_depth": self.max_response_json_depth,
                "max_nodes": self.max_response_json_nodes,
            },
            "artifact_bounds": {
                "max_artifacts_per_page": self.max_artifacts_per_page,
                "max_artifact_bytes": self.max_artifact_bytes,
                "max_markdown_image_bytes": self.max_artifact_bytes,
                "max_raw_response_artifact_bytes": self.max_response_bytes,
                "max_total_artifact_bytes": self.max_total_artifact_bytes,
            },
            "artifact_budget_scope": "parse-request",
            "raw_response_artifact": "exact-http-response-body",
        }


@dataclass(frozen=True, slots=True)
class PaddleOCRVLPageMaterial:
    """One immutable rendered page bound to the verified source version."""

    document_version_id: str
    source_sha256: str
    page_index: int
    media_type: str
    width: int
    height: int
    data: bytes = field(repr=False)
    image_sha256: str = ""
    render_identity: str = "verified-page-render.v1"

    def __post_init__(self) -> None:
        try:
            require_non_empty(self.document_version_id, "document_version_id")
            require_sha256(self.source_sha256, "source_sha256")
            require_non_empty(self.render_identity, "render_identity")
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise PaddleOCRVLServiceContractError(
                "page_index must be a non-negative integer"
            )
        if self.media_type not in _SUPPORTED_PAGE_MEDIA_TYPES:
            raise PaddleOCRVLServiceContractError(
                "rendered page media_type must be image/jpeg, image/png, or image/tiff"
            )
        for value, field_name in ((self.width, "width"), (self.height, "height")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PaddleOCRVLServiceContractError(
                    f"rendered page {field_name} must be a positive integer"
                )
        if not isinstance(self.data, bytes) or not self.data:
            raise PaddleOCRVLServiceContractError(
                "rendered page data must be non-empty immutable bytes"
            )
        actual = hashlib.sha256(self.data).hexdigest()
        declared = self.image_sha256 or actual
        try:
            require_sha256(declared, "image_sha256")
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if declared != actual:
            raise PaddleOCRVLServiceContractError(
                "rendered page digest differs from its bytes"
            )
        object.__setattr__(self, "image_sha256", declared)


class PaddleOCRVLArtifactRole(StrEnum):
    MARKDOWN_IMAGE = "markdown-image"
    RAW_PAGE_RESULT = "raw-page-result"


@dataclass(slots=True)
class _RequestArtifactBudget:
    """Single retained-artifact byte budget shared by every requested page."""

    maximum: int
    used: int = 0

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise AssertionError("artifact reservation must not be negative")
        if amount > self.maximum - self.used:
            raise PaddleOCRVLServiceLimitError(
                "page artifacts exceed the ParseRequest-wide "
                "max_total_artifact_bytes budget"
            )
        self.used += amount


@dataclass(frozen=True, slots=True)
class PaddleOCRVLArtifact:
    """Bounded artifact offered to an injected immutable artifact store."""

    role: PaddleOCRVLArtifactRole
    request_id: str
    document_version_id: str
    page_index: int
    service_log_id: str
    name: str
    media_type: str
    data: bytes = field(repr=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            role = PaddleOCRVLArtifactRole(self.role)
        except (TypeError, ValueError) as exc:
            raise PaddleOCRVLServiceContractError(
                "artifact role is unsupported"
            ) from exc
        for value, field_name in (
            (self.request_id, "request_id"),
            (self.document_version_id, "document_version_id"),
            (self.service_log_id, "service_log_id"),
            (self.name, "name"),
            (self.media_type, "media_type"),
        ):
            try:
                require_non_empty(value, field_name)
            except ContractValidationError as exc:
                raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise PaddleOCRVLServiceContractError(
                "artifact page_index must be a non-negative integer"
            )
        if not isinstance(self.data, bytes) or not self.data:
            raise PaddleOCRVLServiceContractError(
                "artifact data must be non-empty immutable bytes"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "sha256", hashlib.sha256(self.data).hexdigest())


@dataclass(frozen=True, slots=True)
class StoredPaddleOCRVLArtifact:
    """Verified identity returned by the injected immutable artifact store."""

    uri: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        try:
            require_non_empty(self.uri, "artifact uri")
            require_sha256(self.sha256, "artifact sha256")
            require_non_empty(self.media_type, "artifact media_type")
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class PaddleOCRVLHTTPRequest:
    url: str
    body: bytes = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)
    connect_timeout_seconds: float
    operation_timeout_seconds: float
    max_response_bytes: int
    tls_ca_file: str | None = None


@dataclass(frozen=True, slots=True)
class PaddleOCRVLHTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class PaddleOCRVLLoadedPages:
    """Adapter-facing pages plus result-scoped immutable artifact receipts.

    Iteration, indexing, and ``len`` preserve the earlier tuple-like loader
    surface while allowing a page with no blocks to retain its raw evidence.
    """

    pages: tuple[Mapping[str, object], ...]
    artifact_receipts: tuple[ParserArtifactReceipt, ...] = ()

    def __iter__(self):
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | tuple[Mapping[str, object], ...]:
        return self.pages[index]


SourceLoader = Callable[[ParseRequest], VerifiedSourceMaterial]
PageRenderer = Callable[
    [VerifiedSourceMaterial, PageRef], PaddleOCRVLPageMaterial
]
ArtifactSink = Callable[[PaddleOCRVLArtifact], StoredPaddleOCRVLArtifact]
HTTPExchange = Callable[[PaddleOCRVLHTTPRequest], PaddleOCRVLHTTPResponse]
Clock = Callable[[], float]


def _safe_relative_artifact_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise PaddleOCRVLServiceContractError(
            "markdown image name must be a non-empty bounded string"
        )
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or "://" in normalized
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in normalized
        )
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise PaddleOCRVLServiceContractError(
            "markdown image name must be a safe relative path"
        )
    return normalized


def _strict_base64(value: object, field_name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise PaddleOCRVLServiceContractError(
            f"{field_name} must be a non-empty inline Base64 string"
        )
    if value.startswith(("http://", "https://")):
        raise PaddleOCRVLServiceContractError(
            f"{field_name} is a URL; configure the Paddle service to return "
            "inline artifacts so bytes can be hashed and retained"
        )
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PaddleOCRVLServiceContractError(
            f"{field_name} is not valid Base64"
        ) from exc


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> object:
    raise _NonFiniteJSONValueError(
        f"non-finite JSON number {value!r} is not permitted"
    )


def _validate_json_shape(
    value: object, *, max_depth: int, max_nodes: int
) -> None:
    """Bound a decoded JSON tree without recursively walking attacker input."""

    nodes = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise PaddleOCRVLServiceLimitError(
                "service response JSON exceeds max_response_json_depth"
            )
        nodes += 1
        if nodes > max_nodes:
            raise PaddleOCRVLServiceLimitError(
                "service response JSON exceeds max_response_json_nodes"
            )
        if isinstance(current, Mapping):
            # Object member names consume bounded parser/memory work too.
            nodes += len(current)
            if nodes > max_nodes:
                raise PaddleOCRVLServiceLimitError(
                    "service response JSON exceeds max_response_json_nodes"
                )
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _filename_coordinate(value: object) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PaddleOCRVLServiceContractError(
            "block bbox coordinates must be finite numbers"
        )
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def _expected_markdown_asset_name(block: Mapping[str, object]) -> str | None:
    label = block.get("block_label")
    if (
        not isinstance(label, str)
        or label.strip().lower() not in _MARKDOWN_ASSET_LABELS
    ):
        return None
    bbox_value = block.get("block_bbox", block.get("bbox"))
    if not isinstance(bbox_value, Sequence) or isinstance(
        bbox_value, (str, bytes, bytearray)
    ) or len(bbox_value) != 4:
        raise PaddleOCRVLServiceContractError(
            "asset-producing block must have a four-coordinate bbox"
        )
    coordinates = "_".join(_filename_coordinate(value) for value in bbox_value)
    return f"imgs/img_in_{label.strip().lower()}_box_{coordinates}.jpg"


class PaddleOCRVLCompletePipelineLoader:
    """Load strict v1.6 page JSON through Paddle's complete service."""

    def __init__(
        self,
        config: PaddleOCRVLServiceConfig,
        *,
        source_loader: SourceLoader,
        page_renderer: PageRenderer,
        artifact_sink: ArtifactSink,
        inference_options: PaddleOCRVLInferenceOptions | None = None,
        http_exchange: HTTPExchange | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if not isinstance(config, PaddleOCRVLServiceConfig):
            raise TypeError("config must be PaddleOCRVLServiceConfig")
        for callback, field_name in (
            (source_loader, "source_loader"),
            (page_renderer, "page_renderer"),
            (artifact_sink, "artifact_sink"),
            (clock, "clock"),
        ):
            if not callable(callback):
                raise TypeError(f"{field_name} must be callable")
        if inference_options is None:
            inference_options = PaddleOCRVLInferenceOptions()
        if not isinstance(inference_options, PaddleOCRVLInferenceOptions):
            raise TypeError(
                "inference_options must be PaddleOCRVLInferenceOptions"
            )
        if http_exchange is not None and not callable(http_exchange):
            raise TypeError("http_exchange must be callable")
        self._config = config
        self._source_loader = source_loader
        self._page_renderer = page_renderer
        self._artifact_sink = artifact_sink
        self._inference_options = inference_options
        self._exchange = http_exchange or self._stdlib_exchange
        self._clock = clock

    @property
    def config(self) -> PaddleOCRVLServiceConfig:
        return self._config

    @property
    def inference_options(self) -> PaddleOCRVLInferenceOptions:
        return self._inference_options

    def _remaining(self, deadline: float, stage: str) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise PaddleOCRVLServiceTimeoutError(
                f"PaddleOCR-VL complete-pipeline deadline expired during {stage}"
            )
        return remaining

    def _validate_source(
        self, request: ParseRequest, source: object
    ) -> VerifiedSourceMaterial:
        if not isinstance(source, VerifiedSourceMaterial):
            raise PaddleOCRVLServiceContractError(
                "source_loader must return VerifiedSourceMaterial"
            )
        if source.document_version_id != request.document.document_version_id:
            raise PaddleOCRVLServiceContractError(
                "verified source belongs to a different document version"
            )
        if source.sha256 != request.document.content_sha256:
            raise PaddleOCRVLServiceContractError(
                "verified source digest differs from the parse request"
            )
        if source.media_type != request.document.mime_type or source.media_type != "application/pdf":
            raise PaddleOCRVLServiceContractError(
                "verified source must be the requested application/pdf"
            )
        if source.data is None:
            raise PaddleOCRVLServiceContractError(
                "source_loader must retain verified bytes (include_bytes=True)"
            )
        if source.size_bytes != len(source.data):
            raise PaddleOCRVLServiceContractError(
                "verified source size differs from retained bytes"
            )
        if len(source.data) > self._config.max_source_bytes:
            raise PaddleOCRVLServiceLimitError(
                "verified source exceeds max_source_bytes"
            )
        if hashlib.sha256(source.data).hexdigest() != source.sha256:
            raise PaddleOCRVLServiceContractError(
                "verified source bytes no longer match their digest"
            )
        return source

    def _validate_material(
        self,
        request: ParseRequest,
        page: PageRef,
        material: object,
    ) -> PaddleOCRVLPageMaterial:
        if not isinstance(material, PaddleOCRVLPageMaterial):
            raise PaddleOCRVLServiceContractError(
                "page_renderer must return PaddleOCRVLPageMaterial"
            )
        if material.document_version_id != request.document.document_version_id:
            raise PaddleOCRVLServiceContractError(
                "rendered page belongs to a different document version"
            )
        if material.source_sha256 != request.document.content_sha256:
            raise PaddleOCRVLServiceContractError(
                "rendered page source digest differs from the parse request"
            )
        if material.page_index != page.page_index:
            raise PaddleOCRVLServiceContractError(
                "page_renderer returned a different page index"
            )
        if page.image_sha256 is not None and page.image_sha256 != material.image_sha256:
            raise PaddleOCRVLServiceContractError(
                "rendered page digest differs from PageRef.image_sha256"
            )
        if len(material.data) > self._config.max_page_bytes:
            raise PaddleOCRVLServiceLimitError(
                f"rendered page {page.page_index} exceeds max_page_bytes"
            )
        return material

    def __call__(self, request: ParseRequest) -> PaddleOCRVLLoadedPages:
        """Adapter-facing call which preserves permanent/retryable semantics."""

        try:
            return self.load(request)
        except ParserExecutionError:
            raise
        except Exception as exc:
            raise ParserExecutionError(self._execution_issue(exc)) from exc

    @staticmethod
    def _execution_issue(exc: Exception) -> ParseIssue:
        if isinstance(exc, PaddleOCRVLServiceTimeoutError):
            code = ParseErrorCode.TIMEOUT
            classification = ErrorClassification.RETRYABLE
        elif isinstance(exc, PaddleOCRVLServiceConnectionError):
            code = ParseErrorCode.DEPENDENCY_UNAVAILABLE
            classification = ErrorClassification.RETRYABLE
        elif isinstance(exc, PaddleOCRVLRemoteServiceError):
            code = ParseErrorCode.PARSER_FAILURE
            classification = (
                ErrorClassification.RETRYABLE
                if exc.status in {408, 425, 429} or exc.status >= 500
                else ErrorClassification.PERMANENT
            )
        elif isinstance(exc, PaddleOCRVLServiceLimitError):
            code = ParseErrorCode.RESOURCE_EXHAUSTED
            classification = ErrorClassification.PERMANENT
        elif isinstance(
            exc,
            (
                PaddleOCRVLServiceConfigurationError,
                PaddleOCRVLServiceContractError,
                SourceMaterialError,
                ContractValidationError,
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
            message="PaddleOCR-VL complete pipeline failed",
            details={
                "exception_type": type(exc).__name__,
                "service_contract_version": OFFICIAL_SERVICE_CONTRACT_VERSION,
                "complete_pipeline": True,
            },
        )

    def load(self, request: ParseRequest) -> PaddleOCRVLLoadedPages:
        """Execute the strict loader, raising its detailed service exceptions."""

        if not isinstance(request, ParseRequest):
            raise TypeError("request must be ParseRequest")
        if request.document.mime_type != "application/pdf":
            raise PaddleOCRVLServiceContractError(
                "complete PaddleOCR-VL loader accepts application/pdf requests"
            )
        if request.document.page_count is None or request.document.page_count < 1:
            raise PaddleOCRVLServiceContractError(
                "document.page_count is required for page-bound service results"
            )
        if request.regions:
            raise PaddleOCRVLServiceContractError(
                "complete PaddleOCR-VL service loader does not accept regions"
            )
        if request.options.ocr_mode is OcrMode.DISABLED:
            raise PaddleOCRVLServiceContractError(
                "PaddleOCR-VL cannot satisfy ocr_mode=disabled"
            )
        if len(request.pages) > self._config.max_pages_per_request:
            raise PaddleOCRVLServiceLimitError(
                "parse request exceeds max_pages_per_request"
            )
        timeout = self._config.operation_timeout_seconds
        if request.options.timeout_seconds is not None:
            timeout = min(timeout, float(request.options.timeout_seconds))
        deadline = self._clock() + timeout
        source = self._validate_source(request, self._source_loader(request))
        self._remaining(deadline, "source verification")

        total_page_bytes = 0
        artifact_budget = _RequestArtifactBudget(
            self._config.max_total_artifact_bytes
        )
        results: list[Mapping[str, object]] = []
        artifact_receipts: list[ParserArtifactReceipt] = []
        for page in request.pages:
            try:
                material = self._validate_material(
                    request, page, self._page_renderer(source, page)
                )
                total_page_bytes += len(material.data)
                if total_page_bytes > self._config.max_total_page_bytes:
                    raise PaddleOCRVLServiceLimitError(
                        "rendered pages exceed max_total_page_bytes"
                    )
                self._remaining(deadline, f"rendering page {page.page_index}")
                normalized_page, page_receipts = self._load_page(
                    request,
                    page,
                    material,
                    deadline=deadline,
                    artifact_budget=artifact_budget,
                )
                results.append(normalized_page)
                artifact_receipts.extend(page_receipts)
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
        return PaddleOCRVLLoadedPages(
            pages=tuple(results),
            artifact_receipts=tuple(artifact_receipts),
        )

    def _load_page(
        self,
        request: ParseRequest,
        page: PageRef,
        material: PaddleOCRVLPageMaterial,
        *,
        deadline: float,
        artifact_budget: _RequestArtifactBudget,
    ) -> tuple[Mapping[str, object], tuple[ParserArtifactReceipt, ...]]:
        options = self._inference_options.to_dict()
        if not request.options.parse_images:
            options["returnMarkdownImages"] = False
        payload: dict[str, object] = {
            "file": base64.b64encode(material.data).decode("ascii"),
            "fileType": 1,
            **options,
        }
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > self._config.max_request_bytes:
            raise PaddleOCRVLServiceLimitError(
                f"serialized request for page {page.page_index} exceeds max_request_bytes"
            )
        remaining = self._remaining(deadline, f"requesting page {page.page_index}")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-DocARAG-Parse-Request-ID": request.request_id,
            "X-DocARAG-Page-Index": str(page.page_index),
        }
        if request.idempotency_key is not None:
            headers["Idempotency-Key"] = request.idempotency_key
        if self._config.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        response = self._exchange(
            PaddleOCRVLHTTPRequest(
                url=self._service_url(),
                body=body,
                headers=MappingProxyType(headers),
                connect_timeout_seconds=min(
                    self._config.connect_timeout_seconds, remaining
                ),
                operation_timeout_seconds=remaining,
                max_response_bytes=self._config.max_response_bytes,
                tls_ca_file=self._config.tls_ca_file,
            )
        )
        self._remaining(deadline, f"receiving page {page.page_index}")
        return self._normalize_response(
            request,
            page,
            material,
            response,
            deadline,
            artifact_budget,
            effective_options=MappingProxyType(dict(options)),
        )

    def _service_url(self) -> str:
        endpoint = self._config.endpoint
        host = endpoint.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if endpoint.port is not None:
            host = f"{host}:{endpoint.port}"
        return f"{endpoint.scheme}://{host}{OFFICIAL_LAYOUT_PARSING_PATH}"

    def _normalize_response(
        self,
        request: ParseRequest,
        page: PageRef,
        material: PaddleOCRVLPageMaterial,
        response: object,
        deadline: float,
        artifact_budget: _RequestArtifactBudget,
        *,
        effective_options: Mapping[str, object],
    ) -> tuple[Mapping[str, object], tuple[ParserArtifactReceipt, ...]]:
        if not isinstance(response, PaddleOCRVLHTTPResponse):
            raise PaddleOCRVLServiceContractError(
                "http_exchange must return PaddleOCRVLHTTPResponse"
            )
        if (
            isinstance(response.status, bool)
            or not isinstance(response.status, int)
            or not 100 <= response.status <= 599
        ):
            raise PaddleOCRVLServiceContractError(
                "service response status is invalid"
            )
        if not isinstance(response.body, bytes):
            raise PaddleOCRVLServiceContractError(
                "service response body must be immutable bytes"
            )
        if len(response.body) > self._config.max_response_bytes:
            raise PaddleOCRVLServiceLimitError(
                "service response exceeds max_response_bytes"
            )
        if response.status != 200:
            message = f"PaddleOCR-VL complete pipeline returned HTTP {response.status}"
            try:
                error_payload = json.loads(response.body.decode("utf-8"))
                if isinstance(error_payload, Mapping):
                    candidate = error_payload.get("errorMsg", error_payload.get("detail"))
                    if isinstance(candidate, str) and candidate.strip():
                        message = f"{message}: {candidate.strip()[:512]}"
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            raise PaddleOCRVLRemoteServiceError(response.status, message)
        content_type = None
        if isinstance(response.headers, Mapping):
            for key, value in response.headers.items():
                if isinstance(key, str) and key.lower() == "content-type":
                    content_type = value
                    break
        else:
            raise PaddleOCRVLServiceContractError(
                "service response headers must be a mapping"
            )
        if content_type is not None and (
            not isinstance(content_type, str)
            or "application/json" not in content_type.lower()
        ):
            raise PaddleOCRVLServiceContractError(
                "successful service response must be application/json"
            )
        try:
            payload = json.loads(
                response.body.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_non_finite_json,
            )
        except (_DuplicateJSONKeyError, _NonFiniteJSONValueError) as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        except RecursionError as exc:
            raise PaddleOCRVLServiceLimitError(
                "service response JSON exceeds decoder nesting limits"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PaddleOCRVLServiceContractError(
                "service response is not valid UTF-8 JSON"
            ) from exc
        _validate_json_shape(
            payload,
            max_depth=self._config.max_response_json_depth,
            max_nodes=self._config.max_response_json_nodes,
        )
        if not isinstance(payload, Mapping):
            raise PaddleOCRVLServiceContractError(
                "service response must be a JSON object"
            )
        if "choices" in payload or payload.get("object") in {
            "chat.completion",
            "list",
        }:
            raise RawVLMEndpointRejectedError(
                "received an OpenAI/vLLM response; this is region recognition, "
                "not the complete PaddleOCR-VL layout pipeline"
            )
        try:
            reject_unknown_fields(
                payload,
                {"logId", "errorCode", "errorMsg", "result"},
                dto="PaddleOCRVLServiceResponse",
            )
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        log_id = payload.get("logId")
        if not isinstance(log_id, str):
            raise PaddleOCRVLServiceContractError(
                "service logId must be a UUID string"
            )
        try:
            uuid.UUID(log_id)
        except (ValueError, AttributeError) as exc:
            raise PaddleOCRVLServiceContractError(
                "service logId must be a UUID string"
            ) from exc
        if payload.get("errorCode") != 0 or payload.get("errorMsg") != "Success":
            raise PaddleOCRVLServiceContractError(
                "HTTP 200 service response must declare errorCode=0 and errorMsg=Success"
            )
        try:
            result = require_mapping(payload.get("result"), "result")
            reject_unknown_fields(
                result,
                {"layoutParsingResults", "dataInfo"},
                dto="PaddleOCRVLServiceResult",
            )
            require_mapping(result.get("dataInfo"), "result.dataInfo")
            pages = require_sequence(
                result.get("layoutParsingResults"), "result.layoutParsingResults"
            )
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if len(pages) != 1:
            raise PaddleOCRVLServiceContractError(
                "one rendered page request must return exactly one layoutParsingResult"
            )
        try:
            service_page = require_mapping(pages[0], "layoutParsingResults[0]")
            reject_unknown_fields(
                service_page,
                _COMPLETE_RESULT_FIELDS,
                dto="PaddleOCRVLLayoutParsingResult",
            )
            pruned = dict(
                require_mapping(
                    service_page.get("prunedResult"),
                    "layoutParsingResults[0].prunedResult",
                )
            )
            markdown = require_mapping(
                service_page.get("markdown"), "layoutParsingResults[0].markdown"
            )
            reject_unknown_fields(
                markdown,
                {"text", "images"},
                dto="PaddleOCRVLMarkdownResult",
            )
            if not isinstance(markdown.get("text"), str):
                raise ContractValidationError("markdown.text must be a string")
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if "input_path" in pruned or "page_index" in pruned:
            raise PaddleOCRVLServiceContractError(
                "official prunedResult must omit input_path and page_index; refusing "
                "conflicting source identity"
            )
        for field_name in ("model_settings", "layout_det_res"):
            try:
                require_mapping(pruned.get(field_name), f"prunedResult.{field_name}")
            except ContractValidationError as exc:
                raise PaddleOCRVLServiceContractError(str(exc)) from exc
        try:
            blocks = require_sequence(
                pruned.get("parsing_res_list"), "prunedResult.parsing_res_list"
            )
            layout = require_mapping(
                pruned.get("layout_det_res"), "prunedResult.layout_det_res"
            )
            require_sequence(layout.get("boxes"), "prunedResult.layout_det_res.boxes")
            settings = dict(
                require_mapping(
                    pruned.get("model_settings"), "prunedResult.model_settings"
                )
            )
        except ContractValidationError as exc:
            raise PaddleOCRVLServiceContractError(str(exc)) from exc
        if settings.get("use_layout_detection") is not True:
            raise PaddleOCRVLServiceContractError(
                "response does not prove complete layout localization "
                "(model_settings.use_layout_detection must be true)"
            )
        settings["docarag_service_identity"] = self._config.identity_dict()
        settings["docarag_render_identity"] = material.render_identity
        # The public block_content field is ambiguous for visual regions unless
        # the consumer knows whether OCR-for-image-block was requested.  Carry
        # the exact, secret-free invocation options into the normalized page so
        # the adapter can label OCR text and visual descriptions truthfully.
        settings["docarag_inference_options"] = dict(effective_options)
        pruned["model_settings"] = settings
        pruned["input_path"] = (
            f"material://{request.document.document_version_id}/pages/"
            f"{page.page_index}/{material.image_sha256}"
        )
        pruned["source_sha256"] = request.document.content_sha256
        pruned["document_version_id"] = request.document.document_version_id
        pruned["source_uri"] = request.document.source_uri
        pruned["page_index"] = page.page_index
        pruned["page_count"] = request.document.page_count
        pruned["width"] = material.width
        pruned["height"] = material.height
        pruned["service_log_id"] = log_id
        pruned["service_contract_version"] = OFFICIAL_SERVICE_CONTRACT_VERSION

        images_value = markdown.get("images")
        return_markdown_images = effective_options.get("returnMarkdownImages")
        if not isinstance(return_markdown_images, bool):
            raise PaddleOCRVLServiceContractError(
                "effective returnMarkdownImages option must be a boolean"
            )
        if return_markdown_images:
            if images_value is None:
                images: Mapping[str, object] = {}
            else:
                try:
                    images = require_mapping(images_value, "markdown.images")
                except ContractValidationError as exc:
                    raise PaddleOCRVLServiceContractError(str(exc)) from exc
        else:
            if images_value not in (None, {}):
                raise PaddleOCRVLServiceContractError(
                    "service returned markdown images after they were disabled"
                )
            images = {}
        if len(images) > self._config.max_artifacts_per_page:
            raise PaddleOCRVLServiceLimitError(
                "markdown images exceed max_artifacts_per_page"
            )
        decoded_by_name: dict[str, tuple[bytes, str]] = {}
        for raw_name in sorted(images):
            name = _safe_relative_artifact_name(raw_name)
            if name in decoded_by_name:
                raise PaddleOCRVLServiceContractError(
                    "distinct markdown image keys normalize to the same safe path"
                )
            data = _strict_base64(images[raw_name], f"markdown.images[{name!r}]")
            if not data:
                raise PaddleOCRVLServiceContractError(
                    f"markdown.images[{name!r}] decoded to empty bytes"
                )
            if len(data) > self._config.max_artifact_bytes:
                raise PaddleOCRVLServiceLimitError(
                    f"markdown image {name!r} exceeds max_artifact_bytes"
                )
            media_type = mimetypes.guess_type(name)[0]
            if media_type is None or not media_type.startswith("image/"):
                raise PaddleOCRVLServiceContractError(
                    f"markdown image {name!r} has no supported image media type"
                )
            decoded_by_name[name] = (data, media_type)

        output_images = service_page.get("outputImages")
        if output_images not in (None, {}):
            raise PaddleOCRVLServiceContractError(
                "service returned outputImages despite visualize=false"
            )
        if service_page.get("inputImage") is not None:
            raise PaddleOCRVLServiceContractError(
                "service returned inputImage despite visualize=false"
            )
        if service_page.get("exports") is not None:
            raise PaddleOCRVLServiceContractError(
                "service returned exports although outputFormats was not requested"
            )

        block_id_by_asset_name: dict[str, str] = {}
        seen_asset_block_ids: set[str] = set()
        for block_index, raw_block in enumerate(blocks):
            if not isinstance(raw_block, Mapping):
                raise PaddleOCRVLServiceContractError(
                    f"parsing_res_list[{block_index}] must be an object"
                )
            expected_name = _expected_markdown_asset_name(raw_block)
            if expected_name is None:
                continue
            block_id = raw_block.get("block_id")
            if (
                not isinstance(block_id, (str, int))
                or isinstance(block_id, bool)
                or (isinstance(block_id, str) and not block_id.strip())
            ):
                raise PaddleOCRVLServiceContractError(
                    "asset-producing blocks require a non-empty string or integer block_id"
                )
            block_key = str(block_id)
            if block_key in seen_asset_block_ids:
                raise PaddleOCRVLServiceContractError(
                    "asset-producing block IDs collide after normalization"
                )
            if expected_name in block_id_by_asset_name:
                raise PaddleOCRVLServiceContractError(
                    "multiple blocks claim the same markdown image crop"
                )
            seen_asset_block_ids.add(block_key)
            block_id_by_asset_name[expected_name] = block_key

        if return_markdown_images:
            missing = sorted(set(block_id_by_asset_name) - set(decoded_by_name))
            if missing:
                raise PaddleOCRVLServiceContractError(
                    "markdown.images is missing expected block crops: "
                    + ", ".join(missing[:3])
                )
            unmatched = sorted(set(decoded_by_name) - set(block_id_by_asset_name))
            if unmatched:
                raise PaddleOCRVLServiceContractError(
                    "markdown.images contains crops with no unique owning block: "
                    + ", ".join(unmatched[:3])
                )

        # The raw evidence is the exact immutable HTTP body, not a later,
        # source-enriched mutation of prunedResult.  It includes markdown.text,
        # the original model settings, and the original inline-image assertions.
        artifact_budget.reserve(
            len(response.body)
            + sum(len(data) for data, _media_type in decoded_by_name.values())
        )

        producer_task_id = stable_id(
            "paddle-page",
            request.request_id,
            request.document.document_version_id,
            page.page_index,
            material.image_sha256,
            self._config.identity_dict(),
        )
        page_receipts: list[ParserArtifactReceipt] = []
        try:
            stored_by_name: dict[str, StoredPaddleOCRVLArtifact] = {}
            for name in sorted(decoded_by_name):
                data, media_type = decoded_by_name[name]
                artifact = PaddleOCRVLArtifact(
                    role=PaddleOCRVLArtifactRole.MARKDOWN_IMAGE,
                    request_id=request.request_id,
                    document_version_id=request.document.document_version_id,
                    page_index=page.page_index,
                    service_log_id=log_id,
                    name=name,
                    media_type=media_type,
                    data=data,
                )
                stored = self._store(artifact, deadline)
                stored_by_name[name] = stored
                page_receipts.append(
                    ParserArtifactReceipt(
                        artifact_role="paddleocr_vl_markdown_image",
                        availability=ParserArtifactAvailability.EMITTED,
                        artifact_schema=PADDLE_MARKDOWN_IMAGE_ARTIFACT_SCHEMA,
                        producer_task_id=producer_task_id,
                        document_version_id=(
                            request.document.document_version_id
                        ),
                        source_sha256=request.document.content_sha256,
                        artifact_key=name,
                        uri=stored.uri,
                        sha256=stored.sha256,
                        size_bytes=len(data),
                        media_type=stored.media_type,
                        metadata={
                            "receipt_schema_version": (
                                PADDLE_MARKDOWN_IMAGE_RECEIPT_SCHEMA
                            ),
                            "page_index": page.page_index,
                            "owning_block_id": block_id_by_asset_name[name],
                            "service_log_id": log_id,
                            "service_contract_version": (
                                OFFICIAL_SERVICE_CONTRACT_VERSION
                            ),
                        },
                    )
                )

            saved_assets: dict[str, object] = {}
            for expected_name, block_key in block_id_by_asset_name.items():
                stored = stored_by_name.get(expected_name)
                if stored is None:
                    continue
                saved_assets[block_key] = {
                    "uri": stored.uri,
                    "sha256": stored.sha256,
                    "media_type": stored.media_type,
                    "metadata": {
                        "service_log_id": log_id,
                        "service_contract_version": (
                            OFFICIAL_SERVICE_CONTRACT_VERSION
                        ),
                    },
                }
            if saved_assets:
                pruned["saved_assets"] = saved_assets

            raw_artifact = self._store(
                PaddleOCRVLArtifact(
                    role=PaddleOCRVLArtifactRole.RAW_PAGE_RESULT,
                    request_id=request.request_id,
                    document_version_id=request.document.document_version_id,
                    page_index=page.page_index,
                    service_log_id=log_id,
                    name="page-response.json",
                    media_type="application/json",
                    data=response.body,
                ),
                deadline,
            )
            pruned["raw_artifact_uri"] = raw_artifact.uri
            page_receipts.append(
                ParserArtifactReceipt(
                    artifact_role="paddleocr_vl_raw_page_result",
                    availability=ParserArtifactAvailability.EMITTED,
                    artifact_schema=PADDLE_RAW_PAGE_ARTIFACT_SCHEMA,
                    producer_task_id=producer_task_id,
                    document_version_id=request.document.document_version_id,
                    source_sha256=request.document.content_sha256,
                    artifact_key="page-response.json",
                    uri=raw_artifact.uri,
                    sha256=raw_artifact.sha256,
                    size_bytes=len(response.body),
                    media_type=raw_artifact.media_type,
                    metadata={
                        "receipt_schema_version": (
                            PADDLE_RAW_PAGE_RECEIPT_SCHEMA
                        ),
                        "page_index": page.page_index,
                        "service_log_id": log_id,
                        "service_contract_version": (
                            OFFICIAL_SERVICE_CONTRACT_VERSION
                        ),
                        "exact_http_response_body": True,
                    },
                )
            )
            self._remaining(
                deadline,
                f"storing page {page.page_index} artifacts",
            )
        except ParserExecutionError as exc:
            raise ParserExecutionError(
                exc.issue,
                artifacts=tuple(page_receipts) + exc.artifacts,
            ) from exc
        except Exception as exc:
            if page_receipts:
                raise ParserExecutionError(
                    self._execution_issue(exc),
                    artifacts=tuple(page_receipts),
                ) from exc
            raise
        return MappingProxyType(pruned), tuple(page_receipts)

    def _store(
        self, artifact: PaddleOCRVLArtifact, deadline: float
    ) -> StoredPaddleOCRVLArtifact:
        stored = self._artifact_sink(artifact)
        self._remaining(deadline, f"storing {artifact.role.value}")
        if not isinstance(stored, StoredPaddleOCRVLArtifact):
            raise PaddleOCRVLServiceContractError(
                "artifact_sink must return StoredPaddleOCRVLArtifact"
            )
        if stored.sha256 != artifact.sha256:
            raise PaddleOCRVLServiceContractError(
                "artifact store digest differs from offered bytes"
            )
        if stored.media_type != artifact.media_type:
            raise PaddleOCRVLServiceContractError(
                "artifact store media type differs from offered bytes"
            )
        return stored

    @staticmethod
    def _stdlib_exchange(
        request: PaddleOCRVLHTTPRequest,
    ) -> PaddleOCRVLHTTPResponse:
        parsed = urlsplit(request.url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise PaddleOCRVLServiceConfigurationError(
                "HTTP request URL is invalid"
            )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            context = ssl.create_default_context(cafile=request.tls_ca_file)
            connection = http.client.HTTPSConnection(
                parsed.hostname,
                port,
                timeout=request.connect_timeout_seconds,
                context=context,
            )
        else:
            connection = http.client.HTTPConnection(
                parsed.hostname,
                port,
                timeout=request.connect_timeout_seconds,
            )
        try:
            connection.request(
                "POST",
                parsed.path,
                body=request.body,
                headers=dict(request.headers),
            )
            if connection.sock is not None:
                connection.sock.settimeout(request.operation_timeout_seconds)
            response = connection.getresponse()
            raw_length = response.getheader("Content-Length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError as exc:
                    raise PaddleOCRVLServiceContractError(
                        "service Content-Length is invalid"
                    ) from exc
                if declared_length < 0:
                    raise PaddleOCRVLServiceContractError(
                        "service Content-Length is negative"
                    )
                if declared_length > request.max_response_bytes:
                    raise PaddleOCRVLServiceLimitError(
                        "service Content-Length exceeds max_response_bytes"
                    )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(
                    min(64 * 1024, request.max_response_bytes - total + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > request.max_response_bytes:
                    raise PaddleOCRVLServiceLimitError(
                        "streamed service response exceeds max_response_bytes"
                    )
                chunks.append(chunk)
            return PaddleOCRVLHTTPResponse(
                status=response.status,
                headers=MappingProxyType(
                    {key.lower(): value for key, value in response.getheaders()}
                ),
                body=b"".join(chunks),
            )
        except (socket.timeout, TimeoutError) as exc:
            raise PaddleOCRVLServiceTimeoutError(
                "PaddleOCR-VL complete-pipeline HTTP request timed out"
            ) from exc
        except PaddleOCRVLServiceError:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise PaddleOCRVLServiceConnectionError(
                "PaddleOCR-VL complete-pipeline HTTP request failed"
            ) from exc
        finally:
            connection.close()


class PaddleOCRVLServiceAdapter:
    """Parser port backed by the official complete service and page adapter."""

    def __init__(
        self,
        loader: PaddleOCRVLCompletePipelineLoader,
    ) -> None:
        if not isinstance(loader, PaddleOCRVLCompletePipelineLoader):
            raise TypeError("loader must be PaddleOCRVLCompletePipelineLoader")
        self._loader = loader
        service_identity = loader.config.identity_dict()
        adapter_config = {
            "service": service_identity,
            "inference": loader.inference_options.to_dict(),
        }
        self._delegate = PaddleOCRVLPageAdapter(
            loader,
            parser_version=loader.config.parser_version,
            model_version=loader.config.recognition_model_name,
            config=adapter_config,
        )
        delegate_manifest = self._delegate.manifest()
        metadata = dict(delegate_manifest.metadata)
        metadata.update(
            {
                "complete_pipeline": True,
                "complete_pipeline_endpoint": OFFICIAL_LAYOUT_PARSING_PATH,
                "complete_pipeline_contract": OFFICIAL_SERVICE_CONTRACT_VERSION,
                "pipeline_name": loader.config.pipeline_name,
                "pipeline_version": loader.config.pipeline_version,
                "layout_model_name": loader.config.layout_model_name,
                "recognition_model_name": loader.config.recognition_model_name,
                "remote_identity_assurance": "configured-unattested",
                "qualification": "uncertified",
                "source_material": "injected-verified-snapshot-and-page-renderer",
                "raw_vlm_endpoint_accepted": False,
            }
        )
        self._manifest = ParserCapabilityManifest(
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            parser_name=PARSER_NAME,
            parser_version=loader.config.parser_version,
            model_version=loader.config.recognition_model_name,
            supported_mime_types=("application/pdf",),
            modalities=delegate_manifest.modalities,
            block_types=tuple(BlockType),
            accelerator=AcceleratorKind.REMOTE,
            supports_page_selection=True,
            supports_region_selection=False,
            supports_batch=True,
            max_pages_per_request=loader.config.max_pages_per_request,
            # Temperature zero constrains decoding, but the complete remote
            # pipeline is not documented as bitwise deterministic.
            deterministic=False,
            metadata=metadata,
        )

    @property
    def config_sha256(self) -> str:
        return self._delegate.config_sha256

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request: ParseRequest) -> ParseResult:
        return self._delegate.parse(request)


def build_paddleocr_vl_service_adapter(
    config: PaddleOCRVLServiceConfig,
    *,
    source_loader: SourceLoader,
    page_renderer: PageRenderer,
    artifact_sink: ArtifactSink,
    inference_options: PaddleOCRVLInferenceOptions | None = None,
    http_exchange: HTTPExchange | None = None,
    clock: Clock = time.monotonic,
) -> PaddleOCRVLServiceAdapter:
    """Build a complete-pipeline parser adapter from explicit worker callbacks."""

    return PaddleOCRVLServiceAdapter(
        PaddleOCRVLCompletePipelineLoader(
            config,
            source_loader=source_loader,
            page_renderer=page_renderer,
            artifact_sink=artifact_sink,
            inference_options=inference_options,
            http_exchange=http_exchange,
            clock=clock,
        )
    )


__all__ = [
    "ArtifactSink",
    "HTTPExchange",
    "OFFICIAL_LAYOUT_PARSING_PATH",
    "OFFICIAL_SERVICE_CONTRACT_VERSION",
    "PADDLE_MARKDOWN_IMAGE_ARTIFACT_SCHEMA",
    "PADDLE_MARKDOWN_IMAGE_RECEIPT_SCHEMA",
    "PADDLE_RAW_PAGE_ARTIFACT_SCHEMA",
    "PADDLE_RAW_PAGE_RECEIPT_SCHEMA",
    "PageRenderer",
    "PaddleOCRVLArtifact",
    "PaddleOCRVLArtifactRole",
    "PaddleOCRVLCompletePipelineLoader",
    "PaddleOCRVLEndpointKind",
    "PaddleOCRVLHTTPRequest",
    "PaddleOCRVLHTTPResponse",
    "PaddleOCRVLInferenceOptions",
    "PaddleOCRVLLoadedPages",
    "PaddleOCRVLPageMaterial",
    "PaddleOCRVLRemoteServiceError",
    "PaddleOCRVLServiceAdapter",
    "PaddleOCRVLServiceConfig",
    "PaddleOCRVLServiceConfigurationError",
    "PaddleOCRVLServiceConnectionError",
    "PaddleOCRVLServiceContractError",
    "PaddleOCRVLServiceError",
    "PaddleOCRVLServiceLimitError",
    "PaddleOCRVLServiceTimeoutError",
    "RawVLMEndpointRejectedError",
    "SourceLoader",
    "StoredPaddleOCRVLArtifact",
    "SUPPORTED_LAYOUT_MODEL",
    "SUPPORTED_PIPELINE_NAME",
    "SUPPORTED_PIPELINE_VERSION",
    "SUPPORTED_RECOGNITION_MODEL",
    "build_paddleocr_vl_service_adapter",
]
