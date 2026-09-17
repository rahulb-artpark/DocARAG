"""Strict composition for one isolated parser-worker process.

Profiles contain public configuration and environment-variable *names* only.
Loading a profile performs no network calls, imports no parser/model runtime,
does not resolve secrets, and does not open a document.  Composition resolves
secrets once, constructs one source-bound vendor adapter, and gives that
immutable adapter to :class:`document_services.worker_transport.ParserWorkerHTTPServer`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import base64
import binascii
import copy
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
import tomllib
from typing import Protocol, runtime_checkable
from urllib.parse import unquote_to_bytes

from document_services.parser_adapters.docling import DoclingExportAdapter
from document_services.parser_adapters.mineru_compat import MinerUNormalizedAdapter
from document_services.domain.base import canonical_json_bytes
from document_services.domain import (
    ErrorClassification,
    IssueSeverity,
    PageRef,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParserExecutionError,
)
from document_services.parser_clients.docling import (
    DoclingServiceConfig,
    DoclingServiceExportLoader,
)
from document_services.parser_clients.mineru import (
    MinerUAPIClient,
    MinerUBackend,
    MinerUServiceConfig,
    MinerUServiceLoader,
)
from document_services.parser_clients.paddleocr_vl import (
    PaddleOCRVLPageMaterial,
    PaddleOCRVLServiceConfig,
    build_paddleocr_vl_service_adapter,
)
from document_services.evidence.sources import SourceMaterialResolver, VerifiedSourceMaterial
from document_services.worker_transport import ParserWorkerHTTPServer, ParserWorkerServerConfig


PARSER_WORKER_PROFILE_SCHEMA_VERSION = "docarag.parser-worker-profile.v1"
MAX_PARSER_WORKER_PROFILE_BYTES = 256 * 1024
DOCLING_CAS_EXTERNALIZER_VERSION = "docarag.docling-cas-externalizer.v1"
_DOCLING_REQUEST_ENVELOPE_BOUND_BYTES = 64 * 1024

_PROFILE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._+:/-]{0,190}[A-Za-z0-9])?")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "vendor",
        "parser_version",
        "model_version",
        "qualification",
        "server",
        "source",
        "artifacts",
        "service",
        "paddle_page_render",
    }
)
_SERVER_FIELDS = frozenset(
    {
        "host",
        "port",
        "max_request_bytes",
        "max_response_bytes",
        "max_in_flight",
        "request_read_timeout_seconds",
        "bearer_token_env",
        "allow_non_loopback_bind",
    }
)
_SOURCE_FIELDS = frozenset({"allowed_roots", "max_source_bytes", "chunk_size"})
_ARTIFACT_FIELDS = frozenset(
    {
        "root",
        "max_artifact_bytes",
        "max_total_bytes_per_request",
        "max_artifacts_per_request",
        "chunk_size",
    }
)
_PADDLE_RENDER_FIELDS = frozenset({"dpi", "max_page_pixels"})
_DOCLING_FIELDS = frozenset(
    {
        "base_url",
        "api_key_env",
        "connect_timeout_seconds",
        "request_timeout_seconds",
        "operation_timeout_seconds",
        "poll_interval_seconds",
        "max_poll_attempts",
        "max_source_bytes",
        "max_request_bytes",
        "max_response_bytes",
        "max_pages_per_request",
        "expected_document_schema_version",
        "allow_insecure_http",
        "tls_ca_file",
        "ocr_preset",
        "table_mode",
        "pipeline",
        "picture_description_preset",
        "enable_picture_classification",
        "enable_chart_extraction",
        "enable_code_enrichment",
    }
)
_MINERU_FIELDS = frozenset(
    {
        "base_url",
        "backend",
        "effort",
        "bearer_token_env",
        "expected_protocol_version",
        "allow_insecure_http",
        "connect_timeout_seconds",
        "control_timeout_seconds",
        "upload_timeout_seconds",
        "task_timeout_seconds",
        "result_timeout_seconds",
        "poll_interval_seconds",
        "max_poll_attempts",
        "max_source_bytes",
        "max_control_response_bytes",
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
        "max_queued_ahead",
        "max_compression_ratio",
    }
)
_PADDLE_FIELDS = frozenset(
    {
        "endpoint_kind",
        "base_url",
        "pipeline_name",
        "pipeline_version",
        "layout_model_name",
        "recognition_model_name",
        "bearer_token_env",
        "connect_timeout_seconds",
        "operation_timeout_seconds",
        "max_pages_per_request",
        "max_source_bytes",
        "max_page_bytes",
        "max_total_page_bytes",
        "max_request_bytes",
        "max_response_bytes",
        "max_response_json_depth",
        "max_response_json_nodes",
        "max_artifacts_per_page",
        "max_artifact_bytes",
        "max_total_artifact_bytes",
        "allow_insecure_http",
        "tls_ca_file",
    }
)


class ParserWorkerVendor(StrEnum):
    DOCLING = "docling"
    MINERU = "mineru"
    PADDLEOCR_VL = "paddleocr-vl"


class ParserWorkerProfileError(ValueError):
    """Base profile/composition failure safe to show without secret values."""


class ParserWorkerProfileReadError(ParserWorkerProfileError):
    pass


class ParserWorkerProfileSyntaxError(ParserWorkerProfileError):
    pass


class ParserWorkerProfileValidationError(ParserWorkerProfileError):
    pass


class ParserWorkerCompositionError(ParserWorkerProfileError):
    pass


def _mapping(value: object, scope: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ParserWorkerProfileValidationError(f"{scope} must be a TOML table")
    return value  # type: ignore[return-value]


def _table(root: Mapping[str, object], name: str, *, required: bool = True) -> Mapping[str, object]:
    value = root.get(name)
    if value is None and not required:
        return MappingProxyType({})
    if value is None:
        raise ParserWorkerProfileValidationError(f"[{name}] is required")
    return _mapping(value, f"[{name}]")


def _reject_unknown(value: Mapping[str, object], allowed: frozenset[str], scope: str) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ParserWorkerProfileValidationError(
            f"unknown parser-worker field(s) in {scope}: {', '.join(unknown)}"
        )


def _normalized_name(value: str) -> str:
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value.strip())
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _reject_literal_secrets(value: object, path: str = "profile") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = _normalized_name(key)
            if not normalized.endswith("_env") and (
                normalized in _SECRET_NAMES
                or normalized.endswith(("_password", "_secret", "_token", "_api_key"))
            ):
                raise ParserWorkerProfileValidationError(
                    f"literal secret field is forbidden at {path}.{key}; use a supported *_env field"
                )
            _reject_literal_secrets(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_literal_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str) and "-----BEGIN PRIVATE KEY-----" in value:
        raise ParserWorkerProfileValidationError(
            f"literal private key material is forbidden at {path}"
        )


def _env_name(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _ENV_NAME.fullmatch(value) is None:
        raise ParserWorkerProfileValidationError(
            f"{field_name} must be a valid environment-variable name"
        )
    return value


def _absolute_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ParserWorkerProfileValidationError(f"{field_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ParserWorkerProfileValidationError(f"{field_name} must be absolute")
    return Path(os.path.abspath(os.fspath(path)))


def _positive_int(value: object, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ParserWorkerProfileValidationError(
            f"{field_name} must be between 1 and {maximum}"
        )
    return value


def _exact_label(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise ParserWorkerProfileValidationError(
            f"{field_name} must be an exact bounded version/model label"
        )
    mutable = value.lower()
    if mutable in {"latest", "main", "master", "nightly", "edge"} or mutable.startswith(
        ("replace-", "replace_", "replace-with")
    ):
        raise ParserWorkerProfileValidationError(
            f"{field_name} must not be a mutable or placeholder label"
        )
    return value


@dataclass(frozen=True, slots=True)
class WorkerServerProfile:
    host: str
    port: int
    max_request_bytes: int
    max_response_bytes: int
    max_in_flight: int
    request_read_timeout_seconds: float
    bearer_token_env: str | None
    allow_non_loopback_bind: bool

    def resolved(self, environment: Mapping[str, str]) -> ParserWorkerServerConfig:
        token = _resolve_secret(self.bearer_token_env, environment)
        try:
            return ParserWorkerServerConfig(
                host=self.host,
                port=self.port,
                max_request_bytes=self.max_request_bytes,
                max_response_bytes=self.max_response_bytes,
                max_in_flight=self.max_in_flight,
                request_read_timeout_seconds=self.request_read_timeout_seconds,
                bearer_token=token,
                allow_non_loopback_bind=self.allow_non_loopback_bind,
            )
        except (TypeError, ValueError) as exc:
            raise ParserWorkerCompositionError(f"invalid resolved [server] configuration: {exc}") from exc

    def safe_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_in_flight": self.max_in_flight,
            "request_read_timeout_seconds": self.request_read_timeout_seconds,
            "bearer_token_env": self.bearer_token_env,
            "bearer_token_configured": self.bearer_token_env is not None,
            "allow_non_loopback_bind": self.allow_non_loopback_bind,
        }


@dataclass(frozen=True, slots=True)
class WorkerSourceProfile:
    allowed_roots: tuple[Path, ...]
    max_source_bytes: int
    chunk_size: int

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_version": "docarag.local-source-root-binding.v1",
                    "allowed_roots": [os.fspath(path) for path in self.allowed_roots],
                    "max_source_bytes": self.max_source_bytes,
                    "chunk_size": self.chunk_size,
                }
            )
        ).hexdigest()

    def safe_dict(self) -> dict[str, object]:
        return {
            "allowed_roots": [os.fspath(path) for path in self.allowed_roots],
            "max_source_bytes": self.max_source_bytes,
            "chunk_size": self.chunk_size,
            "root_binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkerArtifactProfile:
    root: Path
    max_artifact_bytes: int
    max_total_bytes_per_request: int
    max_artifacts_per_request: int
    chunk_size: int

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_version": "docarag.local-cas-root-binding.v1",
                    "root": os.fspath(self.root),
                    "max_artifact_bytes": self.max_artifact_bytes,
                    "max_total_bytes_per_request": self.max_total_bytes_per_request,
                    "max_artifacts_per_request": self.max_artifacts_per_request,
                    "chunk_size": self.chunk_size,
                }
            )
        ).hexdigest()

    def safe_dict(self) -> dict[str, object]:
        return {
            "root": os.fspath(self.root),
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_total_bytes_per_request": self.max_total_bytes_per_request,
            "max_artifacts_per_request": self.max_artifacts_per_request,
            "chunk_size": self.chunk_size,
            "root_binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class PaddlePageRenderProfile:
    dpi: int = 144
    max_page_pixels: int = 100_000_000

    def __post_init__(self) -> None:
        _positive_int(self.dpi, "paddle_page_render.dpi", 1200)
        _positive_int(
            self.max_page_pixels,
            "paddle_page_render.max_page_pixels",
            1_000_000_000,
        )

    def safe_dict(self) -> dict[str, int]:
        return {"dpi": self.dpi, "max_page_pixels": self.max_page_pixels}


ServiceConfig = DoclingServiceConfig | MinerUServiceConfig | PaddleOCRVLServiceConfig


def _safe_service_dict(
    service: ServiceConfig,
    secret_env: str | None,
) -> dict[str, object]:
    if isinstance(service, DoclingServiceConfig):
        result = service.safe_dict()
        result["api_key_configured"] = (
            secret_env is not None or service.api_key is not None
        )
    elif isinstance(service, MinerUServiceConfig):
        result = service.identity()
        result["resolved_configuration"] = {
            name: (
                getattr(service, name).value
                if isinstance(getattr(service, name), StrEnum)
                else getattr(service, name)
            )
            for name in (
                "base_url",
                "backend",
                "effort",
                "expected_protocol_version",
                "allow_insecure_http",
                "connect_timeout_seconds",
                "control_timeout_seconds",
                "upload_timeout_seconds",
                "task_timeout_seconds",
                "result_timeout_seconds",
                "poll_interval_seconds",
                "max_poll_attempts",
                "max_source_bytes",
                "max_control_response_bytes",
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
                "max_queued_ahead",
                "max_compression_ratio",
            )
        }
    else:
        result = {
            **service.identity_dict(),
            "base_url": service.base_url,
            "allow_insecure_http": service.allow_insecure_http,
            "tls_ca_file": service.tls_ca_file,
            "bounds": {
                name: getattr(service, name)
                for name in (
                    "connect_timeout_seconds",
                    "operation_timeout_seconds",
                    "max_pages_per_request",
                    "max_source_bytes",
                    "max_page_bytes",
                    "max_total_page_bytes",
                    "max_request_bytes",
                    "max_response_bytes",
                    "max_response_json_depth",
                    "max_response_json_nodes",
                    "max_artifacts_per_page",
                    "max_artifact_bytes",
                    "max_total_artifact_bytes",
                )
            },
            "bearer_token_configured": (
                secret_env is not None or service.bearer_token is not None
            ),
        }
    result = dict(result)
    result["secret_env"] = secret_env
    result["secret_configured"] = secret_env is not None
    return result


@dataclass(frozen=True, slots=True)
class ParserWorkerProfile:
    profile_id: str
    vendor: ParserWorkerVendor
    parser_version: str
    model_version: str
    qualification: str
    server: WorkerServerProfile
    source: WorkerSourceProfile
    artifacts: WorkerArtifactProfile
    service: ServiceConfig
    service_secret_env: str | None
    paddle_page_render: PaddlePageRenderProfile | None
    source_path: Path
    source_sha256: str
    configuration_sha256: str
    _source_bytes: bytes = field(repr=False, compare=False)

    @property
    def source_bytes(self) -> bytes:
        return self._source_bytes

    def safe_dict(self) -> dict[str, object]:
        return {
            "schema_version": PARSER_WORKER_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "vendor": self.vendor.value,
            "parser_version": self.parser_version,
            "model_version": self.model_version,
            "qualification": self.qualification,
            "source_path": os.fspath(self.source_path),
            "source_sha256": self.source_sha256,
            "configuration_sha256": self.configuration_sha256,
            "server": self.server.safe_dict(),
            "source": self.source.safe_dict(),
            "artifacts": self.artifacts.safe_dict(),
            "service": _safe_service_dict(
                self.service, self.service_secret_env
            ),
            "paddle_page_render": (
                None if self.paddle_page_render is None else self.paddle_page_render.safe_dict()
            ),
        }


def _read_stable(path: Path, maximum: int) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ParserWorkerProfileReadError(f"parser-worker profile could not be read: {path}: {exc}") from exc
    if len(payload) > maximum:
        raise ParserWorkerProfileReadError(
            f"parser-worker profile exceeds the {maximum}-byte limit: {path}"
        )
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise ParserWorkerProfileReadError(f"parser-worker profile changed while being read: {path}")
    return payload


def _server_profile(values: Mapping[str, object]) -> WorkerServerProfile:
    _reject_unknown(values, _SERVER_FIELDS, "[server]")
    kwargs = dict(values)
    token_env = _env_name(kwargs.pop("bearer_token_env", None), "server.bearer_token_env")
    try:
        probe = ParserWorkerServerConfig(
            **kwargs,
            bearer_token=("profile-secret-reference" if token_env is not None else None),
        )
    except (TypeError, ValueError) as exc:
        raise ParserWorkerProfileValidationError(f"invalid [server]: {exc}") from exc
    return WorkerServerProfile(
        host=probe.host,
        port=probe.port,
        max_request_bytes=probe.max_request_bytes,
        max_response_bytes=probe.max_response_bytes,
        max_in_flight=probe.max_in_flight,
        request_read_timeout_seconds=probe.request_read_timeout_seconds,
        bearer_token_env=token_env,
        allow_non_loopback_bind=probe.allow_non_loopback_bind,
    )


def _source_profile(values: Mapping[str, object]) -> WorkerSourceProfile:
    _reject_unknown(values, _SOURCE_FIELDS, "[source]")
    roots = values.get("allowed_roots")
    if not isinstance(roots, list) or not roots:
        raise ParserWorkerProfileValidationError("source.allowed_roots must be a non-empty TOML array")
    normalized = tuple(_absolute_path(value, "source.allowed_roots[]") for value in roots)
    if len(set(normalized)) != len(normalized):
        raise ParserWorkerProfileValidationError("source.allowed_roots contains duplicates")
    maximum = _positive_int(values.get("max_source_bytes", 512 * 1024 * 1024), "source.max_source_bytes", 4 * 1024**3)
    chunk = _positive_int(values.get("chunk_size", 1024 * 1024), "source.chunk_size", 64 * 1024 * 1024)
    return WorkerSourceProfile(normalized, maximum, chunk)


def _artifact_profile(values: Mapping[str, object]) -> WorkerArtifactProfile:
    _reject_unknown(values, _ARTIFACT_FIELDS, "[artifacts]")
    root = _absolute_path(values.get("root"), "artifacts.root")
    per_item = _positive_int(values.get("max_artifact_bytes", 512 * 1024 * 1024), "artifacts.max_artifact_bytes", 4 * 1024**3)
    total = _positive_int(values.get("max_total_bytes_per_request", 1024 * 1024 * 1024), "artifacts.max_total_bytes_per_request", 16 * 1024**3)
    count = _positive_int(values.get("max_artifacts_per_request", 100_000), "artifacts.max_artifacts_per_request", 10_000_000)
    chunk = _positive_int(values.get("chunk_size", 1024 * 1024), "artifacts.chunk_size", 64 * 1024 * 1024)
    if per_item > total:
        raise ParserWorkerProfileValidationError("artifacts.max_artifact_bytes must not exceed max_total_bytes_per_request")
    return WorkerArtifactProfile(root, per_item, total, count, chunk)


def _service_config(
    vendor: ParserWorkerVendor,
    values: Mapping[str, object],
    *,
    parser_version: str,
    model_version: str,
) -> tuple[ServiceConfig, str | None]:
    if vendor is ParserWorkerVendor.DOCLING:
        allowed, env_field = _DOCLING_FIELDS, "api_key_env"
        factory: Callable[..., ServiceConfig] = DoclingServiceConfig
    elif vendor is ParserWorkerVendor.MINERU:
        allowed, env_field = _MINERU_FIELDS, "bearer_token_env"
        factory = MinerUServiceConfig
    else:
        allowed, env_field = _PADDLE_FIELDS, "bearer_token_env"
        factory = PaddleOCRVLServiceConfig
    _reject_unknown(values, allowed, "[service]")
    kwargs = dict(values)
    secret_env = _env_name(kwargs.pop(env_field, None), f"service.{env_field}")
    if vendor is ParserWorkerVendor.MINERU:
        try:
            kwargs["backend"] = MinerUBackend(kwargs.get("backend"))
        except (TypeError, ValueError) as exc:
            raise ParserWorkerProfileValidationError("service.backend is required and unsupported") from exc
    if vendor is ParserWorkerVendor.PADDLEOCR_VL:
        kwargs["parser_version"] = parser_version
    try:
        config = factory(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ParserWorkerProfileValidationError(f"invalid [{vendor.value}.service]: {exc}") from exc
    if (
        isinstance(config, PaddleOCRVLServiceConfig)
        and model_version != config.recognition_model_name
    ):
        raise ParserWorkerProfileValidationError(
            "Paddle model_version must exactly equal service.recognition_model_name"
        )
    return config, secret_env


def load_parser_worker_profile(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_PARSER_WORKER_PROFILE_BYTES,
) -> ParserWorkerProfile:
    source = Path(path).expanduser()
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise ParserWorkerProfileReadError(f"parser-worker profile could not be resolved: {source}: {exc}") from exc
    if not source.is_file():
        raise ParserWorkerProfileReadError(f"parser-worker profile is not a file: {source}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    raw = _read_stable(source, max_bytes)
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ParserWorkerProfileSyntaxError(f"parser-worker profile is not valid UTF-8 TOML: {source}: {exc}") from exc
    root = _mapping(payload, "profile")
    _reject_literal_secrets(root)
    _reject_unknown(root, _TOP_FIELDS, "the root table")
    if root.get("schema_version") != PARSER_WORKER_PROFILE_SCHEMA_VERSION:
        raise ParserWorkerProfileValidationError(
            f"unsupported parser-worker schema {root.get('schema_version')!r}; expected {PARSER_WORKER_PROFILE_SCHEMA_VERSION!r}"
        )
    profile_id = root.get("profile_id")
    if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
        raise ParserWorkerProfileValidationError("profile_id is invalid")
    try:
        vendor = ParserWorkerVendor(root.get("vendor"))
    except (TypeError, ValueError) as exc:
        raise ParserWorkerProfileValidationError("vendor must be docling, mineru, or paddleocr-vl") from exc
    parser_version = _exact_label(root.get("parser_version"), "parser_version")
    model_version = _exact_label(root.get("model_version"), "model_version")
    if root.get("qualification") != "uncertified":
        raise ParserWorkerProfileValidationError("qualification must be exactly 'uncertified' until this deployment is qualified")
    server = _server_profile(_table(root, "server"))
    source_profile = _source_profile(_table(root, "source"))
    artifacts = _artifact_profile(_table(root, "artifacts"))
    service, secret_env = _service_config(
        vendor,
        _table(root, "service"),
        parser_version=parser_version,
        model_version=model_version,
    )
    if source_profile.max_source_bytes > service.max_source_bytes:
        raise ParserWorkerProfileValidationError(
            "source.max_source_bytes must not exceed service.max_source_bytes"
        )
    if isinstance(service, DoclingServiceConfig):
        encoded_source_bound = 4 * ((source_profile.max_source_bytes + 2) // 3)
        if (
            encoded_source_bound + _DOCLING_REQUEST_ENVELOPE_BOUND_BYTES
            > service.max_request_bytes
        ):
            raise ParserWorkerProfileValidationError(
                "Docling service.max_request_bytes cannot contain the Base64 source "
                "bound plus the bounded request envelope"
            )
        if service.max_response_bytes > artifacts.max_artifact_bytes:
            raise ParserWorkerProfileValidationError(
                "Docling service.max_response_bytes must not exceed "
                "artifacts.max_artifact_bytes for raw-export traceability"
            )
        raw_and_embedded_bound = (
            service.max_response_bytes
            + (3 * service.max_response_bytes + 3) // 4
        )
        if raw_and_embedded_bound > artifacts.max_total_bytes_per_request:
            raise ParserWorkerProfileValidationError(
                "Docling response plus decoded-image bound exceeds "
                "artifacts.max_total_bytes_per_request"
            )
    if isinstance(service, MinerUServiceConfig):
        if service.max_result_archive_bytes > artifacts.max_artifact_bytes:
            raise ParserWorkerProfileValidationError(
                "service.max_result_archive_bytes must not exceed artifacts.max_artifact_bytes"
            )
        if service.max_uncompressed_bytes > artifacts.max_total_bytes_per_request:
            raise ParserWorkerProfileValidationError(
                "service.max_uncompressed_bytes must not exceed artifacts.max_total_bytes_per_request"
            )
        if service.max_archive_members > artifacts.max_artifacts_per_request:
            raise ParserWorkerProfileValidationError(
                "service.max_archive_members must not exceed artifacts.max_artifacts_per_request"
            )
    if isinstance(service, PaddleOCRVLServiceConfig):
        if service.max_response_bytes > artifacts.max_artifact_bytes:
            raise ParserWorkerProfileValidationError(
                "service.max_response_bytes must not exceed "
                "artifacts.max_artifact_bytes because raw responses are retained"
            )
        if service.max_artifact_bytes > artifacts.max_artifact_bytes:
            raise ParserWorkerProfileValidationError(
                "service.max_artifact_bytes must not exceed artifacts.max_artifact_bytes"
            )
        if service.max_total_artifact_bytes > artifacts.max_total_bytes_per_request:
            raise ParserWorkerProfileValidationError(
                "service.max_total_artifact_bytes must not exceed artifacts.max_total_bytes_per_request"
            )
        # Each page may also publish one bounded raw-result artifact in addition
        # to the artifacts declared by the complete pipeline response.
        if (
            (service.max_artifacts_per_page + 1)
            * service.max_pages_per_request
            > artifacts.max_artifacts_per_request
        ):
            raise ParserWorkerProfileValidationError(
                "Paddle page/artifact bounds exceed artifacts.max_artifacts_per_request"
            )
    render_values = _table(root, "paddle_page_render", required=False)
    if vendor is ParserWorkerVendor.PADDLEOCR_VL:
        _reject_unknown(render_values, _PADDLE_RENDER_FIELDS, "[paddle_page_render]")
        render = PaddlePageRenderProfile(**dict(render_values))
    else:
        if render_values:
            raise ParserWorkerProfileValidationError("[paddle_page_render] is allowed only for vendor='paddleocr-vl'")
        render = None
    semantic = {
        "schema_version": PARSER_WORKER_PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "vendor": vendor.value,
        "parser_version": parser_version,
        "model_version": model_version,
        "qualification": "uncertified",
        "server": server.safe_dict(),
        "source": source_profile.safe_dict(),
        "artifacts": artifacts.safe_dict(),
        "service": _safe_service_dict(service, secret_env),
        "paddle_page_render": None if render is None else render.safe_dict(),
    }
    normalized = json.dumps(semantic, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return ParserWorkerProfile(
        profile_id=profile_id,
        vendor=vendor,
        parser_version=parser_version,
        model_version=model_version,
        qualification="uncertified",
        server=server,
        source=source_profile,
        artifacts=artifacts,
        service=service,
        service_secret_env=secret_env,
        paddle_page_render=render,
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        configuration_sha256=hashlib.sha256(normalized).hexdigest(),
        _source_bytes=raw,
    )


def _resolve_secret(name: str | None, environment: Mapping[str, str]) -> str | None:
    if name is None:
        return None
    value = environment.get(name)
    if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
        raise ParserWorkerCompositionError(
            f"required secret environment variable {name!r} is missing, empty, or not single-line"
        )
    return value


def _decode_image_data_uri(uri: object, maximum: int) -> tuple[bytes, str]:
    if not isinstance(uri, str) or not uri.lower().startswith("data:"):
        raise ParserWorkerCompositionError(
            "Docling visual output must be an embedded data URI before local CAS publication"
        )
    try:
        header, payload = uri[5:].split(",", 1)
    except ValueError as exc:
        raise ParserWorkerCompositionError("Docling image data URI has no payload") from exc
    parts = header.split(";")
    media_type = parts[0].lower()
    if not media_type.startswith("image/") or len(media_type) > 127:
        raise ParserWorkerCompositionError("Docling data URI must declare an image media type")
    is_base64 = any(part.lower() == "base64" for part in parts[1:])
    if is_base64:
        estimated = ((len(payload) + 3) // 4) * 3
        if estimated > maximum + 2:
            raise ParserWorkerCompositionError("Docling embedded image exceeds max_artifact_bytes")
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ParserWorkerCompositionError("Docling image data URI Base64 is invalid") from exc
    else:
        hexadecimal = frozenset("0123456789abcdefABCDEF")
        decoded_size = 0
        index = 0
        while index < len(payload):
            if (
                payload[index] == "%"
                and index + 2 < len(payload)
                and payload[index + 1] in hexadecimal
                and payload[index + 2] in hexadecimal
            ):
                decoded_size += 1
                index += 3
            else:
                decoded_size += len(payload[index].encode("utf-8"))
                index += 1
            if decoded_size > maximum:
                raise ParserWorkerCompositionError(
                    "Docling embedded image exceeds max_artifact_bytes"
                )
        data = unquote_to_bytes(payload)
    if not data or len(data) > maximum:
        raise ParserWorkerCompositionError("Docling embedded image is empty or exceeds max_artifact_bytes")
    return data, media_type


def _externalize_docling_images(
    export: Mapping[str, object],
    store: object,
    profile: WorkerArtifactProfile,
    *,
    initial_references: int = 0,
    initial_total_bytes: int = 0,
) -> Mapping[str, object]:
    """Rewrite only adapter-consumed Docling image fields to verified CAS URIs."""

    put_bytes = getattr(store, "put_bytes", None)
    if not callable(put_bytes):
        raise ParserWorkerCompositionError("artifact store does not provide put_bytes")
    try:
        output = copy.deepcopy(dict(export))
    except Exception as exc:
        raise ParserWorkerCompositionError("Docling export could not be safely copied") from exc
    references = initial_references
    total = initial_total_bytes
    if references > profile.max_artifacts_per_request:
        raise ParserWorkerCompositionError(
            "Docling artifacts exceed max_artifacts_per_request"
        )
    if total > profile.max_total_bytes_per_request:
        raise ParserWorkerCompositionError(
            "Docling artifacts exceed max_total_bytes_per_request"
        )

    def publish(mapping: dict[str, object], uri_key: str, sha_keys: tuple[str, ...], media_keys: tuple[str, ...]) -> None:
        nonlocal references, total
        if mapping.get(uri_key) is None:
            return
        references += 1
        if references > profile.max_artifacts_per_request:
            raise ParserWorkerCompositionError("Docling visual references exceed max_artifacts_per_request")
        data, media_type = _decode_image_data_uri(
            mapping[uri_key], profile.max_artifact_bytes
        )
        total += len(data)
        if total > profile.max_total_bytes_per_request:
            raise ParserWorkerCompositionError("Docling visuals exceed max_total_bytes_per_request")
        declared_values: list[str] = []
        for key in sha_keys:
            if key not in mapping or mapping[key] is None:
                continue
            candidate = mapping[key]
            if (
                not isinstance(candidate, str)
                or re.fullmatch(r"[0-9a-f]{64}", candidate) is None
            ):
                raise ParserWorkerCompositionError(
                    "Docling image declares an invalid SHA-256"
                )
            declared_values.append(candidate)
        if len(set(declared_values)) > 1:
            raise ParserWorkerCompositionError(
                "Docling image declares conflicting SHA-256 values"
            )
        declared = declared_values[0] if declared_values else None
        try:
            stored = put_bytes(
                data,
                media_type=media_type,
                expected_sha256=declared,
            )
        except Exception as exc:
            raise ParserWorkerCompositionError(
                f"Docling image could not be published to local CAS ({type(exc).__name__})"
            ) from exc
        uri = getattr(stored, "uri", None)
        digest = getattr(stored, "sha256", None)
        size_bytes = getattr(stored, "size_bytes", None)
        expected_digest = hashlib.sha256(data).hexdigest()
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or digest != expected_digest
            or uri != f"cas://sha256/{digest}"
            or size_bytes != len(data)
        ):
            raise ParserWorkerCompositionError("artifact store returned an invalid CAS identity")
        mapping[uri_key] = uri
        mapping[sha_keys[0]] = digest
        mapping[media_keys[0]] = media_type

    for collection_name in ("texts", "tables", "pictures"):
        collection = output.get(collection_name, [])
        if not isinstance(collection, list):
            # The adapter will produce the canonical schema error; avoid broad
            # recursive rewriting of an untrusted shape here.
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            image = item.get("image")
            if isinstance(image, dict):
                publish(image, "uri", ("sha256", "image_sha256"), ("mimetype", "media_type"))
            if item.get("image_uri") is not None:
                publish(item, "image_uri", ("image_sha256",), ("image_mimetype",))
            assets = item.get("assets", [])
            if isinstance(assets, list):
                for asset in assets:
                    if isinstance(asset, dict):
                        publish(asset, "uri", ("sha256", "image_sha256"), ("mimetype", "media_type"))
    return output


def _publish_docling_export(
    export: Mapping[str, object],
    store: object,
    profile: WorkerArtifactProfile,
    *,
    externalize_images: bool,
) -> Mapping[str, object]:
    """Publish the canonical validated export, then bind blocks and visuals."""

    put_bytes = getattr(store, "put_bytes", None)
    if not callable(put_bytes):
        raise ParserWorkerCompositionError("artifact store does not provide put_bytes")
    try:
        raw_export = canonical_json_bytes(export)
    except Exception as exc:
        raise ParserWorkerCompositionError(
            "Docling export is not canonical JSON"
        ) from exc
    if not raw_export or len(raw_export) > profile.max_artifact_bytes:
        raise ParserWorkerCompositionError(
            "canonical Docling export is empty or exceeds max_artifact_bytes"
        )
    if len(raw_export) > profile.max_total_bytes_per_request:
        raise ParserWorkerCompositionError(
            "canonical Docling export exceeds max_total_bytes_per_request"
        )
    try:
        stored = put_bytes(
            raw_export,
            media_type="application/json",
            expected_sha256=hashlib.sha256(raw_export).hexdigest(),
        )
    except Exception as exc:
        raise ParserWorkerCompositionError(
            f"Docling export could not be published to local CAS ({type(exc).__name__})"
        ) from exc
    raw_uri = getattr(stored, "uri", None)
    digest = getattr(stored, "sha256", None)
    expected_digest = hashlib.sha256(raw_export).hexdigest()
    if (
        digest != expected_digest
        or raw_uri != f"cas://sha256/{digest}"
        or getattr(stored, "size_bytes", None) != len(raw_export)
    ):
        raise ParserWorkerCompositionError(
            "artifact store returned an invalid Docling export CAS identity"
        )

    if externalize_images:
        output = _externalize_docling_images(
            export,
            store,
            profile,
            initial_references=1,
            initial_total_bytes=len(raw_export),
        )
    else:
        try:
            output = copy.deepcopy(dict(export))
        except Exception as exc:
            raise ParserWorkerCompositionError(
                "Docling export could not be safely copied"
            ) from exc
    for collection_name in ("texts", "tables", "pictures"):
        collection = output.get(collection_name, [])
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            self_ref = item.get("self_ref")
            expected_ref = f"#/{collection_name}/{index}"
            if self_ref != expected_ref:
                raise ParserWorkerCompositionError(
                    "Docling item self_ref is not a resolvable canonical JSON pointer"
                )
            item["raw_artifact_uri"] = raw_uri + expected_ref
    return output


class _DoclingCASExportLoader:
    def __init__(
        self,
        loader: DoclingServiceExportLoader,
        store: object,
        profile: WorkerArtifactProfile,
        profile_configuration_sha256: str,
    ) -> None:
        self._loader = loader
        self._store = store
        self._profile = profile
        self._profile_configuration_sha256 = profile_configuration_sha256

    @property
    def safe_config(self) -> Mapping[str, object]:
        root = os.fspath(self._profile.root)
        return MappingProxyType(
            {
                "service": dict(self._loader.safe_config),
                "artifact_externalization": {
                    "schema_version": DOCLING_CAS_EXTERNALIZER_VERSION,
                    "artifact_root": root,
                    "artifact_root_sha256": hashlib.sha256(root.encode()).hexdigest(),
                    "max_artifact_bytes": self._profile.max_artifact_bytes,
                    "max_total_bytes_per_request": self._profile.max_total_bytes_per_request,
                    "max_artifacts_per_request": self._profile.max_artifacts_per_request,
                    "chunk_size": self._profile.chunk_size,
                    "raw_export_media_type": "application/json",
                    "raw_export_serialization": "canonical-json",
                    "orphan_reclamation": "not-implemented-external-quota-gc-required",
                },
                "profile_configuration_sha256": self._profile_configuration_sha256,
                "qualification": "uncertified",
            }
        )

    def __call__(self, request: ParseRequest) -> Mapping[str, object]:
        export = self._loader(request)
        try:
            # Raw export provenance is retained for every request. Image assets
            # are externalized only when the request asked Docling to return
            # them; placeholder exports contain no data URI payloads.
            return _publish_docling_export(
                export,
                self._store,
                self._profile,
                externalize_images=request.options.parse_images,
            )
        except ParserExecutionError:
            raise
        except Exception as exc:
            raise ParserExecutionError(
                ParseIssue(
                    code=ParseErrorCode.CONTRACT_VIOLATION,
                    classification=ErrorClassification.PERMANENT,
                    severity=IssueSeverity.ERROR,
                    message="Docling visual artifacts could not be bound to local CAS",
                    details={"exception_type": type(exc).__name__},
                )
            ) from exc


@dataclass(frozen=True, slots=True)
class WorkerArtifactCallbacks:
    mineru_publisher: Callable[[object], object] | None = None
    paddle_sink: Callable[[object], object] | None = None
    owner: object | None = field(default=None, repr=False)


ArtifactCallbackFactory = Callable[[WorkerArtifactProfile, ParserWorkerVendor], WorkerArtifactCallbacks]
PageRendererFactory = Callable[[PaddlePageRenderProfile, PaddleOCRVLServiceConfig], Callable]


def _default_artifact_callbacks(
    profile: WorkerArtifactProfile, vendor: ParserWorkerVendor
) -> WorkerArtifactCallbacks:
    try:
        module = importlib.import_module("document_services.persistence.artifact_store")
        store_type = getattr(module, "LocalContentAddressedArtifactStore")
    except (ImportError, AttributeError) as exc:
        raise ParserWorkerCompositionError(
            "local artifact-store support is unavailable; install a build containing document_services.persistence.artifact_store"
        ) from exc
    try:
        store = store_type(
            profile.root,
            max_artifact_bytes=profile.max_artifact_bytes,
            chunk_size=profile.chunk_size,
        )
        if vendor is ParserWorkerVendor.MINERU:
            factory = getattr(module, "make_mineru_archive_publisher")
            limits_type = getattr(module, "MinerUArchivePublicationLimits")
            limits = limits_type(
                max_archive_bytes=profile.max_artifact_bytes,
                max_members=profile.max_artifacts_per_request,
                max_member_bytes=profile.max_artifact_bytes,
                max_total_uncompressed_bytes=profile.max_total_bytes_per_request,
                read_chunk_size=profile.chunk_size,
            )
            return WorkerArtifactCallbacks(
                mineru_publisher=factory(store, limits=limits), owner=store
            )
        if vendor is ParserWorkerVendor.PADDLEOCR_VL:
            factory = getattr(module, "make_paddleocr_vl_artifact_sink")
            return WorkerArtifactCallbacks(paddle_sink=factory(store), owner=store)
        return WorkerArtifactCallbacks(owner=store)
    except Exception as exc:
        _close_quietly(locals().get("store"))
        raise ParserWorkerCompositionError(
            f"could not construct the {vendor.value} local artifact callbacks ({type(exc).__name__})"
        ) from exc


class _PyMuPDFPageRenderer:
    def __init__(
        self,
        profile: PaddlePageRenderProfile,
        service: PaddleOCRVLServiceConfig,
    ) -> None:
        self._profile = profile
        self._service = service
        try:
            self._fitz = importlib.import_module("pymupdf")
        except ImportError:
            try:
                self._fitz = importlib.import_module("fitz")
            except ImportError as exc:
                raise ParserWorkerCompositionError(
                    "Paddle page rendering requires optional PyMuPDF in the Paddle worker environment"
                ) from exc

    def __call__(
        self, source: VerifiedSourceMaterial, page: PageRef
    ) -> PaddleOCRVLPageMaterial:
        if source.data is None:
            raise ParserWorkerCompositionError("verified PDF bytes are required for Paddle rendering")
        document = None
        try:
            document = self._fitz.open(stream=source.data, filetype="pdf")
            if page.page_index >= len(document):
                raise ParserWorkerCompositionError("requested page exceeds the verified PDF")
            pdf_page = document.load_page(page.page_index)
            scale = self._profile.dpi / 72.0
            pixmap = pdf_page.get_pixmap(
                matrix=self._fitz.Matrix(scale, scale), alpha=False
            )
            pixels = int(pixmap.width) * int(pixmap.height)
            if pixels > self._profile.max_page_pixels:
                raise ParserWorkerCompositionError("rendered Paddle page exceeds max_page_pixels")
            data = bytes(pixmap.tobytes("png"))
            if len(data) > self._service.max_page_bytes:
                raise ParserWorkerCompositionError("rendered Paddle page exceeds service.max_page_bytes")
            return PaddleOCRVLPageMaterial(
                document_version_id=source.document_version_id,
                source_sha256=source.sha256,
                page_index=page.page_index,
                media_type="image/png",
                width=int(pixmap.width),
                height=int(pixmap.height),
                data=data,
                render_identity=f"pymupdf-png-{self._profile.dpi}dpi.v1",
            )
        except ParserWorkerCompositionError:
            raise
        except Exception as exc:
            raise ParserWorkerCompositionError(
                f"PyMuPDF could not render the verified page ({type(exc).__name__})"
            ) from exc
        finally:
            if document is not None:
                document.close()


def _default_page_renderer(
    profile: PaddlePageRenderProfile,
    service: PaddleOCRVLServiceConfig,
) -> Callable:
    return _PyMuPDFPageRenderer(profile, service)


@runtime_checkable
class WorkerServer(Protocol):
    @property
    def manifest(self): ...
    @property
    def base_url(self) -> str: ...
    def serve_forever(self) -> None: ...
    def close(self) -> None: ...


def _close_quietly(value: object | None) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


class ParserWorkerRuntime:
    """Exact-once owner of server, vendor client, and artifact-store resources."""

    def __init__(self, profile: ParserWorkerProfile, server: WorkerServer, owners: Sequence[object]) -> None:
        self.profile = profile
        self.server = server
        self._owners = tuple(owners)
        self._closed = False

    def summary(self) -> Mapping[str, object]:
        manifest = self.server.manifest
        return MappingProxyType(
            {
                "schema_version": "docarag.parser-worker-startup.v1",
                "profile_id": self.profile.profile_id,
                "vendor": self.profile.vendor.value,
                "qualification": "uncertified",
                "adapter_name": manifest.adapter_name,
                "adapter_version": manifest.adapter_version,
                "parser_version": manifest.parser_version,
                "model_version": manifest.model_version,
                "manifest_sha256": manifest.manifest_sha256,
                "base_url": self.server.base_url,
                "profile_source_sha256": self.profile.source_sha256,
                "profile_configuration_sha256": self.profile.configuration_sha256,
                "source_material": {
                    "handoff": "verified-local-binding",
                    "scope": "same-host-or-identically-mounted-absolute-roots",
                    "allowed_roots": [
                        os.fspath(path) for path in self.profile.source.allowed_roots
                    ],
                    "root_binding_sha256": self.profile.source.binding_sha256,
                    "cross_host_upload_supported": False,
                },
                "artifact_storage": {
                    "uri_scheme": "cas://sha256/",
                    "scope": "same-host-or-identically-mounted-absolute-root",
                    "artifact_root": os.fspath(self.profile.artifacts.root),
                    "root_binding_sha256": self.profile.artifacts.binding_sha256,
                    "cross_host_artifact_endpoint_supported": False,
                    "orphan_reclamation": (
                        "not-implemented-external-quota-gc-required"
                    ),
                },
            }
        )

    def serve_forever(self) -> None:
        if self._closed:
            raise ParserWorkerCompositionError("parser-worker runtime is closed")
        self.server.serve_forever()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[str] = []
        for value in (self.server, *reversed(self._owners)):
            try:
                close = getattr(value, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        if failures:
            raise ParserWorkerCompositionError(
                "parser-worker resources could not all be closed: " + ", ".join(failures)
            )


ServerFactory = Callable[[object, ParserWorkerServerConfig], WorkerServer]


def build_parser_worker_runtime(
    profile: ParserWorkerProfile,
    *,
    environment: Mapping[str, str] | None = None,
    artifact_factory: ArtifactCallbackFactory = _default_artifact_callbacks,
    page_renderer_factory: PageRendererFactory = _default_page_renderer,
    server_factory: ServerFactory = ParserWorkerHTTPServer,
) -> ParserWorkerRuntime:
    if not isinstance(profile, ParserWorkerProfile):
        raise TypeError("profile must be ParserWorkerProfile")
    active_environment = os.environ if environment is None else environment
    if not isinstance(active_environment, Mapping):
        raise TypeError("environment must be a mapping")
    server_config = profile.server.resolved(active_environment)
    service_secret = _resolve_secret(profile.service_secret_env, active_environment)
    try:
        resolver = SourceMaterialResolver(
            allowed_roots=profile.source.allowed_roots,
            max_source_bytes=profile.source.max_source_bytes,
            chunk_size=profile.source.chunk_size,
        )
    except Exception as exc:
        raise ParserWorkerCompositionError(
            f"source resolver could not use the configured roots ({type(exc).__name__})"
        ) from exc
    source_loader = lambda request: resolver.resolve(request.document, include_bytes=True)
    owners: list[object] = []
    server: WorkerServer | None = None
    try:
        if profile.vendor is ParserWorkerVendor.DOCLING:
            assert isinstance(profile.service, DoclingServiceConfig)
            callbacks = artifact_factory(profile.artifacts, profile.vendor)
            if (
                not isinstance(callbacks, WorkerArtifactCallbacks)
                or callbacks.owner is None
                or not callable(getattr(callbacks.owner, "put_bytes", None))
            ):
                raise ParserWorkerCompositionError(
                    "artifact factory did not supply a Docling CAS store"
                )
            owners.append(callbacks.owner)
            service = replace(profile.service, api_key=service_secret)
            service_loader = DoclingServiceExportLoader(
                service, source_loader=source_loader
            )
            loader = _DoclingCASExportLoader(
                service_loader,
                callbacks.owner,
                profile.artifacts,
                profile.configuration_sha256,
            )
            adapter = DoclingExportAdapter(
                loader,
                parser_version=profile.parser_version,
                model_version=profile.model_version,
                config=loader.safe_config,
            )
        elif profile.vendor is ParserWorkerVendor.MINERU:
            assert isinstance(profile.service, MinerUServiceConfig)
            callbacks = artifact_factory(profile.artifacts, profile.vendor)
            if not isinstance(callbacks, WorkerArtifactCallbacks) or callbacks.mineru_publisher is None:
                raise ParserWorkerCompositionError("artifact factory did not supply a MinerU archive publisher")
            if callbacks.owner is not None:
                owners.append(callbacks.owner)
            client = MinerUAPIClient(profile.service, bearer_token=service_secret)
            owners.append(client)
            loader = MinerUServiceLoader(
                client,
                source_loader=source_loader,
                artifact_publisher=callbacks.mineru_publisher,  # type: ignore[arg-type]
            )
            adapter = MinerUNormalizedAdapter(
                loader,
                parser_version=profile.parser_version,
                model_version=profile.model_version,
                config=profile.service.identity(),
            )
        else:
            assert isinstance(profile.service, PaddleOCRVLServiceConfig)
            assert profile.paddle_page_render is not None
            callbacks = artifact_factory(profile.artifacts, profile.vendor)
            if not isinstance(callbacks, WorkerArtifactCallbacks) or callbacks.paddle_sink is None:
                raise ParserWorkerCompositionError("artifact factory did not supply a Paddle artifact sink")
            if callbacks.owner is not None:
                owners.append(callbacks.owner)
            service = replace(profile.service, bearer_token=service_secret)
            renderer = page_renderer_factory(profile.paddle_page_render, service)
            adapter = build_paddleocr_vl_service_adapter(
                service,
                source_loader=source_loader,
                page_renderer=renderer,
                artifact_sink=callbacks.paddle_sink,  # type: ignore[arg-type]
            )
        server = server_factory(adapter, server_config)
        if not isinstance(server, WorkerServer):
            raise ParserWorkerCompositionError("server_factory returned an incompatible server")
        return ParserWorkerRuntime(profile, server, owners)
    except Exception:
        _close_quietly(server)
        for owner in reversed(owners):
            _close_quietly(owner)
        raise


__all__ = [
    "ArtifactCallbackFactory",
    "MAX_PARSER_WORKER_PROFILE_BYTES",
    "PARSER_WORKER_PROFILE_SCHEMA_VERSION",
    "PaddlePageRenderProfile",
    "ParserWorkerCompositionError",
    "ParserWorkerProfile",
    "ParserWorkerProfileError",
    "ParserWorkerProfileReadError",
    "ParserWorkerProfileSyntaxError",
    "ParserWorkerProfileValidationError",
    "ParserWorkerRuntime",
    "ParserWorkerVendor",
    "WorkerArtifactCallbacks",
    "WorkerArtifactProfile",
    "WorkerServerProfile",
    "WorkerSourceProfile",
    "build_parser_worker_runtime",
    "load_parser_worker_profile",
]
