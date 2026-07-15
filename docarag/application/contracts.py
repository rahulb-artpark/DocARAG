"""Versioned application contracts for one local document extraction.

These contracts are intentionally smaller than parser-domain DTOs.  A CLI or
HTTP adapter submits an already admitted, content-bound local PDF snapshot
and receives a compact receipt for the durable canonical artifact.  Canonical
blocks stay in the artifact instead of being duplicated in an API response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import ClassVar, Mapping, Self
from urllib.parse import parse_qsl, urlsplit

from docarag.domain.base import (
    ContractValidationError,
    JsonDTOMixin,
    dump_json,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sha256,
    stable_id,
)
from docarag.domain.evidence import DocumentRef


DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION = (
    "docarag.application.document-extraction.v1"
)
DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION = (
    "docarag.application.document-extraction.v2"
)
# Compatibility alias for callers that treated the former combined constant as
# the response schema.  New code should use the direction-specific constants.
DOCUMENT_EXTRACTION_SCHEMA_VERSION = DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION
MAX_DOCUMENT_EXTRACTION_REQUEST_BYTES = 128 * 1024
MAX_DOCUMENT_EXTRACTION_RESULT_BYTES = 2 * 1024 * 1024

_MAX_PATH_CHARACTERS = 32 * 1024
_MAX_URI_CHARACTERS = 32 * 1024
_MAX_DISPLAY_NAME_CHARACTERS = 1024
_SENSITIVE_URI_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "signature",
        "token",
        "x_amz_credential",
        "x_amz_signature",
    }
)


def _require_application_schema(
    payload: Mapping[str, object],
    *,
    expected: str,
    contract_name: str,
) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise ContractValidationError(
            f"unsupported {contract_name} schema_version "
            f"{actual!r}; expected {expected!r}"
        )


def _bounded_json_object(
    payload: str | bytes | bytearray,
    *,
    maximum_bytes: int,
    field: str,
) -> dict[str, object]:
    if isinstance(payload, str):
        size = len(payload.encode("utf-8"))
    elif isinstance(payload, (bytes, bytearray)):
        size = len(payload)
    else:
        raise ContractValidationError(f"{field} must be JSON text or bytes")
    if size > maximum_bytes:
        raise ContractValidationError(
            f"{field} exceeds the {maximum_bytes}-byte serialized limit"
        )
    return parse_json_object(payload)


def _absolute_path(value: object, field_name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractValidationError(f"{field_name} must be a filesystem path")
    try:
        raw = os.fspath(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} is invalid") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ContractValidationError(f"{field_name} must be a non-empty path")
    if len(raw) > _MAX_PATH_CHARACTERS:
        raise ContractValidationError(f"{field_name} is too long")
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ContractValidationError(f"{field_name} must be absolute")
    return os.path.normpath(os.fspath(expanded))


def _safe_uri(value: object, field_name: str) -> str:
    text = require_non_empty(value, field_name)
    if len(text) > _MAX_URI_CHARACTERS:
        raise ContractValidationError(f"{field_name} is too long")
    try:
        parsed = urlsplit(text)
        query = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} is invalid") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ContractValidationError(f"{field_name} must not contain credentials")
    for key, _ in query:
        normalized = key.lower().replace("-", "_").replace(" ", "_")
        if normalized in _SENSITIVE_URI_KEYS or any(
            normalized.endswith(f"_{suffix}")
            for suffix in ("api_key", "password", "secret", "token")
        ):
            raise ContractValidationError(
                f"{field_name} must not contain secret query parameters"
            )
    return text


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class DocumentExtractionRequest(JsonDTOMixin):
    """One immutable, content-addressed local PDF extraction command.

    Snapshot admission/copying is deliberately outside this use case.  This
    keeps uploads, object-store downloads, and local CLI paths replaceable while
    ensuring the routed parser always receives an expected source digest.
    """

    source_path: str | os.PathLike[str]
    output_root: str | os.PathLike[str]
    source_sha256: str
    source_uri: str
    display_name: str
    force: bool = False
    media_type: str = "application/pdf"
    request_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_path = _absolute_path(self.source_path, "source_path")
        output_root = _absolute_path(self.output_root, "output_root")
        source_sha256 = require_sha256(self.source_sha256, "source_sha256")
        source_uri = _safe_uri(self.source_uri, "source_uri")
        display_name = require_non_empty(self.display_name, "display_name")
        if len(display_name) > _MAX_DISPLAY_NAME_CHARACTERS:
            raise ContractValidationError("display_name is too long")
        if self.media_type != "application/pdf":
            raise ContractValidationError(
                "the routed document extraction use case currently accepts "
                "application/pdf only"
            )
        if not isinstance(self.force, bool):
            raise ContractValidationError("force must be a boolean")
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "source_uri", source_uri)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(
            self,
            "request_id",
            stable_id(
                "document-extraction-request",
                source_path,
                output_root,
                source_sha256,
                source_uri,
                display_name,
                self.media_type,
                self.force,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "source_path": self.source_path,
            "output_root": self.output_root,
            "source_sha256": self.source_sha256,
            "source_uri": self.source_uri,
            "display_name": self.display_name,
            "media_type": self.media_type,
            "force": self.force,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_application_schema(
            payload,
            expected=cls.SCHEMA_VERSION,
            contract_name="document extraction request",
        )
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "request_id",
                "source_path",
                "output_root",
                "source_sha256",
                "source_uri",
                "display_name",
                "media_type",
                "force",
            },
            dto="DocumentExtractionRequest",
        )
        result = cls(
            source_path=payload.get("source_path"),  # type: ignore[arg-type]
            output_root=payload.get("output_root"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            source_uri=payload.get("source_uri"),  # type: ignore[arg-type]
            display_name=payload.get("display_name"),  # type: ignore[arg-type]
            media_type=payload.get("media_type"),  # type: ignore[arg-type]
            force=payload.get("force"),  # type: ignore[arg-type]
        )
        if payload.get("request_id") != result.request_id:
            raise ContractValidationError("DocumentExtractionRequest request_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _bounded_json_object(
                payload,
                maximum_bytes=MAX_DOCUMENT_EXTRACTION_REQUEST_BYTES,
                field="document extraction request",
            )
        )


@dataclass(frozen=True, slots=True)
class DocumentExtractionResult(JsonDTOMixin):
    """Compact success receipt for a verifiable canonical-document handoff.

    The potentially large canonical block collection is intentionally not
    embedded.  Consumers re-hash the referenced artifact and validate its
    producer schema before indexing it; the URI itself is not assumed immutable.
    """

    request_id: str
    backend_configuration_sha256: str
    document: DocumentRef
    canonical_artifact_uri: str
    canonical_artifact_sha256: str
    canonical_artifact_size_bytes: int
    audit_artifact_uri: str
    audit_artifact_sha256: str
    audit_artifact_size_bytes: int
    routed_audit_sha256: str
    canonical_block_count: int
    canonical_blocks_sha256: str
    handoff_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        request_id = require_non_empty(self.request_id, "request_id")
        configuration = require_sha256(
            self.backend_configuration_sha256,
            "backend_configuration_sha256",
        )
        if not isinstance(self.document, DocumentRef):
            raise ContractValidationError("document must be a DocumentRef")
        artifact_uri = _safe_uri(
            self.canonical_artifact_uri,
            "canonical_artifact_uri",
        )
        artifact_sha256 = require_sha256(
            self.canonical_artifact_sha256,
            "canonical_artifact_sha256",
        )
        artifact_size = _non_negative_int(
            self.canonical_artifact_size_bytes,
            "canonical_artifact_size_bytes",
        )
        if artifact_size == 0:
            raise ContractValidationError(
                "canonical_artifact_size_bytes must be positive"
            )
        audit_artifact_uri = _safe_uri(
            self.audit_artifact_uri,
            "audit_artifact_uri",
        )
        audit_artifact_sha256 = require_sha256(
            self.audit_artifact_sha256,
            "audit_artifact_sha256",
        )
        audit_artifact_size = _non_negative_int(
            self.audit_artifact_size_bytes,
            "audit_artifact_size_bytes",
        )
        if audit_artifact_size == 0:
            raise ContractValidationError(
                "audit_artifact_size_bytes must be positive"
            )
        routed_audit_sha256 = require_sha256(
            self.routed_audit_sha256,
            "routed_audit_sha256",
        )
        block_count = _non_negative_int(
            self.canonical_block_count,
            "canonical_block_count",
        )
        if block_count == 0:
            raise ContractValidationError("canonical_block_count must be positive")
        blocks_sha256 = require_sha256(
            self.canonical_blocks_sha256,
            "canonical_blocks_sha256",
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "backend_configuration_sha256", configuration)
        object.__setattr__(self, "canonical_artifact_uri", artifact_uri)
        object.__setattr__(self, "canonical_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "canonical_artifact_size_bytes", artifact_size)
        object.__setattr__(self, "audit_artifact_uri", audit_artifact_uri)
        object.__setattr__(self, "audit_artifact_sha256", audit_artifact_sha256)
        object.__setattr__(self, "audit_artifact_size_bytes", audit_artifact_size)
        object.__setattr__(self, "routed_audit_sha256", routed_audit_sha256)
        object.__setattr__(self, "canonical_block_count", block_count)
        object.__setattr__(self, "canonical_blocks_sha256", blocks_sha256)
        object.__setattr__(
            self,
            "handoff_id",
            stable_id(
                "canonical-document-handoff",
                request_id,
                configuration,
                self.document.document_version_id,
                artifact_uri,
                artifact_sha256,
                artifact_size,
                audit_artifact_uri,
                audit_artifact_sha256,
                audit_artifact_size,
                routed_audit_sha256,
                block_count,
                blocks_sha256,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "handoff_id": self.handoff_id,
            "backend_configuration_sha256": self.backend_configuration_sha256,
            "document": self.document.to_dict(),
            "canonical_artifact_uri": self.canonical_artifact_uri,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "canonical_artifact_size_bytes": self.canonical_artifact_size_bytes,
            "audit_artifact_uri": self.audit_artifact_uri,
            "audit_artifact_sha256": self.audit_artifact_sha256,
            "audit_artifact_size_bytes": self.audit_artifact_size_bytes,
            "routed_audit_sha256": self.routed_audit_sha256,
            "canonical_block_count": self.canonical_block_count,
            "canonical_blocks_sha256": self.canonical_blocks_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_application_schema(
            payload,
            expected=cls.SCHEMA_VERSION,
            contract_name="document extraction result",
        )
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "request_id",
                "handoff_id",
                "backend_configuration_sha256",
                "document",
                "canonical_artifact_uri",
                "canonical_artifact_sha256",
                "canonical_artifact_size_bytes",
                "audit_artifact_uri",
                "audit_artifact_sha256",
                "audit_artifact_size_bytes",
                "routed_audit_sha256",
                "canonical_block_count",
                "canonical_blocks_sha256",
            },
            dto="DocumentExtractionResult",
        )
        result = cls(
            request_id=payload.get("request_id"),  # type: ignore[arg-type]
            backend_configuration_sha256=payload.get(  # type: ignore[arg-type]
                "backend_configuration_sha256"
            ),
            document=DocumentRef.from_dict(
                require_mapping(payload.get("document"), "document")
            ),
            canonical_artifact_uri=payload.get("canonical_artifact_uri"),  # type: ignore[arg-type]
            canonical_artifact_sha256=payload.get("canonical_artifact_sha256"),  # type: ignore[arg-type]
            canonical_artifact_size_bytes=payload.get(  # type: ignore[arg-type]
                "canonical_artifact_size_bytes"
            ),
            audit_artifact_uri=payload.get("audit_artifact_uri"),  # type: ignore[arg-type]
            audit_artifact_sha256=payload.get("audit_artifact_sha256"),  # type: ignore[arg-type]
            audit_artifact_size_bytes=payload.get(  # type: ignore[arg-type]
                "audit_artifact_size_bytes"
            ),
            routed_audit_sha256=payload.get("routed_audit_sha256"),  # type: ignore[arg-type]
            canonical_block_count=payload.get("canonical_block_count"),  # type: ignore[arg-type]
            canonical_blocks_sha256=payload.get("canonical_blocks_sha256"),  # type: ignore[arg-type]
        )
        if payload.get("handoff_id") != result.handoff_id:
            raise ContractValidationError("DocumentExtractionResult handoff_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _bounded_json_object(
                payload,
                maximum_bytes=MAX_DOCUMENT_EXTRACTION_RESULT_BYTES,
                field="document extraction result",
            )
        )


__all__ = [
    "DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION",
    "DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION",
    "DOCUMENT_EXTRACTION_SCHEMA_VERSION",
    "MAX_DOCUMENT_EXTRACTION_REQUEST_BYTES",
    "MAX_DOCUMENT_EXTRACTION_RESULT_BYTES",
    "DocumentExtractionRequest",
    "DocumentExtractionResult",
]
