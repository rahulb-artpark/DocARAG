"""Typed parser capabilities, requests, results, and failure contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import ClassVar, Self

from document_services.domain.base import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    JsonDTOMixin,
    JsonObject,
    canonical_json_bytes,
    content_sha256,
    dump_json,
    freeze_json,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sha256,
    require_schema_version,
    require_sequence,
    stable_id,
)
from document_services.domain.evidence import (
    BlockType,
    DocumentRef,
    PageRef,
    ParserBlockObservation,
    SourceGeometry,
)


class ParserModality(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR_TEXT = "ocr_text"
    LAYOUT = "layout"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    CODE = "code"


class AcceleratorKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    REMOTE = "remote"
    ANY = "any"


class OcrMode(StrEnum):
    AUTO = "auto"
    FORCE = "force"
    DISABLED = "disabled"


class MinerUEffort(StrEnum):
    """Explicit MinerU 3.x accuracy/latency tier in request identity."""

    MEDIUM = "medium"
    HIGH = "high"


class ParseStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class IssueSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class ErrorClassification(StrEnum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    CANCELLED = "cancelled"


class ParseErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_NORMALIZED_BLOCK = "invalid_normalized_block"
    UNSUPPORTED_INPUT = "unsupported_input"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PARSER_FAILURE = "parser_failure"
    CANCELLED = "cancelled"
    CONTRACT_VIOLATION = "contract_violation"


class ParserArtifactAvailability(StrEnum):
    """Whether a parser-native artifact was emitted for one bounded task."""

    EMITTED = "emitted"
    NOT_EMITTED = "not_emitted"


MAX_PARSER_ARTIFACT_RECEIPTS = 4_096
MAX_PARSER_ARTIFACT_METADATA_BYTES = 64 * 1024
MAX_PARSER_ARTIFACT_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_PARSER_ARTIFACT_KEY_CHARS = 1_024


def _enum(enum_type: type[StrEnum], value: object, field_name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class ParserCapabilityManifest(JsonDTOMixin):
    """Machine-readable adapter capabilities used by routing and diagnostics."""

    adapter_name: str
    adapter_version: str
    parser_name: str
    parser_version: str
    model_version: str
    supported_mime_types: tuple[str, ...]
    modalities: tuple[ParserModality, ...]
    block_types: tuple[BlockType, ...]
    accelerator: AcceleratorKind = AcceleratorKind.ANY
    supports_page_selection: bool = True
    supports_region_selection: bool = False
    supports_batch: bool = False
    max_pages_per_request: int | None = None
    deterministic: bool = False
    metadata: JsonObject = field(default_factory=dict)
    manifest_sha256: str = field(init=False)
    capability_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "adapter_name",
            "adapter_version",
            "parser_name",
            "parser_version",
            "model_version",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        mime_types = tuple(sorted(set(self.supported_mime_types)))
        if not mime_types:
            raise ContractValidationError("supported_mime_types must not be empty")
        for mime_type in mime_types:
            require_non_empty(mime_type, "supported_mime_types[]")
        modalities = tuple(
            sorted(
                {_enum(ParserModality, item, "modalities[]") for item in self.modalities},
                key=lambda item: item.value,
            )
        )
        block_types = tuple(
            sorted(
                {_enum(BlockType, item, "block_types[]") for item in self.block_types},
                key=lambda item: item.value,
            )
        )
        if not modalities or not block_types:
            raise ContractValidationError("modalities and block_types must not be empty")
        accelerator = _enum(AcceleratorKind, self.accelerator, "accelerator")
        for field_name in (
            "supports_page_selection",
            "supports_region_selection",
            "supports_batch",
            "deterministic",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")
        if self.max_pages_per_request is not None and (
            isinstance(self.max_pages_per_request, bool)
            or not isinstance(self.max_pages_per_request, int)
            or self.max_pages_per_request < 1
        ):
            raise ContractValidationError("max_pages_per_request must be a positive integer")
        metadata = freeze_json(self.metadata, field="capability.metadata")
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "model_version": self.model_version,
            "supported_mime_types": mime_types,
            "modalities": [item.value for item in modalities],
            "block_types": [item.value for item in block_types],
            "accelerator": accelerator.value,
            "supports_page_selection": self.supports_page_selection,
            "supports_region_selection": self.supports_region_selection,
            "supports_batch": self.supports_batch,
            "max_pages_per_request": self.max_pages_per_request,
            "deterministic": self.deterministic,
            "metadata": metadata,
        }
        digest = content_sha256(identity)
        object.__setattr__(self, "supported_mime_types", mime_types)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "block_types", block_types)
        object.__setattr__(self, "accelerator", accelerator)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(
            self,
            "capability_id",
            stable_id("cap", self.adapter_name, self.adapter_version, digest),
        )

    def supports(
        self,
        *,
        mime_type: str,
        modalities: frozenset[ParserModality] = frozenset(),
        block_types: frozenset[BlockType] = frozenset(),
        regions: bool = False,
    ) -> bool:
        return (
            mime_type in self.supported_mime_types
            and modalities.issubset(self.modalities)
            and block_types.issubset(self.block_types)
            and (not regions or self.supports_region_selection)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capability_id": self.capability_id,
            "manifest_sha256": self.manifest_sha256,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "model_version": self.model_version,
            "supported_mime_types": list(self.supported_mime_types),
            "modalities": [item.value for item in self.modalities],
            "block_types": [item.value for item in self.block_types],
            "accelerator": self.accelerator.value,
            "supports_page_selection": self.supports_page_selection,
            "supports_region_selection": self.supports_region_selection,
            "supports_batch": self.supports_batch,
            "max_pages_per_request": self.max_pages_per_request,
            "deterministic": self.deterministic,
            "metadata": json_ready(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "capability_id",
                "manifest_sha256",
                "adapter_name",
                "adapter_version",
                "parser_name",
                "parser_version",
                "model_version",
                "supported_mime_types",
                "modalities",
                "block_types",
                "accelerator",
                "supports_page_selection",
                "supports_region_selection",
                "supports_batch",
                "max_pages_per_request",
                "deterministic",
                "metadata",
            },
            dto="ParserCapabilityManifest",
        )
        result = cls(
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            adapter_version=payload.get("adapter_version"),  # type: ignore[arg-type]
            parser_name=payload.get("parser_name"),  # type: ignore[arg-type]
            parser_version=payload.get("parser_version"),  # type: ignore[arg-type]
            model_version=payload.get("model_version"),  # type: ignore[arg-type]
            supported_mime_types=tuple(
                require_sequence(payload.get("supported_mime_types"), "supported_mime_types")
            ),  # type: ignore[arg-type]
            modalities=tuple(
                _enum(ParserModality, value, "modalities[]")
                for value in require_sequence(payload.get("modalities"), "modalities")
            ),
            block_types=tuple(
                _enum(BlockType, value, "block_types[]")
                for value in require_sequence(payload.get("block_types"), "block_types")
            ),
            accelerator=_enum(
                AcceleratorKind, payload.get("accelerator"), "accelerator"
            ),
            supports_page_selection=payload.get("supports_page_selection"),  # type: ignore[arg-type]
            supports_region_selection=payload.get("supports_region_selection"),  # type: ignore[arg-type]
            supports_batch=payload.get("supports_batch"),  # type: ignore[arg-type]
            max_pages_per_request=payload.get("max_pages_per_request"),  # type: ignore[arg-type]
            deterministic=payload.get("deterministic"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "capability.metadata"),
        )
        for field_name in ("capability_id", "manifest_sha256"):
            if payload.get(field_name) != getattr(result, field_name):
                raise ContractValidationError(
                    f"ParserCapabilityManifest {field_name} mismatch"
                )
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class ParseOptions(JsonDTOMixin):
    languages: tuple[str, ...] = ()
    ocr_mode: OcrMode = OcrMode.AUTO
    parse_tables: bool = True
    parse_formulas: bool = True
    parse_images: bool = True
    mineru_effort: MinerUEffort | None = None
    timeout_seconds: float | None = None
    extra: JsonObject = field(default_factory=dict)
    options_sha256: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        languages = tuple(dict.fromkeys(self.languages))
        for language in languages:
            require_non_empty(language, "languages[]")
        ocr_mode = _enum(OcrMode, self.ocr_mode, "ocr_mode")
        mineru_effort = (
            None
            if self.mineru_effort is None
            else _enum(MinerUEffort, self.mineru_effort, "mineru_effort")
        )
        for field_name in ("parse_tables", "parse_formulas", "parse_images"):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise ContractValidationError("timeout_seconds must be finite and positive")
        extra = freeze_json(self.extra, field="parse_options.extra")
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "ocr_mode", ocr_mode)
        object.__setattr__(self, "mineru_effort", mineru_effort)
        object.__setattr__(self, "extra", extra)
        object.__setattr__(
            self,
            "options_sha256",
            content_sha256(
                {
                    "languages": languages,
                    "ocr_mode": ocr_mode.value,
                    "parse_tables": self.parse_tables,
                    "parse_formulas": self.parse_formulas,
                    "parse_images": self.parse_images,
                    "mineru_effort": (
                        None if mineru_effort is None else mineru_effort.value
                    ),
                    "timeout_seconds": self.timeout_seconds,
                    "extra": extra,
                }
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "options_sha256": self.options_sha256,
            "languages": list(self.languages),
            "ocr_mode": self.ocr_mode.value,
            "parse_tables": self.parse_tables,
            "parse_formulas": self.parse_formulas,
            "parse_images": self.parse_images,
            "mineru_effort": (
                None if self.mineru_effort is None else self.mineru_effort.value
            ),
            "timeout_seconds": self.timeout_seconds,
            "extra": json_ready(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "options_sha256",
                "languages",
                "ocr_mode",
                "parse_tables",
                "parse_formulas",
                "parse_images",
                "mineru_effort",
                "timeout_seconds",
                "extra",
            },
            dto="ParseOptions",
        )
        result = cls(
            languages=tuple(require_sequence(payload.get("languages", []), "languages")),  # type: ignore[arg-type]
            ocr_mode=_enum(OcrMode, payload.get("ocr_mode"), "ocr_mode"),
            parse_tables=payload.get("parse_tables"),  # type: ignore[arg-type]
            parse_formulas=payload.get("parse_formulas"),  # type: ignore[arg-type]
            parse_images=payload.get("parse_images"),  # type: ignore[arg-type]
            mineru_effort=(
                None
                if payload.get("mineru_effort") is None
                else _enum(
                    MinerUEffort,
                    payload.get("mineru_effort"),
                    "mineru_effort",
                )
            ),
            timeout_seconds=payload.get("timeout_seconds"),  # type: ignore[arg-type]
            extra=require_mapping(payload.get("extra", {}), "parse_options.extra"),
        )
        if payload.get("options_sha256") != result.options_sha256:
            raise ContractValidationError("ParseOptions options_sha256 mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParseRegion(JsonDTOMixin):
    page_index: int
    geometry: SourceGeometry
    purpose: str = "targeted_parse"
    region_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise ContractValidationError("region page_index must be a non-negative integer")
        require_non_empty(self.purpose, "region purpose")
        object.__setattr__(
            self,
            "region_id",
            stable_id("region", self.page_index, self.geometry.to_dict(), self.purpose),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "region_id": self.region_id,
            "page_index": self.page_index,
            "geometry": self.geometry.to_dict(),
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {"schema_version", "region_id", "page_index", "geometry", "purpose"},
            dto="ParseRegion",
        )
        result = cls(
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            geometry=SourceGeometry.from_dict(
                require_mapping(payload.get("geometry"), "region.geometry")
            ),
            purpose=payload.get("purpose"),  # type: ignore[arg-type]
        )
        if payload.get("region_id") != result.region_id:
            raise ContractValidationError("ParseRegion region_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParseRequest(JsonDTOMixin):
    document: DocumentRef
    pages: tuple[PageRef, ...]
    options: ParseOptions = field(default_factory=ParseOptions)
    regions: tuple[ParseRegion, ...] = ()
    run_id: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None
    request_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentRef):
            raise ContractValidationError("document must be DocumentRef")
        if not isinstance(self.options, ParseOptions):
            raise ContractValidationError("options must be ParseOptions")
        pages = tuple(self.pages)
        if not pages:
            raise ContractValidationError("parse request requires at least one page")
        page_indices: set[int] = set()
        for page in pages:
            if not isinstance(page, PageRef):
                raise ContractValidationError("pages must contain PageRef DTOs")
            if page.document_version_id != self.document.document_version_id:
                raise ContractValidationError("request page belongs to a different document version")
            if page.page_index in page_indices:
                raise ContractValidationError("parse request contains a duplicate page")
            page_indices.add(page.page_index)
            if (
                self.document.page_count is not None
                and page.page_index >= self.document.page_count
            ):
                raise ContractValidationError("parse request page exceeds document page_count")
        pages = tuple(sorted(pages, key=lambda page: page.page_index))
        regions = tuple(self.regions)
        for region in regions:
            if not isinstance(region, ParseRegion):
                raise ContractValidationError("regions must contain ParseRegion DTOs")
            if region.page_index not in page_indices:
                raise ContractValidationError("parse region page is not present in request pages")
        regions = tuple(sorted(regions, key=lambda region: region.region_id))
        for field_name in ("run_id", "trace_id", "idempotency_key"):
            value = getattr(self, field_name)
            if value is not None:
                require_non_empty(value, field_name)
        request_id = stable_id(
            "parse",
            self.document.document_version_id,
            [page.page_id for page in pages],
            [region.region_id for region in regions],
            self.options.options_sha256,
            self.idempotency_key,
        )
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "request_id", request_id)

    @property
    def page_by_index(self) -> Mapping[int, PageRef]:
        return {page.page_index: page for page in self.pages}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "document": self.document.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "options": self.options.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "request_id",
                "document",
                "pages",
                "options",
                "regions",
                "run_id",
                "trace_id",
                "idempotency_key",
            },
            dto="ParseRequest",
        )
        result = cls(
            document=DocumentRef.from_dict(
                require_mapping(payload.get("document"), "document")
            ),
            pages=tuple(
                PageRef.from_dict(require_mapping(page, f"pages[{index}]"))
                for index, page in enumerate(
                    require_sequence(payload.get("pages"), "pages")
                )
            ),
            options=ParseOptions.from_dict(
                require_mapping(payload.get("options"), "options")
            ),
            regions=tuple(
                ParseRegion.from_dict(require_mapping(region, f"regions[{index}]"))
                for index, region in enumerate(
                    require_sequence(payload.get("regions", []), "regions")
                )
            ),
            run_id=payload.get("run_id"),  # type: ignore[arg-type]
            trace_id=payload.get("trace_id"),  # type: ignore[arg-type]
            idempotency_key=payload.get("idempotency_key"),  # type: ignore[arg-type]
        )
        if payload.get("request_id") != result.request_id:
            raise ContractValidationError("ParseRequest request_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class ParserArtifactReceipt(JsonDTOMixin):
    """Content-bound parser artifact outcome carried at result scope.

    Result scope avoids repeating a task-level receipt on every block and also
    preserves a truthful ``not_emitted`` outcome when a parser returns no
    blocks.  A receipt is part of ``ParseResult.result_sha256``; metrics remain
    deliberately non-authoritative operational data.  ``artifact_key`` was
    added before this contract's first durable release; payloads carrying the
    older key-less receipt identity are intentionally rejected rather than
    silently migrated under the same schema version.
    """

    artifact_role: str
    availability: ParserArtifactAvailability
    artifact_schema: str
    producer_task_id: str
    document_version_id: str
    source_sha256: str
    artifact_key: str = "primary"
    uri: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str = "application/octet-stream"
    metadata: JsonObject = field(default_factory=dict)
    receipt_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_role",
            "artifact_schema",
            "producer_task_id",
            "document_version_id",
            "artifact_key",
            "media_type",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        if (
            len(self.artifact_key) > MAX_PARSER_ARTIFACT_KEY_CHARS
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.artifact_key
            )
        ):
            raise ContractValidationError(
                "artifact_key must be a bounded printable native artifact name"
            )
        require_sha256(self.source_sha256, "artifact.source_sha256")
        availability = _enum(
            ParserArtifactAvailability,
            self.availability,
            "artifact.availability",
        )
        if availability is ParserArtifactAvailability.EMITTED:
            require_non_empty(self.uri, "artifact.uri")
            require_sha256(self.sha256, "artifact.sha256")
            if (
                isinstance(self.size_bytes, bool)
                or not isinstance(self.size_bytes, int)
                or self.size_bytes < 1
            ):
                raise ContractValidationError(
                    "emitted artifact size_bytes must be a positive integer"
                )
        elif any(
            value is not None for value in (self.uri, self.sha256, self.size_bytes)
        ):
            raise ContractValidationError(
                "not-emitted artifacts cannot declare URI, digest, or size"
            )
        metadata = freeze_json(self.metadata, field="artifact.metadata")
        if len(canonical_json_bytes(metadata)) > MAX_PARSER_ARTIFACT_METADATA_BYTES:
            raise ContractValidationError(
                "artifact metadata exceeds its serialized byte limit"
            )
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "receipt_id",
            stable_id(
                "parser-artifact",
                self.artifact_role,
                availability.value,
                self.artifact_schema,
                self.producer_task_id,
                self.document_version_id,
                self.source_sha256,
                self.artifact_key,
                self.uri,
                self.sha256,
                self.size_bytes,
                self.media_type,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "artifact_role": self.artifact_role,
            "availability": self.availability.value,
            "artifact_schema": self.artifact_schema,
            "producer_task_id": self.producer_task_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "artifact_key": self.artifact_key,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "receipt_id",
                "artifact_role",
                "availability",
                "artifact_schema",
                "producer_task_id",
                "document_version_id",
                "source_sha256",
                "artifact_key",
                "uri",
                "sha256",
                "size_bytes",
                "media_type",
                "metadata",
            },
            dto="ParserArtifactReceipt",
        )
        result = cls(
            artifact_role=payload.get("artifact_role"),  # type: ignore[arg-type]
            availability=_enum(
                ParserArtifactAvailability,
                payload.get("availability"),
                "artifact.availability",
            ),
            artifact_schema=payload.get("artifact_schema"),  # type: ignore[arg-type]
            producer_task_id=payload.get("producer_task_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            artifact_key=payload.get("artifact_key", "primary"),  # type: ignore[arg-type]
            uri=payload.get("uri"),  # type: ignore[arg-type]
            sha256=payload.get("sha256"),  # type: ignore[arg-type]
            size_bytes=payload.get("size_bytes"),  # type: ignore[arg-type]
            media_type=payload.get("media_type"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "artifact.metadata"),
        )
        if payload.get("receipt_id") != result.receipt_id:
            raise ContractValidationError("ParserArtifactReceipt receipt_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParseIssue(JsonDTOMixin):
    code: ParseErrorCode
    classification: ErrorClassification
    severity: IssueSeverity
    message: str
    page_index: int | None = None
    parser_native_id: str | None = None
    details: JsonObject = field(default_factory=dict)
    issue_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        code = _enum(ParseErrorCode, self.code, "issue.code")
        classification = _enum(
            ErrorClassification, self.classification, "issue.classification"
        )
        severity = _enum(IssueSeverity, self.severity, "issue.severity")
        require_non_empty(self.message, "issue.message")
        if self.page_index is not None and (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or self.page_index < 0
        ):
            raise ContractValidationError("issue page_index must be non-negative")
        if self.parser_native_id is not None:
            require_non_empty(self.parser_native_id, "issue.parser_native_id")
        details = freeze_json(self.details, field="issue.details")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "details", details)
        object.__setattr__(
            self,
            "issue_id",
            stable_id(
                "issue",
                code.value,
                classification.value,
                severity.value,
                self.message,
                self.page_index,
                self.parser_native_id,
                details,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "issue_id": self.issue_id,
            "code": self.code.value,
            "classification": self.classification.value,
            "severity": self.severity.value,
            "message": self.message,
            "page_index": self.page_index,
            "parser_native_id": self.parser_native_id,
            "details": json_ready(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "issue_id",
                "code",
                "classification",
                "severity",
                "message",
                "page_index",
                "parser_native_id",
                "details",
            },
            dto="ParseIssue",
        )
        result = cls(
            code=_enum(ParseErrorCode, payload.get("code"), "issue.code"),
            classification=_enum(
                ErrorClassification,
                payload.get("classification"),
                "issue.classification",
            ),
            severity=_enum(IssueSeverity, payload.get("severity"), "issue.severity"),
            message=payload.get("message"),  # type: ignore[arg-type]
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            parser_native_id=payload.get("parser_native_id"),  # type: ignore[arg-type]
            details=require_mapping(payload.get("details", {}), "issue.details"),
        )
        if payload.get("issue_id") != result.issue_id:
            raise ContractValidationError("ParseIssue issue_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParseResult(JsonDTOMixin):
    request_id: str
    adapter_name: str
    adapter_version: str
    status: ParseStatus
    observations: tuple[ParserBlockObservation, ...] = ()
    issues: tuple[ParseIssue, ...] = ()
    artifacts: tuple[ParserArtifactReceipt, ...] = ()
    duration_ms: float | None = None
    metrics: JsonObject = field(default_factory=dict)
    result_sha256: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("request_id", "adapter_name", "adapter_version"):
            require_non_empty(getattr(self, field_name), field_name)
        status = _enum(ParseStatus, self.status, "status")
        observations = tuple(self.observations)
        issues = tuple(self.issues)
        artifacts = tuple(self.artifacts)
        if len(artifacts) > MAX_PARSER_ARTIFACT_RECEIPTS:
            raise ContractValidationError(
                "ParseResult parser artifact receipt limit exceeded"
            )
        if any(not isinstance(item, ParserBlockObservation) for item in observations):
            raise ContractValidationError(
                "observations must contain ParserBlockObservation DTOs"
            )
        if any(not isinstance(item, ParseIssue) for item in issues):
            raise ContractValidationError("issues must contain ParseIssue DTOs")
        if any(not isinstance(item, ParserArtifactReceipt) for item in artifacts):
            raise ContractValidationError(
                "artifacts must contain ParserArtifactReceipt DTOs"
            )
        artifact_ids = [item.receipt_id for item in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ContractValidationError(
                "ParseResult contains duplicate parser artifact receipts"
            )
        artifact_semantic_keys = [
            (
                item.producer_task_id,
                item.artifact_role,
                item.artifact_schema,
                item.artifact_key,
            )
            for item in artifacts
        ]
        if len(artifact_semantic_keys) != len(set(artifact_semantic_keys)):
            raise ContractValidationError(
                "ParseResult contains duplicate semantic parser artifact receipts"
            )
        artifacts = tuple(sorted(artifacts, key=lambda item: item.receipt_id))
        if (
            sum(len(canonical_json_bytes(item.to_dict())) for item in artifacts)
            > MAX_PARSER_ARTIFACT_RECEIPT_BYTES
        ):
            raise ContractValidationError(
                "ParseResult parser artifact receipts exceed their byte limit"
            )
        error_count = sum(issue.severity == IssueSeverity.ERROR for issue in issues)
        if status == ParseStatus.SUCCEEDED and error_count:
            raise ContractValidationError("succeeded ParseResult cannot contain error issues")
        if status == ParseStatus.PARTIAL and (not observations or not error_count):
            raise ContractValidationError(
                "partial ParseResult requires observations and at least one error"
            )
        if status == ParseStatus.FAILED and observations:
            raise ContractValidationError("failed ParseResult cannot contain observations")
        if status == ParseStatus.FAILED and not error_count:
            raise ContractValidationError("failed ParseResult requires an error issue")
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or not math.isfinite(float(self.duration_ms))
            or float(self.duration_ms) < 0
        ):
            raise ContractValidationError("duration_ms must be finite and non-negative")
        observation_ids = [observation.observation_id for observation in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ContractValidationError("ParseResult contains duplicate observation IDs")
        metrics = freeze_json(self.metrics, field="parse_result.metrics")
        digest_payload: dict[str, object] = {
            "request_id": self.request_id,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "status": status.value,
            "observations": [item.to_dict() for item in observations],
            "issues": [item.to_dict() for item in issues],
        }
        if artifacts:
            digest_payload["artifacts"] = [item.to_dict() for item in artifacts]
        digest = content_sha256(digest_payload)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "result_sha256", digest)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "result_sha256": self.result_sha256,
            "request_id": self.request_id,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "status": self.status.value,
            "observations": [item.to_dict() for item in self.observations],
            "issues": [item.to_dict() for item in self.issues],
            "duration_ms": self.duration_ms,
            "metrics": json_ready(self.metrics),
        }
        if self.artifacts:
            payload["artifacts"] = [item.to_dict() for item in self.artifacts]
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "result_sha256",
                "request_id",
                "adapter_name",
                "adapter_version",
                "status",
                "observations",
                "issues",
                "artifacts",
                "duration_ms",
                "metrics",
            },
            dto="ParseResult",
        )
        raw_artifacts = require_sequence(payload.get("artifacts", []), "artifacts")
        if len(raw_artifacts) > MAX_PARSER_ARTIFACT_RECEIPTS:
            raise ContractValidationError(
                "ParseResult parser artifact receipt limit exceeded"
            )
        result = cls(
            request_id=payload.get("request_id"),  # type: ignore[arg-type]
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            adapter_version=payload.get("adapter_version"),  # type: ignore[arg-type]
            status=_enum(ParseStatus, payload.get("status"), "status"),
            observations=tuple(
                ParserBlockObservation.from_dict(
                    require_mapping(observation, f"observations[{index}]")
                )
                for index, observation in enumerate(
                    require_sequence(payload.get("observations", []), "observations")
                )
            ),
            issues=tuple(
                ParseIssue.from_dict(require_mapping(issue, f"issues[{index}]"))
                for index, issue in enumerate(
                    require_sequence(payload.get("issues", []), "issues")
                )
            ),
            artifacts=tuple(
                ParserArtifactReceipt.from_dict(
                    require_mapping(artifact, f"artifacts[{index}]")
                )
                for index, artifact in enumerate(raw_artifacts)
            ),
            duration_ms=payload.get("duration_ms"),  # type: ignore[arg-type]
            metrics=require_mapping(payload.get("metrics", {}), "parse_result.metrics"),
        )
        if payload.get("result_sha256") != result.result_sha256:
            raise ContractValidationError("ParseResult result_sha256 mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


class ParserExecutionError(RuntimeError):
    """Exception boundary with bounded receipts for already-published artifacts.

    Services may fail after completing an earlier page or task.  Carrying only
    the issue would make those immutable publications unreachable from the
    failed ``ParseResult``.  Receipts are therefore allowed on this local
    exception boundary, while their document/source binding is still verified
    by the adapter registry trust boundary.
    """

    def __init__(
        self,
        issue: ParseIssue,
        *,
        artifacts: tuple[ParserArtifactReceipt, ...] = (),
    ) -> None:
        if issue.severity != IssueSeverity.ERROR:
            raise ContractValidationError("ParserExecutionError requires an error issue")
        artifacts = tuple(artifacts)
        if len(artifacts) > MAX_PARSER_ARTIFACT_RECEIPTS:
            raise ContractValidationError(
                "ParserExecutionError parser artifact receipt limit exceeded"
            )
        if any(not isinstance(item, ParserArtifactReceipt) for item in artifacts):
            raise ContractValidationError(
                "ParserExecutionError artifacts must contain ParserArtifactReceipt DTOs"
            )
        receipt_ids = [item.receipt_id for item in artifacts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ContractValidationError(
                "ParserExecutionError contains duplicate parser artifact receipts"
            )
        semantic_keys = [
            (
                item.producer_task_id,
                item.artifact_role,
                item.artifact_schema,
                item.artifact_key,
            )
            for item in artifacts
        ]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ContractValidationError(
                "ParserExecutionError contains duplicate semantic parser artifact receipts"
            )
        if (
            sum(len(canonical_json_bytes(item.to_dict())) for item in artifacts)
            > MAX_PARSER_ARTIFACT_RECEIPT_BYTES
        ):
            raise ContractValidationError(
                "ParserExecutionError parser artifact receipts exceed their byte limit"
            )
        super().__init__(issue.message)
        self.issue = issue
        self.artifacts = tuple(sorted(artifacts, key=lambda item: item.receipt_id))
