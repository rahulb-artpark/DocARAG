"""Versioned source, geometry, parser-observation, and canonical-evidence DTOs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
from typing import ClassVar, Self

from .base import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    JsonDTOMixin,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    content_sha256,
    dump_json,
    freeze_json,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_probability,
    require_schema_version,
    require_sequence,
    require_sha256,
    stable_id,
)


class CoordinateSpace(StrEnum):
    """Coordinate semantics; never infer units from numeric values."""

    NORMALIZED = "normalized"
    PAGE_POINTS = "page_points"
    PAGE_PIXELS = "page_pixels"
    PARSER_PAGE = "parser_page"


class CoordinateOrigin(StrEnum):
    """Origin of page coordinates before canonical normalization.

    Parser adapters must declare this explicitly.  Canonical storage and
    cross-parser reconciliation use ``TOP_LEFT``; ``BOTTOM_LEFT`` exists so a
    raw vendor observation can be transformed without guessing.
    """

    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"


class BlockType(StrEnum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    EQUATION = "equation"
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class RepresentationKind(StrEnum):
    TEXT = "text"
    OCR_TEXT = "ocr_text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    LATEX = "latex"
    MATHML = "mathml"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TABLE_GRID = "table_grid"
    VISUAL_DESCRIPTION = "visual_description"
    VLM_LATEX = "vlm_latex"
    DIAGRAM_GRAPH = "diagram_graph"
    STRUCTURED_DATA = "structured_data"


class AssetKind(StrEnum):
    SOURCE = "source"
    PAGE_RENDER = "page_render"
    REGION_CROP = "region_crop"
    EXTRACTED_IMAGE = "extracted_image"


class QualityStatus(StrEnum):
    ACCEPTED = "accepted"
    DEGRADED = "degraded"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class RelationType(StrEnum):
    """Finite, directed structural relations shared by parser adapters.

    Relations are assertions made by a parser, not geometry-derived guesses.
    The explicit inverse mapping lets graph consumers preserve direction
    without treating inverse labels as equivalent edge identities.
    """

    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    HAS_CAPTION = "has_caption"
    CAPTION_OF = "caption_of"
    HAS_FOOTNOTE = "has_footnote"
    FOOTNOTE_OF = "footnote_of"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    TABLE_CONTINUES = "table_continues"
    TABLE_CONTINUED_FROM = "table_continued_from"
    READING_NEXT = "reading_next"
    READING_PREVIOUS = "reading_previous"

    @property
    def opposite(self) -> "RelationType":
        return RelationType(
            {
                RelationType.PARENT_OF.value: RelationType.CHILD_OF.value,
                RelationType.CHILD_OF.value: RelationType.PARENT_OF.value,
                RelationType.HAS_CAPTION.value: RelationType.CAPTION_OF.value,
                RelationType.CAPTION_OF.value: RelationType.HAS_CAPTION.value,
                RelationType.HAS_FOOTNOTE.value: RelationType.FOOTNOTE_OF.value,
                RelationType.FOOTNOTE_OF.value: RelationType.HAS_FOOTNOTE.value,
                RelationType.REFERENCES.value: RelationType.REFERENCED_BY.value,
                RelationType.REFERENCED_BY.value: RelationType.REFERENCES.value,
                RelationType.TABLE_CONTINUES.value: RelationType.TABLE_CONTINUED_FROM.value,
                RelationType.TABLE_CONTINUED_FROM.value: RelationType.TABLE_CONTINUES.value,
                RelationType.READING_NEXT.value: RelationType.READING_PREVIOUS.value,
                RelationType.READING_PREVIOUS.value: RelationType.READING_NEXT.value,
            }[self.value]
        )


class SemanticStructureKind(StrEnum):
    """Typed block-internal structures shared by every parser adapter."""

    TABLE = "table"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    LAYOUT = "layout"


class SemanticProducerKind(StrEnum):
    """How a semantic graph was produced; never infer this from a model name."""

    PARSER_NATIVE = "parser_native"
    DETERMINISTIC_DERIVED = "deterministic_derived"
    MODEL_DERIVED = "model_derived"


class SemanticStructureAvailability(StrEnum):
    """Availability of an emitted, necessarily non-empty semantic graph.

    An unavailable graph is not represented by an empty successful DTO.  The
    parser's existing issue/capability/readiness surfaces carry that state.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"


class SemanticElementKind(StrEnum):
    """Finite element vocabulary for semantic substructure graphs."""

    TABLE_CELL = "table_cell"
    FORMULA_SYMBOL = "formula_symbol"
    DIAGRAM_ELEMENT = "diagram_element"
    FLOWCHART_ELEMENT = "flowchart_element"
    LAYOUT_REGION = "layout_region"


class SemanticRelationKind(StrEnum):
    """Directed, block-local semantic and spatial edge vocabulary."""

    TABLE_ADJACENT_RIGHT = "table_adjacent_right"
    TABLE_ADJACENT_BELOW = "table_adjacent_below"
    FORMULA_ARGUMENT_OF = "formula_argument_of"
    FORMULA_OPERAND_OF = "formula_operand_of"
    FORMULA_SUPERSCRIPT_OF = "formula_superscript_of"
    FORMULA_SUBSCRIPT_OF = "formula_subscript_of"
    FORMULA_NEXT = "formula_next"
    DIAGRAM_CONNECTS_TO = "diagram_connects_to"
    DIAGRAM_CONTAINS = "diagram_contains"
    FLOW_NEXT = "flow_next"
    FLOW_BRANCH_TRUE = "flow_branch_true"
    FLOW_BRANCH_FALSE = "flow_branch_false"
    SPATIAL_LEFT_OF = "spatial_left_of"
    SPATIAL_RIGHT_OF = "spatial_right_of"
    SPATIAL_ABOVE = "spatial_above"
    SPATIAL_BELOW = "spatial_below"
    SPATIAL_OVERLAPS = "spatial_overlaps"
    SPATIAL_CONTAINS = "spatial_contains"
    READING_NEXT = "reading_next"
    SEMANTIC_ASSOCIATED_WITH = "semantic_associated_with"


SEMANTIC_NORMALIZED_EDGE_TYPE_KEY = "normalized_edge_type"
MAX_SEMANTIC_NORMALIZED_EDGE_TYPE_CHARS = 256


def normalize_semantic_edge_type(value: object) -> str:
    """Return the source-neutral spelling used for explicit graph edge types.

    This deliberately normalizes spelling only.  It does not guess that two
    different edge vocabularies are synonyms: ``depends_on`` and
    ``reports_to`` remain different semantic assertions.
    """

    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            "semantic relation edge type must be non-empty text"
        )
    raw_value = value.strip()
    if len(raw_value) > MAX_SEMANTIC_NORMALIZED_EDGE_TYPE_CHARS:
        raise ContractValidationError(
            "semantic relation edge type exceeds character limit"
        )
    normalized = re.sub(r"[_\W]+", "_", raw_value.casefold()).strip("_")
    if not normalized:
        raise ContractValidationError(
            "semantic relation edge type must contain a word character"
        )
    if len(normalized) > MAX_SEMANTIC_NORMALIZED_EDGE_TYPE_CHARS:
        raise ContractValidationError(
            "semantic relation normalized edge type exceeds character limit"
        )
    return normalized


_OBSERVATION_ID = re.compile(r"obs_[0-9a-f]{32}")
_BLOCK_ID = re.compile(r"block_[0-9a-f]{32}")
_RELATION_ID = re.compile(r"rel_[0-9a-f]{32}")
_REPRESENTATION_ID = re.compile(r"repr_[0-9a-f]{32}")
_ASSET_ID = re.compile(r"asset_[0-9a-f]{32}")
_SEMANTIC_ELEMENT_ID = re.compile(r"selem_[0-9a-f]{32}")
_SEMANTIC_SELECTOR_ID = re.compile(r"ssel_[0-9a-f]{32}")
_SEMANTIC_SUBSTRUCTURE_ID = re.compile(r"sstruct_[0-9a-f]{32}")


# Public, deterministic resource limits.  They are enforced both before nested
# DTO construction and after programmatic construction so a parser worker
# cannot turn one block into an unbounded controller allocation.
MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK = 128
MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE = 4_096
MAX_SEMANTIC_RELATIONS_PER_STRUCTURE = 16_384
MAX_SEMANTIC_ELEMENTS_PER_BLOCK = 8_192
MAX_SEMANTIC_RELATIONS_PER_BLOCK = 32_768
MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT = 16
MAX_SEMANTIC_SOURCE_IDS = 256
MAX_SEMANTIC_STRING_CHARS = 65_536
MAX_SEMANTIC_JSON_NODES = 16_384
MAX_SEMANTIC_JSON_DEPTH = 32
MAX_SEMANTIC_JSON_BYTES = 1_048_576
MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE = 65_536
MAX_SEMANTIC_SUBSTRUCTURE_CANONICAL_BYTES = 16_777_216
MAX_SEMANTIC_BLOCK_CANONICAL_BYTES = 67_108_864
MAX_SEMANTIC_DTO_JSON_BYTES = 20_971_520


def _require_contract_id(value: object, field: str, pattern: re.Pattern[str]) -> str:
    result = require_non_empty(value, field)
    if pattern.fullmatch(result) is None:
        raise ContractValidationError(f"{field} is not a canonical DocARAG ID")
    return result


def _enum(enum_type: type[StrEnum], value: object, field_name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _bounded_semantic_string(
    value: object,
    field_name: str,
    *,
    required: bool = True,
) -> str | None:
    if value is None and not required:
        return None
    result = require_non_empty(value, field_name)
    if len(result) > MAX_SEMANTIC_STRING_CHARS:
        raise ContractValidationError(
            f"{field_name} exceeds {MAX_SEMANTIC_STRING_CHARS} characters"
        )
    return result


def _bounded_semantic_sequence(
    value: object,
    field_name: str,
    maximum: int,
) -> Sequence[object]:
    sequence = require_sequence(value, field_name)
    if len(sequence) > maximum:
        raise ContractValidationError(
            f"{field_name} exceeds the limit of {maximum} items"
        )
    return sequence


def _freeze_bounded_semantic_json(value: object, *, field_name: str) -> JsonValue:
    """Validate semantic payload complexity before recursively freezing it."""

    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SEMANTIC_JSON_NODES:
            raise ContractValidationError(
                f"{field_name} exceeds {MAX_SEMANTIC_JSON_NODES} JSON nodes"
            )
        if depth > MAX_SEMANTIC_JSON_DEPTH:
            raise ContractValidationError(
                f"{field_name} exceeds JSON depth {MAX_SEMANTIC_JSON_DEPTH}"
            )
        if isinstance(current, str):
            if len(current) > MAX_SEMANTIC_STRING_CHARS:
                raise ContractValidationError(
                    f"{field_name} contains a string longer than "
                    f"{MAX_SEMANTIC_STRING_CHARS} characters"
                )
            continue
        if current is None or isinstance(current, (bool, int, float)):
            continue
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise ContractValidationError(
                    f"{field_name} contains a non-string key"
                )
            for key, item in current.items():
                if len(key) > MAX_SEMANTIC_STRING_CHARS:
                    raise ContractValidationError(
                        f"{field_name} contains an overlong object key"
                    )
                stack.append((item, depth + 1))
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            stack.extend((item, depth + 1) for item in current)
            continue
        raise ContractValidationError(
            f"{field_name} must contain JSON-compatible values"
        )

    frozen = freeze_json(value, field=field_name)
    if len(canonical_json_bytes(frozen)) > MAX_SEMANTIC_JSON_BYTES:
        raise ContractValidationError(
            f"{field_name} exceeds {MAX_SEMANTIC_JSON_BYTES} canonical JSON bytes"
        )
    return frozen


def _parse_bounded_semantic_json(
    payload: str | bytes | bytearray,
    *,
    field_name: str,
) -> dict[str, object]:
    if isinstance(payload, str):
        if len(payload) > MAX_SEMANTIC_DTO_JSON_BYTES:
            raise ContractValidationError(
                f"{field_name} exceeds {MAX_SEMANTIC_DTO_JSON_BYTES} JSON bytes"
            )
        size_bytes = len(payload.encode("utf-8"))
    elif isinstance(payload, (bytes, bytearray)):
        size_bytes = len(payload)
    else:
        raise ContractValidationError(
            f"{field_name} must be str, bytes, or bytearray"
        )
    if size_bytes > MAX_SEMANTIC_DTO_JSON_BYTES:
        raise ContractValidationError(
            f"{field_name} exceeds {MAX_SEMANTIC_DTO_JSON_BYTES} JSON bytes"
        )
    return parse_json_object(payload)


@dataclass(frozen=True, slots=True)
class DocumentRef(JsonDTOMixin):
    """Immutable identity of source bytes and their logical document version."""

    document_id: str
    document_version_id: str
    content_sha256: str
    mime_type: str
    source_uri: str
    display_name: str | None = None
    page_count: int | None = None
    metadata: JsonObject = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.document_version_id, "document_version_id")
        require_sha256(self.content_sha256, "content_sha256")
        require_non_empty(self.mime_type, "mime_type")
        require_non_empty(self.source_uri, "source_uri")
        if self.display_name is not None:
            require_non_empty(self.display_name, "display_name")
        if self.page_count is not None and (
            isinstance(self.page_count, bool)
            or not isinstance(self.page_count, int)
            or self.page_count < 0
        ):
            raise ContractValidationError("page_count must be a non-negative integer")
        object.__setattr__(self, "metadata", freeze_json(self.metadata, field="metadata"))

    @classmethod
    def identified(
        cls,
        *,
        content_sha256: str,
        mime_type: str,
        source_uri: str,
        document_id: str | None = None,
        display_name: str | None = None,
        page_count: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Self:
        """Build stable default IDs while allowing a repository-owned document ID."""

        require_sha256(content_sha256, "content_sha256")
        logical_id = document_id or stable_id("doc", source_uri)
        version_id = stable_id("docv", logical_id, content_sha256)
        return cls(
            document_id=logical_id,
            document_version_id=version_id,
            content_sha256=content_sha256,
            mime_type=mime_type,
            source_uri=source_uri,
            display_name=display_name,
            page_count=page_count,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "content_sha256": self.content_sha256,
            "mime_type": self.mime_type,
            "source_uri": self.source_uri,
            "display_name": self.display_name,
            "page_count": self.page_count,
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
                "document_id",
                "document_version_id",
                "content_sha256",
                "mime_type",
                "source_uri",
                "display_name",
                "page_count",
                "metadata",
            },
            dto="DocumentRef",
        )
        return cls(
            document_id=payload.get("document_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            content_sha256=payload.get("content_sha256"),  # type: ignore[arg-type]
            mime_type=payload.get("mime_type"),  # type: ignore[arg-type]
            source_uri=payload.get("source_uri"),  # type: ignore[arg-type]
            display_name=payload.get("display_name"),  # type: ignore[arg-type]
            page_count=payload.get("page_count"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "metadata"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class PageRef(JsonDTOMixin):
    """A zero-based page identity with optional authoritative render geometry."""

    document_version_id: str
    page_index: int
    width: float | None = None
    height: float | None = None
    rotation_degrees: int = 0
    image_uri: str | None = None
    image_sha256: str | None = None
    native_stats: JsonObject = field(default_factory=dict)
    page_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.document_version_id, "document_version_id")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise ContractValidationError("page_index must be a non-negative integer")
        if (self.width is None) != (self.height is None):
            raise ContractValidationError("page width and height must be supplied together")
        if self.width is not None and (
            isinstance(self.width, bool)
            or not isinstance(self.width, (int, float))
            or not math.isfinite(float(self.width))
            or float(self.width) <= 0
        ):
            raise ContractValidationError("page width must be finite and positive")
        if self.height is not None and (
            isinstance(self.height, bool)
            or not isinstance(self.height, (int, float))
            or not math.isfinite(float(self.height))
            or float(self.height) <= 0
        ):
            raise ContractValidationError("page height must be finite and positive")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ContractValidationError("rotation_degrees must be 0, 90, 180, or 270")
        if self.image_uri is not None:
            require_non_empty(self.image_uri, "image_uri")
        if self.image_sha256 is not None:
            require_sha256(self.image_sha256, "image_sha256")
        object.__setattr__(
            self, "native_stats", freeze_json(self.native_stats, field="native_stats")
        )
        object.__setattr__(
            self, "page_id", stable_id("page", self.document_version_id, self.page_index)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "page_id": self.page_id,
            "document_version_id": self.document_version_id,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "rotation_degrees": self.rotation_degrees,
            "image_uri": self.image_uri,
            "image_sha256": self.image_sha256,
            "native_stats": json_ready(self.native_stats),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "page_id",
                "document_version_id",
                "page_index",
                "width",
                "height",
                "rotation_degrees",
                "image_uri",
                "image_sha256",
                "native_stats",
            },
            dto="PageRef",
        )
        result = cls(
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            width=payload.get("width"),  # type: ignore[arg-type]
            height=payload.get("height"),  # type: ignore[arg-type]
            rotation_degrees=payload.get("rotation_degrees", 0),  # type: ignore[arg-type]
            image_uri=payload.get("image_uri"),  # type: ignore[arg-type]
            image_sha256=payload.get("image_sha256"),  # type: ignore[arg-type]
            native_stats=require_mapping(payload.get("native_stats", {}), "native_stats"),
        )
        if payload.get("page_id") != result.page_id:
            raise ContractValidationError("PageRef page_id does not match its stable identity")
        return result


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (self.x, self.y)
        ):
            raise ContractValidationError("point coordinates must be finite numbers")

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        reject_unknown_fields(payload, {"x", "y"}, dto="Point")
        return cls(x=payload.get("x"), y=payload.get("y"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ContractValidationError("bbox coordinates must be finite numbers")
        if float(self.x1) < float(self.x0) or float(self.y1) < float(self.y0):
            raise ContractValidationError("bbox must satisfy x0 <= x1 and y0 <= y1")

    def to_list(self) -> list[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]

    @classmethod
    def from_sequence(cls, payload: Sequence[object]) -> Self:
        if len(payload) != 4:
            raise ContractValidationError("bbox must contain exactly four coordinates")
        return cls(*payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SourceGeometry(JsonDTOMixin):
    """Block geometry bound to an explicit coordinate space and page."""

    coordinate_space: CoordinateSpace
    bbox: BoundingBox | None = None
    polygon: tuple[Point, ...] = ()
    page_width: float | None = None
    page_height: float | None = None
    coordinate_origin: CoordinateOrigin = CoordinateOrigin.TOP_LEFT
    rotation_degrees: int = 0

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinate_space",
            _enum(CoordinateSpace, self.coordinate_space, "coordinate_space"),
        )
        object.__setattr__(
            self,
            "coordinate_origin",
            _enum(CoordinateOrigin, self.coordinate_origin, "coordinate_origin"),
        )
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ContractValidationError(
                "geometry rotation_degrees must be 0, 90, 180, or 270"
            )
        object.__setattr__(self, "polygon", tuple(self.polygon))
        if self.bbox is None and not self.polygon:
            raise ContractValidationError("geometry requires a bbox or polygon")
        if self.polygon and len(self.polygon) < 3:
            raise ContractValidationError("polygon requires at least three points")
        if (self.page_width is None) != (self.page_height is None):
            raise ContractValidationError("geometry page width and height must be supplied together")
        if self.page_width is not None and (
            isinstance(self.page_width, bool)
            or not isinstance(self.page_width, (int, float))
            or not math.isfinite(float(self.page_width))
            or float(self.page_width) <= 0
        ):
            raise ContractValidationError("geometry page width must be positive")
        if self.page_height is not None and (
            isinstance(self.page_height, bool)
            or not isinstance(self.page_height, (int, float))
            or not math.isfinite(float(self.page_height))
            or float(self.page_height) <= 0
        ):
            raise ContractValidationError("geometry page height must be positive")

        coordinates: list[tuple[float, float]] = []
        if self.bbox is not None:
            coordinates.extend(
                [
                    (float(self.bbox.x0), float(self.bbox.y0)),
                    (float(self.bbox.x1), float(self.bbox.y1)),
                ]
            )
        coordinates.extend((float(point.x), float(point.y)) for point in self.polygon)
        if self.coordinate_space == CoordinateSpace.NORMALIZED and any(
            not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in coordinates
        ):
            raise ContractValidationError("normalized geometry coordinates must be in [0, 1]")
        if self.coordinate_space != CoordinateSpace.NORMALIZED and any(
            x < 0 or y < 0 for x, y in coordinates
        ):
            raise ContractValidationError("page geometry coordinates must be non-negative")
        if self.page_width is not None and any(
            x < 0
            or y < 0
            or x > float(self.page_width)
            or y > float(self.page_height)
            for x, y in coordinates
        ):
            raise ContractValidationError("geometry falls outside declared page dimensions")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "coordinate_space": self.coordinate_space.value,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "polygon": [point.to_dict() for point in self.polygon],
            "page_width": self.page_width,
            "page_height": self.page_height,
            "coordinate_origin": self.coordinate_origin.value,
            "rotation_degrees": self.rotation_degrees,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "coordinate_space",
                "bbox",
                "polygon",
                "page_width",
                "page_height",
                "coordinate_origin",
                "rotation_degrees",
            },
            dto="SourceGeometry",
        )
        bbox_payload = payload.get("bbox")
        polygon_payload = require_sequence(payload.get("polygon", []), "polygon")
        return cls(
            coordinate_space=_enum(
                CoordinateSpace, payload.get("coordinate_space"), "coordinate_space"
            ),
            bbox=(
                BoundingBox.from_sequence(require_sequence(bbox_payload, "bbox"))
                if bbox_payload is not None
                else None
            ),
            polygon=tuple(
                Point.from_dict(require_mapping(point, f"polygon[{index}]"))
                for index, point in enumerate(polygon_payload)
            ),
            page_width=payload.get("page_width"),  # type: ignore[arg-type]
            page_height=payload.get("page_height"),  # type: ignore[arg-type]
            coordinate_origin=_enum(
                CoordinateOrigin,
                payload.get("coordinate_origin", CoordinateOrigin.TOP_LEFT.value),
                "coordinate_origin",
            ),
            rotation_degrees=payload.get("rotation_degrees", 0),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SemanticSourceSelector(JsonDTOMixin):
    """Exact, bounded selector into an observation's source material.

    A selector can identify representation character offsets (zero-based,
    end-exclusive Unicode code-point indices), a canonical
    asset, a parser-native JSON/path selector, a spatial region, or a bounded
    combination of those coordinates.  Ownership by the enclosing parser
    observation supplies the observation identity; canonical reconciliation
    adds that identity explicitly to :class:`SemanticSubstructure` lineage.
    """

    representation_id: str | None = None
    asset_id: str | None = None
    native_selector: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    geometry: SourceGeometry | None = None
    metadata: JsonObject = field(default_factory=dict)
    selector_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.representation_id is not None:
            _require_contract_id(
                self.representation_id,
                "semantic_source.representation_id",
                _REPRESENTATION_ID,
            )
        if self.asset_id is not None:
            _require_contract_id(
                self.asset_id, "semantic_source.asset_id", _ASSET_ID
            )
        if self.native_selector is not None:
            _bounded_semantic_string(
                self.native_selector, "semantic_source.native_selector"
            )
        if (self.char_start is None) != (self.char_end is None):
            raise ContractValidationError(
                "semantic source char_start and char_end must be supplied together"
            )
        if self.char_start is not None:
            if self.representation_id is None:
                raise ContractValidationError(
                    "semantic source character offsets require representation_id"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (self.char_start, self.char_end)
            ):
                raise ContractValidationError(
                    "semantic source character offsets must be non-negative integers"
                )
            assert self.char_end is not None
            if self.char_end <= self.char_start:
                raise ContractValidationError(
                    "semantic source char_end must be greater than char_start"
                )
        if self.geometry is not None and not isinstance(
            self.geometry, SourceGeometry
        ):
            raise ContractValidationError(
                "semantic source geometry must be SourceGeometry"
            )
        if (
            self.representation_id is None
            and self.asset_id is None
            and self.native_selector is None
            and self.geometry is None
        ):
            raise ContractValidationError(
                "semantic source requires a representation, asset, native selector, "
                "or geometry"
            )
        metadata = _freeze_bounded_semantic_json(
            self.metadata, field_name="semantic_source.metadata"
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "selector_id",
            stable_id(
                "ssel",
                self.representation_id,
                self.asset_id,
                self.native_selector,
                self.char_start,
                self.char_end,
                None if self.geometry is None else self.geometry.to_dict(),
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "selector_id": self.selector_id,
            "representation_id": self.representation_id,
            "asset_id": self.asset_id,
            "native_selector": self.native_selector,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "selector_id",
                "representation_id",
                "asset_id",
                "native_selector",
                "char_start",
                "char_end",
                "geometry",
                "metadata",
            },
            dto="SemanticSourceSelector",
        )
        geometry_payload = payload.get("geometry")
        result = cls(
            representation_id=payload.get("representation_id"),  # type: ignore[arg-type]
            asset_id=payload.get("asset_id"),  # type: ignore[arg-type]
            native_selector=payload.get("native_selector"),  # type: ignore[arg-type]
            char_start=payload.get("char_start"),  # type: ignore[arg-type]
            char_end=payload.get("char_end"),  # type: ignore[arg-type]
            geometry=(
                None
                if geometry_payload is None
                else SourceGeometry.from_dict(
                    require_mapping(geometry_payload, "semantic_source.geometry")
                )
            ),
            metadata=require_mapping(
                payload.get("metadata", {}), "semantic_source.metadata"
            ),
        )
        if payload.get("selector_id") != result.selector_id:
            raise ContractValidationError(
                "SemanticSourceSelector selector_id mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _parse_bounded_semantic_json(
                payload, field_name="SemanticSourceSelector"
            )
        )


@dataclass(frozen=True, slots=True)
class SemanticElement(JsonDTOMixin):
    """A typed table cell, formula symbol, visual node, or layout region."""

    local_key: str
    kind: SemanticElementKind
    source_selectors: tuple[SemanticSourceSelector, ...]
    content: JsonValue = None
    role: str | None = None
    geometry: SourceGeometry | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_span: int | None = None
    column_span: int | None = None
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    element_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _bounded_semantic_string(self.local_key, "semantic_element.local_key")
        kind = _enum(SemanticElementKind, self.kind, "semantic_element.kind")
        if self.role is not None:
            _bounded_semantic_string(self.role, "semantic_element.role")
        selectors = tuple(self.source_selectors)
        if not selectors:
            raise ContractValidationError(
                "semantic element requires at least one exact source selector"
            )
        if len(selectors) > MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT:
            raise ContractValidationError(
                "semantic element source selector limit exceeded"
            )
        if any(not isinstance(item, SemanticSourceSelector) for item in selectors):
            raise ContractValidationError(
                "source_selectors must contain SemanticSourceSelector DTOs"
            )
        selector_ids = [item.selector_id for item in selectors]
        if len(selector_ids) != len(set(selector_ids)):
            raise ContractValidationError(
                "semantic element contains duplicate source selectors"
            )
        selectors = tuple(sorted(selectors, key=lambda item: item.selector_id))
        if self.geometry is not None and not isinstance(
            self.geometry, SourceGeometry
        ):
            raise ContractValidationError(
                "semantic element geometry must be SourceGeometry"
            )
        table_coordinates = (
            self.row_index,
            self.column_index,
            self.row_span,
            self.column_span,
        )
        if kind is SemanticElementKind.TABLE_CELL:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if index < 2 else 1)
                for index, value in enumerate(table_coordinates)
            ):
                raise ContractValidationError(
                    "table cells require non-negative row/column indices and "
                    "positive row/column spans"
                )
        elif any(value is not None for value in table_coordinates):
            raise ContractValidationError(
                "table coordinates and spans are valid only for table_cell elements"
            )
        require_probability(self.confidence, "semantic_element.confidence")
        content = _freeze_bounded_semantic_json(
            self.content, field_name="semantic_element.content"
        )
        metadata = _freeze_bounded_semantic_json(
            self.metadata, field_name="semantic_element.metadata"
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_selectors", selectors)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "element_id",
            stable_id(
                "selem",
                self.local_key,
                kind.value,
                [item.to_dict() for item in selectors],
                content,
                self.role,
                None if self.geometry is None else self.geometry.to_dict(),
                self.row_index,
                self.column_index,
                self.row_span,
                self.column_span,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "element_id": self.element_id,
            "local_key": self.local_key,
            "kind": self.kind.value,
            "source_selectors": [item.to_dict() for item in self.source_selectors],
            "content": json_ready(self.content),
            "role": self.role,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "row_index": self.row_index,
            "column_index": self.column_index,
            "row_span": self.row_span,
            "column_span": self.column_span,
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "element_id",
                "local_key",
                "kind",
                "source_selectors",
                "content",
                "role",
                "geometry",
                "row_index",
                "column_index",
                "row_span",
                "column_span",
                "confidence",
                "metadata",
            },
            dto="SemanticElement",
        )
        raw_selectors = _bounded_semantic_sequence(
            payload.get("source_selectors", []),
            "semantic_element.source_selectors",
            MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT,
        )
        geometry_payload = payload.get("geometry")
        result = cls(
            local_key=payload.get("local_key"),  # type: ignore[arg-type]
            kind=_enum(
                SemanticElementKind,
                payload.get("kind"),
                "semantic_element.kind",
            ),
            source_selectors=tuple(
                SemanticSourceSelector.from_dict(
                    require_mapping(item, f"source_selectors[{index}]")
                )
                for index, item in enumerate(raw_selectors)
            ),
            content=payload.get("content"),  # type: ignore[arg-type]
            role=payload.get("role"),  # type: ignore[arg-type]
            geometry=(
                None
                if geometry_payload is None
                else SourceGeometry.from_dict(
                    require_mapping(geometry_payload, "semantic_element.geometry")
                )
            ),
            row_index=payload.get("row_index"),  # type: ignore[arg-type]
            column_index=payload.get("column_index"),  # type: ignore[arg-type]
            row_span=payload.get("row_span"),  # type: ignore[arg-type]
            column_span=payload.get("column_span"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(
                payload.get("metadata", {}), "semantic_element.metadata"
            ),
        )
        if payload.get("element_id") != result.element_id:
            raise ContractValidationError("SemanticElement element_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _parse_bounded_semantic_json(payload, field_name="SemanticElement")
        )


@dataclass(frozen=True, slots=True)
class SemanticRelation(JsonDTOMixin):
    """A typed internal edge with selectors proving its source assertion."""

    kind: SemanticRelationKind
    source_element_id: str
    target_element_id: str
    source_selector_ids: tuple[str, ...]
    label: str | None = None
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    relation_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind = _enum(SemanticRelationKind, self.kind, "semantic_relation.kind")
        for value, field_name in (
            (self.source_element_id, "semantic_relation.source_element_id"),
            (self.target_element_id, "semantic_relation.target_element_id"),
        ):
            _require_contract_id(value, field_name, _SEMANTIC_ELEMENT_ID)
        if self.source_element_id == self.target_element_id:
            raise ContractValidationError(
                "semantic relation cannot target its own source element"
            )
        raw_selector_ids = tuple(self.source_selector_ids)
        if not raw_selector_ids:
            raise ContractValidationError(
                "semantic relation requires exact source selector IDs"
            )
        if len(raw_selector_ids) > MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT * 2:
            raise ContractValidationError(
                "semantic relation source selector limit exceeded"
            )
        for selector_id in raw_selector_ids:
            _require_contract_id(
                selector_id,
                "semantic_relation.source_selector_ids[]",
                _SEMANTIC_SELECTOR_ID,
            )
        selector_ids = tuple(sorted(set(raw_selector_ids)))
        if self.label is not None:
            _bounded_semantic_string(self.label, "semantic_relation.label")
        require_probability(self.confidence, "semantic_relation.confidence")
        metadata = _freeze_bounded_semantic_json(
            self.metadata, field_name="semantic_relation.metadata"
        )
        normalized_edge_type = metadata.get(SEMANTIC_NORMALIZED_EDGE_TYPE_KEY)
        if normalized_edge_type is not None:
            canonical_edge_type = normalize_semantic_edge_type(
                normalized_edge_type
            )
            if normalized_edge_type != canonical_edge_type:
                raise ContractValidationError(
                    "semantic relation normalized_edge_type is not canonical"
                )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_selector_ids", selector_ids)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "relation_id",
            stable_id(
                "srel",
                kind.value,
                self.source_element_id,
                self.target_element_id,
                selector_ids,
                self.label,
                self.confidence,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "kind": self.kind.value,
            "source_element_id": self.source_element_id,
            "target_element_id": self.target_element_id,
            "source_selector_ids": list(self.source_selector_ids),
            "label": self.label,
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "relation_id",
                "kind",
                "source_element_id",
                "target_element_id",
                "source_selector_ids",
                "label",
                "confidence",
                "metadata",
            },
            dto="SemanticRelation",
        )
        raw_selector_ids = _bounded_semantic_sequence(
            payload.get("source_selector_ids", []),
            "semantic_relation.source_selector_ids",
            MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT * 2,
        )
        result = cls(
            kind=_enum(
                SemanticRelationKind,
                payload.get("kind"),
                "semantic_relation.kind",
            ),
            source_element_id=payload.get("source_element_id"),  # type: ignore[arg-type]
            target_element_id=payload.get("target_element_id"),  # type: ignore[arg-type]
            source_selector_ids=tuple(raw_selector_ids),  # type: ignore[arg-type]
            label=payload.get("label"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(
                payload.get("metadata", {}), "semantic_relation.metadata"
            ),
        )
        if payload.get("relation_id") != result.relation_id:
            raise ContractValidationError("SemanticRelation relation_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _parse_bounded_semantic_json(payload, field_name="SemanticRelation")
        )


_STRUCTURE_ELEMENT_KINDS: Mapping[
    SemanticStructureKind, frozenset[SemanticElementKind]
] = {
    SemanticStructureKind.TABLE: frozenset({SemanticElementKind.TABLE_CELL}),
    SemanticStructureKind.FORMULA: frozenset({SemanticElementKind.FORMULA_SYMBOL}),
    SemanticStructureKind.DIAGRAM: frozenset({SemanticElementKind.DIAGRAM_ELEMENT}),
    SemanticStructureKind.FLOWCHART: frozenset(
        {SemanticElementKind.FLOWCHART_ELEMENT}
    ),
    SemanticStructureKind.LAYOUT: frozenset({SemanticElementKind.LAYOUT_REGION}),
}

_SPATIAL_RELATION_KINDS = frozenset(
    {
        SemanticRelationKind.SPATIAL_LEFT_OF,
        SemanticRelationKind.SPATIAL_RIGHT_OF,
        SemanticRelationKind.SPATIAL_ABOVE,
        SemanticRelationKind.SPATIAL_BELOW,
        SemanticRelationKind.SPATIAL_OVERLAPS,
        SemanticRelationKind.SPATIAL_CONTAINS,
        SemanticRelationKind.READING_NEXT,
        SemanticRelationKind.SEMANTIC_ASSOCIATED_WITH,
    }
)

_STRUCTURE_RELATION_KINDS: Mapping[
    SemanticStructureKind, frozenset[SemanticRelationKind]
] = {
    SemanticStructureKind.TABLE: frozenset(
        {
            SemanticRelationKind.TABLE_ADJACENT_RIGHT,
            SemanticRelationKind.TABLE_ADJACENT_BELOW,
        }
    )
    | _SPATIAL_RELATION_KINDS,
    SemanticStructureKind.FORMULA: frozenset(
        {
            SemanticRelationKind.FORMULA_ARGUMENT_OF,
            SemanticRelationKind.FORMULA_OPERAND_OF,
            SemanticRelationKind.FORMULA_SUPERSCRIPT_OF,
            SemanticRelationKind.FORMULA_SUBSCRIPT_OF,
            SemanticRelationKind.FORMULA_NEXT,
        }
    )
    | _SPATIAL_RELATION_KINDS,
    SemanticStructureKind.DIAGRAM: frozenset(
        {
            SemanticRelationKind.DIAGRAM_CONNECTS_TO,
            SemanticRelationKind.DIAGRAM_CONTAINS,
        }
    )
    | _SPATIAL_RELATION_KINDS,
    SemanticStructureKind.FLOWCHART: frozenset(
        {
            SemanticRelationKind.FLOW_NEXT,
            SemanticRelationKind.FLOW_BRANCH_TRUE,
            SemanticRelationKind.FLOW_BRANCH_FALSE,
            SemanticRelationKind.DIAGRAM_CONTAINS,
        }
    )
    | _SPATIAL_RELATION_KINDS,
    SemanticStructureKind.LAYOUT: _SPATIAL_RELATION_KINDS,
}


def _semantic_element_geometry(element: SemanticElement) -> SourceGeometry | None:
    if element.geometry is not None:
        return element.geometry
    candidates: dict[str, SourceGeometry] = {}
    for selector in element.source_selectors:
        if selector.geometry is not None:
            candidates.setdefault(
                content_sha256(selector.geometry.to_dict()), selector.geometry
            )
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _normalized_semantic_geometry_bbox(
    geometry: SourceGeometry,
) -> tuple[float, float, float, float]:
    if geometry.bbox is not None:
        x0, y0, x1, y1 = (
            float(geometry.bbox.x0),
            float(geometry.bbox.y0),
            float(geometry.bbox.x1),
            float(geometry.bbox.y1),
        )
    else:
        xs = [float(point.x) for point in geometry.polygon]
        ys = [float(point.y) for point in geometry.polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if geometry.coordinate_space is not CoordinateSpace.NORMALIZED:
        if geometry.page_width is None or geometry.page_height is None:
            raise ContractValidationError(
                "spatial relation page coordinates require declared page dimensions"
            )
        x0 /= float(geometry.page_width)
        x1 /= float(geometry.page_width)
        y0 /= float(geometry.page_height)
        y1 /= float(geometry.page_height)
    if geometry.coordinate_origin is CoordinateOrigin.BOTTOM_LEFT:
        y0, y1 = 1.0 - y1, 1.0 - y0
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if geometry.rotation_degrees == 90:
        corners = tuple((1.0 - y, x) for x, y in corners)
    elif geometry.rotation_degrees == 180:
        corners = tuple((1.0 - x, 1.0 - y) for x, y in corners)
    elif geometry.rotation_degrees == 270:
        corners = tuple((y, 1.0 - x) for x, y in corners)
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _validate_spatial_semantic_relation(
    kind: SemanticRelationKind,
    source: SemanticElement,
    target: SemanticElement,
) -> None:
    source_geometry = _semantic_element_geometry(source)
    target_geometry = _semantic_element_geometry(target)
    if source_geometry is None or target_geometry is None:
        raise ContractValidationError(
            f"{kind.value} requires unambiguous source and target geometry"
        )
    sx0, sy0, sx1, sy1 = _normalized_semantic_geometry_bbox(source_geometry)
    tx0, ty0, tx1, ty1 = _normalized_semantic_geometry_bbox(target_geometry)
    source_x = (sx0 + sx1) / 2.0
    source_y = (sy0 + sy1) / 2.0
    target_x = (tx0 + tx1) / 2.0
    target_y = (ty0 + ty1) / 2.0
    valid = {
        SemanticRelationKind.SPATIAL_LEFT_OF: source_x < target_x,
        SemanticRelationKind.SPATIAL_RIGHT_OF: source_x > target_x,
        SemanticRelationKind.SPATIAL_ABOVE: source_y < target_y,
        SemanticRelationKind.SPATIAL_BELOW: source_y > target_y,
        SemanticRelationKind.SPATIAL_OVERLAPS: (
            max(sx0, tx0) < min(sx1, tx1)
            and max(sy0, ty0) < min(sy1, ty1)
        ),
        SemanticRelationKind.SPATIAL_CONTAINS: (
            sx0 <= tx0 and sy0 <= ty0 and sx1 >= tx1 and sy1 >= ty1
        ),
    }[kind]
    if not valid:
        raise ContractValidationError(
            f"{kind.value} contradicts its endpoint geometry"
        )


_SEMANTIC_TABLE_ROW_DIMENSION_KEYS = ("row_count", "num_rows")
_SEMANTIC_TABLE_COLUMN_DIMENSION_KEYS = (
    "column_count",
    "num_cols",
)


def _semantic_comparison_dimension(
    metadata: Mapping[str, JsonValue],
    keys: Sequence[str],
    *,
    implicit_extent: int,
) -> JsonValue:
    """Normalize only recognized table dimensions, conservatively.

    A missing declaration is equivalent to the occupied extent.  Consistent
    aliases are equivalent to one another.  Invalid or contradictory values
    remain distinguishable without leaking vendor field names into the
    structural comparison.
    """

    values = [metadata[key] for key in keys if key in metadata]
    if not values:
        return implicit_extent
    if all(
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= implicit_extent
        for value in values
    ):
        unique = sorted(set(values))
        if len(unique) == 1:
            return unique[0]
        return {"contradictory_assertions": unique}
    return {
        "invalid_assertion_sha256": sorted(
            content_sha256(json_ready(value)) for value in values
        )
    }


_SEMANTIC_GRAPH_NODE_IDENTITY_FIELDS = frozenset({"id", "node_id", "key"})
_SEMANTIC_GRAPH_NODE_CONFIDENCE_FIELDS = frozenset({"confidence", "score"})
_SEMANTIC_GRAPH_NODE_ROLE_FIELDS = frozenset({"type", "kind", "role"})
_SEMANTIC_GRAPH_NODE_LABEL_FIELDS = ("label", "text", "name", "title")


def _semantic_graph_node_comparison_content(content: JsonValue) -> object:
    """Remove only recognized source-assertion fields from a graph node.

    Parser-native IDs, confidence and free-form metadata identify an exact
    assertion, not the labelled node itself.  Label and role aliases are
    projected to one vocabulary; every unrecognized field remains in the
    comparison so semantic details are never silently discarded.
    """

    if not isinstance(content, Mapping):
        return json_ready(content)
    label_values = [
        content[key]
        for key in _SEMANTIC_GRAPH_NODE_LABEL_FIELDS
        if key in content
    ]
    normalized: dict[str, object] = {
        key: json_ready(value)
        for key, value in content.items()
        if key not in _SEMANTIC_GRAPH_NODE_IDENTITY_FIELDS
        and key not in _SEMANTIC_GRAPH_NODE_CONFIDENCE_FIELDS
        and key not in _SEMANTIC_GRAPH_NODE_ROLE_FIELDS
        and key not in _SEMANTIC_GRAPH_NODE_LABEL_FIELDS
        and key != "metadata"
    }
    if label_values:
        unique_labels = {
            content_sha256(json_ready(value)): json_ready(value)
            for value in label_values
        }
        if len(unique_labels) == 1:
            normalized["label"] = next(iter(unique_labels.values()))
        else:
            normalized["contradictory_labels"] = [
                unique_labels[digest] for digest in sorted(unique_labels)
            ]
    return normalized


def _semantic_element_comparison_payload(
    element: SemanticElement,
    *,
    structure_kind: SemanticStructureKind,
) -> dict[str, object]:
    """Return semantic element truth without source-assertion coordinates."""

    semantic_geometry = _semantic_element_geometry(element)
    return {
        "kind": element.kind.value,
        "content": (
            _semantic_graph_node_comparison_content(element.content)
            if structure_kind
            in {SemanticStructureKind.DIAGRAM, SemanticStructureKind.FLOWCHART}
            else json_ready(element.content)
        ),
        "role": element.role,
        # Geometry is semantic when an adapter asserts it on the element.  It
        # remains exact here: silently rounding or reducing polygons to boxes
        # could merge genuinely different spatial assertions.
        "geometry": (
            None if semantic_geometry is None else semantic_geometry.to_dict()
        ),
        "row_index": element.row_index,
        "column_index": element.column_index,
        "row_span": element.row_span,
        "column_span": element.column_span,
    }


def _semantic_structure_comparison_payload(
    *,
    kind: SemanticStructureKind,
    availability: SemanticStructureAvailability,
    elements: Sequence[SemanticElement],
    relations: Sequence[SemanticRelation],
    metadata: Mapping[str, JsonValue],
) -> dict[str, object]:
    """Build a bounded, source-neutral but conservative graph fingerprint.

    Selectors, representation/asset IDs, native paths, confidence, producer
    identity, and free-form parser metadata are assertion provenance rather
    than semantic structure.  They remain in ``semantic_sha256`` and the full
    DTO, but are excluded here.  Unique elements are matched by their typed
    content/geometry/coordinates.  Structurally indistinguishable duplicate
    elements use hashed local keys as a conservative tie-breaker, preventing
    false equivalence when graph isomorphism would otherwise be ambiguous.
    """

    records: list[tuple[SemanticElement, dict[str, object], str]] = []
    counts: dict[str, int] = {}
    for element in elements:
        payload = _semantic_element_comparison_payload(
            element, structure_kind=kind
        )
        digest = content_sha256(payload)
        records.append((element, payload, digest))
        counts[digest] = counts.get(digest, 0) + 1

    comparison_key_by_element_id: dict[str, str] = {}
    comparison_elements: list[dict[str, object]] = []
    for element, payload, digest in records:
        comparison_key = digest
        record: dict[str, object] = {
            "comparison_key": comparison_key,
            **payload,
        }
        if counts[digest] > 1:
            local_key_sha256 = content_sha256(element.local_key)
            comparison_key = f"{digest}:{local_key_sha256}"
            record["comparison_key"] = comparison_key
            record["ambiguous_local_key_sha256"] = local_key_sha256
        comparison_key_by_element_id[element.element_id] = comparison_key
        comparison_elements.append(record)
    comparison_elements.sort(key=canonical_json_bytes)

    comparison_relations = []
    for relation in relations:
        comparison_relation = {
            "kind": relation.kind.value,
            "source_comparison_key": comparison_key_by_element_id[
                relation.source_element_id
            ],
            "target_comparison_key": comparison_key_by_element_id[
                relation.target_element_id
            ],
            "label": relation.label,
        }
        if kind in {
            SemanticStructureKind.DIAGRAM,
            SemanticStructureKind.FLOWCHART,
        }:
            comparison_relation["normalized_edge_type"] = relation.metadata.get(
                SEMANTIC_NORMALIZED_EDGE_TYPE_KEY
            )
        comparison_relations.append(comparison_relation)
    comparison_relations.sort(key=canonical_json_bytes)

    semantic_metadata: dict[str, object] = {}
    if kind is SemanticStructureKind.TABLE:
        row_extent = max(
            int(element.row_index or 0) + int(element.row_span or 0)
            for element in elements
        )
        column_extent = max(
            int(element.column_index or 0) + int(element.column_span or 0)
            for element in elements
        )
        semantic_metadata["table_dimensions"] = {
            "rows": _semantic_comparison_dimension(
                metadata,
                _SEMANTIC_TABLE_ROW_DIMENSION_KEYS,
                implicit_extent=row_extent,
            ),
            "columns": _semantic_comparison_dimension(
                metadata,
                _SEMANTIC_TABLE_COLUMN_DIMENSION_KEYS,
                implicit_extent=column_extent,
            ),
        }
    return {
        "kind": kind.value,
        "availability": availability.value,
        "elements": comparison_elements,
        "relations": comparison_relations,
        "semantic_metadata": semantic_metadata,
    }


@dataclass(frozen=True, slots=True)
class SemanticSubstructure(JsonDTOMixin):
    """A bounded block-local semantic graph with lossless source lineage.

    Parser observations contain unbound structures (both source tuples empty),
    whose ownership is the enclosing observation.  Canonical blocks contain
    source-bound structures, produced by :meth:`with_source_lineage` during
    reconciliation.  ``semantic_sha256`` identifies the exact source
    assertion, while ``comparison_sha256`` conservatively compares structural
    truth without vendor selectors or metadata.  Reconciliation may represent
    equivalent exact assertions once, but their identities remain separate.
    """

    kind: SemanticStructureKind
    elements: tuple[SemanticElement, ...]
    relations: tuple[SemanticRelation, ...] = ()
    source_observation_ids: tuple[str, ...] = ()
    source_substructure_ids: tuple[str, ...] = ()
    producer_kind: SemanticProducerKind = SemanticProducerKind.PARSER_NATIVE
    availability: SemanticStructureAvailability = (
        SemanticStructureAvailability.COMPLETE
    )
    source_field_paths: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict)
    comparison_sha256: str = field(init=False)
    semantic_sha256: str = field(init=False)
    substructure_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind = _enum(SemanticStructureKind, self.kind, "semantic_structure.kind")
        producer_kind = _enum(
            SemanticProducerKind,
            self.producer_kind,
            "semantic_structure.producer_kind",
        )
        availability = _enum(
            SemanticStructureAvailability,
            self.availability,
            "semantic_structure.availability",
        )
        elements = tuple(self.elements)
        relations = tuple(self.relations)
        if not elements:
            raise ContractValidationError(
                "semantic substructure requires at least one element"
            )
        if len(elements) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
            raise ContractValidationError(
                "semantic substructure element limit exceeded"
            )
        if len(relations) > MAX_SEMANTIC_RELATIONS_PER_STRUCTURE:
            raise ContractValidationError(
                "semantic substructure relation limit exceeded"
            )
        if any(not isinstance(item, SemanticElement) for item in elements):
            raise ContractValidationError(
                "elements must contain SemanticElement DTOs"
            )
        if any(not isinstance(item, SemanticRelation) for item in relations):
            raise ContractValidationError(
                "relations must contain SemanticRelation DTOs"
            )
        if any(
            element.kind not in _STRUCTURE_ELEMENT_KINDS[kind]
            for element in elements
        ):
            raise ContractValidationError(
                f"{kind.value} substructure contains an incompatible element kind"
            )
        local_keys = [item.local_key for item in elements]
        element_ids = [item.element_id for item in elements]
        if len(local_keys) != len(set(local_keys)):
            raise ContractValidationError(
                "semantic substructure contains duplicate element local keys"
            )
        if len(element_ids) != len(set(element_ids)):
            raise ContractValidationError(
                "semantic substructure contains duplicate element IDs"
            )
        elements = tuple(sorted(elements, key=lambda item: item.element_id))
        element_by_id = {item.element_id: item for item in elements}
        if kind is SemanticStructureKind.TABLE:
            occupied: set[tuple[int, int]] = set()
            occupied_count = 0
            for element in elements:
                assert element.row_index is not None
                assert element.column_index is not None
                assert element.row_span is not None
                assert element.column_span is not None
                area = element.row_span * element.column_span
                occupied_count += area
                if occupied_count > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE:
                    raise ContractValidationError(
                        "semantic table expanded-position limit exceeded"
                    )
                for row in range(
                    element.row_index, element.row_index + element.row_span
                ):
                    for column in range(
                        element.column_index,
                        element.column_index + element.column_span,
                    ):
                        position = (row, column)
                        if position in occupied:
                            raise ContractValidationError(
                                "semantic table contains overlapping cell spans"
                            )
                        occupied.add(position)

        relation_ids: set[str] = set()
        for relation in relations:
            if relation.kind not in _STRUCTURE_RELATION_KINDS[kind]:
                raise ContractValidationError(
                    f"{relation.kind.value} is incompatible with {kind.value} structure"
                )
            if (
                SEMANTIC_NORMALIZED_EDGE_TYPE_KEY in relation.metadata
                and kind
                not in {
                    SemanticStructureKind.DIAGRAM,
                    SemanticStructureKind.FLOWCHART,
                }
            ):
                raise ContractValidationError(
                    "normalized_edge_type is valid only for diagram or flowchart relations"
                )
            if (
                relation.source_element_id not in element_by_id
                or relation.target_element_id not in element_by_id
            ):
                raise ContractValidationError(
                    "semantic relation endpoint is outside its substructure"
                )
            source_element = element_by_id[relation.source_element_id]
            target_element = element_by_id[relation.target_element_id]
            endpoint_selector_ids = {
                selector.selector_id
                for element in (source_element, target_element)
                for selector in element.source_selectors
            }
            if any(
                selector_id not in endpoint_selector_ids
                for selector_id in relation.source_selector_ids
            ):
                raise ContractValidationError(
                    "semantic relation source selector is outside its endpoints"
                )
            if relation.kind is SemanticRelationKind.TABLE_ADJACENT_RIGHT:
                assert source_element.column_index is not None
                assert source_element.column_span is not None
                assert source_element.row_index is not None
                assert source_element.row_span is not None
                assert target_element.column_index is not None
                assert target_element.row_index is not None
                assert target_element.row_span is not None
                source_row_end = source_element.row_index + source_element.row_span
                target_row_end = target_element.row_index + target_element.row_span
                if not (
                    source_element.column_index + source_element.column_span
                    == target_element.column_index
                    and max(source_element.row_index, target_element.row_index)
                    < min(source_row_end, target_row_end)
                ):
                    raise ContractValidationError(
                        "table_adjacent_right contradicts cell indices or spans"
                    )
            if relation.kind is SemanticRelationKind.TABLE_ADJACENT_BELOW:
                assert source_element.row_index is not None
                assert source_element.row_span is not None
                assert source_element.column_index is not None
                assert source_element.column_span is not None
                assert target_element.row_index is not None
                assert target_element.column_index is not None
                assert target_element.column_span is not None
                source_column_end = (
                    source_element.column_index + source_element.column_span
                )
                target_column_end = (
                    target_element.column_index + target_element.column_span
                )
                if not (
                    source_element.row_index + source_element.row_span
                    == target_element.row_index
                    and max(
                        source_element.column_index, target_element.column_index
                    )
                    < min(source_column_end, target_column_end)
                ):
                    raise ContractValidationError(
                        "table_adjacent_below contradicts cell indices or spans"
                    )
            if relation.kind in {
                SemanticRelationKind.SPATIAL_LEFT_OF,
                SemanticRelationKind.SPATIAL_RIGHT_OF,
                SemanticRelationKind.SPATIAL_ABOVE,
                SemanticRelationKind.SPATIAL_BELOW,
                SemanticRelationKind.SPATIAL_OVERLAPS,
                SemanticRelationKind.SPATIAL_CONTAINS,
            }:
                _validate_spatial_semantic_relation(
                    relation.kind, source_element, target_element
                )
            if relation.relation_id in relation_ids:
                raise ContractValidationError(
                    "semantic substructure contains duplicate relation IDs"
                )
            relation_ids.add(relation.relation_id)
        relations = tuple(sorted(relations, key=lambda item: item.relation_id))

        raw_source_field_paths = tuple(self.source_field_paths)
        if not raw_source_field_paths:
            raw_source_field_paths = tuple(
                selector.native_selector
                for element in elements
                for selector in element.source_selectors
                if selector.native_selector is not None
            )
        if not raw_source_field_paths:
            raise ContractValidationError(
                "semantic substructure requires source_field_paths or at least "
                "one native source selector"
            )
        if len(raw_source_field_paths) > MAX_SEMANTIC_SOURCE_IDS:
            raise ContractValidationError(
                "semantic substructure source_field_paths limit exceeded"
            )
        for path in raw_source_field_paths:
            _bounded_semantic_string(path, "semantic_structure.source_field_paths[]")
        source_field_paths = tuple(sorted(set(raw_source_field_paths)))

        raw_observation_ids = tuple(self.source_observation_ids)
        raw_substructure_ids = tuple(self.source_substructure_ids)
        if bool(raw_observation_ids) != bool(raw_substructure_ids):
            raise ContractValidationError(
                "semantic source observation and substructure lineage must be "
                "empty together or populated together"
            )
        if len(raw_observation_ids) > MAX_SEMANTIC_SOURCE_IDS or len(
            raw_substructure_ids
        ) > MAX_SEMANTIC_SOURCE_IDS:
            raise ContractValidationError("semantic source lineage limit exceeded")
        for source_id in raw_observation_ids:
            _require_contract_id(
                source_id, "source_observation_ids[]", _OBSERVATION_ID
            )
        for source_id in raw_substructure_ids:
            _require_contract_id(
                source_id,
                "source_substructure_ids[]",
                _SEMANTIC_SUBSTRUCTURE_ID,
            )
        source_observation_ids = tuple(sorted(set(raw_observation_ids)))
        source_substructure_ids = tuple(sorted(set(raw_substructure_ids)))
        if source_observation_ids and len(source_substructure_ids) != 1:
            raise ContractValidationError(
                "a source-bound semantic graph must reference its one exact raw "
                "substructure identity"
            )
        metadata = _freeze_bounded_semantic_json(
            self.metadata, field_name="semantic_structure.metadata"
        )
        semantic_payload = {
            "kind": kind.value,
            "producer_kind": producer_kind.value,
            "availability": availability.value,
            "source_field_paths": source_field_paths,
            "elements": [item.to_dict() for item in elements],
            "relations": [item.to_dict() for item in relations],
            "metadata": metadata,
        }
        canonical_size = len(canonical_json_bytes(semantic_payload))
        if canonical_size > MAX_SEMANTIC_SUBSTRUCTURE_CANONICAL_BYTES:
            raise ContractValidationError(
                "semantic substructure canonical byte limit exceeded"
            )
        comparison_payload = _semantic_structure_comparison_payload(
            kind=kind,
            availability=availability,
            elements=elements,
            relations=relations,
            metadata=metadata,
        )
        if (
            len(canonical_json_bytes(comparison_payload))
            > MAX_SEMANTIC_SUBSTRUCTURE_CANONICAL_BYTES
        ):
            raise ContractValidationError(
                "semantic substructure comparison byte limit exceeded"
            )
        comparison_sha = content_sha256(comparison_payload)
        semantic_sha = content_sha256(semantic_payload)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "producer_kind", producer_kind)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "source_field_paths", source_field_paths)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "source_observation_ids", source_observation_ids)
        object.__setattr__(self, "source_substructure_ids", source_substructure_ids)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "comparison_sha256", comparison_sha)
        object.__setattr__(self, "semantic_sha256", semantic_sha)
        object.__setattr__(
            self,
            "substructure_id",
            stable_id(
                "sstruct",
                semantic_sha,
                source_observation_ids,
                source_substructure_ids,
            ),
        )

    @property
    def is_source_bound(self) -> bool:
        return bool(self.source_observation_ids)

    def to_comparison_dict(self) -> dict[str, object]:
        """Return source-neutral structural truth used only for comparison.

        This payload is not source identity.  Exact selectors, metadata, raw
        graph identity, and lineage remain in :meth:`to_dict`,
        ``semantic_sha256``, and ``substructure_id``.
        """

        return _semantic_structure_comparison_payload(
            kind=self.kind,
            availability=self.availability,
            elements=self.elements,
            relations=self.relations,
            metadata=self.metadata,
        )

    def with_source_lineage(
        self,
        *,
        source_observation_ids: Sequence[str],
        source_substructure_ids: Sequence[str] | None = None,
    ) -> Self:
        """Return the same semantic graph bound to exact parser assertions."""

        if self.is_source_bound:
            raise ContractValidationError(
                "semantic substructure is already source-bound"
            )
        return type(self)(
            kind=self.kind,
            elements=self.elements,
            relations=self.relations,
            source_observation_ids=tuple(source_observation_ids),
            source_substructure_ids=tuple(
                source_substructure_ids
                if source_substructure_ids is not None
                else (self.substructure_id,)
            ),
            metadata=self.metadata,
            producer_kind=self.producer_kind,
            availability=self.availability,
            source_field_paths=self.source_field_paths,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "substructure_id": self.substructure_id,
            "comparison_sha256": self.comparison_sha256,
            "semantic_sha256": self.semantic_sha256,
            "kind": self.kind.value,
            "producer_kind": self.producer_kind.value,
            "availability": self.availability.value,
            "source_field_paths": list(self.source_field_paths),
            "elements": [item.to_dict() for item in self.elements],
            "relations": [item.to_dict() for item in self.relations],
            "source_observation_ids": list(self.source_observation_ids),
            "source_substructure_ids": list(self.source_substructure_ids),
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "substructure_id",
                "comparison_sha256",
                "semantic_sha256",
                "kind",
                "producer_kind",
                "availability",
                "source_field_paths",
                "elements",
                "relations",
                "source_observation_ids",
                "source_substructure_ids",
                "metadata",
            },
            dto="SemanticSubstructure",
        )
        raw_elements = _bounded_semantic_sequence(
            payload.get("elements", []),
            "semantic_structure.elements",
            MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE,
        )
        raw_relations = _bounded_semantic_sequence(
            payload.get("relations", []),
            "semantic_structure.relations",
            MAX_SEMANTIC_RELATIONS_PER_STRUCTURE,
        )
        raw_observation_ids = _bounded_semantic_sequence(
            payload.get("source_observation_ids", []),
            "semantic_structure.source_observation_ids",
            MAX_SEMANTIC_SOURCE_IDS,
        )
        raw_substructure_ids = _bounded_semantic_sequence(
            payload.get("source_substructure_ids", []),
            "semantic_structure.source_substructure_ids",
            MAX_SEMANTIC_SOURCE_IDS,
        )
        raw_source_field_paths = _bounded_semantic_sequence(
            payload.get("source_field_paths", []),
            "semantic_structure.source_field_paths",
            MAX_SEMANTIC_SOURCE_IDS,
        )
        result = cls(
            kind=_enum(
                SemanticStructureKind,
                payload.get("kind"),
                "semantic_structure.kind",
            ),
            elements=tuple(
                SemanticElement.from_dict(
                    require_mapping(item, f"semantic_structure.elements[{index}]")
                )
                for index, item in enumerate(raw_elements)
            ),
            relations=tuple(
                SemanticRelation.from_dict(
                    require_mapping(item, f"semantic_structure.relations[{index}]")
                )
                for index, item in enumerate(raw_relations)
            ),
            source_observation_ids=tuple(raw_observation_ids),  # type: ignore[arg-type]
            source_substructure_ids=tuple(raw_substructure_ids),  # type: ignore[arg-type]
            producer_kind=_enum(
                SemanticProducerKind,
                payload.get(
                    "producer_kind", SemanticProducerKind.PARSER_NATIVE.value
                ),
                "semantic_structure.producer_kind",
            ),
            availability=_enum(
                SemanticStructureAvailability,
                payload.get(
                    "availability", SemanticStructureAvailability.COMPLETE.value
                ),
                "semantic_structure.availability",
            ),
            source_field_paths=tuple(raw_source_field_paths),  # type: ignore[arg-type]
            metadata=require_mapping(
                payload.get("metadata", {}), "semantic_structure.metadata"
            ),
        )
        for field_name in ("semantic_sha256", "substructure_id"):
            if payload.get(field_name) != getattr(result, field_name):
                raise ContractValidationError(
                    f"SemanticSubstructure {field_name} mismatch"
                )
        if (
            payload.get("comparison_sha256") is not None
            and payload.get("comparison_sha256") != result.comparison_sha256
        ):
            raise ContractValidationError(
                "SemanticSubstructure comparison_sha256 mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(
            _parse_bounded_semantic_json(
                payload, field_name="SemanticSubstructure"
            )
        )


@dataclass(frozen=True, slots=True)
class Representation(JsonDTOMixin):
    kind: RepresentationKind
    content: JsonValue
    media_type: str
    language: str | None = None
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    content_sha256: str = field(init=False)
    representation_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind = _enum(RepresentationKind, self.kind, "representation.kind")
        require_non_empty(self.media_type, "representation.media_type")
        if self.language is not None:
            require_non_empty(self.language, "representation.language")
        require_probability(self.confidence, "representation.confidence")
        frozen_content = freeze_json(self.content, field="representation.content")
        if isinstance(frozen_content, str) and not frozen_content.strip():
            raise ContractValidationError("representation.content must not be empty")
        frozen_metadata = freeze_json(self.metadata, field="representation.metadata")
        digest = content_sha256(
            {
                "kind": kind.value,
                "content": frozen_content,
                "media_type": self.media_type,
                "language": self.language,
                "metadata": frozen_metadata,
            }
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "content", frozen_content)
        object.__setattr__(self, "metadata", frozen_metadata)
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(
            self,
            "representation_id",
            stable_id("repr", kind.value, self.media_type, self.language, digest),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "representation_id": self.representation_id,
            "kind": self.kind.value,
            "content": json_ready(self.content),
            "media_type": self.media_type,
            "language": self.language,
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "representation_id",
                "kind",
                "content",
                "media_type",
                "language",
                "confidence",
                "metadata",
                "content_sha256",
            },
            dto="Representation",
        )
        result = cls(
            kind=_enum(RepresentationKind, payload.get("kind"), "representation.kind"),
            content=payload.get("content"),  # type: ignore[arg-type]
            media_type=payload.get("media_type"),  # type: ignore[arg-type]
            language=payload.get("language"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "representation.metadata"),
        )
        if payload.get("content_sha256") != result.content_sha256:
            raise ContractValidationError("Representation content_sha256 mismatch")
        if payload.get("representation_id") != result.representation_id:
            raise ContractValidationError("Representation representation_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class AssetRef(JsonDTOMixin):
    kind: AssetKind
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    asset_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind = _enum(AssetKind, self.kind, "asset.kind")
        require_non_empty(self.uri, "asset.uri")
        if self.media_type is not None:
            require_non_empty(self.media_type, "asset.media_type")
        if self.sha256 is not None:
            require_sha256(self.sha256, "asset.sha256")
        metadata = freeze_json(self.metadata, field="asset.metadata")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "asset_id",
            stable_id("asset", kind.value, self.uri, self.sha256, self.media_type),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "asset_id": self.asset_id,
            "kind": self.kind.value,
            "uri": self.uri,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {"schema_version", "asset_id", "kind", "uri", "media_type", "sha256", "metadata"},
            dto="AssetRef",
        )
        result = cls(
            kind=_enum(AssetKind, payload.get("kind"), "asset.kind"),
            uri=payload.get("uri"),  # type: ignore[arg-type]
            media_type=payload.get("media_type"),  # type: ignore[arg-type]
            sha256=payload.get("sha256"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "asset.metadata"),
        )
        if payload.get("asset_id") != result.asset_id:
            raise ContractValidationError("AssetRef asset_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParserProvenance(JsonDTOMixin):
    parser_name: str
    parser_version: str
    model_version: str
    config_sha256: str
    parser_native_id: str
    raw_artifact_uri: str | None = None
    transformations: tuple[str, ...] = ()

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "parser_name",
            "parser_version",
            "model_version",
            "parser_native_id",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        require_sha256(self.config_sha256, "config_sha256")
        if self.raw_artifact_uri is not None:
            require_non_empty(self.raw_artifact_uri, "raw_artifact_uri")
        transformations = tuple(self.transformations)
        for transformation in transformations:
            require_non_empty(transformation, "transformations[]")
        object.__setattr__(self, "transformations", transformations)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "model_version": self.model_version,
            "config_sha256": self.config_sha256,
            "parser_native_id": self.parser_native_id,
            "raw_artifact_uri": self.raw_artifact_uri,
            "transformations": list(self.transformations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "parser_name",
                "parser_version",
                "model_version",
                "config_sha256",
                "parser_native_id",
                "raw_artifact_uri",
                "transformations",
            },
            dto="ParserProvenance",
        )
        return cls(
            parser_name=payload.get("parser_name"),  # type: ignore[arg-type]
            parser_version=payload.get("parser_version"),  # type: ignore[arg-type]
            model_version=payload.get("model_version"),  # type: ignore[arg-type]
            config_sha256=payload.get("config_sha256"),  # type: ignore[arg-type]
            parser_native_id=payload.get("parser_native_id"),  # type: ignore[arg-type]
            raw_artifact_uri=payload.get("raw_artifact_uri"),  # type: ignore[arg-type]
            transformations=tuple(
                require_sequence(payload.get("transformations", []), "transformations")
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ObservationRelation(JsonDTOMixin):
    relation_type: RelationType
    target_observation_id: str
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    relation_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        relation_type = _enum(RelationType, self.relation_type, "relation_type")
        _require_contract_id(
            self.target_observation_id,
            "target_observation_id",
            _OBSERVATION_ID,
        )
        require_probability(self.confidence, "relation.confidence")
        metadata = freeze_json(self.metadata, field="relation.metadata")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "relation_id",
            stable_id(
                "rel", relation_type.value, self.target_observation_id, metadata
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "target_observation_id": self.target_observation_id,
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {"schema_version", "relation_id", "relation_type", "target_observation_id", "confidence", "metadata"},
            dto="ObservationRelation",
        )
        result = cls(
            relation_type=_enum(
                RelationType, payload.get("relation_type"), "relation_type"
            ),
            target_observation_id=payload.get("target_observation_id"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "relation.metadata"),
        )
        if payload.get("relation_id") != result.relation_id:
            raise ContractValidationError("ObservationRelation relation_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CanonicalRelation(JsonDTOMixin):
    """A resolved canonical edge with exact parser-assertion lineage."""

    relation_type: RelationType
    target_block_id: str
    source_observation_ids: tuple[str, ...]
    source_relation_ids: tuple[str, ...]
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    relation_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        relation_type = _enum(RelationType, self.relation_type, "relation_type")
        _require_contract_id(self.target_block_id, "target_block_id", _BLOCK_ID)
        raw_observation_ids = tuple(self.source_observation_ids)
        raw_relation_ids = tuple(self.source_relation_ids)
        if not raw_observation_ids:
            raise ContractValidationError(
                "canonical relation requires source observations"
            )
        if not raw_relation_ids:
            raise ContractValidationError(
                "canonical relation requires source relation assertions"
            )
        for source_id in raw_observation_ids:
            _require_contract_id(
                source_id, "source_observation_ids[]", _OBSERVATION_ID
            )
        for relation_id in raw_relation_ids:
            _require_contract_id(
                relation_id, "source_relation_ids[]", _RELATION_ID
            )
        observation_ids = tuple(sorted(set(raw_observation_ids)))
        relation_ids = tuple(sorted(set(raw_relation_ids)))
        require_probability(self.confidence, "canonical_relation.confidence")
        metadata = freeze_json(self.metadata, field="canonical_relation.metadata")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "source_observation_ids", observation_ids)
        object.__setattr__(self, "source_relation_ids", relation_ids)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "relation_id",
            stable_id(
                "crel",
                relation_type.value,
                self.target_block_id,
                observation_ids,
                relation_ids,
                self.confidence,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "target_block_id": self.target_block_id,
            "source_observation_ids": list(self.source_observation_ids),
            "source_relation_ids": list(self.source_relation_ids),
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "relation_id",
                "relation_type",
                "target_block_id",
                "source_observation_ids",
                "source_relation_ids",
                "confidence",
                "metadata",
            },
            dto="CanonicalRelation",
        )
        result = cls(
            relation_type=_enum(
                RelationType, payload.get("relation_type"), "relation_type"
            ),
            target_block_id=payload.get("target_block_id"),  # type: ignore[arg-type]
            source_observation_ids=tuple(
                require_sequence(
                    payload.get("source_observation_ids", []),
                    "source_observation_ids",
                )
            ),  # type: ignore[arg-type]
            source_relation_ids=tuple(
                require_sequence(
                    payload.get("source_relation_ids", []),
                    "source_relation_ids",
                )
            ),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(
                payload.get("metadata", {}), "canonical_relation.metadata"
            ),
        )
        if payload.get("relation_id") != result.relation_id:
            raise ContractValidationError("CanonicalRelation relation_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ParserBlockObservation(JsonDTOMixin):
    document_version_id: str
    page_id: str
    page_index: int
    provenance: ParserProvenance
    block_type: BlockType
    representations: tuple[Representation, ...]
    geometry: SourceGeometry | None = None
    semantic_subtype: str | None = None
    reading_order: float | None = None
    assets: tuple[AssetRef, ...] = ()
    relations: tuple[ObservationRelation, ...] = ()
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    semantic_substructures: tuple[SemanticSubstructure, ...] = ()
    observation_id: str = field(init=False)
    content_sha256: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.document_version_id, "document_version_id")
        require_non_empty(self.page_id, "page_id")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise ContractValidationError("page_index must be a non-negative integer")
        expected_page_id = stable_id("page", self.document_version_id, self.page_index)
        if self.page_id != expected_page_id:
            raise ContractValidationError("observation page_id does not match document/page identity")
        block_type = _enum(BlockType, self.block_type, "block_type")
        if self.semantic_subtype is not None:
            require_non_empty(self.semantic_subtype, "semantic_subtype")
        if self.reading_order is not None and (
            isinstance(self.reading_order, bool)
            or not isinstance(self.reading_order, (int, float))
            or not math.isfinite(float(self.reading_order))
        ):
            raise ContractValidationError("reading_order must be a finite number")
        require_probability(self.confidence, "observation.confidence")
        representations = tuple(self.representations)
        if not representations and not self.assets:
            raise ContractValidationError("observation requires a representation or asset")
        assets = tuple(self.assets)
        relations = tuple(self.relations)
        semantic_substructures = tuple(self.semantic_substructures)
        if not isinstance(self.provenance, ParserProvenance):
            raise ContractValidationError("provenance must be ParserProvenance")
        if self.geometry is not None and not isinstance(self.geometry, SourceGeometry):
            raise ContractValidationError("geometry must be SourceGeometry")
        if any(not isinstance(item, Representation) for item in representations):
            raise ContractValidationError("representations must contain Representation DTOs")
        if any(not isinstance(item, AssetRef) for item in assets):
            raise ContractValidationError("assets must contain AssetRef DTOs")
        if any(not isinstance(item, ObservationRelation) for item in relations):
            raise ContractValidationError("relations must contain ObservationRelation DTOs")
        if len(semantic_substructures) > MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK:
            raise ContractValidationError(
                "observation semantic_substructures limit exceeded"
            )
        if any(
            not isinstance(item, SemanticSubstructure)
            for item in semantic_substructures
        ):
            raise ContractValidationError(
                "semantic_substructures must contain SemanticSubstructure DTOs"
            )
        if any(item.is_source_bound for item in semantic_substructures):
            raise ContractValidationError(
                "parser observation semantic substructures must be unbound; "
                "the enclosing observation is their source identity"
            )
        if sum(len(item.elements) for item in semantic_substructures) > (
            MAX_SEMANTIC_ELEMENTS_PER_BLOCK
        ):
            raise ContractValidationError(
                "observation semantic element limit exceeded"
            )
        if sum(len(item.relations) for item in semantic_substructures) > (
            MAX_SEMANTIC_RELATIONS_PER_BLOCK
        ):
            raise ContractValidationError(
                "observation semantic relation limit exceeded"
            )
        if sum(
            len(canonical_json_bytes(item.to_dict()))
            for item in semantic_substructures
        ) > MAX_SEMANTIC_BLOCK_CANONICAL_BYTES:
            raise ContractValidationError(
                "observation semantic canonical byte limit exceeded"
            )
        substructure_ids = [item.substructure_id for item in semantic_substructures]
        if len(substructure_ids) != len(set(substructure_ids)):
            raise ContractValidationError(
                "observation contains duplicate semantic substructure IDs"
            )
        semantic_substructures = tuple(
            sorted(semantic_substructures, key=lambda item: item.substructure_id)
        )
        representation_by_id = {
            item.representation_id: item for item in representations
        }
        asset_ids = {item.asset_id for item in assets}
        for structure in semantic_substructures:
            for element in structure.elements:
                for selector in element.source_selectors:
                    if (
                        selector.representation_id is not None
                        and selector.representation_id not in representation_by_id
                    ):
                        raise ContractValidationError(
                            "semantic source selector references a representation "
                            "outside its observation"
                        )
                    if selector.char_start is not None:
                        assert selector.representation_id is not None
                        source_representation = representation_by_id.get(
                            selector.representation_id
                        )
                        if source_representation is not None and not isinstance(
                            source_representation.content, str
                        ):
                            raise ContractValidationError(
                                "semantic character selector requires a string "
                                "representation"
                            )
                        if (
                            source_representation is not None
                            and selector.char_end is not None
                            and selector.char_end
                            > len(source_representation.content)
                        ):
                            raise ContractValidationError(
                                "semantic char_end exceeds its representation length"
                            )
                    if (
                        selector.asset_id is not None
                        and selector.asset_id not in asset_ids
                    ):
                        raise ContractValidationError(
                            "semantic source selector references an asset outside "
                            "its observation"
                        )
        metadata = freeze_json(self.metadata, field="observation.metadata")
        identity = stable_id(
            "obs",
            self.document_version_id,
            self.page_id,
            self.provenance.parser_name,
            self.provenance.parser_version,
            self.provenance.parser_native_id,
        )
        relation_ids = [item.relation_id for item in relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ContractValidationError("relations contains duplicate relation IDs")
        if any(item.target_observation_id == identity for item in relations):
            raise ContractValidationError("an observation relation cannot target itself")
        digest_payload: dict[str, object] = {
                "document_version_id": self.document_version_id,
                "page_id": self.page_id,
                "page_index": self.page_index,
                "provenance": self.provenance.to_dict(),
                "block_type": block_type.value,
                "semantic_subtype": self.semantic_subtype,
                "geometry": self.geometry.to_dict() if self.geometry else None,
                "reading_order": self.reading_order,
                "representations": [item.to_dict() for item in representations],
                "assets": [item.to_dict() for item in assets],
                "relations": [item.to_dict() for item in relations],
                "confidence": self.confidence,
                "metadata": metadata,
            }
        # Hash-neutral when absent: existing parser artifacts retain their
        # exact content hashes and generated revision identities.
        if semantic_substructures:
            digest_payload["semantic_substructures"] = [
                item.to_dict() for item in semantic_substructures
            ]
        digest = content_sha256(digest_payload)
        object.__setattr__(self, "block_type", block_type)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "semantic_substructures", semantic_substructures)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "observation_id", identity)
        object.__setattr__(self, "content_sha256", digest)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "observation_id": self.observation_id,
            "content_sha256": self.content_sha256,
            "document_version_id": self.document_version_id,
            "page_id": self.page_id,
            "page_index": self.page_index,
            "provenance": self.provenance.to_dict(),
            "block_type": self.block_type.value,
            "semantic_subtype": self.semantic_subtype,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "reading_order": self.reading_order,
            "representations": [item.to_dict() for item in self.representations],
            "assets": [item.to_dict() for item in self.assets],
            "relations": [item.to_dict() for item in self.relations],
            "confidence": self.confidence,
            "metadata": json_ready(self.metadata),
        }
        if self.semantic_substructures:
            payload["semantic_substructures"] = [
                item.to_dict() for item in self.semantic_substructures
            ]
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
                "observation_id",
                "content_sha256",
                "document_version_id",
                "page_id",
                "page_index",
                "provenance",
                "block_type",
                "semantic_subtype",
                "geometry",
                "reading_order",
                "representations",
                "assets",
                "relations",
                "confidence",
                "metadata",
                "semantic_substructures",
            },
            dto="ParserBlockObservation",
        )
        geometry_payload = payload.get("geometry")
        raw_semantic_substructures = _bounded_semantic_sequence(
            payload.get("semantic_substructures", []),
            "semantic_substructures",
            MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
        )
        result = cls(
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            page_id=payload.get("page_id"),  # type: ignore[arg-type]
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            provenance=ParserProvenance.from_dict(
                require_mapping(payload.get("provenance"), "provenance")
            ),
            block_type=_enum(BlockType, payload.get("block_type"), "block_type"),
            semantic_subtype=payload.get("semantic_subtype"),  # type: ignore[arg-type]
            geometry=(
                SourceGeometry.from_dict(require_mapping(geometry_payload, "geometry"))
                if geometry_payload is not None
                else None
            ),
            reading_order=payload.get("reading_order"),  # type: ignore[arg-type]
            representations=tuple(
                Representation.from_dict(require_mapping(item, f"representations[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("representations", []), "representations")
                )
            ),
            assets=tuple(
                AssetRef.from_dict(require_mapping(item, f"assets[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("assets", []), "assets")
                )
            ),
            relations=tuple(
                ObservationRelation.from_dict(require_mapping(item, f"relations[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("relations", []), "relations")
                )
            ),
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "metadata"),
            semantic_substructures=tuple(
                SemanticSubstructure.from_dict(
                    require_mapping(item, f"semantic_substructures[{index}]")
                )
                for index, item in enumerate(raw_semantic_substructures)
            ),
        )
        if payload.get("observation_id") != result.observation_id:
            raise ContractValidationError("ParserBlockObservation observation_id mismatch")
        if payload.get("content_sha256") != result.content_sha256:
            raise ContractValidationError("ParserBlockObservation content_sha256 mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class CanonicalBlock(JsonDTOMixin):
    """A reconciled evidence revision; never overwrites parser observations."""

    canonical_key: str
    revision: int
    document_version_id: str
    page_id: str
    page_index: int
    block_type: BlockType
    representations: tuple[Representation, ...]
    source_observation_ids: tuple[str, ...]
    reconciliation_method: str
    quality_status: QualityStatus
    geometry: SourceGeometry | None = None
    semantic_subtype: str | None = None
    reading_order: float | None = None
    assets: tuple[AssetRef, ...] = ()
    relations: tuple[CanonicalRelation, ...] = ()
    parent_block_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    semantic_substructures: tuple[SemanticSubstructure, ...] = ()
    block_id: str = field(init=False)
    content_sha256: str = field(init=False)
    revision_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.canonical_key, "canonical_key")
        require_non_empty(self.document_version_id, "document_version_id")
        require_non_empty(self.page_id, "page_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ContractValidationError("canonical revision must be a positive integer")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int) or self.page_index < 0:
            raise ContractValidationError("page_index must be a non-negative integer")
        expected_page_id = stable_id("page", self.document_version_id, self.page_index)
        if self.page_id != expected_page_id:
            raise ContractValidationError("canonical block page_id does not match page identity")
        block_type = _enum(BlockType, self.block_type, "block_type")
        status = _enum(QualityStatus, self.quality_status, "quality_status")
        require_non_empty(self.reconciliation_method, "reconciliation_method")
        if self.semantic_subtype is not None:
            require_non_empty(self.semantic_subtype, "semantic_subtype")
        if self.reading_order is not None and (
            isinstance(self.reading_order, bool)
            or not isinstance(self.reading_order, (int, float))
            or not math.isfinite(float(self.reading_order))
        ):
            raise ContractValidationError("reading_order must be a finite number")
        raw_source_ids = tuple(self.source_observation_ids)
        if not raw_source_ids:
            raise ContractValidationError("canonical block requires source observations")
        for source_id in raw_source_ids:
            require_non_empty(source_id, "source_observation_ids[]")
        source_ids = tuple(sorted(set(raw_source_ids)))
        representations = tuple(self.representations)
        assets = tuple(self.assets)
        relations = tuple(self.relations)
        semantic_substructures = tuple(self.semantic_substructures)
        if self.geometry is not None and not isinstance(self.geometry, SourceGeometry):
            raise ContractValidationError("geometry must be SourceGeometry")
        if any(not isinstance(item, Representation) for item in representations):
            raise ContractValidationError("representations must contain Representation DTOs")
        if any(not isinstance(item, AssetRef) for item in assets):
            raise ContractValidationError("assets must contain AssetRef DTOs")
        if any(not isinstance(item, CanonicalRelation) for item in relations):
            raise ContractValidationError(
                "relations must contain CanonicalRelation DTOs"
            )
        if len(semantic_substructures) > MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK:
            raise ContractValidationError(
                "canonical semantic_substructures limit exceeded"
            )
        if any(
            not isinstance(item, SemanticSubstructure)
            for item in semantic_substructures
        ):
            raise ContractValidationError(
                "semantic_substructures must contain SemanticSubstructure DTOs"
            )
        if any(not item.is_source_bound for item in semantic_substructures):
            raise ContractValidationError(
                "canonical semantic substructures require exact source lineage"
            )
        if sum(len(item.elements) for item in semantic_substructures) > (
            MAX_SEMANTIC_ELEMENTS_PER_BLOCK
        ):
            raise ContractValidationError("canonical semantic element limit exceeded")
        if sum(len(item.relations) for item in semantic_substructures) > (
            MAX_SEMANTIC_RELATIONS_PER_BLOCK
        ):
            raise ContractValidationError("canonical semantic relation limit exceeded")
        if sum(
            len(canonical_json_bytes(item.to_dict()))
            for item in semantic_substructures
        ) > MAX_SEMANTIC_BLOCK_CANONICAL_BYTES:
            raise ContractValidationError(
                "canonical semantic byte limit exceeded"
            )
        substructure_ids = [item.substructure_id for item in semantic_substructures]
        if len(substructure_ids) != len(set(substructure_ids)):
            raise ContractValidationError(
                "canonical block contains duplicate semantic substructure IDs"
            )
        semantic_substructures = tuple(
            sorted(semantic_substructures, key=lambda item: item.substructure_id)
        )
        availability_by_kind: dict[
            SemanticStructureKind, set[SemanticStructureAvailability]
        ] = {}
        for structure in semantic_substructures:
            availability_by_kind.setdefault(structure.kind, set()).add(
                structure.availability
            )
        has_uncovered_partial = any(
            SemanticStructureAvailability.PARTIAL in availabilities
            and SemanticStructureAvailability.COMPLETE not in availabilities
            for availabilities in availability_by_kind.values()
        )
        if status is QualityStatus.ACCEPTED and has_uncovered_partial:
            status = QualityStatus.DEGRADED
        if not representations and not assets:
            raise ContractValidationError("canonical block requires a representation or asset")
        if self.parent_block_id is not None:
            require_non_empty(self.parent_block_id, "parent_block_id")
        metadata = freeze_json(self.metadata, field="canonical.metadata")
        block_id = stable_id(
            "block", self.document_version_id, self.page_id, self.canonical_key
        )
        relation_ids = [item.relation_id for item in relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ContractValidationError("relations contains duplicate relation IDs")
        if any(item.target_block_id == block_id for item in relations):
            raise ContractValidationError("a canonical relation cannot target its own block")
        if any(
            source_id not in source_ids
            for relation in relations
            for source_id in relation.source_observation_ids
        ):
            raise ContractValidationError(
                "canonical relation source observations must belong to the block"
            )
        if any(
            source_id not in source_ids
            for structure in semantic_substructures
            for source_id in structure.source_observation_ids
        ):
            raise ContractValidationError(
                "semantic substructure source observations must belong to the block"
            )
        representation_by_id = {
            item.representation_id: item for item in representations
        }
        asset_ids = {item.asset_id for item in assets}
        for structure in semantic_substructures:
            for element in structure.elements:
                for selector in element.source_selectors:
                    if (
                        selector.representation_id is not None
                        and selector.representation_id not in representation_by_id
                    ):
                        raise ContractValidationError(
                            "canonical semantic selector references a representation "
                            "outside its block"
                        )
                    if selector.char_start is not None:
                        assert selector.representation_id is not None
                        source_representation = representation_by_id.get(
                            selector.representation_id
                        )
                        if source_representation is not None and not isinstance(
                            source_representation.content, str
                        ):
                            raise ContractValidationError(
                                "canonical semantic character selector requires a "
                                "string representation"
                            )
                        if (
                            source_representation is not None
                            and selector.char_end is not None
                            and selector.char_end
                            > len(source_representation.content)
                        ):
                            raise ContractValidationError(
                                "canonical semantic char_end exceeds its "
                                "representation length"
                            )
                    if (
                        selector.asset_id is not None
                        and selector.asset_id not in asset_ids
                    ):
                        raise ContractValidationError(
                            "canonical semantic selector references an asset outside "
                            "its block"
                        )
        relations = tuple(
            sorted(
                relations,
                key=lambda item: (
                    item.relation_type.value,
                    item.target_block_id,
                    item.relation_id,
                ),
            )
        )
        digest_payload: dict[str, object] = {
            "block_id": block_id,
            "block_type": block_type.value,
            "semantic_subtype": self.semantic_subtype,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "reading_order": self.reading_order,
            "representations": [item.to_dict() for item in representations],
            "assets": [item.to_dict() for item in assets],
            "source_observation_ids": source_ids,
            "reconciliation_method": self.reconciliation_method,
            "quality_status": status.value,
            "parent_block_id": self.parent_block_id,
            "metadata": metadata,
        }
        # Omitting the empty field from the digest preserves hashes of v1
        # canonical artifacts written before structural relations were added.
        if relations:
            digest_payload["relations"] = [item.to_dict() for item in relations]
        if semantic_substructures:
            digest_payload["semantic_substructures"] = [
                item.to_dict() for item in semantic_substructures
            ]
        digest = content_sha256(digest_payload)
        object.__setattr__(self, "block_type", block_type)
        object.__setattr__(self, "quality_status", status)
        object.__setattr__(self, "source_observation_ids", source_ids)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "semantic_substructures", semantic_substructures)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(
            self,
            "revision_id",
            stable_id("blockrev", block_id, self.revision, digest),
        )

    @classmethod
    def from_observation(
        cls,
        observation: ParserBlockObservation,
        *,
        canonical_key: str | None = None,
        revision: int = 1,
        reconciliation_method: str = "single_observation",
        quality_status: QualityStatus = QualityStatus.ACCEPTED,
    ) -> Self:
        if observation.relations:
            raise ContractValidationError(
                "CanonicalBlock.from_observation cannot resolve relation targets; "
                "use reconcile_observations with the complete observation set"
            )
        return cls(
            canonical_key=canonical_key or observation.observation_id,
            revision=revision,
            document_version_id=observation.document_version_id,
            page_id=observation.page_id,
            page_index=observation.page_index,
            block_type=observation.block_type,
            semantic_subtype=observation.semantic_subtype,
            geometry=observation.geometry,
            reading_order=observation.reading_order,
            representations=observation.representations,
            assets=observation.assets,
            semantic_substructures=tuple(
                item.with_source_lineage(
                    source_observation_ids=(observation.observation_id,)
                )
                for item in observation.semantic_substructures
            ),
            source_observation_ids=(observation.observation_id,),
            reconciliation_method=reconciliation_method,
            quality_status=quality_status,
            metadata={"source_content_sha256": observation.content_sha256},
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "block_id": self.block_id,
            "revision_id": self.revision_id,
            "content_sha256": self.content_sha256,
            "canonical_key": self.canonical_key,
            "revision": self.revision,
            "document_version_id": self.document_version_id,
            "page_id": self.page_id,
            "page_index": self.page_index,
            "block_type": self.block_type.value,
            "semantic_subtype": self.semantic_subtype,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "reading_order": self.reading_order,
            "representations": [item.to_dict() for item in self.representations],
            "assets": [item.to_dict() for item in self.assets],
            "relations": [item.to_dict() for item in self.relations],
            "source_observation_ids": list(self.source_observation_ids),
            "reconciliation_method": self.reconciliation_method,
            "quality_status": self.quality_status.value,
            "parent_block_id": self.parent_block_id,
            "metadata": json_ready(self.metadata),
        }
        if self.semantic_substructures:
            payload["semantic_substructures"] = [
                item.to_dict() for item in self.semantic_substructures
            ]
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
                "block_id",
                "revision_id",
                "content_sha256",
                "canonical_key",
                "revision",
                "document_version_id",
                "page_id",
                "page_index",
                "block_type",
                "semantic_subtype",
                "geometry",
                "reading_order",
                "representations",
                "assets",
                "relations",
                "source_observation_ids",
                "reconciliation_method",
                "quality_status",
                "parent_block_id",
                "metadata",
                "semantic_substructures",
            },
            dto="CanonicalBlock",
        )
        geometry_payload = payload.get("geometry")
        raw_semantic_substructures = _bounded_semantic_sequence(
            payload.get("semantic_substructures", []),
            "semantic_substructures",
            MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
        )
        result = cls(
            canonical_key=payload.get("canonical_key"),  # type: ignore[arg-type]
            revision=payload.get("revision"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            page_id=payload.get("page_id"),  # type: ignore[arg-type]
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            block_type=_enum(BlockType, payload.get("block_type"), "block_type"),
            semantic_subtype=payload.get("semantic_subtype"),  # type: ignore[arg-type]
            geometry=(
                SourceGeometry.from_dict(require_mapping(geometry_payload, "geometry"))
                if geometry_payload is not None
                else None
            ),
            reading_order=payload.get("reading_order"),  # type: ignore[arg-type]
            representations=tuple(
                Representation.from_dict(require_mapping(item, f"representations[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("representations", []), "representations")
                )
            ),
            assets=tuple(
                AssetRef.from_dict(require_mapping(item, f"assets[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("assets", []), "assets")
                )
            ),
            relations=tuple(
                CanonicalRelation.from_dict(
                    require_mapping(item, f"relations[{index}]")
                )
                for index, item in enumerate(
                    require_sequence(payload.get("relations", []), "relations")
                )
            ),
            source_observation_ids=tuple(
                require_sequence(payload.get("source_observation_ids", []), "source_observation_ids")
            ),  # type: ignore[arg-type]
            reconciliation_method=payload.get("reconciliation_method"),  # type: ignore[arg-type]
            quality_status=_enum(
                QualityStatus, payload.get("quality_status"), "quality_status"
            ),
            parent_block_id=payload.get("parent_block_id"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "metadata"),
            semantic_substructures=tuple(
                SemanticSubstructure.from_dict(
                    require_mapping(item, f"semantic_substructures[{index}]")
                )
                for index, item in enumerate(raw_semantic_substructures)
            ),
        )
        for field_name in ("block_id", "revision_id", "content_sha256"):
            if payload.get(field_name) != getattr(result, field_name):
                raise ContractValidationError(f"CanonicalBlock {field_name} mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))
