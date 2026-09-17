"""Deterministic, page-local fusion of parser observations.

Parser adapters produce immutable :class:`ParserBlockObservation` objects.
This module is the only application-layer component that combines observations
into canonical evidence.  It deliberately never joins blocks by global list
position: candidates must share a page and match by geometry and/or content.

The implementation is dependency-free so reconciliation can be replayed and
audited without installing any parser or ML runtime.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from html.parser import HTMLParser
import math
import re
import unicodedata

from document_services.domain.base import (
    ContractValidationError,
    UnsupportedSchemaVersionError,
    content_sha256,
    dump_json,
    freeze_json,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
    stable_id,
)
from document_services.domain.evidence import (
    AssetRef,
    BlockType,
    CanonicalBlock,
    CanonicalRelation,
    CoordinateOrigin,
    CoordinateSpace,
    ObservationRelation,
    ParserBlockObservation,
    QualityStatus,
    Representation,
    RepresentationKind,
    SemanticStructureKind,
    SemanticSubstructure,
    SourceGeometry,
)


RECONCILIATION_SCHEMA_VERSION = "docarag.reconciliation.v1"
RECONCILIATION_METHOD = "page_local_geometry_text_fusion.v2"


def _require_reconciliation_schema(payload: Mapping[str, object]) -> None:
    actual = payload.get("schema_version")
    if actual != RECONCILIATION_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            "unsupported reconciliation schema_version "
            f"{actual!r}; expected {RECONCILIATION_SCHEMA_VERSION!r}"
        )


class ReconciliationError(ContractValidationError):
    """Observations cannot be reconciled without violating source identity."""


class ReconciliationIssueCode(StrEnum):
    DUPLICATE_OBSERVATION = "duplicate_observation"
    CONFLICTING_OBSERVATION = "conflicting_observation"
    CANDIDATE_LIMIT = "candidate_limit"
    TYPE_DISAGREEMENT = "type_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_GEOMETRY_FREE = "ambiguous_geometry_free"
    PRIOR_IDENTITY_CANDIDATE_LIMIT = "prior_identity_candidate_limit"
    AMBIGUOUS_PRIOR_IDENTITY = "ambiguous_prior_identity"
    CANONICAL_IDENTITY_COLLISION = "canonical_identity_collision"
    UNRESOLVED_RELATION_TARGET = "unresolved_relation_target"
    COLLAPSED_RELATION_TARGET = "collapsed_relation_target"


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    """Auditable thresholds and deterministic parser preferences.

    ``parser_priority`` determines the stable anchor used for a canonical key;
    it is not an unconditional quality ranking.  Rich representations and
    confidence still determine the selected canonical type and geometry.
    """

    minimum_iou: float = 0.55
    high_iou: float = 0.85
    minimum_text_similarity: float = 0.65
    cross_type_iou: float = 0.78
    cross_type_text_similarity: float = 0.78
    accepted_confidence: float = 0.60
    grid_size: int = 16
    max_candidates_per_observation: int = 512
    max_raw_candidate_probes_per_observation: int = 4096
    parser_quality_weights: Mapping[str, float] = field(default_factory=dict)
    parser_priority: tuple[str, ...] = (
        "docling",
        "mineru",
        "paddleocr-vl",
        "paddleocr_vl",
    )
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "minimum_iou",
            "high_iou",
            "minimum_text_similarity",
            "cross_type_iou",
            "cross_type_text_similarity",
            "accepted_confidence",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ContractValidationError(f"{name} must be between 0 and 1")
        if self.high_iou < self.minimum_iou:
            raise ContractValidationError("high_iou must be >= minimum_iou")
        for name in (
            "grid_size",
            "max_candidates_per_observation",
            "max_raw_candidate_probes_per_observation",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractValidationError(f"{name} must be a positive integer")
        if self.grid_size > 128:
            raise ContractValidationError("grid_size must not exceed 128")
        if self.max_candidates_per_observation > 100_000:
            raise ContractValidationError(
                "max_candidates_per_observation must not exceed 100000"
            )
        if self.max_raw_candidate_probes_per_observation > 1_000_000:
            raise ContractValidationError(
                "max_raw_candidate_probes_per_observation must not exceed 1000000"
            )
        if (
            self.max_raw_candidate_probes_per_observation
            <= self.max_candidates_per_observation
        ):
            raise ContractValidationError(
                "max_raw_candidate_probes_per_observation must be greater than "
                "max_candidates_per_observation"
            )
        priorities = tuple(dict.fromkeys(self.parser_priority))
        for parser_name in priorities:
            require_non_empty(parser_name, "parser_priority[]")
        quality_weights: dict[str, float] = {}
        for parser_name, raw_weight in self.parser_quality_weights.items():
            require_non_empty(parser_name, "parser_quality_weights key")
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or not 0.0 < float(raw_weight) <= 2.0
            ):
                raise ContractValidationError(
                    "parser quality weights must be finite numbers in (0, 2]"
                )
            quality_weights[parser_name] = float(raw_weight)
        frozen_weights = freeze_json(
            quality_weights, field="parser_quality_weights"
        )
        object.__setattr__(self, "parser_priority", priorities)
        object.__setattr__(self, "parser_quality_weights", frozen_weights)
        object.__setattr__(
            self,
            "policy_sha256",
            content_sha256(
                {
                    "schema_version": RECONCILIATION_SCHEMA_VERSION,
                    "minimum_iou": self.minimum_iou,
                    "high_iou": self.high_iou,
                    "minimum_text_similarity": self.minimum_text_similarity,
                    "cross_type_iou": self.cross_type_iou,
                    "cross_type_text_similarity": self.cross_type_text_similarity,
                    "accepted_confidence": self.accepted_confidence,
                    "grid_size": self.grid_size,
                    "max_candidates_per_observation": self.max_candidates_per_observation,
                    "max_raw_candidate_probes_per_observation": (
                        self.max_raw_candidate_probes_per_observation
                    ),
                    "parser_priority": priorities,
                    "parser_quality_weights": frozen_weights,
                }
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "policy_sha256": self.policy_sha256,
            "minimum_iou": self.minimum_iou,
            "high_iou": self.high_iou,
            "minimum_text_similarity": self.minimum_text_similarity,
            "cross_type_iou": self.cross_type_iou,
            "cross_type_text_similarity": self.cross_type_text_similarity,
            "accepted_confidence": self.accepted_confidence,
            "grid_size": self.grid_size,
            "max_candidates_per_observation": self.max_candidates_per_observation,
            "max_raw_candidate_probes_per_observation": (
                self.max_raw_candidate_probes_per_observation
            ),
            "parser_priority": list(self.parser_priority),
            "parser_quality_weights": json_ready(self.parser_quality_weights),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReconciliationPolicy":
        _require_reconciliation_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "policy_sha256",
                "minimum_iou",
                "high_iou",
                "minimum_text_similarity",
                "cross_type_iou",
                "cross_type_text_similarity",
                "accepted_confidence",
                "grid_size",
                "max_candidates_per_observation",
                "max_raw_candidate_probes_per_observation",
                "parser_priority",
                "parser_quality_weights",
            },
            dto="ReconciliationPolicy",
        )
        result = cls(
            minimum_iou=payload.get("minimum_iou"),  # type: ignore[arg-type]
            high_iou=payload.get("high_iou"),  # type: ignore[arg-type]
            minimum_text_similarity=payload.get("minimum_text_similarity"),  # type: ignore[arg-type]
            cross_type_iou=payload.get("cross_type_iou"),  # type: ignore[arg-type]
            cross_type_text_similarity=payload.get("cross_type_text_similarity"),  # type: ignore[arg-type]
            accepted_confidence=payload.get("accepted_confidence"),  # type: ignore[arg-type]
            grid_size=payload.get("grid_size"),  # type: ignore[arg-type]
            max_candidates_per_observation=payload.get("max_candidates_per_observation"),  # type: ignore[arg-type]
            max_raw_candidate_probes_per_observation=payload.get(
                "max_raw_candidate_probes_per_observation", 4096
            ),  # type: ignore[arg-type]
            parser_priority=tuple(
                require_sequence(payload.get("parser_priority"), "parser_priority")
            ),  # type: ignore[arg-type]
            parser_quality_weights=require_mapping(
                payload.get("parser_quality_weights", {}),
                "parser_quality_weights",
            ),  # type: ignore[arg-type]
        )
        if payload.get("policy_sha256") != result.policy_sha256:
            raise ContractValidationError("ReconciliationPolicy policy_sha256 mismatch")
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ReconciliationPolicy":
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    code: ReconciliationIssueCode
    message: str
    page_index: int
    observation_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    issue_id: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            code = ReconciliationIssueCode(self.code)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("unsupported reconciliation issue code") from exc
        require_non_empty(self.message, "reconciliation issue message")
        if (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or self.page_index < 0
        ):
            raise ContractValidationError("issue page_index must be non-negative")
        observation_ids = tuple(sorted(set(self.observation_ids)))
        for observation_id in observation_ids:
            require_non_empty(observation_id, "issue observation_ids[]")
        metadata = freeze_json(self.metadata, field="reconciliation_issue.metadata")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "issue_id",
            stable_id(
                "reconcile-issue",
                code.value,
                self.page_index,
                observation_ids,
                self.message,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "issue_id": self.issue_id,
            "code": self.code.value,
            "message": self.message,
            "page_index": self.page_index,
            "observation_ids": list(self.observation_ids),
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReconciliationIssue":
        _require_reconciliation_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "issue_id",
                "code",
                "message",
                "page_index",
                "observation_ids",
                "metadata",
            },
            dto="ReconciliationIssue",
        )
        result = cls(
            code=payload.get("code"),  # type: ignore[arg-type]
            message=payload.get("message"),  # type: ignore[arg-type]
            page_index=payload.get("page_index"),  # type: ignore[arg-type]
            observation_ids=tuple(
                require_sequence(payload.get("observation_ids", []), "observation_ids")
            ),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "metadata"),
        )
        if payload.get("issue_id") != result.issue_id:
            raise ContractValidationError("ReconciliationIssue issue_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    document_version_id: str
    policy_sha256: str
    canonical_blocks: tuple[CanonicalBlock, ...]
    issues: tuple[ReconciliationIssue, ...]
    candidate_pair_count: int
    matched_pair_count: int
    cluster_merge_count: int
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        require_non_empty(self.document_version_id, "document_version_id")
        require_sha256(self.policy_sha256, "policy_sha256")
        for field_name in (
            "candidate_pair_count",
            "matched_pair_count",
            "cluster_merge_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.matched_pair_count > self.candidate_pair_count:
            raise ContractValidationError("matched pairs cannot exceed candidate pairs")
        if self.cluster_merge_count > self.matched_pair_count:
            raise ContractValidationError("cluster merges cannot exceed matched pairs")
        blocks = tuple(self.canonical_blocks)
        issues = tuple(self.issues)
        if any(not isinstance(block, CanonicalBlock) for block in blocks):
            raise ContractValidationError("canonical_blocks must contain CanonicalBlock DTOs")
        if any(not isinstance(issue, ReconciliationIssue) for issue in issues):
            raise ContractValidationError("issues must contain ReconciliationIssue DTOs")
        if not blocks:
            raise ContractValidationError("canonical_blocks must not be empty")
        if any(block.document_version_id != self.document_version_id for block in blocks):
            raise ReconciliationError("canonical blocks span multiple documents")
        block_ids = [block.block_id for block in blocks]
        revision_ids = [block.revision_id for block in blocks]
        issue_ids = [issue.issue_id for issue in issues]
        if len(block_ids) != len(set(block_ids)):
            raise ContractValidationError("canonical_blocks contains duplicate block IDs")
        if len(revision_ids) != len(set(revision_ids)):
            raise ContractValidationError("canonical_blocks contains duplicate revision IDs")
        if len(issue_ids) != len(set(issue_ids)):
            raise ContractValidationError("issues contains duplicate issue IDs")
        known_block_ids = set(block_ids)
        if any(
            relation.target_block_id not in known_block_ids
            for block in blocks
            for relation in block.relations
        ):
            raise ContractValidationError(
                "canonical relation references a block outside the result"
            )
        source_ids = {
            source_id for block in blocks for source_id in block.source_observation_ids
        }
        if any(
            source_id not in source_ids
            for issue in issues
            for source_id in issue.observation_ids
        ):
            raise ContractValidationError(
                "reconciliation issue references an observation outside canonical blocks"
            )
        object.__setattr__(self, "canonical_blocks", blocks)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "reconciliation",
                self.document_version_id,
                self.policy_sha256,
                [block.revision_id for block in blocks],
                [issue.issue_id for issue in issues],
                self.candidate_pair_count,
                self.matched_pair_count,
                self.cluster_merge_count,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "result_id": self.result_id,
            "document_version_id": self.document_version_id,
            "policy_sha256": self.policy_sha256,
            "candidate_pair_count": self.candidate_pair_count,
            "matched_pair_count": self.matched_pair_count,
            "cluster_merge_count": self.cluster_merge_count,
            "canonical_blocks": [block.to_dict() for block in self.canonical_blocks],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReconciliationResult":
        _require_reconciliation_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "result_id",
                "document_version_id",
                "policy_sha256",
                "candidate_pair_count",
                "matched_pair_count",
                "cluster_merge_count",
                "canonical_blocks",
                "issues",
            },
            dto="ReconciliationResult",
        )
        result = cls(
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            policy_sha256=payload.get("policy_sha256"),  # type: ignore[arg-type]
            canonical_blocks=tuple(
                CanonicalBlock.from_dict(
                    require_mapping(item, f"canonical_blocks[{index}]")
                )
                for index, item in enumerate(
                    require_sequence(payload.get("canonical_blocks", []), "canonical_blocks")
                )
            ),
            issues=tuple(
                ReconciliationIssue.from_dict(require_mapping(item, f"issues[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("issues", []), "issues")
                )
            ),
            candidate_pair_count=payload.get("candidate_pair_count"),  # type: ignore[arg-type]
            matched_pair_count=payload.get("matched_pair_count"),  # type: ignore[arg-type]
            cluster_merge_count=payload.get("cluster_merge_count"),  # type: ignore[arg-type]
        )
        if payload.get("result_id") != result.result_id:
            raise ContractValidationError("ReconciliationResult result_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ReconciliationResult":
        return cls.from_dict(parse_json_object(payload))


_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:%|[eE][-+]?\d+)?")
_TEXT_TYPES = frozenset({BlockType.TITLE, BlockType.PARAGRAPH, BlockType.LIST_ITEM})
_VISUAL_TYPES = frozenset({BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM})
_FURNITURE_TYPES = frozenset(
    {BlockType.CAPTION, BlockType.FOOTNOTE, BlockType.HEADER, BlockType.FOOTER}
)
_TYPE_SPECIFICITY = {
    BlockType.DIAGRAM: 6,
    BlockType.CHART: 5,
    BlockType.TABLE: 5,
    BlockType.EQUATION: 5,
    BlockType.TITLE: 4,
    BlockType.CODE: 4,
    BlockType.LIST_ITEM: 3,
    BlockType.FIGURE: 2,
    BlockType.PARAGRAPH: 2,
    BlockType.CAPTION: 2,
    BlockType.FOOTNOTE: 2,
    BlockType.HEADER: 2,
    BlockType.FOOTER: 2,
    BlockType.UNKNOWN: 0,
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.values.append(data)


def _flatten_content(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, Mapping):
        values: list[str] = []
        for key in sorted(value):
            values.extend(_flatten_content(value[key]))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_flatten_content(item))
        return values
    return []


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE.sub(" ", value).strip()


def _representation_text(representation: Representation) -> str:
    values = _flatten_content(representation.content)
    if representation.kind is RepresentationKind.HTML and values:
        parser = _HTMLTextExtractor()
        try:
            parser.feed(" ".join(values))
            parser.close()
        except Exception:
            # Malformed vendor HTML remains comparable as plain text; parsing
            # quality is handled by the adapter and canonical quality status.
            pass
        if parser.values:
            values = parser.values
    return _normalize(" ".join(values))


def _content_channels(observation: ParserBlockObservation) -> Mapping[str, str]:
    channels: dict[str, list[str]] = defaultdict(list)
    for representation in observation.representations:
        value = _representation_text(representation)
        if not value:
            continue
        if representation.kind in {
            RepresentationKind.TABLE_GRID,
            RepresentationKind.CSV,
            RepresentationKind.HTML,
        } or (
            observation.block_type is BlockType.TABLE
            and representation.kind is RepresentationKind.MARKDOWN
        ):
            channels["table"].append(value)
        elif representation.kind in {
            RepresentationKind.LATEX,
            RepresentationKind.MATHML,
        } or (
            observation.block_type is BlockType.EQUATION
            and representation.kind
            in {
                RepresentationKind.TEXT,
                RepresentationKind.OCR_TEXT,
                RepresentationKind.MARKDOWN,
            }
        ):
            channels["formula"].append(value)
        elif representation.kind in {
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
            RepresentationKind.MARKDOWN,
            RepresentationKind.CAPTION,
            RepresentationKind.FOOTNOTE,
            RepresentationKind.VISUAL_DESCRIPTION,
        }:
            channels["text"].append(value)
    return {
        channel: _normalize(" ".join(dict.fromkeys(values)))
        for channel, values in channels.items()
        if values
    }


def _normalized_text(observation: ParserBlockObservation) -> str:
    channels = _content_channels(observation)
    return _normalize(" ".join(channels[key] for key in sorted(channels)))


def _text_similarity(left: str, right: str) -> float | None:
    if not left or not right:
        return None
    if left == right:
        return 1.0
    left_tokens = set(_TOKEN.findall(left))
    right_tokens = set(_TOKEN.findall(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _bbox(geometry: SourceGeometry | None) -> tuple[float, float, float, float] | None:
    if geometry is None:
        return None
    if geometry.bbox is not None:
        x0, y0, x1, y1 = geometry.bbox.to_list()
    elif geometry.polygon:
        xs = [float(point.x) for point in geometry.polygon]
        ys = [float(point.y) for point in geometry.polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:  # guarded by SourceGeometry, retained for defensive replay
        return None
    if geometry.coordinate_space != CoordinateSpace.NORMALIZED:
        if geometry.page_width is None or geometry.page_height is None:
            return None
        x0 /= float(geometry.page_width)
        x1 /= float(geometry.page_width)
        y0 /= float(geometry.page_height)
        y1 /= float(geometry.page_height)
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    if geometry.coordinate_origin is CoordinateOrigin.BOTTOM_LEFT:
        y0, y1 = 1.0 - y1, 1.0 - y0
    elif geometry.coordinate_origin is not CoordinateOrigin.TOP_LEFT:
        raise ReconciliationError(
            f"unsupported coordinate origin: {geometry.coordinate_origin!r}"
        )
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    rotation = geometry.rotation_degrees
    if rotation == 90:
        corners = tuple((1.0 - y, x) for x, y in corners)
    elif rotation == 180:
        corners = tuple((1.0 - x, 1.0 - y) for x, y in corners)
    elif rotation == 270:
        corners = tuple((y, 1.0 - x) for x, y in corners)
    elif rotation != 0:  # SourceGeometry validates; retained for hostile replay.
        raise ReconciliationError(f"unsupported geometry rotation: {rotation!r}")
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    result = (min(xs), min(ys), max(xs), max(ys))
    if any(value < -1e-9 or value > 1.0 + 1e-9 for value in result):
        raise ReconciliationError("normalized geometry falls outside the source page")
    return tuple(min(max(value, 0.0), 1.0) for value in result)  # type: ignore[return-value]


def _iou(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else (1.0 if left == right else 0.0)


def _has_kind(observation: ParserBlockObservation, *kinds: RepresentationKind) -> bool:
    expected = set(kinds)
    return any(representation.kind in expected for representation in observation.representations)


def _types_compatible_types(left: BlockType, right: BlockType) -> bool:
    if left == right:
        return True
    pair = {left, right}
    if pair <= _TEXT_TYPES or pair <= _VISUAL_TYPES:
        return True
    if BlockType.UNKNOWN in pair:
        return not bool(pair & _FURNITURE_TYPES)
    if BlockType.TABLE in pair and pair & _VISUAL_TYPES:
        return True
    if BlockType.EQUATION in pair and pair & _TEXT_TYPES:
        return True
    return False


def _types_compatible(
    left: ParserBlockObservation, right: ParserBlockObservation
) -> bool:
    pair = {left.block_type, right.block_type}
    if not _types_compatible_types(left.block_type, right.block_type):
        return False
    if left.block_type == right.block_type and left.block_type in _FURNITURE_TYPES:
        return _normalized_text(left) == _normalized_text(right)
    if BlockType.TABLE in pair and pair & _VISUAL_TYPES:
        table = left if left.block_type == BlockType.TABLE else right
        return _has_kind(
            table,
            RepresentationKind.TABLE_GRID,
            RepresentationKind.HTML,
            RepresentationKind.CSV,
            RepresentationKind.MARKDOWN,
        )
    if BlockType.EQUATION in pair and pair & _TEXT_TYPES:
        equation = left if left.block_type == BlockType.EQUATION else right
        return _has_kind(equation, RepresentationKind.LATEX, RepresentationKind.MATHML)
    return True


def _numbers(value: str) -> frozenset[str]:
    return frozenset(match.group(0).replace(",", "") for match in _NUMBER.finditer(value))


def _comparable_content(
    left: ParserBlockObservation,
    right: ParserBlockObservation,
) -> tuple[str | None, str | None, str | None]:
    left_channels = _content_channels(left)
    right_channels = _content_channels(right)
    for channel in ("formula", "table", "text"):
        if channel in left_channels and channel in right_channels:
            return channel, left_channels[channel], right_channels[channel]
    pair = {left.block_type, right.block_type}
    if BlockType.EQUATION in pair and pair & _TEXT_TYPES:
        equation_channels = left_channels if left.block_type is BlockType.EQUATION else right_channels
        text_channels = right_channels if left.block_type is BlockType.EQUATION else left_channels
        formula = equation_channels.get("formula")
        text = text_channels.get("text")
        if formula and text:
            return "formula_text", formula, text
    return None, None, None


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    score: float
    metadata: Mapping[str, object]
    conflict: bool


_BLOCK_COMPARISON_STRUCTURE_KINDS: Mapping[
    BlockType, frozenset[SemanticStructureKind]
] = {
    BlockType.TABLE: frozenset({SemanticStructureKind.TABLE}),
    BlockType.EQUATION: frozenset({SemanticStructureKind.FORMULA}),
    BlockType.DIAGRAM: frozenset(
        {SemanticStructureKind.DIAGRAM, SemanticStructureKind.FLOWCHART}
    ),
}


def _semantic_comparison_assertions(
    observation: ParserBlockObservation,
) -> Mapping[SemanticStructureKind, frozenset[str]]:
    values: dict[SemanticStructureKind, set[str]] = defaultdict(set)
    for structure in observation.semantic_substructures:
        values[structure.kind].add(structure.comparison_sha256)
    return {kind: frozenset(digests) for kind, digests in values.items()}


def _semantic_pair_comparison(
    left: ParserBlockObservation,
    right: ParserBlockObservation,
    *,
    left_values: Mapping[SemanticStructureKind, frozenset[str]],
    right_values: Mapping[SemanticStructureKind, frozenset[str]],
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool]:
    """Return exact set agreement/disagreement for shared semantic families.

    Representation-level content may be encoded in different vendor JSON
    shapes.  Only complete equality of the relevant comparison-digest sets can
    override that representation-level mismatch.  Any changed relevant set is
    an explicit structural conflict.
    """

    shared = set(left_values) & set(right_values)
    equivalent = tuple(
        sorted(
            kind.value
            for kind in shared
            if left_values[kind] == right_values[kind]
        )
    )
    disagreed = tuple(
        sorted(
            kind.value
            for kind in shared
            if left_values[kind] != right_values[kind]
        )
    )
    relevant = (
        _BLOCK_COMPARISON_STRUCTURE_KINDS.get(left.block_type, frozenset())
        if left.block_type == right.block_type
        else frozenset()
    )
    left_relevant = {
        kind: left_values[kind] for kind in relevant if kind in left_values
    }
    right_relevant = {
        kind: right_values[kind] for kind in relevant if kind in right_values
    }
    return (
        equivalent,
        disagreed,
        bool(left_relevant) and left_relevant == right_relevant,
        bool(left_relevant)
        and bool(right_relevant)
        and left_relevant != right_relevant,
    )


def _pair_evaluation(
    left: ParserBlockObservation,
    right: ParserBlockObservation,
    *,
    left_bbox: tuple[float, float, float, float] | None,
    right_bbox: tuple[float, float, float, float] | None,
    left_text: str,
    right_text: str,
    policy: ReconciliationPolicy,
    geometry_free_unique: bool,
    left_semantic_assertions: Mapping[
        SemanticStructureKind, frozenset[str]
    ],
    right_semantic_assertions: Mapping[
        SemanticStructureKind, frozenset[str]
    ],
) -> _PairEvaluation | None:
    if not _types_compatible(left, right):
        return None
    overlap = _iou(left_bbox, right_bbox)
    similarity = _text_similarity(left_text, right_text)
    same_type = left.block_type == right.block_type
    channel, comparable_left, comparable_right = _comparable_content(left, right)
    comparable_similarity = (
        _text_similarity(comparable_left, comparable_right)
        if comparable_left is not None and comparable_right is not None
        else None
    )
    representation_number_conflict = bool(
        comparable_left
        and comparable_right
        and _numbers(comparable_left)
        and _numbers(comparable_right)
        and _numbers(comparable_left) != _numbers(comparable_right)
    )
    representation_formula_conflict = bool(
        channel in {"formula", "formula_text"}
        and comparable_left
        and comparable_right
        and re.sub(r"\s+", "", comparable_left)
        != re.sub(r"\s+", "", comparable_right)
    )
    representation_content_conflict = bool(
        comparable_similarity is not None
        and comparable_similarity
        < (policy.minimum_text_similarity if same_type else policy.cross_type_text_similarity)
    ) or representation_number_conflict or representation_formula_conflict
    (
        equivalent_semantic_kinds,
        disagreed_semantic_kinds,
        relevant_semantic_match,
        relevant_semantic_conflict,
    ) = _semantic_pair_comparison(
        left,
        right,
        left_values=left_semantic_assertions,
        right_values=right_semantic_assertions,
    )
    content_conflict = (
        False
        if relevant_semantic_match
        else representation_content_conflict or relevant_semantic_conflict
    )
    number_conflict = (
        False if relevant_semantic_match else representation_number_conflict
    )
    formula_conflict = (
        False if relevant_semantic_match else representation_formula_conflict
    )

    accepted = False
    conflict = False
    reason = ""
    if overlap is not None:
        geometry_threshold = policy.minimum_iou if same_type else policy.cross_type_iou
        content_threshold = (
            policy.minimum_text_similarity
            if same_type
            else policy.cross_type_text_similarity
        )
        if overlap >= geometry_threshold and content_conflict:
            accepted = True
            conflict = True
            reason = (
                "same_region_semantic_structure_conflict"
                if relevant_semantic_conflict
                else "same_region_content_conflict"
            )
        elif same_type and overlap >= geometry_threshold:
            accepted = (
                relevant_semantic_match
                or comparable_similarity is None
                or comparable_similarity >= content_threshold
            )
            reason = (
                "same_type_semantic_structure"
                if relevant_semantic_match
                else "same_type_geometry"
            )
        elif not same_type and overlap >= geometry_threshold:
            accepted = comparable_similarity is None or comparable_similarity >= content_threshold
            reason = "compatible_cross_type_geometry"
    elif geometry_free_unique and similarity is not None and similarity == 1.0:
        # Geometry-free evidence is merged only on exact normalized content.
        accepted = same_type or ({left.block_type, right.block_type} <= _TEXT_TYPES)
        reason = "exact_text_without_geometry"
    if not accepted:
        return None
    score = 0.0
    if overlap is not None:
        score += 0.70 * overlap
    if similarity is not None:
        score += 0.25 * similarity
    score += 0.05 if same_type else 0.0
    return _PairEvaluation(
        score=min(score, 1.0),
        conflict=conflict,
        metadata={
            "iou": overlap,
            "text_similarity": similarity,
            "comparable_channel": channel,
            "comparable_similarity": comparable_similarity,
            "number_conflict": number_conflict,
            "formula_conflict": formula_conflict,
            "representation_number_conflict": representation_number_conflict,
            "representation_formula_conflict": representation_formula_conflict,
            "representation_content_conflict": representation_content_conflict,
            "equivalent_semantic_kinds": list(equivalent_semantic_kinds),
            "disagreed_semantic_kinds": list(disagreed_semantic_kinds),
            "semantic_structure_override": relevant_semantic_match,
            "same_type": same_type,
            "disposition": "conflict" if conflict else "match",
            "reason": reason,
        },
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.members = [{index} for index in range(size)]
        self.parsers = [set() for _ in range(size)]

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != index:
            next_index = self.parent[index]
            self.parent[index] = root
            index = next_index
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if len(self.members[left_root]) < len(self.members[right_root]):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.members[left_root].update(self.members[right_root])
        self.parsers[left_root].update(self.parsers[right_root])
        self.members[right_root].clear()
        self.parsers[right_root].clear()
        return left_root


def _grid_cells(
    bbox: tuple[float, float, float, float], grid_size: int
) -> tuple[tuple[int, int], ...]:
    def cell(value: float) -> int:
        return min(grid_size - 1, max(0, int(value * grid_size)))

    x0, y0, x1, y1 = bbox
    return tuple(
        (x, y)
        for x in range(cell(x0), cell(x1) + 1)
        for y in range(cell(y0), cell(y1) + 1)
    )


def _priority_key(
    observation: ParserBlockObservation, policy: ReconciliationPolicy
) -> tuple[object, ...]:
    try:
        parser_rank = policy.parser_priority.index(observation.provenance.parser_name)
    except ValueError:
        parser_rank = len(policy.parser_priority)
    return (
        parser_rank,
        -_calibrated_confidence(observation, policy),
        observation.observation_id,
    )


def _calibrated_confidence(
    observation: ParserBlockObservation, policy: ReconciliationPolicy
) -> float:
    if observation.confidence is None:
        return 0.0
    weight = policy.parser_quality_weights.get(
        observation.provenance.parser_name, 1.0
    )
    return min(float(observation.confidence) * float(weight), 1.0)


def _type_support(observation: ParserBlockObservation) -> int:
    if observation.block_type is BlockType.TABLE and _has_kind(
        observation,
        RepresentationKind.TABLE_GRID,
        RepresentationKind.HTML,
        RepresentationKind.CSV,
    ):
        return 2
    if observation.block_type is BlockType.EQUATION and _has_kind(
        observation, RepresentationKind.LATEX, RepresentationKind.MATHML
    ):
        return 2
    if observation.block_type is BlockType.DIAGRAM and _has_kind(
        observation, RepresentationKind.DIAGRAM_GRAPH
    ):
        return 2
    if observation.block_type is BlockType.CHART and _has_kind(
        observation, RepresentationKind.STRUCTURED_DATA
    ):
        return 2
    return 1 if observation.block_type is not BlockType.UNKNOWN else 0


def _selection_key(
    observation: ParserBlockObservation, policy: ReconciliationPolicy
) -> tuple[object, ...]:
    rich_kinds = {
        RepresentationKind.TABLE_GRID,
        RepresentationKind.LATEX,
        RepresentationKind.MATHML,
        RepresentationKind.DIAGRAM_GRAPH,
        RepresentationKind.STRUCTURED_DATA,
        RepresentationKind.HTML,
        RepresentationKind.CSV,
    }
    richness = sum(rep.kind in rich_kinds for rep in observation.representations)
    specificity = _TYPE_SPECIFICITY.get(observation.block_type, 0)
    return (
        -_type_support(observation),
        -_calibrated_confidence(observation, policy),
        -specificity,
        -richness,
        *_priority_key(observation, policy),
    )


def _quality_status(
    cluster: Sequence[ParserBlockObservation],
    *,
    selected: ParserBlockObservation,
    uncertain_ids: set[str],
    conflict_ids: set[str],
    policy: ReconciliationPolicy,
) -> QualityStatus:
    ids = {observation.observation_id for observation in cluster}
    if ids & (uncertain_ids | conflict_ids):
        return QualityStatus.NEEDS_REVIEW
    types = {observation.block_type for observation in cluster}
    if len(types) > 1:
        return QualityStatus.NEEDS_REVIEW
    if selected.confidence is None:
        return QualityStatus.DEGRADED
    if _calibrated_confidence(selected, policy) < policy.accepted_confidence:
        return QualityStatus.DEGRADED
    return QualityStatus.ACCEPTED


def _canonical_family(block_type: BlockType) -> str:
    if block_type in _TEXT_TYPES:
        return "text"
    if block_type in _VISUAL_TYPES:
        return "visual"
    return block_type.value


def _new_canonical_key(
    cluster: Sequence[ParserBlockObservation],
    selected: ParserBlockObservation,
) -> str:
    normalized_bbox = _bbox(selected.geometry)
    bbox_identity = (
        None
        if normalized_bbox is None
        else [round(value, 4) for value in normalized_bbox]
    )
    text = _normalized_text(selected)
    return stable_id(
        "canonical-key",
        selected.document_version_id,
        selected.page_id,
        _canonical_family(selected.block_type),
        bbox_identity,
        None if selected.reading_order is None else round(selected.reading_order, 3),
        content_sha256(text) if text else None,
    )


def _best_representation(
    entries: Sequence[tuple[ParserBlockObservation, Representation]],
    policy: ReconciliationPolicy,
) -> tuple[ParserBlockObservation, Representation]:
    return min(
        entries,
        key=lambda entry: (
            -(entry[1].confidence if entry[1].confidence is not None else 0.0),
            -_calibrated_confidence(entry[0], policy),
            *_priority_key(entry[0], policy),
            entry[1].representation_id,
        ),
    )


def _canonical_block(
    cluster: Sequence[ParserBlockObservation],
    *,
    pair_metadata: Mapping[tuple[str, str], Mapping[str, object]],
    uncertain_ids: set[str],
    conflict_ids: set[str],
    policy: ReconciliationPolicy,
    canonical_key: str | None = None,
    revision: int = 1,
) -> CanonicalBlock:
    ordered = sorted(cluster, key=lambda item: _priority_key(item, policy))
    anchor = ordered[0]
    selected = min(cluster, key=lambda item: _selection_key(item, policy))

    representation_entries: dict[
        str, list[tuple[ParserBlockObservation, Representation]]
    ] = defaultdict(list)
    for observation in ordered:
        for representation in observation.representations:
            representation_entries[representation.representation_id].append(
                (observation, representation)
            )
    chosen_representations = {
        representation_id: _best_representation(entries, policy)
        for representation_id, entries in representation_entries.items()
    }
    representations = [
        entry[1]
        for _, entry in sorted(
            chosen_representations.items(),
            key=lambda item: (
                item[1][0].observation_id != selected.observation_id,
                item[1][1].kind.value,
                item[0],
            ),
        )
    ]
    asset_entries: dict[
        str, list[tuple[ParserBlockObservation, AssetRef]]
    ] = defaultdict(list)
    for observation in ordered:
        for asset in observation.assets:
            asset_entries[asset.asset_id].append((observation, asset))
    assets = [
        min(
            entries,
            key=lambda entry: (
                -_calibrated_confidence(entry[0], policy),
                *_priority_key(entry[0], policy),
            ),
        )[1]
        for _, entries in sorted(asset_entries.items())
    ]

    semantic_entries: dict[
        str, list[tuple[ParserBlockObservation, SemanticSubstructure]]
    ] = defaultdict(list)
    for observation in ordered:
        for structure in observation.semantic_substructures:
            semantic_entries[structure.comparison_sha256].append(
                (observation, structure)
            )
    semantic_substructures: list[SemanticSubstructure] = []
    semantic_equivalence_sources: dict[str, dict[str, object]] = {}
    for comparison_sha, entries in sorted(semantic_entries.items()):
        first_comparison = entries[0][1].to_comparison_dict()
        if any(
            structure.to_comparison_dict() != first_comparison
            for _, structure in entries[1:]
        ):
            raise ReconciliationError(
                "semantic substructures share a comparison digest but differ "
                f"in structural content: {comparison_sha}"
            )
        exact_entries: dict[
            str, list[tuple[ParserBlockObservation, SemanticSubstructure]]
        ] = defaultdict(list)
        for observation, structure in entries:
            exact_entries[structure.semantic_sha256].append(
                (observation, structure)
            )
        for semantic_sha, exact_group in exact_entries.items():
            first_exact = exact_group[0][1].to_dict()
            if any(
                structure.to_dict() != first_exact
                for _, structure in exact_group[1:]
            ):
                raise ReconciliationError(
                    "semantic substructures share an exact digest but differ "
                    f"in source assertion content: {semantic_sha}"
                )

        representative_observation, representative = min(
            entries,
            key=lambda entry: (
                entry[0].observation_id != selected.observation_id,
                _selection_key(entry[0], policy),
                entry[1].semantic_sha256,
                entry[1].substructure_id,
            ),
        )
        representative_entries = exact_entries[representative.semantic_sha256]
        canonical_structure = representative.with_source_lineage(
            source_observation_ids=tuple(
                observation.observation_id
                for observation, _ in representative_entries
            ),
            source_substructure_ids=tuple(
                structure.substructure_id for _, structure in representative_entries
            ),
        )
        semantic_substructures.append(canonical_structure)
        if len(exact_entries) > 1:
            semantic_equivalence_sources[canonical_structure.substructure_id] = {
                "comparison_sha256": comparison_sha,
                "representative_observation_id": (
                    representative_observation.observation_id
                ),
                "representative_substructure_id": representative.substructure_id,
                "representative_semantic_sha256": representative.semantic_sha256,
                "source_assertions": [
                    {
                        "observation_id": observation.observation_id,
                        "substructure_id": structure.substructure_id,
                        "semantic_sha256": structure.semantic_sha256,
                        "comparison_sha256": structure.comparison_sha256,
                        "represented_by_canonical_graph": (
                            structure.semantic_sha256
                            == representative.semantic_sha256
                        ),
                    }
                    for observation, structure in sorted(
                        entries,
                        key=lambda entry: (
                            entry[0].observation_id,
                            entry[1].substructure_id,
                        ),
                    )
                ],
            }

    semantic_assertions_by_kind: dict[
        str, dict[str, tuple[str, ...]]
    ] = defaultdict(dict)
    semantic_exact_assertions_by_kind: dict[
        str, dict[str, tuple[str, ...]]
    ] = defaultdict(dict)
    for observation in ordered:
        local_by_kind: dict[str, set[str]] = defaultdict(set)
        local_exact_by_kind: dict[str, set[str]] = defaultdict(set)
        for structure in observation.semantic_substructures:
            local_by_kind[structure.kind.value].add(structure.comparison_sha256)
            local_exact_by_kind[structure.kind.value].add(
                structure.semantic_sha256
            )
        for structure_kind, digests in local_by_kind.items():
            semantic_assertions_by_kind[structure_kind][
                observation.observation_id
            ] = tuple(sorted(digests))
            semantic_exact_assertions_by_kind[structure_kind][
                observation.observation_id
            ] = tuple(sorted(local_exact_by_kind[structure_kind]))
    semantic_conflicts = tuple(
        {
            "kind": structure_kind,
            "observation_assertions": [
                {
                    "observation_id": observation_id,
                    "comparison_sha256": list(digests),
                    "semantic_sha256": list(
                        semantic_exact_assertions_by_kind[structure_kind][
                            observation_id
                        ]
                    ),
                }
                for observation_id, digests in sorted(assertions.items())
            ],
        }
        for structure_kind, assertions in sorted(
            semantic_assertions_by_kind.items()
        )
        if len(assertions) > 1 and len(set(assertions.values())) > 1
    )

    primary_representations: dict[str, str] = {}
    for representation_id, (source, representation) in sorted(
        chosen_representations.items()
    ):
        current_id = primary_representations.get(representation.kind.value)
        if current_id is None:
            primary_representations[representation.kind.value] = representation_id
            continue
        current_source, current_representation = chosen_representations[current_id]
        if _best_representation(
            ((current_source, current_representation), (source, representation)),
            policy,
        )[1].representation_id == representation_id:
            primary_representations[representation.kind.value] = representation_id

    source_ids = tuple(observation.observation_id for observation in ordered)
    relevant_pairs = [
        {
            "observation_ids": list(pair),
            **json_ready(metadata),
        }
        for pair, metadata in sorted(pair_metadata.items())
        if pair[0] in source_ids and pair[1] in source_ids
    ]
    status = _quality_status(
        cluster,
        selected=selected,
        uncertain_ids=uncertain_ids,
        conflict_ids=conflict_ids,
        policy=policy,
    )
    if semantic_conflicts and status is not QualityStatus.REJECTED:
        status = QualityStatus.NEEDS_REVIEW
    method = "single_observation" if len(cluster) == 1 else RECONCILIATION_METHOD
    return CanonicalBlock(
        canonical_key=canonical_key or _new_canonical_key(cluster, selected),
        revision=revision,
        document_version_id=anchor.document_version_id,
        page_id=anchor.page_id,
        page_index=anchor.page_index,
        block_type=selected.block_type,
        semantic_subtype=selected.semantic_subtype,
        geometry=selected.geometry or anchor.geometry,
        reading_order=(
            selected.reading_order
            if selected.reading_order is not None
            else anchor.reading_order
        ),
        representations=tuple(representations),
        assets=tuple(assets),
        semantic_substructures=tuple(semantic_substructures),
        source_observation_ids=source_ids,
        reconciliation_method=method,
        quality_status=status,
        metadata={
            "reconciliation_schema_version": RECONCILIATION_SCHEMA_VERSION,
            "policy_sha256": policy.policy_sha256,
            "anchor_observation_id": anchor.observation_id,
            "selected_observation_id": selected.observation_id,
            "primary_representations": primary_representations,
            "representation_sources": {
                representation_id: sorted(
                    {entry[0].observation_id for entry in entries}
                )
                for representation_id, entries in sorted(representation_entries.items())
            },
            "asset_sources": {
                asset_id: sorted({entry[0].observation_id for entry in entries})
                for asset_id, entries in sorted(asset_entries.items())
            },
            **(
                {
                    "semantic_substructure_sources": {
                        structure.substructure_id: {
                            "source_observation_ids": list(
                                structure.source_observation_ids
                            ),
                            "source_substructure_ids": list(
                                structure.source_substructure_ids
                            ),
                            "comparison_sha256": structure.comparison_sha256,
                            "semantic_sha256": structure.semantic_sha256,
                        }
                        for structure in semantic_substructures
                    }
                }
                if semantic_substructures
                else {}
            ),
            **(
                {
                    "semantic_substructure_equivalence_sources": (
                        semantic_equivalence_sources
                    )
                }
                if semantic_equivalence_sources
                else {}
            ),
            **(
                {"semantic_substructure_conflicts": list(semantic_conflicts)}
                if semantic_conflicts
                else {}
            ),
            "source_content_sha256": {
                observation.observation_id: observation.content_sha256
                for observation in ordered
            },
            "source_parsers": [
                {
                    "observation_id": observation.observation_id,
                    "parser_name": observation.provenance.parser_name,
                    "parser_version": observation.provenance.parser_version,
                    "model_version": observation.provenance.model_version,
                    "config_sha256": observation.provenance.config_sha256,
                    "parser_native_id": observation.provenance.parser_native_id,
                    "raw_artifact_uri": observation.provenance.raw_artifact_uri,
                    "transformations": list(
                        observation.provenance.transformations
                    ),
                }
                for observation in ordered
            ],
            "match_evidence": relevant_pairs,
        },
    )


def _copy_block_identity(
    block: CanonicalBlock,
    *,
    canonical_key: str,
    revision: int,
) -> CanonicalBlock:
    return CanonicalBlock(
        canonical_key=canonical_key,
        revision=revision,
        document_version_id=block.document_version_id,
        page_id=block.page_id,
        page_index=block.page_index,
        block_type=block.block_type,
        representations=block.representations,
        source_observation_ids=block.source_observation_ids,
        reconciliation_method=block.reconciliation_method,
        quality_status=block.quality_status,
        geometry=block.geometry,
        semantic_subtype=block.semantic_subtype,
        reading_order=block.reading_order,
        assets=block.assets,
        semantic_substructures=block.semantic_substructures,
        relations=block.relations,
        parent_block_id=block.parent_block_id,
        metadata=block.metadata,
    )


def _resolve_canonical_relations(
    blocks: Sequence[CanonicalBlock],
    observations: Mapping[str, ParserBlockObservation],
    prior_blocks: Sequence[CanonicalBlock],
) -> tuple[tuple[CanonicalBlock, ...], tuple[ReconciliationIssue, ...]]:
    """Resolve parser assertions after final canonical identities are known."""

    owner: dict[str, CanonicalBlock] = {}
    for block in blocks:
        for source_id in block.source_observation_ids:
            if source_id in owner:
                raise ReconciliationError(
                    "a source observation is owned by multiple canonical blocks"
                )
            owner[source_id] = block

    grouped: dict[
        str,
        dict[tuple[str, str], list[tuple[str, ObservationRelation]]],
    ] = defaultdict(lambda: defaultdict(list))
    affected_blocks: set[str] = set()
    issues: list[ReconciliationIssue] = []
    for source_id in sorted(owner):
        source_block = owner[source_id]
        observation = observations[source_id]
        for relation in observation.relations:
            target_block = owner.get(relation.target_observation_id)
            if target_block is None:
                affected_blocks.add(source_block.block_id)
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.UNRESOLVED_RELATION_TARGET,
                        message=(
                            "parser relation target is absent from the reconciled "
                            "observation set; the edge was not guessed"
                        ),
                        page_index=source_block.page_index,
                        observation_ids=(source_id,),
                        metadata={
                            "source_relation_id": relation.relation_id,
                            "relation_type": relation.relation_type.value,
                            "target_observation_id": relation.target_observation_id,
                        },
                    )
                )
                continue
            if target_block.block_id == source_block.block_id:
                affected_blocks.add(source_block.block_id)
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.COLLAPSED_RELATION_TARGET,
                        message=(
                            "parser relation endpoints reconciled to one canonical "
                            "block; a self-edge was not emitted"
                        ),
                        page_index=source_block.page_index,
                        observation_ids=(source_id, relation.target_observation_id),
                        metadata={
                            "source_relation_id": relation.relation_id,
                            "relation_type": relation.relation_type.value,
                            "canonical_block_id": source_block.block_id,
                        },
                    )
                )
                continue
            grouped[source_block.block_id][
                (relation.relation_type.value, target_block.block_id)
            ].append((source_id, relation))

    prior_by_key = {block.canonical_key: block for block in prior_blocks}
    resolved: list[CanonicalBlock] = []
    for block in blocks:
        canonical_relations: list[CanonicalRelation] = []
        for (_relation_type, target_block_id), assertions in sorted(
            grouped.get(block.block_id, {}).items()
        ):
            ordered_assertions = sorted(
                assertions, key=lambda item: (item[0], item[1].relation_id)
            )
            confidences = [
                relation.confidence for _, relation in ordered_assertions
            ]
            confidence = (
                max(value for value in confidences if value is not None)
                if any(value is not None for value in confidences)
                else None
            )
            first_relation = ordered_assertions[0][1]
            canonical_relations.append(
                CanonicalRelation(
                    relation_type=first_relation.relation_type,
                    target_block_id=target_block_id,
                    source_observation_ids=tuple(
                        source_id for source_id, _ in ordered_assertions
                    ),
                    source_relation_ids=tuple(
                        relation.relation_id
                        for _, relation in ordered_assertions
                    ),
                    confidence=confidence,
                    metadata={
                        "source_assertions": [
                            {
                                "source_observation_id": source_id,
                                "source_relation_id": relation.relation_id,
                                "confidence": relation.confidence,
                                "metadata": json_ready(relation.metadata),
                            }
                            for source_id, relation in ordered_assertions
                        ]
                    },
                )
            )
        status = (
            QualityStatus.NEEDS_REVIEW
            if block.block_id in affected_blocks
            and block.quality_status != QualityStatus.REJECTED
            else block.quality_status
        )
        candidate = CanonicalBlock(
            canonical_key=block.canonical_key,
            revision=block.revision,
            document_version_id=block.document_version_id,
            page_id=block.page_id,
            page_index=block.page_index,
            block_type=block.block_type,
            representations=block.representations,
            source_observation_ids=block.source_observation_ids,
            reconciliation_method=block.reconciliation_method,
            quality_status=status,
            geometry=block.geometry,
            semantic_subtype=block.semantic_subtype,
            reading_order=block.reading_order,
            assets=block.assets,
            semantic_substructures=block.semantic_substructures,
            relations=tuple(canonical_relations),
            parent_block_id=block.parent_block_id,
            metadata=block.metadata,
        )
        prior = prior_by_key.get(candidate.canonical_key)
        if prior is not None:
            probe = _copy_block_identity(
                candidate,
                canonical_key=candidate.canonical_key,
                revision=prior.revision,
            )
            revision = (
                prior.revision
                if probe.content_sha256 == prior.content_sha256
                else prior.revision + 1
            )
            candidate = _copy_block_identity(
                probe,
                canonical_key=probe.canonical_key,
                revision=revision,
            )
        resolved.append(candidate)
    return tuple(resolved), tuple(issues)


def _canonical_text(block: CanonicalBlock) -> str:
    return _normalize(
        " ".join(
            value
            for value in (_representation_text(item) for item in block.representations)
            if value
        )
    )


def _prior_match_score(
    current: CanonicalBlock,
    prior: CanonicalBlock,
    policy: ReconciliationPolicy,
    *,
    current_bbox: tuple[float, float, float, float] | None,
    prior_bbox: tuple[float, float, float, float] | None,
    current_text: str,
    prior_text: str,
    geometry_free_unique: bool,
) -> float | None:
    if current.page_id != prior.page_id or not _types_compatible_types(
        current.block_type, prior.block_type
    ):
        return None
    overlap = _iou(current_bbox, prior_bbox)
    similarity = _text_similarity(current_text, prior_text)
    same_type = current.block_type == prior.block_type
    threshold = policy.minimum_iou if same_type else policy.cross_type_iou
    if overlap is not None and overlap >= threshold:
        return min(0.80 * overlap + 0.20 * (similarity or 0.0), 1.0)
    if overlap is None and geometry_free_unique and similarity == 1.0 and same_type:
        return 1.0
    return None


def _compatible_identity_families(block_type: BlockType) -> frozenset[str]:
    """Return the finite type buckets that can retain this block's identity."""

    return frozenset(
        _canonical_family(candidate)
        for candidate in BlockType
        if _types_compatible_types(block_type, candidate)
    )


def _apply_prior_revisions(
    blocks: Sequence[CanonicalBlock],
    prior_blocks: Iterable[CanonicalBlock],
    *,
    document_version_id: str,
    policy: ReconciliationPolicy,
) -> tuple[tuple[CanonicalBlock, ...], tuple[ReconciliationIssue, ...]]:
    """Bind current blocks to prior identities with bounded page-local probes.

    Both current and prior block content/geometry are normalized once.  Spatial
    candidates are read from a page/type/grid index and exact-text candidates
    from a page/type index.  No current block scores more than the configured
    candidate limit.  Crowded or tied identity matches fail closed and are
    surfaced as reconciliation issues instead of receiving an arbitrary ID.
    """

    priors = tuple(
        sorted(
            prior_blocks,
            key=lambda block: (
                block.page_index if isinstance(block, CanonicalBlock) else -1,
                block.reading_order is None
                if isinstance(block, CanonicalBlock)
                else False,
                block.reading_order or 0.0
                if isinstance(block, CanonicalBlock)
                else 0.0,
                block.block_id if isinstance(block, CanonicalBlock) else "",
            ),
        )
    )
    if any(not isinstance(block, CanonicalBlock) for block in priors):
        raise TypeError("prior_blocks must contain CanonicalBlock DTOs")
    if any(block.document_version_id != document_version_id for block in priors):
        raise ReconciliationError("prior canonical blocks belong to another document")
    prior_ids = [block.block_id for block in priors]
    prior_keys = [block.canonical_key for block in priors]
    if len(prior_ids) != len(set(prior_ids)) or len(prior_keys) != len(set(prior_keys)):
        raise ReconciliationError("prior canonical blocks contain duplicate identities")

    prior_source_owner: dict[str, int] = {}
    for prior_index, prior in enumerate(priors):
        for source_id in prior.source_observation_ids:
            existing = prior_source_owner.get(source_id)
            if existing is not None:
                raise ReconciliationError(
                    "a prior source observation is owned by multiple canonical blocks: "
                    f"{source_id}"
                )
            prior_source_owner[source_id] = prior_index
    current_source_owner: dict[str, int] = {}
    for current_index, current in enumerate(blocks):
        for source_id in current.source_observation_ids:
            existing = current_source_owner.get(source_id)
            if existing is not None:
                raise ReconciliationError(
                    "a current source observation is owned by multiple canonical blocks: "
                    f"{source_id}"
                )
            current_source_owner[source_id] = current_index

    current_boxes = tuple(_bbox(block.geometry) for block in blocks)
    prior_boxes = tuple(_bbox(block.geometry) for block in priors)
    current_texts = tuple(_canonical_text(block) for block in blocks)
    prior_texts = tuple(_canonical_text(block) for block in priors)

    prior_grid: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    prior_exact_text: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    prior_geometry_free_text: dict[
        tuple[str, str, str], list[int]
    ] = defaultdict(list)
    for prior_index, prior in enumerate(priors):
        prior_bbox = prior_boxes[prior_index]
        if prior_bbox is not None:
            for x, y in _grid_cells(prior_bbox, policy.grid_size):
                prior_grid[
                    (prior.page_id, _canonical_family(prior.block_type), x, y)
                ].append(prior_index)
        prior_text = prior_texts[prior_index]
        if prior_text:
            exact_key = (prior.page_id, prior.block_type.value, prior_text)
            prior_exact_text[exact_key].append(prior_index)
            if prior_bbox is None:
                prior_geometry_free_text[exact_key].append(prior_index)

    current_exact_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for current_index, current in enumerate(blocks):
        current_text = current_texts[current_index]
        if current_text:
            current_exact_counts[
                (current.page_id, current.block_type.value, current_text)
            ] += 1

    # Immutable parser-observation identity is stronger evidence than fuzzy
    # geometry/text.  It preserves a canonical block through corrected content
    # or geometry while still incrementing the canonical revision below.
    exact_source_matches: dict[int, int] = {}
    ambiguity_reasons: dict[int, set[str]] = defaultdict(set)
    candidate_prior_ids: dict[int, set[str]] = defaultdict(set)
    for current_index, current in enumerate(blocks):
        owners = {
            prior_source_owner[source_id]
            for source_id in current.source_observation_ids
            if source_id in prior_source_owner
        }
        if len(owners) == 1:
            exact_source_matches[current_index] = next(iter(owners))
        elif len(owners) > 1:
            ambiguity_reasons[current_index].add(
                "current_sources_span_multiple_prior_blocks"
            )
            candidate_prior_ids[current_index].update(
                priors[index].block_id for index in owners
            )
    exact_by_prior: dict[int, list[int]] = defaultdict(list)
    for current_index, prior_index in exact_source_matches.items():
        exact_by_prior[prior_index].append(current_index)
    for prior_index, current_indices in exact_by_prior.items():
        if len(current_indices) < 2:
            continue
        for current_index in current_indices:
            ambiguity_reasons[current_index].add(
                "prior_sources_split_across_current_blocks"
            )
            candidate_prior_ids[current_index].add(priors[prior_index].block_id)
            exact_source_matches.pop(current_index, None)

    exact_assigned_priors = set(exact_source_matches.values())
    candidates: list[tuple[float, int, int]] = []
    issues: list[ReconciliationIssue] = []
    candidate_limit = policy.max_candidates_per_observation
    for current_index, current in enumerate(blocks):
        if current_index in exact_source_matches or current_index in ambiguity_reasons:
            continue
        current_bbox = current_boxes[current_index]
        current_text = current_texts[current_index]
        candidate_indices: set[int] = set()
        exceeded_limit = False

        if current_bbox is not None:
            for x, y in _grid_cells(current_bbox, policy.grid_size):
                for family in sorted(
                    _compatible_identity_families(current.block_type)
                ):
                    for prior_index in prior_grid.get(
                        (current.page_id, family, x, y), ()
                    ):
                        candidate_indices.add(prior_index)
                        if len(candidate_indices) > candidate_limit:
                            exceeded_limit = True
                            break
                    if exceeded_limit:
                        break
                if exceeded_limit:
                    break

        exact_key = (current.page_id, current.block_type.value, current_text)
        if not exceeded_limit and current_text:
            # Exact text adds identity evidence only when at least one side has
            # no geometry.  Two geometric blocks must share a spatial region;
            # indexing distant repeated headers/labels here creates false
            # candidate pressure without ever producing a valid match.
            exact_candidates = (
                prior_exact_text.get(exact_key, ())
                if current_bbox is None
                else prior_geometry_free_text.get(exact_key, ())
            )
            if len(exact_candidates) > candidate_limit:
                exceeded_limit = True
            else:
                candidate_indices.update(exact_candidates)
                exceeded_limit = len(candidate_indices) > candidate_limit

        if exceeded_limit:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.PRIOR_IDENTITY_CANDIDATE_LIMIT,
                    message=(
                        "prior identity candidate limit was exceeded; a new "
                        "identity was issued instead of guessing"
                    ),
                    page_index=current.page_index,
                    observation_ids=current.source_observation_ids,
                    metadata={
                        "candidate_limit": candidate_limit,
                        "current_block_id": current.block_id,
                    },
                )
            )
            continue

        geometry_free_unique = bool(
            current_text
            and current_exact_counts[exact_key] == 1
            and len(prior_exact_text.get(exact_key, ())) == 1
        )
        for prior_index in sorted(candidate_indices):
            if prior_index in exact_assigned_priors:
                continue
            prior = priors[prior_index]
            score = _prior_match_score(
                current,
                prior,
                policy,
                current_bbox=current_bbox,
                prior_bbox=prior_boxes[prior_index],
                current_text=current_text,
                prior_text=prior_texts[prior_index],
                geometry_free_unique=geometry_free_unique,
            )
            if score is not None:
                candidates.append((score, current_index, prior_index))

    # A tied best match has no defensible source identity.  Exclude every such
    # current block before the one-to-one assignment and make the ambiguity
    # visible to operators and downstream quality gates.
    by_current: dict[int, list[tuple[float, int]]] = defaultdict(list)
    by_prior: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for score, current_index, prior_index in candidates:
        by_current[current_index].append((score, prior_index))
        by_prior[prior_index].append((score, current_index))
    for current_index, values in by_current.items():
        best = max(score for score, _ in values)
        tied = [
            prior_index
            for score, prior_index in values
            if math.isclose(score, best, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(tied) > 1:
            ambiguity_reasons[current_index].add("multiple_equal_prior_matches")
            candidate_prior_ids[current_index].update(
                priors[index].block_id for index in tied
            )
    for prior_index, values in by_prior.items():
        best = max(score for score, _ in values)
        tied = [
            current_index
            for score, current_index in values
            if math.isclose(score, best, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(tied) > 1:
            for current_index in tied:
                ambiguity_reasons[current_index].add(
                    "multiple_current_blocks_compete_for_prior"
                )
                candidate_prior_ids[current_index].add(priors[prior_index].block_id)
    for current_index, reasons in sorted(ambiguity_reasons.items()):
        current = blocks[current_index]
        issues.append(
            ReconciliationIssue(
                code=ReconciliationIssueCode.AMBIGUOUS_PRIOR_IDENTITY,
                message=(
                    "prior identity match was ambiguous; a new identity was "
                    "issued instead of choosing by ordering"
                ),
                page_index=current.page_index,
                observation_ids=current.source_observation_ids,
                metadata={
                    "current_block_id": current.block_id,
                    "reasons": sorted(reasons),
                    "candidate_prior_block_ids": sorted(
                        candidate_prior_ids[current_index]
                    ),
                },
            )
        )

    assigned_current: dict[int, int] = dict(exact_source_matches)
    assigned_prior: set[int] = set(exact_source_matches.values())
    for score, current_index, prior_index in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            blocks[item[1]].block_id,
            priors[item[2]].block_id,
        ),
    ):
        if current_index in ambiguity_reasons:
            continue
        if current_index in assigned_current or prior_index in assigned_prior:
            continue
        assigned_current[current_index] = prior_index
        assigned_prior.add(prior_index)

    # Greedy fuzzy assignment can lose a one-to-one contest even when the
    # current block had plausible prior candidates.  Never let that identity
    # churn be silent; a later implementation may replace this with a
    # component-level maximum-weight matching policy without changing the
    # source-truth contract.
    for current_index, values in sorted(by_current.items()):
        if current_index in assigned_current or current_index in ambiguity_reasons:
            continue
        current = blocks[current_index]
        competing_prior_ids = sorted(
            {priors[prior_index].block_id for _, prior_index in values}
        )
        issues.append(
            ReconciliationIssue(
                code=ReconciliationIssueCode.AMBIGUOUS_PRIOR_IDENTITY,
                message=(
                    "fuzzy prior candidates were consumed by stronger one-to-one "
                    "matches; a new identity was issued"
                ),
                page_index=current.page_index,
                observation_ids=current.source_observation_ids,
                metadata={
                    "current_block_id": current.block_id,
                    "reasons": ["fuzzy_candidate_lost_one_to_one_assignment"],
                    "candidate_prior_block_ids": competing_prior_ids,
                },
            )
        )

    revised: list[CanonicalBlock] = []
    used_keys: set[str] = set()
    reserved_prior_keys = set(prior_keys)
    for index, block in enumerate(blocks):
        prior_index = assigned_current.get(index)
        if prior_index is None:
            key = block.canonical_key
            if key in reserved_prior_keys or key in used_keys:
                original_key = key
                collision_scope = (
                    "reserved_prior_identity"
                    if key in reserved_prior_keys
                    else "current_result_identity"
                )
                key = stable_id(
                    "canonical-key-collision",
                    key,
                    block.page_id,
                    block.source_observation_ids,
                    block.content_sha256,
                )
                collision_ordinal = 1
                while key in reserved_prior_keys or key in used_keys:
                    key = stable_id(
                        "canonical-key-collision",
                        block.canonical_key,
                        block.page_id,
                        block.source_observation_ids,
                        block.content_sha256,
                        collision_ordinal,
                    )
                    collision_ordinal += 1
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.CANONICAL_IDENTITY_COLLISION,
                        message=(
                            "canonical identity inputs were not unique; a "
                            "source-bound disambiguated key was issued"
                        ),
                        page_index=block.page_index,
                        observation_ids=block.source_observation_ids,
                        metadata={
                            "collision_scope": collision_scope,
                            "original_canonical_key": original_key,
                            "disambiguated_canonical_key": key,
                        },
                    )
                )
            current = _copy_block_identity(block, canonical_key=key, revision=1)
        else:
            prior = priors[prior_index]
            provisional = _copy_block_identity(
                block,
                canonical_key=prior.canonical_key,
                revision=prior.revision,
            )
            revision = (
                prior.revision
                if provisional.content_sha256 == prior.content_sha256
                else prior.revision + 1
            )
            current = _copy_block_identity(
                provisional,
                canonical_key=prior.canonical_key,
                revision=revision,
            )
        if current.canonical_key in used_keys:
            raise ReconciliationError("canonical identity assignment produced a collision")
        used_keys.add(current.canonical_key)
        revised.append(current)
    return tuple(revised), tuple(issues)


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    entries: tuple[tuple[int, tuple[str, ...]], ...]
    raw_probe_count: int
    plausible_count_lower_bound: int
    overflow_reason: str | None = None


def _bounded_candidates(
    index: int,
    *,
    ordered: Sequence[ParserBlockObservation],
    normalized_boxes: Sequence[tuple[float, float, float, float] | None],
    normalized_texts: Sequence[str],
    grid: Mapping[
        tuple[int, int, str], Mapping[str, Sequence[int]]
    ],
    exact_text: Mapping[
        tuple[str, str], Mapping[str, Sequence[int]]
    ],
    geometry_free_text: Mapping[
        tuple[str, str], Mapping[str, Sequence[int]]
    ],
    policy: ReconciliationPolicy,
) -> _CandidateSelection:
    """Return a bounded, parser-diverse candidate selection.

    Candidate buckets are partitioned by parser and coarse type family, so
    same-parser siblings and incompatible families consume no probe budget.
    The function stops at the first ``K+1`` plausible match or ``B+1`` unique
    raw probes and returns no candidates on overflow.  This fail-closed rule
    prevents an arbitrary prefix from becoming canonical lineage.
    """

    observation = ordered[index]
    bbox = normalized_boxes[index]
    text = normalized_texts[index]
    family = _canonical_family(observation.block_type)
    candidate_limit = policy.max_candidates_per_observation
    raw_probe_limit = policy.max_raw_candidate_probes_per_observation
    channels: dict[int, set[str]] = defaultdict(set)
    raw_seen: set[int] = set()

    compatible_families = _compatible_identity_families(observation.block_type)
    spatial_buckets: list[Mapping[str, Sequence[int]]] = []
    if bbox is not None:
        for cell in _grid_cells(bbox, policy.grid_size):
            for compatible_family in sorted(compatible_families):
                bucket = grid.get((*cell, compatible_family))
                if bucket:
                    spatial_buckets.append(bucket)
    exact_buckets: list[Mapping[str, Sequence[int]]] = []
    if text:
        exact_index = exact_text if bbox is None else geometry_free_text
        bucket = exact_index.get((family, text))
        if bucket:
            exact_buckets.append(bucket)

    def parser_names(
        buckets: Sequence[Mapping[str, Sequence[int]]],
    ) -> tuple[tuple[str, ...], bool]:
        values: set[str] = set()
        for bucket in buckets:
            for parser_name in bucket:
                if parser_name == observation.provenance.parser_name:
                    continue
                values.add(parser_name)
                if len(values) > raw_probe_limit:
                    return (), True
        return tuple(sorted(values)), False

    def plausible(candidate: int, source_channels: set[str]) -> bool:
        prior = ordered[candidate]
        if not _types_compatible(prior, observation):
            return False
        overlap = _iou(normalized_boxes[candidate], bbox)
        same_type = prior.block_type == observation.block_type
        threshold = policy.minimum_iou if same_type else policy.cross_type_iou
        if "spatial" in source_channels and overlap is not None and overlap >= threshold:
            return True
        if "exact_text" in source_channels and overlap is None:
            exact_match = (
                normalized_texts[candidate]
                and normalized_texts[candidate] == text
            )
            return bool(
                exact_match
                and (same_type or {prior.block_type, observation.block_type} <= _TEXT_TYPES)
            )
        return False

    def consume(
        buckets: Sequence[Mapping[str, Sequence[int]]], channel: str
    ) -> str | None:
        names, parser_overflow = parser_names(buckets)
        if parser_overflow:
            return "raw_probe_limit"
        for parser_name in names:
            for bucket in buckets:
                for candidate in bucket.get(parser_name, ()):
                    if candidate in raw_seen:
                        if candidate in channels:
                            channels[candidate].add(channel)
                        continue
                    raw_seen.add(candidate)
                    if len(raw_seen) > raw_probe_limit:
                        return "raw_probe_limit"
                    source_channels = {channel}
                    if plausible(candidate, source_channels):
                        channels[candidate].add(channel)
                        if len(channels) > candidate_limit:
                            return "plausible_candidate_limit"
        return None

    overflow_reason = consume(spatial_buckets, "spatial")
    if overflow_reason is None:
        overflow_reason = consume(exact_buckets, "exact_text")
    if overflow_reason is not None:
        plausible_lower_bound = (
            candidate_limit + 1
            if overflow_reason == "plausible_candidate_limit"
            else len(channels)
        )
        return _CandidateSelection(
            entries=(),
            raw_probe_count=min(len(raw_seen), raw_probe_limit + 1),
            plausible_count_lower_bound=plausible_lower_bound,
            overflow_reason=overflow_reason,
        )

    ranked_by_parser: dict[
        str, list[tuple[tuple[object, ...], int, tuple[str, ...]]]
    ] = defaultdict(list)
    for candidate, source_channels in channels.items():
        prior = ordered[candidate]
        overlap = _iou(normalized_boxes[candidate], bbox)
        exact = "exact_text" in source_channels
        spatial_tier = (
            3
            if overlap is not None and overlap >= policy.high_iou
            else 2
            if overlap is not None
            and overlap
            >= (
                policy.minimum_iou
                if prior.block_type == observation.block_type
                else policy.cross_type_iou
            )
            else 1
            if overlap is not None and overlap > 0
            else 0
        )
        priority = (
            spatial_tier,
            -1.0 if overlap is None else overlap,
            exact,
            prior.block_type == observation.block_type,
            -candidate,
        )
        ranked_by_parser[prior.provenance.parser_name].append(
            (priority, candidate, tuple(sorted(source_channels)))
        )

    for values in ranked_by_parser.values():
        values.sort(key=lambda item: item[0], reverse=True)

    # Reserve one slot per parser before filling by global score.  This avoids
    # a prolific parser starving independent corroboration from another model.
    selected: list[tuple[tuple[object, ...], int, tuple[str, ...]]] = []
    heads = sorted(
        (values[0] for values in ranked_by_parser.values() if values),
        key=lambda item: item[0],
        reverse=True,
    )
    selected.extend(heads[:candidate_limit])
    if len(selected) < candidate_limit:
        selected_ids = {item[1] for item in selected}
        remaining = sorted(
            (
                item
                for values in ranked_by_parser.values()
                for item in values
                if item[1] not in selected_ids
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        selected.extend(remaining[: candidate_limit - len(selected)])
    selected.sort(key=lambda item: ordered[item[1]].observation_id)
    return _CandidateSelection(
        entries=tuple((item[1], item[2]) for item in selected),
        raw_probe_count=len(raw_seen),
        plausible_count_lower_bound=len(channels),
    )


def reconcile_observations(
    observations: Iterable[ParserBlockObservation],
    *,
    policy: ReconciliationPolicy | None = None,
    prior_blocks: Iterable[CanonicalBlock] = (),
) -> ReconciliationResult:
    """Reconcile observations without losing parser-specific source evidence.

    Candidate generation uses a page-local grid and a bounded, geometry-ranked
    top-k per observation.  With ``n`` observations, ``p`` prior blocks,
    maximum grid span ``G``, semantic structures per observation ``S``,
    plausible-candidate cap ``K`` and raw-probe cap ``B``, the parameterized
    bound is ``O(content + n*S + m*S + (n+p) log(n+p) + (n+p)*G^2 +
    n*B*G^2 + m log m + m*K^2 + n*K log(n*K))`` for ``m`` retained parser
    candidate pairs; memory is
    ``O(content + n*S + (n+p)*G^2 + m + n*K + B)``.  ``S`` is
    contract-bounded at 128 and each observation's digest sets are built once.
    Crowded parser pages fail closed
    at ``K+1`` plausible candidates or ``B+1`` raw probes; prior identity
    probes fail closed at ``K+1``.  Both emit explicit review issues.
    Prior canonical revisions, when supplied, own stable block identity.
    """

    active_policy = policy or ReconciliationPolicy()
    prior_block_values = tuple(prior_blocks)
    unique: dict[str, ParserBlockObservation] = {}
    issues: list[ReconciliationIssue] = []
    for observation in observations:
        if not isinstance(observation, ParserBlockObservation):
            raise TypeError("observations must contain ParserBlockObservation DTOs")
        prior = unique.get(observation.observation_id)
        if prior is None:
            unique[observation.observation_id] = observation
        elif prior.content_sha256 == observation.content_sha256:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.DUPLICATE_OBSERVATION,
                    message="identical parser observation was supplied more than once",
                    page_index=observation.page_index,
                    observation_ids=(observation.observation_id,),
                )
            )
        else:
            raise ReconciliationError(
                "the same observation_id was supplied with different content"
            )
    ordered = sorted(
        unique.values(),
        key=lambda item: (item.page_index, item.reading_order is None, item.reading_order or 0.0, item.observation_id),
    )
    if not ordered:
        raise ReconciliationError("at least one parser observation is required")
    document_ids = {observation.document_version_id for observation in ordered}
    if len(document_ids) != 1:
        raise ReconciliationError("observations span multiple document versions")

    union_find = _UnionFind(len(ordered))
    for index, observation in enumerate(ordered):
        union_find.parsers[index].add(observation.provenance.parser_name)
    normalized_boxes = [_bbox(observation.geometry) for observation in ordered]
    normalized_texts = [_normalized_text(observation) for observation in ordered]
    semantic_assertions = [
        _semantic_comparison_assertions(observation) for observation in ordered
    ]
    pair_metadata: dict[tuple[str, str], Mapping[str, object]] = {}
    scored_pairs: list[tuple[float, int, int]] = []
    uncertain_ids: set[str] = set()
    conflict_ids: set[str] = set()
    candidate_pair_count = 0

    pages: dict[int, list[int]] = defaultdict(list)
    for index, observation in enumerate(ordered):
        pages[observation.page_index].append(index)

    text_occurrences: dict[
        tuple[int, str, str], dict[str, list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    keys_with_geometry_free: set[tuple[int, str, str]] = set()
    for index, observation in enumerate(ordered):
        text = normalized_texts[index]
        if not text:
            continue
        key = (
            observation.page_index,
            _canonical_family(observation.block_type),
            text,
        )
        text_occurrences[key][observation.provenance.parser_name].append(index)
        if normalized_boxes[index] is None:
            keys_with_geometry_free.add(key)
    ambiguous_geometry_free: set[int] = set()
    for (page_index, family, text), by_parser in sorted(
        text_occurrences.items(), key=lambda item: item[0]
    ):
        key = (page_index, family, text)
        indices = sorted(index for values in by_parser.values() for index in values)
        if key not in keys_with_geometry_free or len(indices) < 2:
            continue
        duplicate_parser_occurrence = any(
            len(values) > 1 for values in by_parser.values()
        )
        geometry_indices = [
            index for index in indices if normalized_boxes[index] is not None
        ]
        spatially_distinct_regions = False
        if not duplicate_parser_occurrence:
            for position, left_index in enumerate(geometry_indices):
                for right_index in geometry_indices[position + 1 :]:
                    left = ordered[left_index]
                    right = ordered[right_index]
                    overlap = _iou(
                        normalized_boxes[left_index], normalized_boxes[right_index]
                    )
                    threshold = (
                        active_policy.minimum_iou
                        if left.block_type == right.block_type
                        else active_policy.cross_type_iou
                    )
                    if overlap is None or overlap < threshold:
                        spatially_distinct_regions = True
                        break
                if spatially_distinct_regions:
                    break
        if not duplicate_parser_occurrence and not spatially_distinct_regions:
            continue
        ambiguous_geometry_free.update(indices)
        ids = tuple(ordered[index].observation_id for index in indices)
        uncertain_ids.update(ids)
        issues.append(
            ReconciliationIssue(
                code=ReconciliationIssueCode.AMBIGUOUS_GEOMETRY_FREE,
                message=(
                    "ambiguous repeated geometry-free text cannot be paired "
                    "without occurrence evidence"
                ),
                page_index=page_index,
                observation_ids=ids,
                metadata={
                    "family": family,
                    "text_sha256": content_sha256(text),
                    "occurrences_by_parser": {
                        parser: len(values) for parser, values in sorted(by_parser.items())
                    },
                    "duplicate_parser_occurrence": duplicate_parser_occurrence,
                    "spatially_distinct_regions": spatially_distinct_regions,
                },
            )
        )

    for page_index, page_indices in sorted(pages.items()):
        grid: dict[
            tuple[int, int, str], dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        exact_text: dict[
            tuple[str, str], dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        geometry_free_text: dict[
            tuple[str, str], dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        for index in page_indices:
            observation = ordered[index]
            bbox = normalized_boxes[index]
            text = normalized_texts[index]
            family = _canonical_family(observation.block_type)
            candidate_selection = _bounded_candidates(
                index,
                ordered=ordered,
                normalized_boxes=normalized_boxes,
                normalized_texts=normalized_texts,
                grid=grid,
                exact_text=exact_text,
                geometry_free_text=geometry_free_text,
                policy=active_policy,
            )
            candidate_entries = candidate_selection.entries
            if candidate_selection.overflow_reason is not None:
                uncertain_ids.add(observation.observation_id)
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.CANDIDATE_LIMIT,
                        message=(
                            "candidate probing exceeded a configured bound; the "
                            "affected observation was left unmerged for review"
                        ),
                        page_index=page_index,
                        observation_ids=(observation.observation_id,),
                        metadata={
                            "overflow_reason": candidate_selection.overflow_reason,
                            "raw_candidate_probe_count": (
                                candidate_selection.raw_probe_count
                            ),
                            "plausible_candidate_count_lower_bound": (
                                candidate_selection.plausible_count_lower_bound
                            ),
                            "retained_candidate_count": len(candidate_entries),
                            "candidate_limit": active_policy.max_candidates_per_observation,
                            "raw_probe_limit": (
                                active_policy.max_raw_candidate_probes_per_observation
                            ),
                        },
                    )
                )
            for candidate, channels in candidate_entries:
                candidate_pair_count += 1
                evaluation = _pair_evaluation(
                    ordered[candidate],
                    observation,
                    left_bbox=normalized_boxes[candidate],
                    right_bbox=bbox,
                    left_text=normalized_texts[candidate],
                    right_text=text,
                    policy=active_policy,
                    geometry_free_unique=(
                        candidate not in ambiguous_geometry_free
                        and index not in ambiguous_geometry_free
                    ),
                    left_semantic_assertions=semantic_assertions[candidate],
                    right_semantic_assertions=semantic_assertions[index],
                )
                if evaluation is None:
                    continue
                left_id, right_id = sorted(
                    (ordered[candidate].observation_id, observation.observation_id)
                )
                metadata = dict(evaluation.metadata)
                metadata["candidate_channels"] = list(channels)
                pair_metadata[(left_id, right_id)] = metadata
                scored_pairs.append((evaluation.score, candidate, index))
                if evaluation.conflict:
                    conflict_ids.update((left_id, right_id))
                    issues.append(
                        ReconciliationIssue(
                            code=ReconciliationIssueCode.CONFLICTING_OBSERVATION,
                            message=(
                                "parsers produced conflicting content for the same "
                                "page region"
                            ),
                            page_index=page_index,
                            observation_ids=(left_id, right_id),
                            metadata=metadata,
                        )
                    )

            if bbox is not None:
                for cell in _grid_cells(bbox, active_policy.grid_size):
                    grid[(*cell, family)][
                        observation.provenance.parser_name
                    ].append(index)
            if text:
                exact_text[(family, text)][
                    observation.provenance.parser_name
                ].append(index)
                if bbox is None:
                    geometry_free_text[(family, text)][
                        observation.provenance.parser_name
                    ].append(index)

    cluster_merge_count = 0
    for score, left, right in sorted(
        scored_pairs,
        key=lambda item: (
            -item[0],
            ordered[item[1]].observation_id,
            ordered[item[2]].observation_id,
        ),
    ):
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue
        # One observation per parser in a canonical cluster avoids collapsing
        # overlapping sibling blocks emitted by the same parser.
        if union_find.parsers[left_root] & union_find.parsers[right_root]:
            continue
        # Complete-link clustering prevents A~B~C transitive joins when A and C
        # have no direct evidence of representing the same source region.
        complete_link = True
        for left_member in union_find.members[left_root]:
            for right_member in union_find.members[right_root]:
                ids = tuple(
                    sorted(
                        (
                            ordered[left_member].observation_id,
                            ordered[right_member].observation_id,
                        )
                    )
                )
                if ids not in pair_metadata:
                    complete_link = False
                    break
            if not complete_link:
                break
        if not complete_link:
            continue
        union_find.union(left_root, right_root)
        cluster_merge_count += 1

    clusters = [
        [ordered[index] for index in sorted(union_find.members[root])]
        for root in range(len(ordered))
        if union_find.find(root) == root and union_find.members[root]
    ]
    blocks = [
        _canonical_block(
            cluster,
            pair_metadata=pair_metadata,
            uncertain_ids=uncertain_ids,
            conflict_ids=conflict_ids,
            policy=active_policy,
        )
        for cluster in clusters
    ]
    document_version_id = next(iter(document_ids))
    revised_blocks, prior_identity_issues = _apply_prior_revisions(
        blocks,
        prior_block_values,
        document_version_id=document_version_id,
        policy=active_policy,
    )
    related_blocks, relation_issues = _resolve_canonical_relations(
        revised_blocks,
        unique,
        prior_block_values,
    )
    blocks = list(related_blocks)
    issues.extend(prior_identity_issues)
    issues.extend(relation_issues)
    blocks.sort(
        key=lambda block: (
            block.page_index,
            block.reading_order is None,
            block.reading_order or 0.0,
            block.block_id,
        )
    )
    for block in blocks:
        if len({unique[source].block_type for source in block.source_observation_ids}) > 1:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.TYPE_DISAGREEMENT,
                    message="compatible parsers assigned different block types",
                    page_index=block.page_index,
                    observation_ids=block.source_observation_ids,
                    metadata={
                        "selected_type": block.block_type.value,
                        "quality_status": block.quality_status.value,
                    },
                )
            )
        if block.quality_status == QualityStatus.DEGRADED:
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.LOW_CONFIDENCE,
                    message=(
                        "selected canonical evidence has no calibrated confidence "
                        "or is below the acceptance threshold"
                    ),
                    page_index=block.page_index,
                    observation_ids=block.source_observation_ids,
                )
            )

    issues.sort(key=lambda issue: (issue.page_index, issue.code.value, issue.issue_id))
    return ReconciliationResult(
        document_version_id=document_version_id,
        policy_sha256=active_policy.policy_sha256,
        canonical_blocks=tuple(blocks),
        issues=tuple(issues),
        candidate_pair_count=candidate_pair_count,
        matched_pair_count=len(scored_pairs),
        cluster_merge_count=cluster_merge_count,
    )


__all__ = [
    "RECONCILIATION_METHOD",
    "RECONCILIATION_SCHEMA_VERSION",
    "ReconciliationError",
    "ReconciliationIssue",
    "ReconciliationIssueCode",
    "ReconciliationPolicy",
    "ReconciliationResult",
    "reconcile_observations",
]
