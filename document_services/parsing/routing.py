"""Deterministic, dependency-free page routing for parser adapters.

The router is an application component: it consumes cheap page-probe features
and parser capability manifests, but never imports or invokes a parser vendor.
It selects one primary adapter and, when policy allows, at most one conditional
fallback or adjudicator.  Execution and post-parse quality checks remain the
responsibility of the orchestration layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import ClassVar, Self

from document_services.domain.base import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    JsonDTOMixin,
    JsonObject,
    content_sha256,
    freeze_json,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_probability,
    require_schema_version,
    require_sequence,
    stable_id,
)
from document_services.domain.evidence import BlockType, PageRef
from document_services.domain.parsing import ParserCapabilityManifest, ParserModality
from document_services.ports.parser import ParserRegistry


class PageClass(StrEnum):
    """Coarse page class produced by deterministic probe rules."""

    NATIVE_TEXT = "native_text"
    TECHNICAL = "technical"
    VISUAL = "visual"
    AMBIGUOUS = "ambiguous"


class SecondaryRouteRole(StrEnum):
    """Why the optional second adapter may be invoked by orchestration."""

    FALLBACK = "fallback"
    ADJUDICATOR = "adjudicator"


class RouteReasonCode(StrEnum):
    """Stable machine-readable routing and candidate-audit reason codes."""

    STRONG_NATIVE_TEXT = "strong_native_text"
    TECHNICAL_STRUCTURE = "technical_structure"
    VISUAL_PAGE = "visual_page"
    LOW_NATIVE_TEXT = "low_native_text"
    HIGH_RASTER_COVERAGE = "high_raster_coverage"
    HIGH_TEXT_NOISE = "high_text_noise"
    HIGH_DUPLICATED_TEXT = "high_duplicated_text"
    HIGH_OCR_DISAGREEMENT = "high_ocr_disagreement"
    COMPLEX_COLUMNS = "complex_columns"
    SCANNED_PAGE = "scanned_page"
    SLIDE_LIKE_PAGE = "slide_like_page"
    PHOTOGRAPHED_PAGE = "photographed_page"
    AMBIGUOUS_PAGE = "ambiguous_page"
    PROBE_FEATURES_INCOMPLETE = "probe_features_incomplete"

    ADAPTER_NOT_REGISTERED = "adapter_not_registered"
    PROFILE_NOT_CONFIGURED = "profile_not_configured"
    PAGE_CLASS_NOT_CONFIGURED = "page_class_not_configured"
    MIME_TYPE_UNSUPPORTED = "mime_type_unsupported"
    PAGE_SELECTION_UNSUPPORTED = "page_selection_unsupported"
    REQUIRED_MODALITY_UNSUPPORTED = "required_modality_unsupported"
    TEXT_MODALITY_UNSUPPORTED = "text_modality_unsupported"
    REQUIRED_BLOCK_TYPE_UNSUPPORTED = "required_block_type_unsupported"
    PRIMARY_COST_BUDGET_EXCEEDED = "primary_cost_budget_exceeded"
    PRIMARY_LATENCY_BUDGET_EXCEEDED = "primary_latency_budget_exceeded"

    PRIMARY_SELECTED = "primary_selected"
    FALLBACK_PLANNED = "fallback_planned"
    ADJUDICATOR_PLANNED = "adjudicator_planned"
    SINGLE_CAPABLE_ADAPTER = "single_capable_adapter"
    SECONDARY_COST_BUDGET_EXCEEDED = "secondary_cost_budget_exceeded"
    SECONDARY_LATENCY_BUDGET_EXCEEDED = "secondary_latency_budget_exceeded"
    SECONDARY_DISABLED = "secondary_disabled"


def _enum(enum_type: type[StrEnum], value: object, field_name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(f"{field_name} must be one of: {allowed}") from exc


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_non_negative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field_name)


def _finite_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ContractValidationError(f"{field_name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ContractValidationError(f"{field_name} must be at most {maximum}")
    return result


def _optional_rank(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field_name)


@dataclass(frozen=True, slots=True)
class PageFeatures(JsonDTOMixin):
    """Cheap, model-free signals used to route one source-bound page."""

    page: PageRef
    mime_type: str
    native_character_count: int | None = None
    native_word_count: int | None = None
    text_coverage_ratio: float | None = None
    raster_image_coverage_ratio: float | None = None
    replacement_character_ratio: float | None = None
    control_character_ratio: float | None = None
    duplicated_text_ratio: float | None = None
    ocr_native_disagreement_ratio: float | None = None
    connected_component_density: float | None = None
    font_count: int | None = None
    vector_path_density: float | None = None
    skew_degrees: float | None = None
    candidate_table_regions: int | None = None
    candidate_equation_regions: int | None = None
    candidate_figure_regions: int | None = None
    candidate_chart_regions: int | None = None
    candidate_diagram_regions: int | None = None
    column_count: int | None = None
    is_scanned_page: bool | None = None
    is_slide_like: bool | None = None
    is_photographed_page: bool | None = None
    metadata: JsonObject = field(default_factory=dict)
    feature_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION
    NATIVE_STAT_FIELDS: ClassVar[tuple[str, ...]] = (
        "native_character_count",
        "native_word_count",
        "text_coverage_ratio",
        "raster_image_coverage_ratio",
        "replacement_character_ratio",
        "control_character_ratio",
        "duplicated_text_ratio",
        "ocr_native_disagreement_ratio",
        "connected_component_density",
        "font_count",
        "vector_path_density",
        "skew_degrees",
        "candidate_table_regions",
        "candidate_equation_regions",
        "candidate_figure_regions",
        "candidate_chart_regions",
        "candidate_diagram_regions",
        "column_count",
        "is_scanned_page",
        "is_slide_like",
        "is_photographed_page",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.page, PageRef):
            raise ContractValidationError("page must be PageRef")
        require_non_empty(self.mime_type, "mime_type")
        for field_name in (
            "native_character_count",
            "native_word_count",
            "font_count",
            "candidate_table_regions",
            "candidate_equation_regions",
            "candidate_figure_regions",
            "candidate_chart_regions",
            "candidate_diagram_regions",
        ):
            _optional_non_negative_int(getattr(self, field_name), field_name)
        if self.column_count is not None:
            _positive_int(self.column_count, "column_count")
        for field_name in (
            "text_coverage_ratio",
            "raster_image_coverage_ratio",
            "replacement_character_ratio",
            "control_character_ratio",
            "duplicated_text_ratio",
        ):
            require_probability(getattr(self, field_name), field_name)
        require_probability(
            self.ocr_native_disagreement_ratio, "ocr_native_disagreement_ratio"
        )
        component_density = (
            None
            if self.connected_component_density is None
            else _finite_number(
                self.connected_component_density,
                "connected_component_density",
                minimum=0.0,
            )
        )
        path_density = (
            None
            if self.vector_path_density is None
            else _finite_number(
                self.vector_path_density, "vector_path_density", minimum=0.0
            )
        )
        skew = (
            None
            if self.skew_degrees is None
            else _finite_number(
                self.skew_degrees,
                "skew_degrees",
                minimum=-90.0,
                maximum=90.0,
            )
        )
        for field_name in (
            "is_scanned_page",
            "is_slide_like",
            "is_photographed_page",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ContractValidationError(f"{field_name} must be a boolean or null")
        metadata = freeze_json(self.metadata, field="page_features.metadata")
        normalized_ratios = {
            field_name: (
                None
                if getattr(self, field_name) is None
                else float(getattr(self, field_name))
            )
            for field_name in (
                "text_coverage_ratio",
                "raster_image_coverage_ratio",
                "replacement_character_ratio",
                "control_character_ratio",
                "duplicated_text_ratio",
                "ocr_native_disagreement_ratio",
            )
        }
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "page": self.page.to_dict(),
            "mime_type": self.mime_type,
            "native_character_count": self.native_character_count,
            "native_word_count": self.native_word_count,
            "text_coverage_ratio": normalized_ratios["text_coverage_ratio"],
            "raster_image_coverage_ratio": normalized_ratios[
                "raster_image_coverage_ratio"
            ],
            "replacement_character_ratio": normalized_ratios[
                "replacement_character_ratio"
            ],
            "control_character_ratio": normalized_ratios[
                "control_character_ratio"
            ],
            "duplicated_text_ratio": normalized_ratios["duplicated_text_ratio"],
            "ocr_native_disagreement_ratio": normalized_ratios[
                "ocr_native_disagreement_ratio"
            ],
            "connected_component_density": component_density,
            "font_count": self.font_count,
            "vector_path_density": path_density,
            "skew_degrees": skew,
            "candidate_table_regions": self.candidate_table_regions,
            "candidate_equation_regions": self.candidate_equation_regions,
            "candidate_figure_regions": self.candidate_figure_regions,
            "candidate_chart_regions": self.candidate_chart_regions,
            "candidate_diagram_regions": self.candidate_diagram_regions,
            "column_count": self.column_count,
            "is_scanned_page": self.is_scanned_page,
            "is_slide_like": self.is_slide_like,
            "is_photographed_page": self.is_photographed_page,
            "metadata": metadata,
        }
        for field_name in (
            "text_coverage_ratio",
            "raster_image_coverage_ratio",
            "replacement_character_ratio",
            "control_character_ratio",
            "duplicated_text_ratio",
            "ocr_native_disagreement_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, float(value))
        object.__setattr__(self, "connected_component_density", component_density)
        object.__setattr__(self, "vector_path_density", path_density)
        object.__setattr__(self, "skew_degrees", skew)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "feature_id",
            stable_id("pagefeat", self.page.page_id, content_sha256(identity)),
        )

    @classmethod
    def from_page_ref(
        cls,
        page: PageRef,
        mime_type: str,
        **overrides: object,
    ) -> Self:
        """Build features from canonical ``PageRef.native_stats`` keys.

        Missing signals remain ``None`` (unknown), never guessed as zero.  Keys
        outside the routing contract are retained in metadata for diagnostics.
        Explicit keyword overrides take precedence over ``native_stats`` and
        are validated by the normal DTO constructor.
        """

        if not isinstance(page, PageRef):
            raise ContractValidationError("page must be PageRef")
        require_non_empty(mime_type, "mime_type")
        allowed_overrides = set(cls.NATIVE_STAT_FIELDS) | {"metadata"}
        unknown_overrides = sorted(set(overrides) - allowed_overrides)
        if unknown_overrides:
            raise ContractValidationError(
                "PageFeatures override has unknown field(s): "
                + ", ".join(unknown_overrides)
            )
        values = {
            key: page.native_stats[key]
            for key in cls.NATIVE_STAT_FIELDS
            if key in page.native_stats
        }
        values.update({key: value for key, value in overrides.items() if key != "metadata"})
        unmapped = {
            key: value
            for key, value in page.native_stats.items()
            if key not in cls.NATIVE_STAT_FIELDS
        }
        supplied_metadata = require_mapping(overrides.get("metadata", {}), "metadata")
        metadata: dict[str, object] = {
            "probe_source": "page_ref.native_stats",
            "unmapped_native_stats": json_ready(unmapped),
        }
        metadata.update(supplied_metadata)
        return cls(
            page=page,
            mime_type=mime_type,
            metadata=metadata,
            **values,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "feature_id": self.feature_id,
            "page": self.page.to_dict(),
            "mime_type": self.mime_type,
            "native_character_count": self.native_character_count,
            "native_word_count": self.native_word_count,
            "text_coverage_ratio": self.text_coverage_ratio,
            "raster_image_coverage_ratio": self.raster_image_coverage_ratio,
            "replacement_character_ratio": self.replacement_character_ratio,
            "control_character_ratio": self.control_character_ratio,
            "duplicated_text_ratio": self.duplicated_text_ratio,
            "ocr_native_disagreement_ratio": self.ocr_native_disagreement_ratio,
            "connected_component_density": self.connected_component_density,
            "font_count": self.font_count,
            "vector_path_density": self.vector_path_density,
            "skew_degrees": self.skew_degrees,
            "candidate_table_regions": self.candidate_table_regions,
            "candidate_equation_regions": self.candidate_equation_regions,
            "candidate_figure_regions": self.candidate_figure_regions,
            "candidate_chart_regions": self.candidate_chart_regions,
            "candidate_diagram_regions": self.candidate_diagram_regions,
            "column_count": self.column_count,
            "is_scanned_page": self.is_scanned_page,
            "is_slide_like": self.is_slide_like,
            "is_photographed_page": self.is_photographed_page,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        fields = {
            "schema_version",
            "feature_id",
            "page",
            "mime_type",
            "native_character_count",
            "native_word_count",
            "text_coverage_ratio",
            "raster_image_coverage_ratio",
            "replacement_character_ratio",
            "control_character_ratio",
            "duplicated_text_ratio",
            "ocr_native_disagreement_ratio",
            "connected_component_density",
            "font_count",
            "vector_path_density",
            "skew_degrees",
            "candidate_table_regions",
            "candidate_equation_regions",
            "candidate_figure_regions",
            "candidate_chart_regions",
            "candidate_diagram_regions",
            "column_count",
            "is_scanned_page",
            "is_slide_like",
            "is_photographed_page",
            "metadata",
        }
        reject_unknown_fields(payload, fields, dto="PageFeatures")
        result = cls(
            page=PageRef.from_dict(require_mapping(payload.get("page"), "page")),
            mime_type=payload.get("mime_type"),  # type: ignore[arg-type]
            native_character_count=payload.get("native_character_count"),  # type: ignore[arg-type]
            native_word_count=payload.get("native_word_count"),  # type: ignore[arg-type]
            text_coverage_ratio=payload.get("text_coverage_ratio"),  # type: ignore[arg-type]
            raster_image_coverage_ratio=payload.get("raster_image_coverage_ratio"),  # type: ignore[arg-type]
            replacement_character_ratio=payload.get("replacement_character_ratio"),  # type: ignore[arg-type]
            control_character_ratio=payload.get("control_character_ratio"),  # type: ignore[arg-type]
            duplicated_text_ratio=payload.get("duplicated_text_ratio"),  # type: ignore[arg-type]
            ocr_native_disagreement_ratio=payload.get("ocr_native_disagreement_ratio"),  # type: ignore[arg-type]
            connected_component_density=payload.get("connected_component_density"),  # type: ignore[arg-type]
            font_count=payload.get("font_count"),  # type: ignore[arg-type]
            vector_path_density=payload.get("vector_path_density"),  # type: ignore[arg-type]
            skew_degrees=payload.get("skew_degrees"),  # type: ignore[arg-type]
            candidate_table_regions=payload.get("candidate_table_regions"),  # type: ignore[arg-type]
            candidate_equation_regions=payload.get("candidate_equation_regions"),  # type: ignore[arg-type]
            candidate_figure_regions=payload.get("candidate_figure_regions"),  # type: ignore[arg-type]
            candidate_chart_regions=payload.get("candidate_chart_regions"),  # type: ignore[arg-type]
            candidate_diagram_regions=payload.get("candidate_diagram_regions"),  # type: ignore[arg-type]
            column_count=payload.get("column_count"),  # type: ignore[arg-type]
            is_scanned_page=payload.get("is_scanned_page"),  # type: ignore[arg-type]
            is_slide_like=payload.get("is_slide_like"),  # type: ignore[arg-type]
            is_photographed_page=payload.get("is_photographed_page"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "metadata"),
        )
        if payload.get("feature_id") != result.feature_id:
            raise ContractValidationError("PageFeatures feature_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class AdapterRoutingProfile(JsonDTOMixin):
    """Policy-owned ranking and bounded-work estimates for one adapter."""

    adapter_name: str
    native_text_rank: int | None
    technical_rank: int | None
    visual_rank: int | None
    ambiguous_rank: int | None
    estimated_latency_ms: int
    estimated_cost_microusd: int
    heavy: bool = False
    profile_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.adapter_name, "adapter_name")
        ranks: dict[PageClass, int | None] = {
            PageClass.NATIVE_TEXT: _optional_rank(
                self.native_text_rank, "native_text_rank"
            ),
            PageClass.TECHNICAL: _optional_rank(self.technical_rank, "technical_rank"),
            PageClass.VISUAL: _optional_rank(self.visual_rank, "visual_rank"),
            PageClass.AMBIGUOUS: _optional_rank(
                self.ambiguous_rank, "ambiguous_rank"
            ),
        }
        if all(rank is None for rank in ranks.values()):
            raise ContractValidationError("adapter profile must rank at least one page class")
        _non_negative_int(self.estimated_latency_ms, "estimated_latency_ms")
        _non_negative_int(self.estimated_cost_microusd, "estimated_cost_microusd")
        if not isinstance(self.heavy, bool):
            raise ContractValidationError("heavy must be a boolean")
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "adapter_name": self.adapter_name,
            "ranks": {key.value: value for key, value in ranks.items()},
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost_microusd": self.estimated_cost_microusd,
            "heavy": self.heavy,
        }
        object.__setattr__(
            self,
            "profile_id",
            stable_id("routeprof", self.adapter_name, content_sha256(identity)),
        )

    def rank_for(self, page_class: PageClass) -> int | None:
        return {
            PageClass.NATIVE_TEXT: self.native_text_rank,
            PageClass.TECHNICAL: self.technical_rank,
            PageClass.VISUAL: self.visual_rank,
            PageClass.AMBIGUOUS: self.ambiguous_rank,
        }[page_class]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "adapter_name": self.adapter_name,
            "native_text_rank": self.native_text_rank,
            "technical_rank": self.technical_rank,
            "visual_rank": self.visual_rank,
            "ambiguous_rank": self.ambiguous_rank,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost_microusd": self.estimated_cost_microusd,
            "heavy": self.heavy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "profile_id",
                "adapter_name",
                "native_text_rank",
                "technical_rank",
                "visual_rank",
                "ambiguous_rank",
                "estimated_latency_ms",
                "estimated_cost_microusd",
                "heavy",
            },
            dto="AdapterRoutingProfile",
        )
        result = cls(
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            native_text_rank=payload.get("native_text_rank"),  # type: ignore[arg-type]
            technical_rank=payload.get("technical_rank"),  # type: ignore[arg-type]
            visual_rank=payload.get("visual_rank"),  # type: ignore[arg-type]
            ambiguous_rank=payload.get("ambiguous_rank"),  # type: ignore[arg-type]
            estimated_latency_ms=payload.get("estimated_latency_ms"),  # type: ignore[arg-type]
            estimated_cost_microusd=payload.get("estimated_cost_microusd"),  # type: ignore[arg-type]
            heavy=payload.get("heavy", False),  # type: ignore[arg-type]
        )
        if payload.get("profile_id") != result.profile_id:
            raise ContractValidationError("AdapterRoutingProfile profile_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PageRoutingPolicy(JsonDTOMixin):
    """Versioned deterministic quality, cost, and latency routing policy."""

    policy_name: str
    policy_version: str
    adapter_profiles: tuple[AdapterRoutingProfile, ...]
    required_modalities: tuple[ParserModality, ...] = (ParserModality.LAYOUT,)
    required_block_types: tuple[BlockType, ...] = ()
    min_native_characters: int = 80
    min_native_words: int = 12
    min_native_text_coverage_ratio: float = 0.02
    max_native_noise_ratio: float = 0.03
    max_duplicated_text_ratio: float = 0.40
    min_visual_raster_coverage_ratio: float = 0.45
    max_visual_text_coverage_ratio: float = 0.10
    min_ocr_disagreement_ratio: float = 0.25
    technical_region_threshold: int = 1
    complex_column_count: int = 2
    max_primary_latency_ms: int | None = None
    max_primary_cost_microusd: int | None = None
    max_total_latency_ms: int | None = None
    max_total_cost_microusd: int | None = None
    allow_fallback: bool = True
    allow_adjudicator: bool = True
    prefer_non_heavy: bool = True
    policy_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.policy_name, "policy_name")
        require_non_empty(self.policy_version, "policy_version")
        profiles = tuple(sorted(self.adapter_profiles, key=lambda item: item.adapter_name))
        if not profiles:
            raise ContractValidationError("adapter_profiles must not be empty")
        if any(not isinstance(profile, AdapterRoutingProfile) for profile in profiles):
            raise ContractValidationError(
                "adapter_profiles must contain AdapterRoutingProfile DTOs"
            )
        names = [profile.adapter_name for profile in profiles]
        if len(names) != len(set(names)):
            raise ContractValidationError("adapter_profiles contains a duplicate adapter")
        modalities = tuple(
            sorted(
                {_enum(ParserModality, item, "required_modalities[]") for item in self.required_modalities},
                key=lambda item: item.value,
            )
        )
        block_types = tuple(
            sorted(
                {_enum(BlockType, item, "required_block_types[]") for item in self.required_block_types},
                key=lambda item: item.value,
            )
        )
        _non_negative_int(self.min_native_characters, "min_native_characters")
        _non_negative_int(self.min_native_words, "min_native_words")
        for field_name in (
            "min_native_text_coverage_ratio",
            "max_native_noise_ratio",
            "max_duplicated_text_ratio",
            "min_visual_raster_coverage_ratio",
            "max_visual_text_coverage_ratio",
            "min_ocr_disagreement_ratio",
        ):
            require_probability(getattr(self, field_name), field_name)
        _positive_int(self.technical_region_threshold, "technical_region_threshold")
        _positive_int(self.complex_column_count, "complex_column_count")
        for field_name in (
            "max_primary_latency_ms",
            "max_primary_cost_microusd",
            "max_total_latency_ms",
            "max_total_cost_microusd",
        ):
            _optional_non_negative_int(getattr(self, field_name), field_name)
        if (
            self.max_primary_latency_ms is not None
            and self.max_total_latency_ms is not None
            and self.max_total_latency_ms < self.max_primary_latency_ms
        ):
            raise ContractValidationError(
                "max_total_latency_ms must be at least max_primary_latency_ms"
            )
        if (
            self.max_primary_cost_microusd is not None
            and self.max_total_cost_microusd is not None
            and self.max_total_cost_microusd < self.max_primary_cost_microusd
        ):
            raise ContractValidationError(
                "max_total_cost_microusd must be at least max_primary_cost_microusd"
            )
        for field_name in ("allow_fallback", "allow_adjudicator", "prefer_non_heavy"):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "adapter_profiles": [item.to_dict() for item in profiles],
            "required_modalities": [item.value for item in modalities],
            "required_block_types": [item.value for item in block_types],
            "min_native_characters": self.min_native_characters,
            "min_native_words": self.min_native_words,
            "min_native_text_coverage_ratio": self.min_native_text_coverage_ratio,
            "max_native_noise_ratio": self.max_native_noise_ratio,
            "max_duplicated_text_ratio": self.max_duplicated_text_ratio,
            "min_visual_raster_coverage_ratio": self.min_visual_raster_coverage_ratio,
            "max_visual_text_coverage_ratio": self.max_visual_text_coverage_ratio,
            "min_ocr_disagreement_ratio": self.min_ocr_disagreement_ratio,
            "technical_region_threshold": self.technical_region_threshold,
            "complex_column_count": self.complex_column_count,
            "max_primary_latency_ms": self.max_primary_latency_ms,
            "max_primary_cost_microusd": self.max_primary_cost_microusd,
            "max_total_latency_ms": self.max_total_latency_ms,
            "max_total_cost_microusd": self.max_total_cost_microusd,
            "allow_fallback": self.allow_fallback,
            "allow_adjudicator": self.allow_adjudicator,
            "prefer_non_heavy": self.prefer_non_heavy,
        }
        object.__setattr__(self, "adapter_profiles", profiles)
        object.__setattr__(self, "required_modalities", modalities)
        object.__setattr__(self, "required_block_types", block_types)
        object.__setattr__(
            self,
            "policy_id",
            stable_id(
                "routepolicy",
                self.policy_name,
                self.policy_version,
                content_sha256(identity),
            ),
        )

    @property
    def profile_by_adapter(self) -> Mapping[str, AdapterRoutingProfile]:
        return {profile.adapter_name: profile for profile in self.adapter_profiles}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "adapter_profiles": [item.to_dict() for item in self.adapter_profiles],
            "required_modalities": [item.value for item in self.required_modalities],
            "required_block_types": [item.value for item in self.required_block_types],
            "min_native_characters": self.min_native_characters,
            "min_native_words": self.min_native_words,
            "min_native_text_coverage_ratio": self.min_native_text_coverage_ratio,
            "max_native_noise_ratio": self.max_native_noise_ratio,
            "max_duplicated_text_ratio": self.max_duplicated_text_ratio,
            "min_visual_raster_coverage_ratio": self.min_visual_raster_coverage_ratio,
            "max_visual_text_coverage_ratio": self.max_visual_text_coverage_ratio,
            "min_ocr_disagreement_ratio": self.min_ocr_disagreement_ratio,
            "technical_region_threshold": self.technical_region_threshold,
            "complex_column_count": self.complex_column_count,
            "max_primary_latency_ms": self.max_primary_latency_ms,
            "max_primary_cost_microusd": self.max_primary_cost_microusd,
            "max_total_latency_ms": self.max_total_latency_ms,
            "max_total_cost_microusd": self.max_total_cost_microusd,
            "allow_fallback": self.allow_fallback,
            "allow_adjudicator": self.allow_adjudicator,
            "prefer_non_heavy": self.prefer_non_heavy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        fields = {
            "schema_version",
            "policy_id",
            "policy_name",
            "policy_version",
            "adapter_profiles",
            "required_modalities",
            "required_block_types",
            "min_native_characters",
            "min_native_words",
            "min_native_text_coverage_ratio",
            "max_native_noise_ratio",
            "max_duplicated_text_ratio",
            "min_visual_raster_coverage_ratio",
            "max_visual_text_coverage_ratio",
            "min_ocr_disagreement_ratio",
            "technical_region_threshold",
            "complex_column_count",
            "max_primary_latency_ms",
            "max_primary_cost_microusd",
            "max_total_latency_ms",
            "max_total_cost_microusd",
            "allow_fallback",
            "allow_adjudicator",
            "prefer_non_heavy",
        }
        reject_unknown_fields(payload, fields, dto="PageRoutingPolicy")
        result = cls(
            policy_name=payload.get("policy_name"),  # type: ignore[arg-type]
            policy_version=payload.get("policy_version"),  # type: ignore[arg-type]
            adapter_profiles=tuple(
                AdapterRoutingProfile.from_dict(
                    require_mapping(item, f"adapter_profiles[{index}]")
                )
                for index, item in enumerate(
                    require_sequence(payload.get("adapter_profiles"), "adapter_profiles")
                )
            ),
            required_modalities=tuple(
                _enum(ParserModality, item, "required_modalities[]")
                for item in require_sequence(
                    payload.get("required_modalities", []), "required_modalities"
                )
            ),
            required_block_types=tuple(
                _enum(BlockType, item, "required_block_types[]")
                for item in require_sequence(
                    payload.get("required_block_types", []), "required_block_types"
                )
            ),
            min_native_characters=payload.get("min_native_characters", 80),  # type: ignore[arg-type]
            min_native_words=payload.get("min_native_words", 12),  # type: ignore[arg-type]
            min_native_text_coverage_ratio=payload.get("min_native_text_coverage_ratio", 0.02),  # type: ignore[arg-type]
            max_native_noise_ratio=payload.get("max_native_noise_ratio", 0.03),  # type: ignore[arg-type]
            max_duplicated_text_ratio=payload.get("max_duplicated_text_ratio", 0.40),  # type: ignore[arg-type]
            min_visual_raster_coverage_ratio=payload.get("min_visual_raster_coverage_ratio", 0.45),  # type: ignore[arg-type]
            max_visual_text_coverage_ratio=payload.get("max_visual_text_coverage_ratio", 0.10),  # type: ignore[arg-type]
            min_ocr_disagreement_ratio=payload.get("min_ocr_disagreement_ratio", 0.25),  # type: ignore[arg-type]
            technical_region_threshold=payload.get("technical_region_threshold", 1),  # type: ignore[arg-type]
            complex_column_count=payload.get("complex_column_count", 2),  # type: ignore[arg-type]
            max_primary_latency_ms=payload.get("max_primary_latency_ms"),  # type: ignore[arg-type]
            max_primary_cost_microusd=payload.get("max_primary_cost_microusd"),  # type: ignore[arg-type]
            max_total_latency_ms=payload.get("max_total_latency_ms"),  # type: ignore[arg-type]
            max_total_cost_microusd=payload.get("max_total_cost_microusd"),  # type: ignore[arg-type]
            allow_fallback=payload.get("allow_fallback", True),  # type: ignore[arg-type]
            allow_adjudicator=payload.get("allow_adjudicator", True),  # type: ignore[arg-type]
            prefer_non_heavy=payload.get("prefer_non_heavy", True),  # type: ignore[arg-type]
        )
        if payload.get("policy_id") != result.policy_id:
            raise ContractValidationError("PageRoutingPolicy policy_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RouteCandidate(JsonDTOMixin):
    """Auditable eligibility result for one configured or registered adapter."""

    adapter_name: str
    capability_id: str | None
    parser_name: str | None
    profile_id: str | None
    preference_rank: int | None
    estimated_latency_ms: int | None
    estimated_cost_microusd: int | None
    heavy: bool | None
    eligible: bool
    rejection_reason_codes: tuple[RouteReasonCode, ...] = ()
    candidate_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.adapter_name, "adapter_name")
        for field_name in ("capability_id", "parser_name", "profile_id"):
            value = getattr(self, field_name)
            if value is not None:
                require_non_empty(value, field_name)
        rank = _optional_rank(self.preference_rank, "preference_rank")
        latency = _optional_non_negative_int(
            self.estimated_latency_ms, "estimated_latency_ms"
        )
        cost = _optional_non_negative_int(
            self.estimated_cost_microusd, "estimated_cost_microusd"
        )
        if self.heavy is not None and not isinstance(self.heavy, bool):
            raise ContractValidationError("heavy must be a boolean or null")
        if not isinstance(self.eligible, bool):
            raise ContractValidationError("eligible must be a boolean")
        reasons = tuple(
            sorted(
                {
                    _enum(RouteReasonCode, item, "rejection_reason_codes[]")
                    for item in self.rejection_reason_codes
                },
                key=lambda item: item.value,
            )
        )
        complete = all(
            item is not None
            for item in (
                self.capability_id,
                self.parser_name,
                self.profile_id,
                rank,
                latency,
                cost,
                self.heavy,
            )
        )
        if self.eligible and (not complete or reasons):
            raise ContractValidationError(
                "eligible route candidate requires complete estimates and no rejections"
            )
        if not self.eligible and not reasons:
            raise ContractValidationError(
                "ineligible route candidate requires at least one rejection reason"
            )
        object.__setattr__(self, "rejection_reason_codes", reasons)
        object.__setattr__(
            self,
            "candidate_id",
            stable_id(
                "routecand",
                self.adapter_name,
                self.capability_id,
                self.profile_id,
                rank,
                latency,
                cost,
                self.heavy,
                self.eligible,
                [item.value for item in reasons],
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "adapter_name": self.adapter_name,
            "capability_id": self.capability_id,
            "parser_name": self.parser_name,
            "profile_id": self.profile_id,
            "preference_rank": self.preference_rank,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost_microusd": self.estimated_cost_microusd,
            "heavy": self.heavy,
            "eligible": self.eligible,
            "rejection_reason_codes": [
                item.value for item in self.rejection_reason_codes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "candidate_id",
                "adapter_name",
                "capability_id",
                "parser_name",
                "profile_id",
                "preference_rank",
                "estimated_latency_ms",
                "estimated_cost_microusd",
                "heavy",
                "eligible",
                "rejection_reason_codes",
            },
            dto="RouteCandidate",
        )
        result = cls(
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            capability_id=payload.get("capability_id"),  # type: ignore[arg-type]
            parser_name=payload.get("parser_name"),  # type: ignore[arg-type]
            profile_id=payload.get("profile_id"),  # type: ignore[arg-type]
            preference_rank=payload.get("preference_rank"),  # type: ignore[arg-type]
            estimated_latency_ms=payload.get("estimated_latency_ms"),  # type: ignore[arg-type]
            estimated_cost_microusd=payload.get("estimated_cost_microusd"),  # type: ignore[arg-type]
            heavy=payload.get("heavy"),  # type: ignore[arg-type]
            eligible=payload.get("eligible"),  # type: ignore[arg-type]
            rejection_reason_codes=tuple(
                _enum(RouteReasonCode, item, "rejection_reason_codes[]")
                for item in require_sequence(
                    payload.get("rejection_reason_codes", []),
                    "rejection_reason_codes",
                )
            ),
        )
        if payload.get("candidate_id") != result.candidate_id:
            raise ContractValidationError("RouteCandidate candidate_id mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RouteDecision(JsonDTOMixin):
    """Stable route plan that binds policy, probe, and capability revisions."""

    page_id: str
    feature_id: str
    policy_id: str
    page_class: PageClass
    primary_adapter_name: str
    primary_capability_id: str
    secondary_adapter_name: str | None
    secondary_capability_id: str | None
    secondary_role: SecondaryRouteRole | None
    required_modalities: tuple[ParserModality, ...]
    text_modality_any_of: tuple[ParserModality, ...]
    required_block_types: tuple[BlockType, ...]
    reason_codes: tuple[RouteReasonCode, ...]
    candidates: tuple[RouteCandidate, ...]
    decision_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "page_id",
            "feature_id",
            "policy_id",
            "primary_adapter_name",
            "primary_capability_id",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        page_class = _enum(PageClass, self.page_class, "page_class")
        secondary_role = (
            None
            if self.secondary_role is None
            else _enum(SecondaryRouteRole, self.secondary_role, "secondary_role")
        )
        secondary_values = (
            self.secondary_adapter_name,
            self.secondary_capability_id,
            secondary_role,
        )
        if any(item is None for item in secondary_values) and any(
            item is not None for item in secondary_values
        ):
            raise ContractValidationError(
                "secondary adapter, capability, and role must be supplied together"
            )
        if self.secondary_adapter_name == self.primary_adapter_name:
            raise ContractValidationError("primary and secondary adapters must differ")
        modalities = tuple(
            sorted(
                {_enum(ParserModality, item, "required_modalities[]") for item in self.required_modalities},
                key=lambda item: item.value,
            )
        )
        text_modalities = tuple(
            sorted(
                {_enum(ParserModality, item, "text_modality_any_of[]") for item in self.text_modality_any_of},
                key=lambda item: item.value,
            )
        )
        if any(
            item not in {ParserModality.NATIVE_TEXT, ParserModality.OCR_TEXT}
            for item in text_modalities
        ):
            raise ContractValidationError(
                "text_modality_any_of may contain native_text and ocr_text only"
            )
        block_types = tuple(
            sorted(
                {_enum(BlockType, item, "required_block_types[]") for item in self.required_block_types},
                key=lambda item: item.value,
            )
        )
        reasons = tuple(
            sorted(
                {_enum(RouteReasonCode, item, "reason_codes[]") for item in self.reason_codes},
                key=lambda item: item.value,
            )
        )
        candidates = tuple(sorted(self.candidates, key=lambda item: item.adapter_name))
        if any(not isinstance(candidate, RouteCandidate) for candidate in candidates):
            raise ContractValidationError("candidates must contain RouteCandidate DTOs")
        names = [candidate.adapter_name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ContractValidationError("candidates contains a duplicate adapter")
        by_name = {candidate.adapter_name: candidate for candidate in candidates}
        primary = by_name.get(self.primary_adapter_name)
        if (
            primary is None
            or not primary.eligible
            or primary.capability_id != self.primary_capability_id
        ):
            raise ContractValidationError(
                "primary adapter must reference an eligible audited candidate"
            )
        if self.secondary_adapter_name is not None:
            secondary = by_name.get(self.secondary_adapter_name)
            if (
                secondary is None
                or not secondary.eligible
                or secondary.capability_id != self.secondary_capability_id
            ):
                raise ContractValidationError(
                    "secondary adapter must reference an eligible audited candidate"
                )
        object.__setattr__(self, "page_class", page_class)
        object.__setattr__(self, "secondary_role", secondary_role)
        object.__setattr__(self, "required_modalities", modalities)
        object.__setattr__(self, "text_modality_any_of", text_modalities)
        object.__setattr__(self, "required_block_types", block_types)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "decision_id",
            stable_id(
                "route",
                self.page_id,
                self.feature_id,
                self.policy_id,
                page_class.value,
                self.primary_adapter_name,
                self.primary_capability_id,
                self.secondary_adapter_name,
                self.secondary_capability_id,
                None if secondary_role is None else secondary_role.value,
                [item.value for item in modalities],
                [item.value for item in text_modalities],
                [item.value for item in block_types],
                [item.value for item in reasons],
                [item.candidate_id for item in candidates],
            ),
        )

    @property
    def fallback_adapter_name(self) -> str | None:
        return (
            self.secondary_adapter_name
            if self.secondary_role is SecondaryRouteRole.FALLBACK
            else None
        )

    @property
    def adjudicator_adapter_name(self) -> str | None:
        return (
            self.secondary_adapter_name
            if self.secondary_role is SecondaryRouteRole.ADJUDICATOR
            else None
        )

    @property
    def planned_adapter_names(self) -> tuple[str, ...]:
        if self.secondary_adapter_name is None:
            return (self.primary_adapter_name,)
        return (self.primary_adapter_name, self.secondary_adapter_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "page_id": self.page_id,
            "feature_id": self.feature_id,
            "policy_id": self.policy_id,
            "page_class": self.page_class.value,
            "primary_adapter_name": self.primary_adapter_name,
            "primary_capability_id": self.primary_capability_id,
            "secondary_adapter_name": self.secondary_adapter_name,
            "secondary_capability_id": self.secondary_capability_id,
            "secondary_role": (
                None if self.secondary_role is None else self.secondary_role.value
            ),
            "required_modalities": [item.value for item in self.required_modalities],
            "text_modality_any_of": [item.value for item in self.text_modality_any_of],
            "required_block_types": [item.value for item in self.required_block_types],
            "reason_codes": [item.value for item in self.reason_codes],
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "decision_id",
                "page_id",
                "feature_id",
                "policy_id",
                "page_class",
                "primary_adapter_name",
                "primary_capability_id",
                "secondary_adapter_name",
                "secondary_capability_id",
                "secondary_role",
                "required_modalities",
                "text_modality_any_of",
                "required_block_types",
                "reason_codes",
                "candidates",
            },
            dto="RouteDecision",
        )
        role = payload.get("secondary_role")
        result = cls(
            page_id=payload.get("page_id"),  # type: ignore[arg-type]
            feature_id=payload.get("feature_id"),  # type: ignore[arg-type]
            policy_id=payload.get("policy_id"),  # type: ignore[arg-type]
            page_class=_enum(PageClass, payload.get("page_class"), "page_class"),
            primary_adapter_name=payload.get("primary_adapter_name"),  # type: ignore[arg-type]
            primary_capability_id=payload.get("primary_capability_id"),  # type: ignore[arg-type]
            secondary_adapter_name=payload.get("secondary_adapter_name"),  # type: ignore[arg-type]
            secondary_capability_id=payload.get("secondary_capability_id"),  # type: ignore[arg-type]
            secondary_role=(
                None
                if role is None
                else _enum(SecondaryRouteRole, role, "secondary_role")
            ),
            required_modalities=tuple(
                _enum(ParserModality, item, "required_modalities[]")
                for item in require_sequence(
                    payload.get("required_modalities", []), "required_modalities"
                )
            ),
            text_modality_any_of=tuple(
                _enum(ParserModality, item, "text_modality_any_of[]")
                for item in require_sequence(
                    payload.get("text_modality_any_of", []), "text_modality_any_of"
                )
            ),
            required_block_types=tuple(
                _enum(BlockType, item, "required_block_types[]")
                for item in require_sequence(
                    payload.get("required_block_types", []), "required_block_types"
                )
            ),
            reason_codes=tuple(
                _enum(RouteReasonCode, item, "reason_codes[]")
                for item in require_sequence(
                    payload.get("reason_codes", []), "reason_codes"
                )
            ),
            candidates=tuple(
                RouteCandidate.from_dict(require_mapping(item, f"candidates[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("candidates", []), "candidates")
                )
            ),
        )
        if payload.get("decision_id") != result.decision_id:
            raise ContractValidationError("RouteDecision decision_id mismatch")
        return result


class NoCapableParserError(LookupError):
    """No registered adapter satisfies capability, profile, and budget gates."""

    def __init__(self, page_id: str, candidates: tuple[RouteCandidate, ...]) -> None:
        self.page_id = page_id
        self.candidates = candidates
        details = "; ".join(
            f"{candidate.adapter_name}="
            + ",".join(reason.value for reason in candidate.rejection_reason_codes)
            for candidate in candidates
        )
        if not details:
            details = "no registered adapters or configured profiles"
        super().__init__(f"no capable parser for page {page_id}: {details}")


@dataclass(frozen=True, slots=True)
class _Requirements:
    modalities: frozenset[ParserModality]
    text_any_of: frozenset[ParserModality]
    block_types: frozenset[BlockType]


@dataclass(frozen=True, slots=True)
class _ClassifiedPage:
    page_class: PageClass
    reason_codes: tuple[RouteReasonCode, ...]
    ambiguous: bool


class DeterministicPageRouter:
    """Route pages from immutable manifests without calling parser runtimes."""

    def __init__(self, registry: ParserRegistry, policy: PageRoutingPolicy) -> None:
        if not isinstance(registry, ParserRegistry):
            raise TypeError("registry must be ParserRegistry")
        if not isinstance(policy, PageRoutingPolicy):
            raise TypeError("policy must be PageRoutingPolicy")
        self._registry = registry
        self._policy = policy

    @property
    def policy(self) -> PageRoutingPolicy:
        return self._policy

    def route(self, features: PageFeatures) -> RouteDecision:
        if not isinstance(features, PageFeatures):
            raise TypeError("features must be PageFeatures")
        classified = self._classify(features)
        requirements = self._requirements(features, classified.page_class)
        candidates, manifests = self._candidates(
            features, classified.page_class, requirements
        )
        eligible = [candidate for candidate in candidates if candidate.eligible]
        if not eligible:
            raise NoCapableParserError(features.page.page_id, candidates)
        eligible.sort(key=self._rank_key)
        primary = eligible[0]
        primary_manifest = manifests[primary.adapter_name]
        secondary, secondary_role, secondary_reasons = self._secondary(
            primary,
            primary_manifest,
            eligible[1:],
            manifests,
            classified,
        )
        reasons = list(classified.reason_codes)
        reasons.append(RouteReasonCode.PRIMARY_SELECTED)
        reasons.extend(secondary_reasons)
        if secondary is None and len(eligible) == 1:
            reasons.append(RouteReasonCode.SINGLE_CAPABLE_ADAPTER)
        return RouteDecision(
            page_id=features.page.page_id,
            feature_id=features.feature_id,
            policy_id=self._policy.policy_id,
            page_class=classified.page_class,
            primary_adapter_name=primary.adapter_name,
            primary_capability_id=primary.capability_id or "",
            secondary_adapter_name=(None if secondary is None else secondary.adapter_name),
            secondary_capability_id=(None if secondary is None else secondary.capability_id),
            secondary_role=secondary_role,
            required_modalities=tuple(requirements.modalities),
            text_modality_any_of=tuple(requirements.text_any_of),
            required_block_types=tuple(requirements.block_types),
            reason_codes=tuple(reasons),
            candidates=candidates,
        )

    def _classify(self, features: PageFeatures) -> _ClassifiedPage:
        policy = self._policy
        reasons: list[RouteReasonCode] = []
        required_probe_values = (
            features.native_character_count,
            features.native_word_count,
            features.text_coverage_ratio,
            features.raster_image_coverage_ratio,
            features.replacement_character_ratio,
            features.control_character_ratio,
            features.duplicated_text_ratio,
        )
        incomplete = any(value is None for value in required_probe_values)
        if incomplete:
            reasons.append(RouteReasonCode.PROBE_FEATURES_INCOMPLETE)
        noise = (
            None
            if features.replacement_character_ratio is None
            or features.control_character_ratio is None
            else features.replacement_character_ratio
            + features.control_character_ratio
        )
        strong_native = (
            features.native_character_count is not None
            and features.native_character_count >= policy.min_native_characters
            and features.native_word_count is not None
            and features.native_word_count >= policy.min_native_words
            and features.text_coverage_ratio is not None
            and features.text_coverage_ratio >= policy.min_native_text_coverage_ratio
            and noise is not None
            and noise <= policy.max_native_noise_ratio
            and features.duplicated_text_ratio is not None
            and features.duplicated_text_ratio <= policy.max_duplicated_text_ratio
        )
        if strong_native:
            reasons.append(RouteReasonCode.STRONG_NATIVE_TEXT)
        elif (
            features.native_character_count is not None
            and features.native_character_count < policy.min_native_characters
        ) or (
            features.native_word_count is not None
            and features.native_word_count < policy.min_native_words
        ) or (
            features.text_coverage_ratio is not None
            and features.text_coverage_ratio < policy.min_native_text_coverage_ratio
        ):
            reasons.append(RouteReasonCode.LOW_NATIVE_TEXT)
        if noise is not None and noise > policy.max_native_noise_ratio:
            reasons.append(RouteReasonCode.HIGH_TEXT_NOISE)
        if (
            features.duplicated_text_ratio is not None
            and features.duplicated_text_ratio > policy.max_duplicated_text_ratio
        ):
            reasons.append(RouteReasonCode.HIGH_DUPLICATED_TEXT)
        disagreement = (
            features.ocr_native_disagreement_ratio is not None
            and features.ocr_native_disagreement_ratio
            >= policy.min_ocr_disagreement_ratio
        )
        if disagreement:
            reasons.append(RouteReasonCode.HIGH_OCR_DISAGREEMENT)
        if (
            features.raster_image_coverage_ratio is not None
            and features.raster_image_coverage_ratio
            >= policy.min_visual_raster_coverage_ratio
        ):
            reasons.append(RouteReasonCode.HIGH_RASTER_COVERAGE)
        if features.is_scanned_page:
            reasons.append(RouteReasonCode.SCANNED_PAGE)
        if features.is_slide_like:
            reasons.append(RouteReasonCode.SLIDE_LIKE_PAGE)
        if features.is_photographed_page:
            reasons.append(RouteReasonCode.PHOTOGRAPHED_PAGE)
        if (
            features.column_count is not None
            and features.column_count >= policy.complex_column_count
        ):
            reasons.append(RouteReasonCode.COMPLEX_COLUMNS)

        visual = (
            features.is_scanned_page is True
            or features.is_slide_like is True
            or features.is_photographed_page is True
            or (
                features.raster_image_coverage_ratio is not None
                and features.raster_image_coverage_ratio
                >= policy.min_visual_raster_coverage_ratio
                and (
                    not strong_native
                    or (
                        features.text_coverage_ratio is not None
                        and features.text_coverage_ratio
                        <= policy.max_visual_text_coverage_ratio
                    )
                )
            )
        )
        technical_region_values = (
            features.candidate_table_regions,
            features.candidate_equation_regions,
            features.candidate_chart_regions,
            features.candidate_diagram_regions,
        )
        known_technical_regions = tuple(
            value for value in technical_region_values if value is not None
        )
        technical = (
            bool(known_technical_regions)
            and sum(known_technical_regions)
            >= policy.technical_region_threshold
        ) or (
            features.column_count is not None
            and features.column_count >= policy.complex_column_count
        )
        ambiguous = bool(
            incomplete
            or disagreement
            or (
                noise is not None and noise > policy.max_native_noise_ratio
            )
            or (
                features.duplicated_text_ratio is not None
                and features.duplicated_text_ratio
                > policy.max_duplicated_text_ratio
            )
        )
        if incomplete:
            page_class = PageClass.AMBIGUOUS
            ambiguous = True
            reasons.append(RouteReasonCode.AMBIGUOUS_PAGE)
        elif visual:
            page_class = PageClass.VISUAL
            reasons.append(RouteReasonCode.VISUAL_PAGE)
        elif technical:
            page_class = PageClass.TECHNICAL
            reasons.append(RouteReasonCode.TECHNICAL_STRUCTURE)
        elif strong_native and not ambiguous:
            page_class = PageClass.NATIVE_TEXT
        else:
            page_class = PageClass.AMBIGUOUS
            ambiguous = True
            reasons.append(RouteReasonCode.AMBIGUOUS_PAGE)
        return _ClassifiedPage(
            page_class,
            tuple(sorted(set(reasons), key=lambda item: item.value)),
            ambiguous,
        )

    def _requirements(
        self, features: PageFeatures, page_class: PageClass
    ) -> _Requirements:
        modalities = set(self._policy.required_modalities)
        block_types = set(self._policy.required_block_types)
        if page_class is PageClass.VISUAL:
            modalities.add(ParserModality.OCR_TEXT)
            text_any_of: set[ParserModality] = set()
        else:
            text_any_of = {
                ParserModality.NATIVE_TEXT,
                ParserModality.OCR_TEXT,
            }
        if features.candidate_table_regions:
            modalities.add(ParserModality.TABLE)
            block_types.add(BlockType.TABLE)
        if features.candidate_equation_regions:
            modalities.add(ParserModality.FORMULA)
            block_types.add(BlockType.EQUATION)
        if features.candidate_figure_regions:
            modalities.add(ParserModality.IMAGE)
            block_types.add(BlockType.FIGURE)
        if features.candidate_chart_regions:
            modalities.add(ParserModality.CHART)
            block_types.add(BlockType.CHART)
        if features.candidate_diagram_regions:
            modalities.add(ParserModality.DIAGRAM)
            block_types.add(BlockType.DIAGRAM)
        return _Requirements(
            frozenset(modalities), frozenset(text_any_of), frozenset(block_types)
        )

    def _candidates(
        self,
        features: PageFeatures,
        page_class: PageClass,
        requirements: _Requirements,
    ) -> tuple[tuple[RouteCandidate, ...], Mapping[str, ParserCapabilityManifest]]:
        manifests = {
            manifest.adapter_name: manifest for manifest in self._registry.manifests()
        }
        profiles = self._policy.profile_by_adapter
        adapter_names = sorted(set(manifests) | set(profiles))
        candidates: list[RouteCandidate] = []
        for adapter_name in adapter_names:
            manifest = manifests.get(adapter_name)
            profile = profiles.get(adapter_name)
            reasons: list[RouteReasonCode] = []
            if manifest is None:
                reasons.append(RouteReasonCode.ADAPTER_NOT_REGISTERED)
            if profile is None:
                reasons.append(RouteReasonCode.PROFILE_NOT_CONFIGURED)
            rank = None if profile is None else profile.rank_for(page_class)
            if profile is not None and rank is None:
                reasons.append(RouteReasonCode.PAGE_CLASS_NOT_CONFIGURED)
            if manifest is not None:
                if features.mime_type not in manifest.supported_mime_types:
                    reasons.append(RouteReasonCode.MIME_TYPE_UNSUPPORTED)
                if not manifest.supports_page_selection:
                    reasons.append(RouteReasonCode.PAGE_SELECTION_UNSUPPORTED)
                if not requirements.modalities.issubset(manifest.modalities):
                    reasons.append(RouteReasonCode.REQUIRED_MODALITY_UNSUPPORTED)
                if requirements.text_any_of and not requirements.text_any_of.intersection(
                    manifest.modalities
                ):
                    reasons.append(RouteReasonCode.TEXT_MODALITY_UNSUPPORTED)
                if not requirements.block_types.issubset(manifest.block_types):
                    reasons.append(RouteReasonCode.REQUIRED_BLOCK_TYPE_UNSUPPORTED)
            if profile is not None:
                primary_latency_budget = self._effective_primary_latency_budget()
                primary_cost_budget = self._effective_primary_cost_budget()
                if (
                    primary_latency_budget is not None
                    and profile.estimated_latency_ms > primary_latency_budget
                ):
                    reasons.append(RouteReasonCode.PRIMARY_LATENCY_BUDGET_EXCEEDED)
                if (
                    primary_cost_budget is not None
                    and profile.estimated_cost_microusd > primary_cost_budget
                ):
                    reasons.append(RouteReasonCode.PRIMARY_COST_BUDGET_EXCEEDED)
            candidates.append(
                RouteCandidate(
                    adapter_name=adapter_name,
                    capability_id=(None if manifest is None else manifest.capability_id),
                    parser_name=(None if manifest is None else manifest.parser_name),
                    profile_id=(None if profile is None else profile.profile_id),
                    preference_rank=rank,
                    estimated_latency_ms=(
                        None if profile is None else profile.estimated_latency_ms
                    ),
                    estimated_cost_microusd=(
                        None if profile is None else profile.estimated_cost_microusd
                    ),
                    heavy=None if profile is None else profile.heavy,
                    eligible=not reasons,
                    rejection_reason_codes=tuple(reasons),
                )
            )
        return tuple(candidates), manifests

    def _effective_primary_latency_budget(self) -> int | None:
        values = [
            value
            for value in (
                self._policy.max_primary_latency_ms,
                self._policy.max_total_latency_ms,
            )
            if value is not None
        ]
        return min(values) if values else None

    def _effective_primary_cost_budget(self) -> int | None:
        values = [
            value
            for value in (
                self._policy.max_primary_cost_microusd,
                self._policy.max_total_cost_microusd,
            )
            if value is not None
        ]
        return min(values) if values else None

    def _rank_key(self, candidate: RouteCandidate) -> tuple[object, ...]:
        return (
            candidate.preference_rank,
            bool(candidate.heavy) if self._policy.prefer_non_heavy else False,
            candidate.estimated_cost_microusd,
            candidate.estimated_latency_ms,
            candidate.adapter_name,
            candidate.capability_id,
        )

    def _secondary(
        self,
        primary: RouteCandidate,
        primary_manifest: ParserCapabilityManifest,
        alternatives: Sequence[RouteCandidate],
        manifests: Mapping[str, ParserCapabilityManifest],
        classified: _ClassifiedPage,
    ) -> tuple[
        RouteCandidate | None,
        SecondaryRouteRole | None,
        tuple[RouteReasonCode, ...],
    ]:
        role: SecondaryRouteRole | None
        if classified.ambiguous:
            role = (
                SecondaryRouteRole.ADJUDICATOR
                if self._policy.allow_adjudicator
                else None
            )
        else:
            role = (
                SecondaryRouteRole.FALLBACK if self._policy.allow_fallback else None
            )
        if role is None:
            return None, None, (RouteReasonCode.SECONDARY_DISABLED,)
        ordered = sorted(
            alternatives,
            key=lambda candidate: (
                manifests[candidate.adapter_name].parser_name
                == primary_manifest.parser_name,
                *self._rank_key(candidate),
            ),
        )
        budget_reasons: set[RouteReasonCode] = set()
        for candidate in ordered:
            if (
                self._policy.max_total_latency_ms is not None
                and (primary.estimated_latency_ms or 0)
                + (candidate.estimated_latency_ms or 0)
                > self._policy.max_total_latency_ms
            ):
                budget_reasons.add(
                    RouteReasonCode.SECONDARY_LATENCY_BUDGET_EXCEEDED
                )
                continue
            if (
                self._policy.max_total_cost_microusd is not None
                and (primary.estimated_cost_microusd or 0)
                + (candidate.estimated_cost_microusd or 0)
                > self._policy.max_total_cost_microusd
            ):
                budget_reasons.add(RouteReasonCode.SECONDARY_COST_BUDGET_EXCEEDED)
                continue
            planned_reason = (
                RouteReasonCode.ADJUDICATOR_PLANNED
                if role is SecondaryRouteRole.ADJUDICATOR
                else RouteReasonCode.FALLBACK_PLANNED
            )
            return candidate, role, (planned_reason,)
        return None, None, tuple(sorted(budget_reasons, key=lambda item: item.value))


__all__ = [
    "AdapterRoutingProfile",
    "DeterministicPageRouter",
    "NoCapableParserError",
    "PageClass",
    "PageFeatures",
    "PageRoutingPolicy",
    "RouteCandidate",
    "RouteDecision",
    "RouteReasonCode",
    "SecondaryRouteRole",
]
