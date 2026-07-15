"""Maintainable composition service for routed, auditable document parsing.

This module coordinates existing parser ports, deterministic page routing, and
canonical reconciliation.  It deliberately contains no parser-vendor or
BookRAG imports.  Execution is sequential and batches are explicitly bounded;
resource-aware worker scheduling can therefore be added outside this service
without changing its source/evidence contracts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from docarag.domain.base import (
    ContractValidationError,
    JsonDTOMixin,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_probability,
    require_sequence,
    require_sha256,
    stable_id,
)
from docarag.domain.evidence import (
    CanonicalBlock,
    DocumentRef,
    QualityStatus,
)
from docarag.domain.parsing import (
    IssueSeverity,
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParseStatus,
)
from docarag.ports.parser import ParserRegistry
from docarag.reconciliation import (
    ReconciliationPolicy,
    ReconciliationResult,
    reconcile_observations,
)
from docarag.routing import (
    DeterministicPageRouter,
    PageClass,
    PageFeatures,
    RouteDecision,
    SecondaryRouteRole,
)


ORCHESTRATION_SCHEMA_VERSION = "docarag.parsing-orchestration.v1"


class OrchestrationContractError(ContractValidationError):
    """Composition inputs or route/registry bindings are inconsistent."""


class UnsupportedOrchestrationSchemaError(ContractValidationError):
    """A serialized orchestration audit uses an unsupported schema."""


class InvocationRole(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    ADJUDICATOR = "adjudicator"


class InvocationTrigger(StrEnum):
    PRIMARY_ROUTE = "primary_route"
    PRIMARY_FAILED = "primary_failed"
    PRIMARY_LOW_QUALITY = "primary_low_quality"
    PRIMARY_MISSING_OUTPUT = "primary_missing_output"
    AMBIGUOUS_ADJUDICATION = "ambiguous_adjudication"


class PageParseQuality(StrEnum):
    ACCEPTED = "accepted"
    LOW_QUALITY = "low_quality"
    MISSING_OUTPUT = "missing_output"
    FAILED = "failed"


class PageResolution(StrEnum):
    RESOLVED = "resolved"
    NEEDS_REVIEW = "needs_review"
    UNRESOLVED = "unresolved"


class RoutedParsingStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class OrchestrationReasonCode(StrEnum):
    PRIMARY_ACCEPTED = "primary_accepted"
    PRIMARY_LOW_QUALITY = "primary_low_quality"
    PRIMARY_MISSING_OUTPUT = "primary_missing_output"
    PRIMARY_FAILED = "primary_failed"
    SECONDARY_ACCEPTED = "secondary_accepted"
    SECONDARY_LOW_QUALITY = "secondary_low_quality"
    SECONDARY_MISSING_OUTPUT = "secondary_missing_output"
    SECONDARY_FAILED = "secondary_failed"
    SECONDARY_NOT_REQUIRED = "secondary_not_required"
    NO_SECONDARY_AVAILABLE = "no_secondary_available"
    NO_CANONICAL_OUTPUT = "no_canonical_output"
    CANONICAL_DEGRADED = "canonical_degraded"
    CANONICAL_NEEDS_REVIEW = "canonical_needs_review"
    RECONCILIATION_ISSUE = "reconciliation_issue"
    RECONCILIATION_FAILED = "reconciliation_failed"
    AMBIGUITY_UNRESOLVED = "ambiguity_unresolved"


def _require_schema(payload: Mapping[str, object]) -> None:
    actual = payload.get("schema_version")
    if actual != ORCHESTRATION_SCHEMA_VERSION:
        raise UnsupportedOrchestrationSchemaError(
            "unsupported orchestration schema_version "
            f"{actual!r}; expected {ORCHESTRATION_SCHEMA_VERSION!r}"
        )


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


@dataclass(frozen=True, slots=True)
class ParsingOrchestrationPolicy(JsonDTOMixin):
    """Bounded execution and adapter-independent page quality gates."""

    max_batch_pages: int = 16
    minimum_observations_per_page: int = 1
    minimum_explicit_confidence: float = 0.50
    require_expected_block_types: bool = True
    review_degraded_canonical: bool = True
    review_reconciliation_issues: bool = True
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.max_batch_pages, "max_batch_pages")
        if self.max_batch_pages > 1024:
            raise ContractValidationError("max_batch_pages must not exceed 1024")
        _positive_int(
            self.minimum_observations_per_page,
            "minimum_observations_per_page",
        )
        require_probability(
            self.minimum_explicit_confidence,
            "minimum_explicit_confidence",
        )
        confidence = float(self.minimum_explicit_confidence)
        for field_name in (
            "require_expected_block_types",
            "review_degraded_canonical",
            "review_reconciliation_issues",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractValidationError(f"{field_name} must be a boolean")
        object.__setattr__(self, "minimum_explicit_confidence", confidence)
        object.__setattr__(
            self,
            "policy_id",
            stable_id(
                "parse-orchestration-policy",
                self.max_batch_pages,
                self.minimum_observations_per_page,
                confidence,
                self.require_expected_block_types,
                self.review_degraded_canonical,
                self.review_reconciliation_issues,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "max_batch_pages": self.max_batch_pages,
            "minimum_observations_per_page": self.minimum_observations_per_page,
            "minimum_explicit_confidence": self.minimum_explicit_confidence,
            "require_expected_block_types": self.require_expected_block_types,
            "review_degraded_canonical": self.review_degraded_canonical,
            "review_reconciliation_issues": self.review_reconciliation_issues,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "policy_id",
                "max_batch_pages",
                "minimum_observations_per_page",
                "minimum_explicit_confidence",
                "require_expected_block_types",
                "review_degraded_canonical",
                "review_reconciliation_issues",
            },
            dto="ParsingOrchestrationPolicy",
        )
        result = cls(
            max_batch_pages=payload.get("max_batch_pages"),  # type: ignore[arg-type]
            minimum_observations_per_page=payload.get("minimum_observations_per_page"),  # type: ignore[arg-type]
            minimum_explicit_confidence=payload.get("minimum_explicit_confidence"),  # type: ignore[arg-type]
            require_expected_block_types=payload.get("require_expected_block_types"),  # type: ignore[arg-type]
            review_degraded_canonical=payload.get("review_degraded_canonical"),  # type: ignore[arg-type]
            review_reconciliation_issues=payload.get("review_reconciliation_issues"),  # type: ignore[arg-type]
        )
        if payload.get("policy_id") != result.policy_id:
            raise ContractValidationError(
                "ParsingOrchestrationPolicy policy_id mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class ParseInvocationAudit(JsonDTOMixin):
    """Exact request/result or captured boundary failure for one adapter call."""

    sequence: int
    role: InvocationRole
    adapter_name: str
    capability_id: str
    triggers: tuple[InvocationTrigger, ...]
    request: ParseRequest
    result: ParseResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    invocation_id: str = field(init=False)

    def __post_init__(self) -> None:
        _non_negative_int(self.sequence, "sequence")
        role = _enum(InvocationRole, self.role, "role")
        require_non_empty(self.adapter_name, "adapter_name")
        require_non_empty(self.capability_id, "capability_id")
        triggers = tuple(
            sorted(
                {
                    _enum(InvocationTrigger, item, "triggers[]")
                    for item in self.triggers
                },
                key=lambda item: item.value,
            )
        )
        if not triggers:
            raise ContractValidationError("triggers must not be empty")
        if role is InvocationRole.PRIMARY and triggers != (
            InvocationTrigger.PRIMARY_ROUTE,
        ):
            raise ContractValidationError(
                "primary invocation must use the primary_route trigger"
            )
        if role is InvocationRole.ADJUDICATOR and triggers != (
            InvocationTrigger.AMBIGUOUS_ADJUDICATION,
        ):
            raise ContractValidationError(
                "adjudicator invocation must use the ambiguous trigger"
            )
        if role is InvocationRole.FALLBACK and any(
            trigger
            not in {
                InvocationTrigger.PRIMARY_FAILED,
                InvocationTrigger.PRIMARY_LOW_QUALITY,
                InvocationTrigger.PRIMARY_MISSING_OUTPUT,
            }
            for trigger in triggers
        ):
            raise ContractValidationError("fallback invocation has an invalid trigger")
        if not isinstance(self.request, ParseRequest):
            raise ContractValidationError("request must be ParseRequest")
        if self.result is None:
            require_non_empty(self.error_type, "error_type")
            require_non_empty(self.error_message, "error_message")
        else:
            if not isinstance(self.result, ParseResult):
                raise ContractValidationError("result must be ParseResult or null")
            if self.result.request_id != self.request.request_id:
                raise OrchestrationContractError(
                    "invocation result belongs to a different request"
                )
            if self.result.adapter_name != self.adapter_name:
                raise OrchestrationContractError(
                    "invocation result belongs to a different adapter"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ContractValidationError(
                    "successful invocation audit cannot also contain an exception"
                )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "triggers", triggers)
        object.__setattr__(
            self,
            "invocation_id",
            stable_id(
                "parse-invocation",
                self.sequence,
                role.value,
                self.adapter_name,
                self.capability_id,
                [item.value for item in triggers],
                self.request.request_id,
                None if self.result is None else self.result.result_sha256,
                self.error_type,
                self.error_message,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "invocation_id": self.invocation_id,
            "sequence": self.sequence,
            "role": self.role.value,
            "adapter_name": self.adapter_name,
            "capability_id": self.capability_id,
            "triggers": [item.value for item in self.triggers],
            "request": self.request.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "invocation_id",
                "sequence",
                "role",
                "adapter_name",
                "capability_id",
                "triggers",
                "request",
                "result",
                "error_type",
                "error_message",
            },
            dto="ParseInvocationAudit",
        )
        result_payload = payload.get("result")
        result = cls(
            sequence=payload.get("sequence"),  # type: ignore[arg-type]
            role=_enum(InvocationRole, payload.get("role"), "role"),
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            capability_id=payload.get("capability_id"),  # type: ignore[arg-type]
            triggers=tuple(
                _enum(InvocationTrigger, item, "triggers[]")
                for item in require_sequence(payload.get("triggers"), "triggers")
            ),
            request=ParseRequest.from_dict(
                require_mapping(payload.get("request"), "request")
            ),
            result=(
                None
                if result_payload is None
                else ParseResult.from_dict(require_mapping(result_payload, "result"))
            ),
            error_type=payload.get("error_type"),  # type: ignore[arg-type]
            error_message=payload.get("error_message"),  # type: ignore[arg-type]
        )
        if payload.get("invocation_id") != result.invocation_id:
            raise ContractValidationError("ParseInvocationAudit invocation_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class PageParsingAudit(JsonDTOMixin):
    """Route, invocation, evidence, and final disposition for one page."""

    features: PageFeatures
    route: RouteDecision
    primary_invocation_id: str
    primary_quality: PageParseQuality
    secondary_invocation_id: str | None
    secondary_quality: PageParseQuality | None
    observation_ids: tuple[str, ...]
    canonical_revision_ids: tuple[str, ...]
    resolution: PageResolution
    reason_codes: tuple[OrchestrationReasonCode, ...]
    page_audit_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.features, PageFeatures):
            raise ContractValidationError("features must be PageFeatures")
        if not isinstance(self.route, RouteDecision):
            raise ContractValidationError("route must be RouteDecision")
        if self.route.page_id != self.features.page.page_id:
            raise OrchestrationContractError("route belongs to a different page")
        if self.route.feature_id != self.features.feature_id:
            raise OrchestrationContractError("route belongs to different page features")
        require_non_empty(self.primary_invocation_id, "primary_invocation_id")
        primary_quality = _enum(
            PageParseQuality, self.primary_quality, "primary_quality"
        )
        secondary_quality = (
            None
            if self.secondary_quality is None
            else _enum(
                PageParseQuality,
                self.secondary_quality,
                "secondary_quality",
            )
        )
        if (self.secondary_invocation_id is None) != (secondary_quality is None):
            raise ContractValidationError(
                "secondary invocation and quality must be supplied together"
            )
        if self.secondary_invocation_id is not None:
            require_non_empty(self.secondary_invocation_id, "secondary_invocation_id")
            if self.route.secondary_adapter_name is None:
                raise OrchestrationContractError(
                    "page audit invokes a secondary absent from its route"
                )
        if (
            self.route.secondary_role is SecondaryRouteRole.ADJUDICATOR
            and self.secondary_invocation_id is None
        ):
            raise OrchestrationContractError(
                "adjudicator route requires a secondary invocation"
            )
        if self.route.secondary_role is SecondaryRouteRole.FALLBACK:
            if (
                primary_quality is PageParseQuality.ACCEPTED
                and self.secondary_invocation_id is not None
            ):
                raise OrchestrationContractError(
                    "accepted primary must not eagerly invoke fallback"
                )
            if (
                primary_quality is not PageParseQuality.ACCEPTED
                and self.secondary_invocation_id is None
            ):
                raise OrchestrationContractError(
                    "failed/low/missing primary requires its planned fallback"
                )
        observation_ids = tuple(sorted(set(self.observation_ids)))
        canonical_revision_ids = tuple(sorted(set(self.canonical_revision_ids)))
        for field_name, values in (
            ("observation_ids", observation_ids),
            ("canonical_revision_ids", canonical_revision_ids),
        ):
            for value in values:
                require_non_empty(value, f"{field_name}[]")
        resolution = _enum(PageResolution, self.resolution, "resolution")
        if resolution is PageResolution.UNRESOLVED and canonical_revision_ids:
            raise ContractValidationError(
                "unresolved page cannot reference canonical output"
            )
        if resolution is not PageResolution.UNRESOLVED and not canonical_revision_ids:
            raise ContractValidationError(
                "resolved/review page requires canonical output"
            )
        reason_codes = tuple(
            sorted(
                {
                    _enum(OrchestrationReasonCode, item, "reason_codes[]")
                    for item in self.reason_codes
                },
                key=lambda item: item.value,
            )
        )
        if not reason_codes:
            raise ContractValidationError("reason_codes must not be empty")
        expected_primary_reason = {
            PageParseQuality.ACCEPTED: OrchestrationReasonCode.PRIMARY_ACCEPTED,
            PageParseQuality.LOW_QUALITY: OrchestrationReasonCode.PRIMARY_LOW_QUALITY,
            PageParseQuality.MISSING_OUTPUT: OrchestrationReasonCode.PRIMARY_MISSING_OUTPUT,
            PageParseQuality.FAILED: OrchestrationReasonCode.PRIMARY_FAILED,
        }[primary_quality]
        if expected_primary_reason not in reason_codes:
            raise ContractValidationError(
                "reason_codes does not describe primary_quality"
            )
        if secondary_quality is not None:
            expected_secondary_reason = {
                PageParseQuality.ACCEPTED: OrchestrationReasonCode.SECONDARY_ACCEPTED,
                PageParseQuality.LOW_QUALITY: OrchestrationReasonCode.SECONDARY_LOW_QUALITY,
                PageParseQuality.MISSING_OUTPUT: OrchestrationReasonCode.SECONDARY_MISSING_OUTPUT,
                PageParseQuality.FAILED: OrchestrationReasonCode.SECONDARY_FAILED,
            }[secondary_quality]
            if expected_secondary_reason not in reason_codes:
                raise ContractValidationError(
                    "reason_codes does not describe secondary_quality"
                )
        if resolution is PageResolution.UNRESOLVED and (
            OrchestrationReasonCode.NO_CANONICAL_OUTPUT not in reason_codes
        ):
            raise ContractValidationError(
                "unresolved page requires the no_canonical_output reason"
            )
        if resolution is not PageResolution.UNRESOLVED and (
            OrchestrationReasonCode.NO_CANONICAL_OUTPUT in reason_codes
        ):
            raise ContractValidationError(
                "page with canonical output cannot use no_canonical_output"
            )
        object.__setattr__(self, "primary_quality", primary_quality)
        object.__setattr__(self, "secondary_quality", secondary_quality)
        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "canonical_revision_ids", canonical_revision_ids)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(
            self,
            "page_audit_id",
            stable_id(
                "page-parse-audit",
                self.features.feature_id,
                self.route.decision_id,
                self.primary_invocation_id,
                primary_quality.value,
                self.secondary_invocation_id,
                None if secondary_quality is None else secondary_quality.value,
                observation_ids,
                canonical_revision_ids,
                resolution.value,
                [item.value for item in reason_codes],
            ),
        )

    @property
    def page_id(self) -> str:
        return self.features.page.page_id

    @property
    def page_index(self) -> int:
        return self.features.page.page_index

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "page_audit_id": self.page_audit_id,
            "features": self.features.to_dict(),
            "route": self.route.to_dict(),
            "primary_invocation_id": self.primary_invocation_id,
            "primary_quality": self.primary_quality.value,
            "secondary_invocation_id": self.secondary_invocation_id,
            "secondary_quality": (
                None if self.secondary_quality is None else self.secondary_quality.value
            ),
            "observation_ids": list(self.observation_ids),
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "resolution": self.resolution.value,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "page_audit_id",
                "features",
                "route",
                "primary_invocation_id",
                "primary_quality",
                "secondary_invocation_id",
                "secondary_quality",
                "observation_ids",
                "canonical_revision_ids",
                "resolution",
                "reason_codes",
            },
            dto="PageParsingAudit",
        )
        secondary_quality = payload.get("secondary_quality")
        result = cls(
            features=PageFeatures.from_dict(
                require_mapping(payload.get("features"), "features")
            ),
            route=RouteDecision.from_dict(
                require_mapping(payload.get("route"), "route")
            ),
            primary_invocation_id=payload.get("primary_invocation_id"),  # type: ignore[arg-type]
            primary_quality=_enum(
                PageParseQuality,
                payload.get("primary_quality"),
                "primary_quality",
            ),
            secondary_invocation_id=payload.get("secondary_invocation_id"),  # type: ignore[arg-type]
            secondary_quality=(
                None
                if secondary_quality is None
                else _enum(
                    PageParseQuality,
                    secondary_quality,
                    "secondary_quality",
                )
            ),
            observation_ids=tuple(
                require_sequence(payload.get("observation_ids", []), "observation_ids")
            ),  # type: ignore[arg-type]
            canonical_revision_ids=tuple(
                require_sequence(
                    payload.get("canonical_revision_ids", []),
                    "canonical_revision_ids",
                )
            ),  # type: ignore[arg-type]
            resolution=_enum(
                PageResolution, payload.get("resolution"), "resolution"
            ),
            reason_codes=tuple(
                _enum(OrchestrationReasonCode, item, "reason_codes[]")
                for item in require_sequence(
                    payload.get("reason_codes", []), "reason_codes"
                )
            ),
        )
        if payload.get("page_audit_id") != result.page_audit_id:
            raise ContractValidationError("PageParsingAudit page_audit_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class RoutedParsingResult(JsonDTOMixin):
    """Strict replayable result of one routed parsing composition run."""

    document: DocumentRef
    options: ParseOptions
    orchestration_policy: ParsingOrchestrationPolicy
    reconciliation_policy_sha256: str
    prior_revision_ids: tuple[str, ...]
    status: RoutedParsingStatus
    invocations: tuple[ParseInvocationAudit, ...]
    pages: tuple[PageParsingAudit, ...]
    reconciliation: ReconciliationResult | None
    reconciliation_error_type: str | None = None
    reconciliation_error_message: str | None = None
    unresolved_page_ids: tuple[str, ...] = ()
    review_page_ids: tuple[str, ...] = ()
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentRef):
            raise ContractValidationError("document must be DocumentRef")
        if not isinstance(self.options, ParseOptions):
            raise ContractValidationError("options must be ParseOptions")
        if not isinstance(self.orchestration_policy, ParsingOrchestrationPolicy):
            raise ContractValidationError(
                "orchestration_policy must be ParsingOrchestrationPolicy"
            )
        require_sha256(
            self.reconciliation_policy_sha256,
            "reconciliation_policy_sha256",
        )
        prior_revision_ids = tuple(sorted(set(self.prior_revision_ids)))
        for revision_id in prior_revision_ids:
            require_non_empty(revision_id, "prior_revision_ids[]")
        invocations = tuple(sorted(self.invocations, key=lambda item: item.sequence))
        if not invocations or any(
            not isinstance(item, ParseInvocationAudit) for item in invocations
        ):
            raise ContractValidationError(
                "invocations must contain at least one ParseInvocationAudit"
            )
        if [item.sequence for item in invocations] != list(range(len(invocations))):
            raise ContractValidationError("invocation sequence must be contiguous from zero")
        invocation_ids = [item.invocation_id for item in invocations]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ContractValidationError("invocations contains duplicate IDs")
        for invocation in invocations:
            if invocation.request.document != self.document:
                raise OrchestrationContractError(
                    "invocation request belongs to another document"
                )
            if invocation.request.options != self.options:
                raise OrchestrationContractError(
                    "invocation request uses different parse options"
                )
        pages = tuple(sorted(self.pages, key=lambda item: item.page_index))
        if not pages or any(not isinstance(item, PageParsingAudit) for item in pages):
            raise ContractValidationError(
                "pages must contain at least one PageParsingAudit"
            )
        page_ids = [item.page_id for item in pages]
        page_indices = [item.page_index for item in pages]
        if len(page_ids) != len(set(page_ids)) or len(page_indices) != len(
            set(page_indices)
        ):
            raise ContractValidationError("pages contains duplicate source pages")
        invocation_by_id = {item.invocation_id: item for item in invocations}
        referenced_invocations: set[str] = set()
        for page in pages:
            if (
                page.features.page.document_version_id
                != self.document.document_version_id
            ):
                raise OrchestrationContractError("page audit belongs to another document")
            if page.features.mime_type != self.document.mime_type:
                raise OrchestrationContractError(
                    "page audit MIME type differs from its document"
                )
            invocation_bindings = (
                (
                    page.primary_invocation_id,
                    InvocationRole.PRIMARY,
                    page.route.primary_adapter_name,
                    page.route.primary_capability_id,
                ),
                (
                    page.secondary_invocation_id,
                    (
                        InvocationRole.ADJUDICATOR
                        if page.route.secondary_role
                        is SecondaryRouteRole.ADJUDICATOR
                        else InvocationRole.FALLBACK
                    ),
                    page.route.secondary_adapter_name,
                    page.route.secondary_capability_id,
                ),
            )
            page_invocations: list[ParseInvocationAudit] = []
            for invocation_id, expected_role, adapter_name, capability_id in invocation_bindings:
                if invocation_id is None:
                    continue
                invocation = invocation_by_id.get(invocation_id)
                if invocation is None:
                    raise OrchestrationContractError(
                        "page audit references an unknown invocation"
                    )
                request_pages = {
                    request_page.page_id: request_page
                    for request_page in invocation.request.pages
                }
                if request_pages.get(page.page_id) != page.features.page:
                    raise OrchestrationContractError(
                        "page audit source page differs from its invocation"
                    )
                if invocation.role is not expected_role:
                    raise OrchestrationContractError(
                        "page audit references an invocation with the wrong role"
                    )
                if invocation.adapter_name != adapter_name:
                    raise OrchestrationContractError(
                        "page route and invocation adapter differ"
                    )
                if invocation.capability_id != capability_id:
                    raise OrchestrationContractError(
                        "page route and invocation capability differ"
                    )
                referenced_invocations.add(invocation_id)
                page_invocations.append(invocation)
            expected_observation_ids = {
                observation.observation_id
                for invocation in page_invocations
                if invocation.result is not None
                for observation in invocation.result.observations
                if observation.page_id == page.page_id
            }
            if set(page.observation_ids) != expected_observation_ids:
                raise OrchestrationContractError(
                    "page audit observations differ from invocation results"
                )
        if referenced_invocations != set(invocation_ids):
            raise OrchestrationContractError(
                "invocations must each be referenced by at least one page audit"
            )
        reconciliation = self.reconciliation
        error_values = (
            self.reconciliation_error_type,
            self.reconciliation_error_message,
        )
        if reconciliation is None:
            if any(value is None for value in error_values) and any(
                value is not None for value in error_values
            ):
                raise ContractValidationError(
                    "reconciliation error type and message must be supplied together"
                )
        else:
            if not isinstance(reconciliation, ReconciliationResult):
                raise ContractValidationError(
                    "reconciliation must be ReconciliationResult or null"
                )
            if any(value is not None for value in error_values):
                raise ContractValidationError(
                    "successful reconciliation cannot contain an error"
                )
            if (
                reconciliation.document_version_id
                != self.document.document_version_id
            ):
                raise OrchestrationContractError(
                    "reconciliation belongs to another document"
                )
            if reconciliation.policy_sha256 != self.reconciliation_policy_sha256:
                raise OrchestrationContractError(
                    "reconciliation policy binding mismatch"
                )
        expected_unresolved = tuple(
            page.page_id
            for page in pages
            if page.resolution is PageResolution.UNRESOLVED
        )
        expected_review = tuple(
            page.page_id
            for page in pages
            if page.resolution is PageResolution.NEEDS_REVIEW
        )
        if tuple(self.unresolved_page_ids) != expected_unresolved:
            raise ContractValidationError("unresolved_page_ids does not match page audits")
        if tuple(self.review_page_ids) != expected_review:
            raise ContractValidationError("review_page_ids does not match page audits")
        canonical_ids = {
            block.revision_id
            for block in (() if reconciliation is None else reconciliation.canonical_blocks)
        }
        audited_canonical_ids = {
            revision_id for page in pages for revision_id in page.canonical_revision_ids
        }
        if canonical_ids != audited_canonical_ids:
            raise OrchestrationContractError(
                "page audits do not cover reconciliation canonical output"
            )
        expected_status = _aggregate_status(pages)
        status = _enum(RoutedParsingStatus, self.status, "status")
        if status is not expected_status:
            raise ContractValidationError(
                f"status must be {expected_status.value} for the page outcomes"
            )
        object.__setattr__(self, "prior_revision_ids", prior_revision_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "invocations", invocations)
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "unresolved_page_ids", expected_unresolved)
        object.__setattr__(self, "review_page_ids", expected_review)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "routed-parsing-result",
                self.document.document_version_id,
                self.document.content_sha256,
                self.options.options_sha256,
                self.orchestration_policy.policy_id,
                self.reconciliation_policy_sha256,
                prior_revision_ids,
                status.value,
                [item.invocation_id for item in invocations],
                [item.page_audit_id for item in pages],
                None if reconciliation is None else reconciliation.result_id,
                self.reconciliation_error_type,
                self.reconciliation_error_message,
            ),
        )

    @property
    def canonical_blocks(self) -> tuple[CanonicalBlock, ...]:
        return () if self.reconciliation is None else self.reconciliation.canonical_blocks

    @property
    def manual_review_page_ids(self) -> tuple[str, ...]:
        return tuple(
            page.page_id
            for page in self.pages
            if page.resolution is not PageResolution.RESOLVED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "result_id": self.result_id,
            "document": self.document.to_dict(),
            "options": self.options.to_dict(),
            "orchestration_policy": self.orchestration_policy.to_dict(),
            "reconciliation_policy_sha256": self.reconciliation_policy_sha256,
            "prior_revision_ids": list(self.prior_revision_ids),
            "status": self.status.value,
            "invocations": [item.to_dict() for item in self.invocations],
            "pages": [item.to_dict() for item in self.pages],
            "reconciliation": (
                None if self.reconciliation is None else self.reconciliation.to_dict()
            ),
            "reconciliation_error_type": self.reconciliation_error_type,
            "reconciliation_error_message": self.reconciliation_error_message,
            "unresolved_page_ids": list(self.unresolved_page_ids),
            "review_page_ids": list(self.review_page_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "result_id",
                "document",
                "options",
                "orchestration_policy",
                "reconciliation_policy_sha256",
                "prior_revision_ids",
                "status",
                "invocations",
                "pages",
                "reconciliation",
                "reconciliation_error_type",
                "reconciliation_error_message",
                "unresolved_page_ids",
                "review_page_ids",
            },
            dto="RoutedParsingResult",
        )
        reconciliation_payload = payload.get("reconciliation")
        result = cls(
            document=DocumentRef.from_dict(
                require_mapping(payload.get("document"), "document")
            ),
            options=ParseOptions.from_dict(
                require_mapping(payload.get("options"), "options")
            ),
            orchestration_policy=ParsingOrchestrationPolicy.from_dict(
                require_mapping(
                    payload.get("orchestration_policy"),
                    "orchestration_policy",
                )
            ),
            reconciliation_policy_sha256=payload.get("reconciliation_policy_sha256"),  # type: ignore[arg-type]
            prior_revision_ids=tuple(
                require_sequence(
                    payload.get("prior_revision_ids", []), "prior_revision_ids"
                )
            ),  # type: ignore[arg-type]
            status=_enum(RoutedParsingStatus, payload.get("status"), "status"),
            invocations=tuple(
                ParseInvocationAudit.from_dict(
                    require_mapping(item, f"invocations[{index}]")
                )
                for index, item in enumerate(
                    require_sequence(payload.get("invocations"), "invocations")
                )
            ),
            pages=tuple(
                PageParsingAudit.from_dict(require_mapping(item, f"pages[{index}]"))
                for index, item in enumerate(
                    require_sequence(payload.get("pages"), "pages")
                )
            ),
            reconciliation=(
                None
                if reconciliation_payload is None
                else ReconciliationResult.from_dict(
                    require_mapping(reconciliation_payload, "reconciliation")
                )
            ),
            reconciliation_error_type=payload.get("reconciliation_error_type"),  # type: ignore[arg-type]
            reconciliation_error_message=payload.get("reconciliation_error_message"),  # type: ignore[arg-type]
            unresolved_page_ids=tuple(
                require_sequence(
                    payload.get("unresolved_page_ids", []), "unresolved_page_ids"
                )
            ),  # type: ignore[arg-type]
            review_page_ids=tuple(
                require_sequence(
                    payload.get("review_page_ids", []), "review_page_ids"
                )
            ),  # type: ignore[arg-type]
        )
        if payload.get("result_id") != result.result_id:
            raise ContractValidationError("RoutedParsingResult result_id mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class _Target:
    features: PageFeatures
    route: RouteDecision
    adapter_name: str
    capability_id: str
    role: InvocationRole
    trigger: InvocationTrigger


@dataclass(frozen=True, slots=True)
class _Assessment:
    quality: PageParseQuality
    observation_ids: tuple[str, ...]


def _aggregate_status(pages: Sequence[PageParsingAudit]) -> RoutedParsingStatus:
    if all(page.resolution is PageResolution.RESOLVED for page in pages):
        return RoutedParsingStatus.SUCCEEDED
    if all(page.resolution is PageResolution.UNRESOLVED for page in pages):
        return RoutedParsingStatus.FAILED
    return RoutedParsingStatus.PARTIAL


class ParsingOrchestrator:
    """Sequentially execute routed parser work and reconcile source evidence."""

    def __init__(
        self,
        registry: ParserRegistry,
        router: DeterministicPageRouter,
        *,
        policy: ParsingOrchestrationPolicy | None = None,
        reconciliation_policy: ReconciliationPolicy | None = None,
    ) -> None:
        if not isinstance(registry, ParserRegistry):
            raise TypeError("registry must be ParserRegistry")
        if not isinstance(router, DeterministicPageRouter):
            raise TypeError("router must be DeterministicPageRouter")
        if policy is not None and not isinstance(policy, ParsingOrchestrationPolicy):
            raise TypeError("policy must be ParsingOrchestrationPolicy")
        if reconciliation_policy is not None and not isinstance(
            reconciliation_policy, ReconciliationPolicy
        ):
            raise TypeError("reconciliation_policy must be ReconciliationPolicy")
        self._registry = registry
        self._router = router
        self._policy = policy or ParsingOrchestrationPolicy()
        self._reconciliation_policy = reconciliation_policy or ReconciliationPolicy()

    @property
    def policy(self) -> ParsingOrchestrationPolicy:
        return self._policy

    def parse(
        self,
        document: DocumentRef,
        page_features: Iterable[PageFeatures],
        *,
        options: ParseOptions | None = None,
        prior_blocks: Iterable[CanonicalBlock] = (),
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> RoutedParsingResult:
        features = self._validate_inputs(document, page_features)
        active_options = options or ParseOptions()
        if not isinstance(active_options, ParseOptions):
            raise TypeError("options must be ParseOptions")
        priors = tuple(prior_blocks)
        self._validate_priors(document, priors)

        routes = tuple(self._router.route(item) for item in features)
        self._preflight_routes(features, routes)
        primary_targets = tuple(
            _Target(
                features=feature,
                route=route,
                adapter_name=route.primary_adapter_name,
                capability_id=route.primary_capability_id,
                role=InvocationRole.PRIMARY,
                trigger=InvocationTrigger.PRIMARY_ROUTE,
            )
            for feature, route in zip(features, routes, strict=True)
        )
        primary_invocations, primary_by_page = self._execute_targets(
            document,
            active_options,
            primary_targets,
            sequence_start=0,
            run_id=run_id,
            trace_id=trace_id,
        )
        primary_assessments = {
            feature.page.page_id: self._assess_page(
                primary_by_page[feature.page.page_id], feature, route
            )
            for feature, route in zip(features, routes, strict=True)
        }

        secondary_targets: list[_Target] = []
        for feature, route in zip(features, routes, strict=True):
            if route.secondary_adapter_name is None:
                continue
            primary_quality = primary_assessments[feature.page.page_id].quality
            if route.secondary_role is SecondaryRouteRole.ADJUDICATOR:
                trigger = InvocationTrigger.AMBIGUOUS_ADJUDICATION
                role = InvocationRole.ADJUDICATOR
            elif (
                route.secondary_role is SecondaryRouteRole.FALLBACK
                and primary_quality is not PageParseQuality.ACCEPTED
            ):
                trigger = {
                    PageParseQuality.FAILED: InvocationTrigger.PRIMARY_FAILED,
                    PageParseQuality.LOW_QUALITY: InvocationTrigger.PRIMARY_LOW_QUALITY,
                    PageParseQuality.MISSING_OUTPUT: InvocationTrigger.PRIMARY_MISSING_OUTPUT,
                }[primary_quality]
                role = InvocationRole.FALLBACK
            else:
                continue
            secondary_targets.append(
                _Target(
                    features=feature,
                    route=route,
                    adapter_name=route.secondary_adapter_name,
                    capability_id=route.secondary_capability_id or "",
                    role=role,
                    trigger=trigger,
                )
            )
        secondary_invocations, secondary_by_page = self._execute_targets(
            document,
            active_options,
            tuple(secondary_targets),
            sequence_start=len(primary_invocations),
            run_id=run_id,
            trace_id=trace_id,
        )
        secondary_assessments = {
            target.features.page.page_id: self._assess_page(
                secondary_by_page[target.features.page.page_id],
                target.features,
                target.route,
            )
            for target in secondary_targets
        }
        invocations = primary_invocations + secondary_invocations
        observations = tuple(
            observation
            for invocation in invocations
            if invocation.result is not None
            for observation in invocation.result.observations
        )
        reconciliation: ReconciliationResult | None = None
        reconciliation_error_type: str | None = None
        reconciliation_error_message: str | None = None
        if observations:
            try:
                reconciliation = reconcile_observations(
                    observations,
                    policy=self._reconciliation_policy,
                    prior_blocks=priors,
                )
            except ContractValidationError as exc:
                reconciliation_error_type = type(exc).__name__
                reconciliation_error_message = (
                    str(exc).strip() or "reconciliation failed without a message"
                )

        canonical_by_page: dict[str, tuple[CanonicalBlock, ...]] = {}
        reconciliation_issue_pages: set[int] = set()
        if reconciliation is not None:
            grouped_blocks: dict[str, list[CanonicalBlock]] = defaultdict(list)
            for block in reconciliation.canonical_blocks:
                grouped_blocks[block.page_id].append(block)
            canonical_by_page = {
                page_id: tuple(sorted(blocks, key=lambda block: block.revision_id))
                for page_id, blocks in grouped_blocks.items()
            }
            reconciliation_issue_pages = {
                issue.page_index for issue in reconciliation.issues
            }

        page_audits: list[PageParsingAudit] = []
        for feature, route in zip(features, routes, strict=True):
            page_id = feature.page.page_id
            primary_invocation = primary_by_page[page_id]
            primary_assessment = primary_assessments[page_id]
            secondary_invocation = secondary_by_page.get(page_id)
            secondary_assessment = secondary_assessments.get(page_id)
            page_observations = {
                observation.observation_id
                for invocation in (primary_invocation, secondary_invocation)
                if invocation is not None and invocation.result is not None
                for observation in invocation.result.observations
                if observation.page_id == page_id
            }
            blocks = canonical_by_page.get(page_id, ())
            resolution, reason_codes = self._page_resolution(
                route=route,
                primary=primary_assessment,
                secondary=secondary_assessment,
                blocks=blocks,
                reconciliation_has_issue=(
                    feature.page.page_index in reconciliation_issue_pages
                ),
                reconciliation_failed=(
                    reconciliation_error_type is not None
                    and bool(page_observations)
                ),
            )
            page_audits.append(
                PageParsingAudit(
                    features=feature,
                    route=route,
                    primary_invocation_id=primary_invocation.invocation_id,
                    primary_quality=primary_assessment.quality,
                    secondary_invocation_id=(
                        None
                        if secondary_invocation is None
                        else secondary_invocation.invocation_id
                    ),
                    secondary_quality=(
                        None
                        if secondary_assessment is None
                        else secondary_assessment.quality
                    ),
                    observation_ids=tuple(page_observations),
                    canonical_revision_ids=tuple(
                        block.revision_id for block in blocks
                    ),
                    resolution=resolution,
                    reason_codes=reason_codes,
                )
            )
        status = _aggregate_status(page_audits)
        unresolved_page_ids = tuple(
            page.page_id
            for page in page_audits
            if page.resolution is PageResolution.UNRESOLVED
        )
        review_page_ids = tuple(
            page.page_id
            for page in page_audits
            if page.resolution is PageResolution.NEEDS_REVIEW
        )
        return RoutedParsingResult(
            document=document,
            options=active_options,
            orchestration_policy=self._policy,
            reconciliation_policy_sha256=self._reconciliation_policy.policy_sha256,
            prior_revision_ids=tuple(block.revision_id for block in priors),
            status=status,
            invocations=invocations,
            pages=tuple(page_audits),
            reconciliation=reconciliation,
            reconciliation_error_type=reconciliation_error_type,
            reconciliation_error_message=reconciliation_error_message,
            unresolved_page_ids=unresolved_page_ids,
            review_page_ids=review_page_ids,
        )

    def _validate_inputs(
        self,
        document: DocumentRef,
        page_features: Iterable[PageFeatures],
    ) -> tuple[PageFeatures, ...]:
        if not isinstance(document, DocumentRef):
            raise TypeError("document must be DocumentRef")
        features = tuple(page_features)
        if not features:
            raise OrchestrationContractError("page_features must not be empty")
        if any(not isinstance(item, PageFeatures) for item in features):
            raise TypeError("page_features must contain PageFeatures DTOs")
        features = tuple(sorted(features, key=lambda item: item.page.page_index))
        page_ids = [item.page.page_id for item in features]
        page_indices = [item.page.page_index for item in features]
        if len(page_ids) != len(set(page_ids)) or len(page_indices) != len(
            set(page_indices)
        ):
            raise OrchestrationContractError("page_features contains duplicate pages")
        for feature in features:
            if feature.page.document_version_id != document.document_version_id:
                raise OrchestrationContractError(
                    "page feature belongs to another document version"
                )
            if feature.mime_type != document.mime_type:
                raise OrchestrationContractError(
                    "page feature MIME type differs from its document"
                )
            if (
                document.page_count is not None
                and feature.page.page_index >= document.page_count
            ):
                raise OrchestrationContractError(
                    "page feature exceeds document page_count"
                )
        return features

    @staticmethod
    def _validate_priors(
        document: DocumentRef, priors: Sequence[CanonicalBlock]
    ) -> None:
        if any(not isinstance(block, CanonicalBlock) for block in priors):
            raise TypeError("prior_blocks must contain CanonicalBlock DTOs")
        if any(
            block.document_version_id != document.document_version_id
            for block in priors
        ):
            raise OrchestrationContractError(
                "prior canonical block belongs to another document"
            )
        revision_ids = [block.revision_id for block in priors]
        block_ids = [block.block_id for block in priors]
        canonical_keys = [block.canonical_key for block in priors]
        if (
            len(revision_ids) != len(set(revision_ids))
            or len(block_ids) != len(set(block_ids))
            or len(canonical_keys) != len(set(canonical_keys))
        ):
            raise OrchestrationContractError(
                "prior_blocks contains duplicate canonical identities"
            )

    def _preflight_routes(
        self,
        features: Sequence[PageFeatures],
        routes: Sequence[RouteDecision],
    ) -> None:
        for feature, route in zip(features, routes, strict=True):
            if route.page_id != feature.page.page_id:
                raise OrchestrationContractError("router returned a decision for another page")
            if route.feature_id != feature.feature_id:
                raise OrchestrationContractError(
                    "router returned a decision for stale features"
                )
            planned = (
                (route.primary_adapter_name, route.primary_capability_id),
                (route.secondary_adapter_name, route.secondary_capability_id),
            )
            for adapter_name, capability_id in planned:
                if adapter_name is None:
                    continue
                manifest = self._registry.get(adapter_name).manifest()
                if manifest.capability_id != capability_id:
                    raise OrchestrationContractError(
                        f"route capability changed before execution: {adapter_name}"
                    )

    def _execute_targets(
        self,
        document: DocumentRef,
        options: ParseOptions,
        targets: Sequence[_Target],
        *,
        sequence_start: int,
        run_id: str | None,
        trace_id: str | None,
    ) -> tuple[
        tuple[ParseInvocationAudit, ...],
        Mapping[str, ParseInvocationAudit],
    ]:
        if not targets:
            return (), {}
        grouped: dict[tuple[str, InvocationRole], list[_Target]] = defaultdict(list)
        for target in targets:
            grouped[(target.adapter_name, target.role)].append(target)
        invocations: list[ParseInvocationAudit] = []
        by_page: dict[str, ParseInvocationAudit] = {}
        sequence = sequence_start
        for adapter_name, role in sorted(
            grouped, key=lambda item: (item[0], item[1].value)
        ):
            group = sorted(
                grouped[(adapter_name, role)],
                key=lambda item: item.features.page.page_index,
            )
            manifest = self._registry.get(adapter_name).manifest()
            if any(target.capability_id != manifest.capability_id for target in group):
                raise OrchestrationContractError(
                    f"route capability changed before execution: {adapter_name}"
                )
            batch_size = 1
            if manifest.supports_batch:
                batch_size = self._policy.max_batch_pages
                if manifest.max_pages_per_request is not None:
                    batch_size = min(batch_size, manifest.max_pages_per_request)
            # Vendor page-selection APIs commonly accept one contiguous range.
            # Never combine sparse routed pages into a request merely because
            # they share an adapter: doing so either violates that contract or
            # forces the worker to parse every intervening page.  Split into
            # maximal contiguous runs first, then apply the advertised batch
            # bound. This remains O(number_of_targets) after the deterministic
            # sort above.
            chunks: list[list[_Target]] = []
            run: list[_Target] = []
            for target in group:
                if run and (
                    target.features.page.page_index
                    != run[-1].features.page.page_index + 1
                ):
                    chunks.extend(
                        run[offset : offset + batch_size]
                        for offset in range(0, len(run), batch_size)
                    )
                    run = []
                run.append(target)
            if run:
                chunks.extend(
                    run[offset : offset + batch_size]
                    for offset in range(0, len(run), batch_size)
                )
            for chunk in chunks:
                triggers = tuple(target.trigger for target in chunk)
                idempotency_key = stable_id(
                    "routed-parser-call",
                    self._policy.policy_id,
                    role.value,
                    adapter_name,
                    manifest.capability_id,
                    [target.route.decision_id for target in chunk],
                    [trigger.value for trigger in triggers],
                    options.options_sha256,
                )
                request = ParseRequest(
                    document=document,
                    pages=tuple(target.features.page for target in chunk),
                    options=options,
                    run_id=run_id,
                    trace_id=trace_id,
                    idempotency_key=idempotency_key,
                )
                try:
                    result = self._registry.parse(adapter_name, request)
                    invocation = ParseInvocationAudit(
                        sequence=sequence,
                        role=role,
                        adapter_name=adapter_name,
                        capability_id=manifest.capability_id,
                        triggers=triggers,
                        request=request,
                        result=result,
                    )
                except Exception as exc:  # parser/transport/contract trust boundary
                    invocation = ParseInvocationAudit(
                        sequence=sequence,
                        role=role,
                        adapter_name=adapter_name,
                        capability_id=manifest.capability_id,
                        triggers=triggers,
                        request=request,
                        error_type=type(exc).__name__,
                        error_message=(
                            str(exc).strip()
                            or "adapter invocation failed without a message"
                        ),
                    )
                sequence += 1
                invocations.append(invocation)
                for target in chunk:
                    page_id = target.features.page.page_id
                    if page_id in by_page:
                        raise OrchestrationContractError(
                            "page was scheduled twice in the same execution stage"
                        )
                    by_page[page_id] = invocation
        return tuple(invocations), by_page

    def _assess_page(
        self,
        invocation: ParseInvocationAudit,
        features: PageFeatures,
        route: RouteDecision,
    ) -> _Assessment:
        if invocation.result is None or invocation.result.status is ParseStatus.FAILED:
            return _Assessment(PageParseQuality.FAILED, ())
        observations = tuple(
            observation
            for observation in invocation.result.observations
            if observation.page_id == features.page.page_id
        )
        observation_ids = tuple(
            sorted(observation.observation_id for observation in observations)
        )
        if not observations:
            return _Assessment(PageParseQuality.MISSING_OUTPUT, ())
        low_quality = len(observations) < self._policy.minimum_observations_per_page
        relevant_errors = tuple(
            issue
            for issue in invocation.result.issues
            if issue.severity is IssueSeverity.ERROR
            and (
                issue.page_index is None
                or issue.page_index == features.page.page_index
            )
        )
        if relevant_errors:
            low_quality = True
        explicit_confidences = tuple(
            observation.confidence
            for observation in observations
            if observation.confidence is not None
        )
        if (
            explicit_confidences
            and max(explicit_confidences)
            < self._policy.minimum_explicit_confidence
        ):
            low_quality = True
        if self._policy.require_expected_block_types:
            observed_types = {observation.block_type for observation in observations}
            if not set(route.required_block_types).issubset(observed_types):
                low_quality = True
        return _Assessment(
            PageParseQuality.LOW_QUALITY
            if low_quality
            else PageParseQuality.ACCEPTED,
            observation_ids,
        )

    def _page_resolution(
        self,
        *,
        route: RouteDecision,
        primary: _Assessment,
        secondary: _Assessment | None,
        blocks: Sequence[CanonicalBlock],
        reconciliation_has_issue: bool,
        reconciliation_failed: bool,
    ) -> tuple[PageResolution, tuple[OrchestrationReasonCode, ...]]:
        reasons: list[OrchestrationReasonCode] = [
            {
                PageParseQuality.ACCEPTED: OrchestrationReasonCode.PRIMARY_ACCEPTED,
                PageParseQuality.LOW_QUALITY: OrchestrationReasonCode.PRIMARY_LOW_QUALITY,
                PageParseQuality.MISSING_OUTPUT: OrchestrationReasonCode.PRIMARY_MISSING_OUTPUT,
                PageParseQuality.FAILED: OrchestrationReasonCode.PRIMARY_FAILED,
            }[primary.quality]
        ]
        if secondary is not None:
            reasons.append(
                {
                    PageParseQuality.ACCEPTED: OrchestrationReasonCode.SECONDARY_ACCEPTED,
                    PageParseQuality.LOW_QUALITY: OrchestrationReasonCode.SECONDARY_LOW_QUALITY,
                    PageParseQuality.MISSING_OUTPUT: OrchestrationReasonCode.SECONDARY_MISSING_OUTPUT,
                    PageParseQuality.FAILED: OrchestrationReasonCode.SECONDARY_FAILED,
                }[secondary.quality]
            )
        elif (
            route.secondary_role is SecondaryRouteRole.FALLBACK
            and primary.quality is PageParseQuality.ACCEPTED
        ):
            reasons.append(OrchestrationReasonCode.SECONDARY_NOT_REQUIRED)
        elif route.secondary_adapter_name is None:
            reasons.append(OrchestrationReasonCode.NO_SECONDARY_AVAILABLE)

        if not blocks:
            reasons.append(OrchestrationReasonCode.NO_CANONICAL_OUTPUT)
            if reconciliation_failed:
                reasons.append(OrchestrationReasonCode.RECONCILIATION_FAILED)
            return PageResolution.UNRESOLVED, tuple(reasons)

        review = False
        if any(block.quality_status is QualityStatus.NEEDS_REVIEW for block in blocks):
            reasons.append(OrchestrationReasonCode.CANONICAL_NEEDS_REVIEW)
            review = True
        if (
            self._policy.review_degraded_canonical
            and any(block.quality_status is QualityStatus.DEGRADED for block in blocks)
        ):
            reasons.append(OrchestrationReasonCode.CANONICAL_DEGRADED)
            review = True
        if self._policy.review_reconciliation_issues and reconciliation_has_issue:
            reasons.append(OrchestrationReasonCode.RECONCILIATION_ISSUE)
            review = True
        if route.page_class is PageClass.AMBIGUOUS and (
            route.secondary_role is not SecondaryRouteRole.ADJUDICATOR
            or secondary is None
            or secondary.quality is not PageParseQuality.ACCEPTED
        ):
            reasons.append(OrchestrationReasonCode.AMBIGUITY_UNRESOLVED)
            review = True
        if (
            primary.quality is not PageParseQuality.ACCEPTED
            and (
                secondary is None
                or secondary.quality is not PageParseQuality.ACCEPTED
            )
        ):
            review = True
        return (
            PageResolution.NEEDS_REVIEW if review else PageResolution.RESOLVED,
            tuple(reasons),
        )


__all__ = [
    "InvocationRole",
    "InvocationTrigger",
    "OrchestrationContractError",
    "OrchestrationReasonCode",
    "ORCHESTRATION_SCHEMA_VERSION",
    "PageParseQuality",
    "PageParsingAudit",
    "PageResolution",
    "ParseInvocationAudit",
    "ParsingOrchestrationPolicy",
    "ParsingOrchestrator",
    "RoutedParsingResult",
    "RoutedParsingStatus",
    "UnsupportedOrchestrationSchemaError",
]
