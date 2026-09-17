"""Public dependency-light facade from routed parsing to BookRAG rows.

The facade is a composition root, not a service locator.  Callers inject a
``ParserRegistry`` or an explicit adapter iterable.  The routed factory
constructs ordinary policy objects and keeps no module-level runtime state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast, Self
import warnings

from document_services.evidence.projections.bookrag import (
    PROJECTION_NAME,
    BookRAGAssetMaterializationLease,
    BookRAGImageMaterialPolicy,
    project_to_bookrag_pdf_list,
)
from document_services.evidence.artifacts import ROUTED_PARSING_AUDIT_SCHEMA_VERSION
from document_services.domain.base import (
    ContractValidationError,
    JsonDTOMixin,
    JsonObject,
    freeze_json,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    stable_id,
)
from document_services.domain.evidence import CanonicalBlock, DocumentRef
from document_services.domain.parsing import ParseOptions, ParseRequest
from document_services.parsing.orchestrator import (
    ParsingOrchestrationPolicy,
    ParsingOrchestrator,
    RoutedParsingResult,
    RoutedParsingStatus,
)
from document_services.ports.parser import ParserAdapter, ParserRegistry
from document_services.parsing.routing_presets import balanced_pdf_routing_policy
from document_services.parsing.reconciliation import ReconciliationPolicy
from document_services.parsing.routing import (
    DeterministicPageRouter,
    PageFeatures,
    PageRoutingPolicy,
)


COMPOSITION_SCHEMA_VERSION = ROUTED_PARSING_AUDIT_SCHEMA_VERSION


class CompositionBindingError(ContractValidationError):
    """Injected policies, source requests, or projected rows do not bind."""


class UnsupportedCompositionSchemaError(ContractValidationError):
    """A serialized facade result uses an unsupported schema."""


class BookRAGHandoffNotReadyError(CompositionBindingError):
    """Strict BookRAG handoff was attempted with incomplete parsing evidence."""


class BookRAGHandoffReadiness(StrEnum):
    """Fail-closed readiness state for the legacy BookRAG indexing seam."""

    READY = "ready"
    BLOCKED_FAILED = "blocked_failed"
    BLOCKED_UNRESOLVED = "blocked_unresolved"
    BLOCKED_REVIEW = "blocked_review"
    BLOCKED_PARTIAL = "blocked_partial"
    BLOCKED_NO_CANONICAL_OUTPUT = "blocked_no_canonical_output"


def _require_schema(payload: Mapping[str, object]) -> None:
    actual = payload.get("schema_version")
    if actual != COMPOSITION_SCHEMA_VERSION:
        raise UnsupportedCompositionSchemaError(
            "unsupported composition schema_version "
            f"{actual!r}; expected {COMPOSITION_SCHEMA_VERSION!r}"
        )


@dataclass(frozen=True, slots=True)
class RoutedBookRAGResult(JsonDTOMixin):
    """Replayable parsing audit with a fail-closed BookRAG handoff.

    Compatibility rows are deliberately not stored or serialized.  A ready
    result projects them through :meth:`to_bookrag_pdf_list`; unresolved or
    review output requires the explicitly audit-only partial export method.
    """

    request: ParseRequest
    parsing: RoutedParsingResult
    projection_name: str = PROJECTION_NAME
    projection_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ParseRequest):
            raise ContractValidationError("request must be ParseRequest")
        if not isinstance(self.parsing, RoutedParsingResult):
            raise ContractValidationError("parsing must be RoutedParsingResult")
        require_non_empty(self.projection_name, "projection_name")
        if self.projection_name != PROJECTION_NAME:
            raise CompositionBindingError(
                f"projection_name must be {PROJECTION_NAME!r}"
            )
        if self.request.document != self.parsing.document:
            raise CompositionBindingError(
                "outer request and parsing result bind different documents"
            )
        if self.request.options != self.parsing.options:
            raise CompositionBindingError(
                "outer request and parsing result bind different parse options"
            )
        parsed_pages = tuple(page.features.page for page in self.parsing.pages)
        if self.request.pages != parsed_pages:
            raise CompositionBindingError(
                "outer request and parsing result bind different pages or order"
            )
        object.__setattr__(
            self,
            "projection_id",
            stable_id(
                "bookrag-projection",
                self.request.request_id,
                self.parsing.result_id,
                self.projection_name,
                [block.revision_id for block in self.parsing.canonical_blocks],
            ),
        )

    @property
    def bookrag_handoff_readiness(self) -> BookRAGHandoffReadiness:
        """Return the strict, deterministic readiness classification."""

        if self.parsing.status is RoutedParsingStatus.FAILED:
            return BookRAGHandoffReadiness.BLOCKED_FAILED
        if self.parsing.unresolved_page_ids:
            return BookRAGHandoffReadiness.BLOCKED_UNRESOLVED
        if self.parsing.review_page_ids:
            return BookRAGHandoffReadiness.BLOCKED_REVIEW
        if self.parsing.status is not RoutedParsingStatus.SUCCEEDED:
            return BookRAGHandoffReadiness.BLOCKED_PARTIAL
        if not self.parsing.canonical_blocks:
            return BookRAGHandoffReadiness.BLOCKED_NO_CANONICAL_OUTPUT
        return BookRAGHandoffReadiness.READY

    @property
    def is_ready_for_bookrag_handoff(self) -> bool:
        return self.bookrag_handoff_readiness is BookRAGHandoffReadiness.READY

    @property
    def bookrag_handoff_blockers(self) -> tuple[str, ...]:
        """Human-readable, audit-safe reasons the strict handoff is blocked."""

        blockers: list[str] = []
        if self.parsing.status is not RoutedParsingStatus.SUCCEEDED:
            blockers.append(f"parsing status is {self.parsing.status.value}")
        if self.parsing.unresolved_page_ids:
            blockers.append(
                "unresolved pages: " + ", ".join(self.parsing.unresolved_page_ids)
            )
        if self.parsing.review_page_ids:
            blockers.append(
                "pages require review: " + ", ".join(self.parsing.review_page_ids)
            )
        if not self.parsing.canonical_blocks:
            blockers.append("no canonical output")
        return tuple(blockers)

    def to_bookrag_pdf_list(
        self,
        *,
        asset_materialization_lease: BookRAGAssetMaterializationLease | None = None,
        image_material_policy: BookRAGImageMaterialPolicy = (
            BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
        ),
    ) -> list[dict[str, object]]:
        """Return rows only when every requested page is resolved.

        Image paths default to verified canonical-CAS lease material. Passing
        the explicitly named legacy-inspection policy marks every returned row
        as inspection-only; those rows are not indexing/evidence authority.
        """

        if not self.is_ready_for_bookrag_handoff:
            blockers = "; ".join(self.bookrag_handoff_blockers)
            raise BookRAGHandoffNotReadyError(
                "BookRAG handoff is not ready: "
                f"{self.bookrag_handoff_readiness.value} ({blockers}). "
                "Inspect parsing for the routed audit; use "
                "to_audit_partial_bookrag_pdf_list() only for diagnosis."
            )
        return self._project_bookrag_rows(
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )

    def to_audit_partial_bookrag_pdf_list(
        self,
        *,
        asset_materialization_lease: BookRAGAssetMaterializationLease | None = None,
        image_material_policy: BookRAGImageMaterialPolicy = (
            BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
        ),
    ) -> list[dict[str, object]]:
        """Explicitly project available evidence for diagnosis, never indexing.

        Rows from this method may omit unresolved pages or contain evidence
        flagged for review.  The method name intentionally prevents it from
        masquerading as the normal BookRAG ingestion handoff.
        """

        return self._project_bookrag_rows(
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )

    def _project_bookrag_rows(
        self,
        *,
        asset_materialization_lease: BookRAGAssetMaterializationLease | None,
        image_material_policy: BookRAGImageMaterialPolicy,
    ) -> list[dict[str, object]]:
        rows = project_to_bookrag_pdf_list(
            self.request.document,
            self.parsing.canonical_blocks,
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )
        frozen_rows: list[JsonObject] = []
        for index, row in enumerate(rows):
            frozen = freeze_json(row, field=f"projected_rows[{index}]")
            if not isinstance(frozen, Mapping):
                raise ContractValidationError(
                    "BookRAG projection must contain JSON objects"
                )
            frozen_rows.append(cast(JsonObject, frozen))
        self._validate_projected_rows(tuple(frozen_rows))

        values = json_ready(tuple(frozen_rows))
        if not isinstance(values, list):  # defensive: tuple always converts to list
            raise AssertionError("bookrag row conversion did not produce a list")
        return [dict(cast(Mapping[str, object], row)) for row in values]

    def _validate_projected_rows(self, rows: tuple[JsonObject, ...]) -> None:
        blocks = tuple(self.parsing.canonical_blocks)
        if len(rows) != len(blocks):
            raise CompositionBindingError(
                "BookRAG row count differs from canonical block count"
            )
        expected_revision_ids = {block.revision_id for block in blocks}
        actual_revision_ids: set[str] = set()
        for index, row in enumerate(rows):
            if row.get("pdf_id") != index + 1:
                raise CompositionBindingError(
                    "BookRAG pdf_id values must be contiguous from one"
                )
            if (
                row.get("document_version_id")
                != self.request.document.document_version_id
            ):
                raise CompositionBindingError(
                    "BookRAG row belongs to another document version"
                )
            if row.get("source_sha256") != self.request.document.content_sha256:
                raise CompositionBindingError(
                    "BookRAG row source hash differs from its request"
                )
            revision_id = row.get("canonical_revision_id")
            if not isinstance(revision_id, str) or not revision_id:
                raise CompositionBindingError("BookRAG row lacks canonical_revision_id")
            if revision_id in actual_revision_ids:
                raise CompositionBindingError(
                    "BookRAG rows contain a duplicate canonical revision"
                )
            actual_revision_ids.add(revision_id)
        if actual_revision_ids != expected_revision_ids:
            raise CompositionBindingError(
                "BookRAG rows do not cover canonical parsing revisions"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPOSITION_SCHEMA_VERSION,
            "projection_id": self.projection_id,
            "projection_name": self.projection_name,
            "request": self.request.to_dict(),
            "parsing": self.parsing.to_dict(),
            "bookrag_handoff_readiness": self.bookrag_handoff_readiness.value,
            "bookrag_handoff_blockers": list(self.bookrag_handoff_blockers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "projection_id",
                "projection_name",
                "request",
                "parsing",
                "bookrag_handoff_readiness",
                "bookrag_handoff_blockers",
            },
            dto="RoutedBookRAGResult",
        )
        result = cls(
            request=ParseRequest.from_dict(
                require_mapping(payload.get("request"), "request")
            ),
            parsing=RoutedParsingResult.from_dict(
                require_mapping(payload.get("parsing"), "parsing")
            ),
            projection_name=payload.get("projection_name"),  # type: ignore[arg-type]
        )
        if payload.get("projection_id") != result.projection_id:
            raise ContractValidationError("RoutedBookRAGResult projection_id mismatch")
        if payload.get("bookrag_handoff_readiness") != (
            result.bookrag_handoff_readiness.value
        ):
            raise ContractValidationError(
                "RoutedBookRAGResult bookrag_handoff_readiness mismatch"
            )
        if payload.get("bookrag_handoff_blockers") != list(
            result.bookrag_handoff_blockers
        ):
            raise ContractValidationError(
                "RoutedBookRAGResult bookrag_handoff_blockers mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


class RoutedBookRAGFacade:
    """Explicit composition of router, parser orchestration, and projection."""

    def __init__(
        self,
        registry: ParserRegistry,
        routing_policy: PageRoutingPolicy,
        *,
        orchestration_policy: ParsingOrchestrationPolicy | None = None,
        reconciliation_policy: ReconciliationPolicy | None = None,
        require_all_policy_adapters: bool = True,
    ) -> None:
        if not isinstance(registry, ParserRegistry):
            raise TypeError("registry must be ParserRegistry")
        if not isinstance(routing_policy, PageRoutingPolicy):
            raise TypeError("routing_policy must be PageRoutingPolicy")
        if not isinstance(require_all_policy_adapters, bool):
            raise TypeError("require_all_policy_adapters must be a boolean")
        registered = {manifest.adapter_name for manifest in registry.manifests()}
        configured = set(routing_policy.profile_by_adapter)
        missing = tuple(sorted(configured - registered))
        if require_all_policy_adapters and missing:
            raise CompositionBindingError(
                "routing policy adapter(s) are not registered: " + ", ".join(missing)
            )
        self._registry = registry
        self._routing_policy = routing_policy
        self._orchestration_policy = (
            orchestration_policy or ParsingOrchestrationPolicy()
        )
        self._reconciliation_policy = reconciliation_policy or ReconciliationPolicy()
        router = DeterministicPageRouter(registry, routing_policy)
        self._orchestrator = ParsingOrchestrator(
            registry,
            router,
            policy=self._orchestration_policy,
            reconciliation_policy=self._reconciliation_policy,
        )

    @classmethod
    def routed_default(
        cls,
        *,
        registry: ParserRegistry | None = None,
        adapters: Iterable[ParserAdapter] | None = None,
        orchestration_policy: ParsingOrchestrationPolicy | None = None,
        reconciliation_policy: ReconciliationPolicy | None = None,
        estimated_latency_ms: Mapping[str, int] | None = None,
        estimated_cost_microusd: Mapping[str, int] | None = None,
        max_primary_latency_ms: int | None = None,
        max_total_latency_ms: int | None = None,
        max_primary_cost_microusd: int | None = None,
        max_total_cost_microusd: int | None = None,
    ) -> Self:
        """Build the default routed PDF policy from explicit dependencies.

        Exactly one of ``registry`` or ``adapters`` is required.  Supplying an
        adapter iterable constructs a private registry owned by this facade;
        no plugin discovery or process-global registration occurs.
        """

        if (registry is None) == (adapters is None):
            raise CompositionBindingError(
                "provide exactly one of registry or adapters"
            )
        active_registry = registry
        if active_registry is None:
            active_registry = ParserRegistry(tuple(adapters or ()))
        routing_policy = balanced_pdf_routing_policy(
            estimated_latency_ms=estimated_latency_ms,
            estimated_cost_microusd=estimated_cost_microusd,
            max_primary_latency_ms=max_primary_latency_ms,
            max_total_latency_ms=max_total_latency_ms,
            max_primary_cost_microusd=max_primary_cost_microusd,
            max_total_cost_microusd=max_total_cost_microusd,
        )
        return cls(
            active_registry,
            routing_policy,
            orchestration_policy=orchestration_policy,
            reconciliation_policy=reconciliation_policy,
            require_all_policy_adapters=True,
        )

    @classmethod
    def balanced(
        cls,
        *,
        registry: ParserRegistry | None = None,
        adapters: Iterable[ParserAdapter] | None = None,
        orchestration_policy: ParsingOrchestrationPolicy | None = None,
        reconciliation_policy: ReconciliationPolicy | None = None,
        estimated_latency_ms: Mapping[str, int] | None = None,
        estimated_cost_microusd: Mapping[str, int] | None = None,
        max_primary_latency_ms: int | None = None,
        max_total_latency_ms: int | None = None,
        max_primary_cost_microusd: int | None = None,
        max_total_cost_microusd: int | None = None,
    ) -> Self:
        """Deprecated alias for :meth:`routed_default`.

        ``balanced`` is the request-time query-budget name in the target API;
        retaining it as a build-router name would conflate independent axes.
        """

        warnings.warn(
            "RoutedBookRAGFacade.balanced() is deprecated; use "
            "RoutedBookRAGFacade.routed_default()",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.routed_default(
            registry=registry,
            adapters=adapters,
            orchestration_policy=orchestration_policy,
            reconciliation_policy=reconciliation_policy,
            estimated_latency_ms=estimated_latency_ms,
            estimated_cost_microusd=estimated_cost_microusd,
            max_primary_latency_ms=max_primary_latency_ms,
            max_total_latency_ms=max_total_latency_ms,
            max_primary_cost_microusd=max_primary_cost_microusd,
            max_total_cost_microusd=max_total_cost_microusd,
        )

    @property
    def registry(self) -> ParserRegistry:
        return self._registry

    @property
    def routing_policy(self) -> PageRoutingPolicy:
        return self._routing_policy

    @property
    def orchestration_policy(self) -> ParsingOrchestrationPolicy:
        return self._orchestration_policy

    @property
    def reconciliation_policy(self) -> ReconciliationPolicy:
        return self._reconciliation_policy

    def run(
        self,
        request: ParseRequest,
        page_features: Iterable[PageFeatures],
        *,
        prior_blocks: Iterable[CanonicalBlock] = (),
    ) -> RoutedBookRAGResult:
        """Run one exact source request through parsing and BookRAG projection."""

        features = self._validate_request(request, page_features)
        parsed = self._orchestrator.parse(
            request.document,
            features,
            options=request.options,
            prior_blocks=prior_blocks,
            run_id=request.run_id,
            trace_id=request.trace_id,
        )
        if tuple(page.features.page for page in parsed.pages) != request.pages:
            raise CompositionBindingError(
                "orchestrator returned different pages or page order"
            )
        return RoutedBookRAGResult(
            request=request,
            parsing=parsed,
        )

    def run_pages(
        self,
        document: DocumentRef,
        page_features: Iterable[PageFeatures],
        *,
        options: ParseOptions | None = None,
        prior_blocks: Iterable[CanonicalBlock] = (),
        run_id: str | None = None,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RoutedBookRAGResult:
        """Convenience wrapper that creates an explicit outer ``ParseRequest``."""

        features = tuple(page_features)
        # ParseRequest owns the authoritative DocumentRef type validation.
        request = ParseRequest(
            document=document,
            pages=tuple(feature.page for feature in features),
            options=options or ParseOptions(),
            run_id=run_id,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )
        return self.run(request, features, prior_blocks=prior_blocks)

    @staticmethod
    def _validate_request(
        request: ParseRequest,
        page_features: Iterable[PageFeatures],
    ) -> tuple[PageFeatures, ...]:
        if not isinstance(request, ParseRequest):
            raise TypeError("request must be ParseRequest")
        if request.regions:
            raise CompositionBindingError(
                "routed document facade does not accept region ParseRequests"
            )
        features = tuple(page_features)
        if not features:
            raise CompositionBindingError("page_features must not be empty")
        if any(not isinstance(feature, PageFeatures) for feature in features):
            raise TypeError("page_features must contain PageFeatures DTOs")
        feature_pages = tuple(feature.page for feature in features)
        if feature_pages != request.pages:
            raise CompositionBindingError(
                "page features must exactly match the outer request pages"
            )
        if any(feature.mime_type != request.document.mime_type for feature in features):
            raise CompositionBindingError(
                "page feature MIME type differs from the outer request"
            )
        return features


__all__ = [
    "BookRAGHandoffNotReadyError",
    "BookRAGHandoffReadiness",
    "COMPOSITION_SCHEMA_VERSION",
    "CompositionBindingError",
    "RoutedBookRAGFacade",
    "RoutedBookRAGResult",
    "UnsupportedCompositionSchemaError",
]
