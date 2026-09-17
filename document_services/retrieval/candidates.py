"""Bounded, backend-neutral retrieval candidates and deterministic RRF fusion.

This module is intentionally dependency-light.  Search backends translate their
results into :class:`CandidateHit` values, while a controller supplies the exact
tenant/corpus, document-version, policy, filter, and ready-index scope.  Fusion
then happens without importing a vector database, graph database, or model SDK.

The implementation uses ordinary reciprocal-rank fusion (RRF): each retrieval
leg contributes ``1 / (rrf_k + rank)`` at most once for an evidence identity.
Raw backend scores are retained for audit but never compared across backends.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import heapq
import math
import re
from types import MappingProxyType
from typing import ClassVar, Self

from document_services.domain.base import (
    ContractValidationError,
    JsonDTOMixin,
    content_sha256,
    json_ready,
    parse_json_object,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sha256,
)


RETRIEVAL_CANDIDATE_SCHEMA_VERSION = "docarag.retrieval-candidate.v1"

# Absolute guards stop a caller from turning an explicitly bounded request into
# an accidental unbounded allocation.  Deployments can choose lower values in
# every FusionBounds instance.
ABSOLUTE_MAX_LEGS = 64
ABSOLUTE_MAX_HITS_PER_LEG = 10_000
ABSOLUTE_MAX_OUTPUT = 10_000
ABSOLUTE_MAX_RRF_K = 1_000_000
MAX_IDENTIFIER_UTF8_BYTES = 512
MAX_MODALITY_UTF8_BYTES = 128
MAX_CANONICAL_REVISIONS_PER_HIT = 64
MAX_MATERIAL_IDS_PER_HIT = 128
MAX_CANONICAL_IDS_PER_DOCUMENT = 65_536
MAX_SELECTED_DOCUMENTS = 1_024
MAX_READY_INDEXES = 64
MAX_SUPPORTING_HITS_PER_CONTRIBUTION = ABSOLUTE_MAX_HITS_PER_LEG
MAX_BINDINGS_PER_FUSED_CANDIDATE = 8_192
MAX_MODALITIES_PER_FUSED_CANDIDATE = 128
MAX_SCOPE_IDENTITY_UTF8_BYTES = 4 * 1024 * 1024
MAX_TRACE_IDENTITY_UTF8_BYTES = 64 * 1024 * 1024
MAX_LEG_BUDGET_MS = 600_000
MAX_RECORDED_LATENCY_MS = 3_600_000
MAX_ATTEMPTS_PER_LEG = 8

# Wire-level limits are deliberately explicit and class-specific.  Cardinality
# checks cannot protect a process until after ``json.loads`` has already built
# the input tree, so every public retrieval DTO checks encoded bytes first.
# Composite DTOs receive larger envelopes while the small scalar contracts stay
# small enough to reject an obviously misrouted response cheaply.  Deployments
# should still enforce a lower endpoint/request body limit where appropriate.
_KIB = 1024
_MIB = 1024 * _KIB
MAX_DOCUMENT_EVIDENCE_BINDING_JSON_BYTES = 256 * _MIB
MAX_CANDIDATE_HIT_JSON_BYTES = 4 * _MIB
MAX_PLANNED_RETRIEVAL_LEG_JSON_BYTES = 64 * _KIB
MAX_CANDIDATE_QUERY_SCOPE_JSON_BYTES = 64 * _MIB
MAX_CANDIDATE_LEG_JSON_BYTES = 128 * _MIB
MAX_FUSION_BOUNDS_JSON_BYTES = 16 * _KIB
MAX_LEG_RANK_TRACE_JSON_BYTES = 128 * _MIB
MAX_RRF_CONTRIBUTION_JSON_BYTES = 64 * _MIB
MAX_FUSED_CANDIDATE_JSON_BYTES = 128 * _MIB
MAX_CANDIDATE_FUSION_RESULT_JSON_BYTES = 512 * _MIB

_LEG_STATUSES = frozenset({"success", "empty", "timeout", "error"})
_LEG_KINDS = frozenset({"bm25", "dense", "sparse", "entity", "graph"})
_GRAPH_LEG_KINDS = frozenset({"entity", "graph"})
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

_FILTER_REASONS = frozenset(
    {
        "document_access",
        "document_version",
        "duplicate_hit_id",
        "evidence_binding",
        "evidence_graph_release",
        "canonical_material",
        "filter_identity",
        "index_readiness",
        "policy_identity",
    }
)


def _require_bounded_json_payload(
    payload: object,
    *,
    dto: str,
    max_bytes: int,
) -> str | bytes | bytearray:
    """Reject an oversized serialized DTO before invoking ``json.loads``.

    For text input the character count is a zero-allocation lower bound on its
    UTF-8 size.  Only text no longer than the byte limit is encoded, and then in
    small chunks, so a multibyte string cannot create one additional payload-
    sized allocation merely to measure it.
    """

    if isinstance(payload, (bytes, bytearray)):
        payload_bytes = len(payload)
    elif isinstance(payload, str):
        if len(payload) > max_bytes:
            raise ContractValidationError(
                f"{dto} serialized JSON exceeds {max_bytes} bytes"
            )
        payload_bytes = 0
        chunk_chars = 64 * _KIB
        for start in range(0, len(payload), chunk_chars):
            try:
                encoded_chunk = payload[start : start + chunk_chars].encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ContractValidationError(
                    f"{dto} serialized JSON must be valid UTF-8 text"
                ) from exc
            payload_bytes += len(encoded_chunk)
            if payload_bytes > max_bytes:
                break
    else:
        raise ContractValidationError(
            f"{dto} serialized JSON must be str, bytes, or bytearray"
        )
    if payload_bytes > max_bytes:
        raise ContractValidationError(
            f"{dto} serialized JSON exceeds {max_bytes} bytes"
        )
    return payload


class _BoundedJsonDTOMixin(JsonDTOMixin):
    """JSON DTO surface with a mandatory pre-decode wire-size guard."""

    __slots__ = ()
    MAX_JSON_BYTES: ClassVar[int]

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        bounded = _require_bounded_json_payload(
            payload,
            dto=cls.__name__,
            max_bytes=cls.MAX_JSON_BYTES,
        )
        return cls.from_dict(parse_json_object(bounded))


def _require_schema(payload: Mapping[str, object]) -> None:
    actual = payload.get("schema_version")
    if actual != RETRIEVAL_CANDIDATE_SCHEMA_VERSION:
        raise ContractValidationError(
            "unsupported retrieval candidate schema_version "
            f"{actual!r}; expected {RETRIEVAL_CANDIDATE_SCHEMA_VERSION!r}"
        )


def _bounded_text(
    value: object,
    field_name: str,
    *,
    max_utf8_bytes: int = MAX_IDENTIFIER_UTF8_BYTES,
) -> str:
    result = require_non_empty(value, field_name)
    size = len(result.encode("utf-8"))
    if size > max_utf8_bytes:
        raise ContractValidationError(
            f"{field_name} exceeds {max_utf8_bytes} UTF-8 bytes"
        )
    return result


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, field_name)


def _optional_error_code(value: object) -> str | None:
    if value is None:
        return None
    result = _bounded_text(value, "error_code", max_utf8_bytes=64)
    if _ERROR_CODE.fullmatch(result) is None:
        raise ContractValidationError(
            "error_code must be a privacy-safe symbolic code"
        )
    return result


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _finite_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be a finite number")
    return result


def _bounded_sequence(
    value: object, field_name: str, *, max_items: int
) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{field_name} must be a finite sequence")
    if len(value) > max_items:
        raise ContractValidationError(
            f"{field_name} exceeds maximum cardinality {max_items}"
        )
    return value


def _preflight_utf8_payload(
    value: object,
    field_name: str,
    *,
    max_utf8_bytes: int,
    max_nesting: int = 32,
) -> int:
    """Bound nested string identity bytes without copying nested containers.

    DTO constructors normalize arrays into tuples.  Running this streaming
    walk first prevents a syntactically bounded outer payload from causing
    large nested allocations before its aggregate byte budget is checked.
    Mapping keys are schema field names, not identity content, so only values
    contribute to the budget.  Cyclic/non-tree inputs are rejected even though
    they cannot be produced by JSON decoding.
    """

    iterators = [iter((value,))]
    container_ids: list[int | None] = [None]
    active_container_ids: set[int] = set()
    total = 0
    while iterators:
        try:
            current = next(iterators[-1])
        except StopIteration:
            iterators.pop()
            container_id = container_ids.pop()
            if container_id is not None:
                active_container_ids.remove(container_id)
            continue

        if isinstance(current, str):
            text = _bounded_text(current, f"{field_name} string")
            total += len(text.encode("utf-8"))
            if total > max_utf8_bytes:
                raise ContractValidationError(
                    f"{field_name} exceeds the absolute UTF-8 byte limit"
                )
            continue

        nested: object | None = None
        if isinstance(current, Mapping):
            nested = current.values()
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            nested = current
        if nested is None:
            continue

        container_id = id(current)
        if container_id in active_container_ids:
            raise ContractValidationError(
                f"{field_name} must be an acyclic JSON tree"
            )
        if len(iterators) >= max_nesting:
            raise ContractValidationError(
                f"{field_name} exceeds maximum nesting depth {max_nesting}"
            )
        active_container_ids.add(container_id)
        iterators.append(iter(nested))  # type: ignore[arg-type]
        container_ids.append(container_id)
    return total


def _string_map(
    value: object, field_name: str, *, max_items: int
) -> Mapping[str, str]:
    mapping = require_mapping(value, field_name)
    if len(mapping) > max_items:
        raise ContractValidationError(
            f"{field_name} exceeds maximum cardinality {max_items}"
        )
    if any(not isinstance(key, str) for key in mapping):
        raise ContractValidationError(f"{field_name} contains a non-string key")
    normalized: dict[str, str] = {}
    for key in sorted(mapping):
        normalized[_bounded_text(key, f"{field_name} key")] = _bounded_text(
            mapping[key], f"{field_name}[{key!r}]"
        )
    return MappingProxyType(normalized)


def _stable_id(kind: str, *parts: object) -> str:
    require_non_empty(kind, "kind")
    digest = content_sha256(
        {
            "schema_version": RETRIEVAL_CANDIDATE_SCHEMA_VERSION,
            "kind": kind,
            "parts": parts,
        }
    )
    return f"{kind}_{digest[:32]}"


def _verify_generated(payload: Mapping[str, object], field: str, expected: str) -> None:
    if payload.get(field) != expected:
        raise ContractValidationError(f"{field} does not match canonical content")


def _bounded_ids(
    values: Sequence[object],
    field_name: str,
    *,
    max_items: int,
    non_empty: bool = False,
) -> tuple[str, ...]:
    values = _bounded_sequence(values, field_name, max_items=max_items)
    result = tuple(
        sorted({_bounded_text(value, f"{field_name}[]") for value in values})
    )
    if non_empty and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return result


def _add_bounded_unique_id(
    target: set[str],
    value: str,
    field_name: str,
    *,
    max_items: int,
) -> None:
    target.add(value)
    if len(target) > max_items:
        raise ContractValidationError(
            f"{field_name} exceeds maximum cardinality {max_items}"
        )


def _extend_bounded_unique_ids(
    target: set[str],
    values: Sequence[str],
    field_name: str,
    *,
    max_items: int,
) -> None:
    """Accumulate a bounded unique union without expanding repeated values."""

    for value in values:
        _add_bounded_unique_id(
            target,
            value,
            field_name,
            max_items=max_items,
        )


def _binding_id(
    *,
    document_id: str,
    document_version_id: str,
    source_sha256: str,
    canonical_artifact_sha256: str,
    tree_revision_id: str,
) -> str:
    return _stable_id(
        "evidence_binding",
        document_id,
        document_version_id,
        source_sha256,
        canonical_artifact_sha256,
        tree_revision_id,
    )


@dataclass(frozen=True, slots=True)
class DocumentEvidenceBinding(_BoundedJsonDTOMixin):
    """Authoritative source/canonical/tree binding for one selected document.

    The evidence-graph release is optional so base BM25 and dense indexes can be
    queried before knowledge-graph construction completes.  Entity and graph
    retrieval plans require the optional release pair to be adopted.
    """

    document_id: str
    document_version_id: str
    source_sha256: str
    canonical_artifact_sha256: str
    tree_revision_id: str
    evidence_graph_release_id: str | None = None
    evidence_graph_release_sha256: str | None = None
    canonical_revision_ids: tuple[str, ...] = ()
    canonical_representation_ids: tuple[str, ...] = ()
    canonical_asset_ids: tuple[str, ...] = ()
    evidence_binding_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_DOCUMENT_EVIDENCE_BINDING_JSON_BYTES

    def __post_init__(self) -> None:
        _bounded_text(self.document_id, "document_id")
        _bounded_text(self.document_version_id, "document_version_id")
        require_sha256(self.source_sha256, "source_sha256")
        require_sha256(
            self.canonical_artifact_sha256, "canonical_artifact_sha256"
        )
        _bounded_text(self.tree_revision_id, "tree_revision_id")
        graph_release_id = _optional_id(
            self.evidence_graph_release_id, "evidence_graph_release_id"
        )
        graph_release_sha = self.evidence_graph_release_sha256
        if (graph_release_id is None) != (graph_release_sha is None):
            raise ContractValidationError(
                "evidence graph release ID and SHA-256 must be supplied together"
            )
        if graph_release_sha is not None:
            require_sha256(
                graph_release_sha, "evidence_graph_release_sha256"
            )
        object.__setattr__(self, "evidence_graph_release_id", graph_release_id)
        canonical_revision_ids = _bounded_ids(
            self.canonical_revision_ids,
            "canonical_revision_ids",
            max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
        )
        canonical_representation_ids = _bounded_ids(
            self.canonical_representation_ids,
            "canonical_representation_ids",
            max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
        )
        canonical_asset_ids = _bounded_ids(
            self.canonical_asset_ids,
            "canonical_asset_ids",
            max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
        )
        object.__setattr__(self, "canonical_revision_ids", canonical_revision_ids)
        object.__setattr__(
            self, "canonical_representation_ids", canonical_representation_ids
        )
        object.__setattr__(self, "canonical_asset_ids", canonical_asset_ids)
        object.__setattr__(
            self,
            "evidence_binding_id",
            _binding_id(
                document_id=self.document_id,
                document_version_id=self.document_version_id,
                source_sha256=self.source_sha256,
                canonical_artifact_sha256=self.canonical_artifact_sha256,
                tree_revision_id=self.tree_revision_id,
            ),
        )

    @property
    def identity_utf8_bytes(self) -> int:
        scalar_bytes = sum(
            len(value.encode("utf-8"))
            for value in (
                self.document_id,
                self.document_version_id,
                self.source_sha256,
                self.canonical_artifact_sha256,
                self.tree_revision_id,
                self.evidence_graph_release_id or "",
                self.evidence_graph_release_sha256 or "",
                self.evidence_binding_id,
            )
        )
        material_bytes = sum(
            len(value.encode("utf-8"))
            for value in (
                self.canonical_revision_ids
                + self.canonical_representation_ids
                + self.canonical_asset_ids
            )
        )
        return scalar_bytes + material_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "evidence_binding_id": self.evidence_binding_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "tree_revision_id": self.tree_revision_id,
            "evidence_graph_release_id": self.evidence_graph_release_id,
            "evidence_graph_release_sha256": self.evidence_graph_release_sha256,
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "canonical_representation_ids": list(
                self.canonical_representation_ids
            ),
            "canonical_asset_ids": list(self.canonical_asset_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "evidence_binding_id",
                "document_id",
                "document_version_id",
                "source_sha256",
                "canonical_artifact_sha256",
                "tree_revision_id",
                "evidence_graph_release_id",
                "evidence_graph_release_sha256",
                "canonical_revision_ids",
                "canonical_representation_ids",
                "canonical_asset_ids",
            },
            dto="DocumentEvidenceBinding",
        )
        result = cls(
            document_id=payload.get("document_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            canonical_artifact_sha256=payload.get("canonical_artifact_sha256"),  # type: ignore[arg-type]
            tree_revision_id=payload.get("tree_revision_id"),  # type: ignore[arg-type]
            evidence_graph_release_id=payload.get("evidence_graph_release_id"),  # type: ignore[arg-type]
            evidence_graph_release_sha256=payload.get("evidence_graph_release_sha256"),  # type: ignore[arg-type]
            canonical_revision_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_revision_ids", ()),
                    "canonical_revision_ids",
                    max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
                )
            ),  # type: ignore[arg-type]
            canonical_representation_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_representation_ids", ()),
                    "canonical_representation_ids",
                    max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
                )
            ),  # type: ignore[arg-type]
            canonical_asset_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_asset_ids", ()),
                    "canonical_asset_ids",
                    max_items=MAX_CANONICAL_IDS_PER_DOCUMENT,
                )
            ),  # type: ignore[arg-type]
        )
        _verify_generated(
            payload, "evidence_binding_id", result.evidence_binding_id
        )
        return result


@dataclass(frozen=True, slots=True)
class CandidateHit(_BoundedJsonDTOMixin):
    """One immutable ranked hit returned by a retrieval backend.

    Exactly one of ``canonical_evidence_id`` and ``derived_evidence_id`` is
    required.  ``hit_id`` is assigned by the producing retrieval leg and must
    be stable for that result.  The fusion boundary rejects reuse of a hit ID
    with conflicting content.
    """

    hit_id: str
    tenant_id: str
    corpus_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    canonical_artifact_sha256: str
    tree_revision_id: str
    tree_node_id: str
    modality: str
    backend_id: str
    leg_kind: str
    index_revision_id: str
    raw_score: float
    rank: int
    payload_identity: str
    filter_identity: str
    policy_identity: str
    canonical_evidence_id: str | None = None
    derived_evidence_id: str | None = None
    evidence_graph_release_id: str | None = None
    evidence_graph_release_sha256: str | None = None
    canonical_revision_ids: tuple[str, ...] = ()
    canonical_representation_ids: tuple[str, ...] = ()
    canonical_asset_ids: tuple[str, ...] = ()
    selected_representation_ids: tuple[str, ...] = ()
    selected_asset_ids: tuple[str, ...] = ()
    evidence_binding_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_CANDIDATE_HIT_JSON_BYTES

    def __post_init__(self) -> None:
        for field_name in (
            "hit_id",
            "tenant_id",
            "corpus_id",
            "document_id",
            "document_version_id",
            "tree_revision_id",
            "tree_node_id",
            "modality",
            "backend_id",
            "leg_kind",
            "index_revision_id",
            "payload_identity",
            "filter_identity",
            "policy_identity",
        ):
            max_bytes = (
                MAX_MODALITY_UTF8_BYTES
                if field_name == "modality"
                else MAX_IDENTIFIER_UTF8_BYTES
            )
            _bounded_text(
                getattr(self, field_name), field_name, max_utf8_bytes=max_bytes
            )
        require_sha256(self.source_sha256, "source_sha256")
        require_sha256(
            self.canonical_artifact_sha256, "canonical_artifact_sha256"
        )
        require_sha256(self.payload_identity, "payload_identity")
        if self.leg_kind not in _LEG_KINDS:
            raise ContractValidationError(
                f"leg_kind must be one of {sorted(_LEG_KINDS)}"
            )
        graph_release_id = _optional_id(
            self.evidence_graph_release_id, "evidence_graph_release_id"
        )
        graph_release_sha = self.evidence_graph_release_sha256
        if (graph_release_id is None) != (graph_release_sha is None):
            raise ContractValidationError(
                "evidence graph release ID and SHA-256 must be supplied together"
            )
        if graph_release_sha is not None:
            require_sha256(
                graph_release_sha, "evidence_graph_release_sha256"
            )
        if self.leg_kind in _GRAPH_LEG_KINDS and graph_release_id is None:
            raise ContractValidationError(
                "entity/graph hits require an adopted evidence graph release"
            )
        object.__setattr__(self, "evidence_graph_release_id", graph_release_id)

        canonical_id = _optional_id(
            self.canonical_evidence_id, "canonical_evidence_id"
        )
        derived_id = _optional_id(self.derived_evidence_id, "derived_evidence_id")
        if (canonical_id is None) == (derived_id is None):
            raise ContractValidationError(
                "CandidateHit requires exactly one of canonical_evidence_id or "
                "derived_evidence_id"
            )

        object.__setattr__(self, "canonical_evidence_id", canonical_id)
        object.__setattr__(self, "derived_evidence_id", derived_id)
        canonical_revision_ids = _bounded_ids(
            self.canonical_revision_ids,
            "canonical_revision_ids",
            max_items=MAX_CANONICAL_REVISIONS_PER_HIT,
            non_empty=True,
        )
        canonical_representation_ids = _bounded_ids(
            self.canonical_representation_ids,
            "canonical_representation_ids",
            max_items=MAX_MATERIAL_IDS_PER_HIT,
        )
        canonical_asset_ids = _bounded_ids(
            self.canonical_asset_ids,
            "canonical_asset_ids",
            max_items=MAX_MATERIAL_IDS_PER_HIT,
        )
        selected_representation_ids = _bounded_ids(
            self.selected_representation_ids,
            "selected_representation_ids",
            max_items=MAX_MATERIAL_IDS_PER_HIT,
        )
        selected_asset_ids = _bounded_ids(
            self.selected_asset_ids,
            "selected_asset_ids",
            max_items=MAX_MATERIAL_IDS_PER_HIT,
        )
        if not set(selected_representation_ids).issubset(
            canonical_representation_ids
        ):
            raise ContractValidationError(
                "selected_representation_ids must be a subset of "
                "canonical_representation_ids"
            )
        if not set(selected_asset_ids).issubset(canonical_asset_ids):
            raise ContractValidationError(
                "selected_asset_ids must be a subset of canonical_asset_ids"
            )
        object.__setattr__(self, "canonical_revision_ids", canonical_revision_ids)
        object.__setattr__(
            self, "canonical_representation_ids", canonical_representation_ids
        )
        object.__setattr__(self, "canonical_asset_ids", canonical_asset_ids)
        object.__setattr__(
            self, "selected_representation_ids", selected_representation_ids
        )
        object.__setattr__(self, "selected_asset_ids", selected_asset_ids)
        object.__setattr__(
            self, "raw_score", _finite_score(self.raw_score, "raw_score")
        )
        rank = _positive_int(self.rank, "rank")
        if rank > ABSOLUTE_MAX_HITS_PER_LEG:
            raise ContractValidationError(
                f"rank exceeds absolute limit {ABSOLUTE_MAX_HITS_PER_LEG}"
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(
            self,
            "evidence_binding_id",
            _binding_id(
                document_id=self.document_id,
                document_version_id=self.document_version_id,
                source_sha256=self.source_sha256,
                canonical_artifact_sha256=self.canonical_artifact_sha256,
                tree_revision_id=self.tree_revision_id,
            ),
        )

    @property
    def evidence_kind(self) -> str:
        return "canonical" if self.canonical_evidence_id is not None else "derived"

    @property
    def evidence_id(self) -> str:
        value = self.canonical_evidence_id or self.derived_evidence_id
        assert value is not None  # established by __post_init__
        return value

    @property
    def dedup_key(self) -> tuple[str, str, str, str, str, str, str]:
        """Stable scope-aware identity used for cross-leg deduplication."""

        return (
            self.tenant_id,
            self.corpus_id,
            self.document_id,
            self.document_version_id,
            self.evidence_binding_id,
            self.evidence_kind,
            self.evidence_id,
        )

    @property
    def identity_utf8_bytes(self) -> int:
        scalar_values = (
            self.hit_id,
            self.tenant_id,
            self.corpus_id,
            self.document_id,
            self.document_version_id,
            self.source_sha256,
            self.canonical_artifact_sha256,
            self.tree_revision_id,
            self.evidence_graph_release_id or "",
            self.evidence_graph_release_sha256 or "",
            self.evidence_binding_id,
            self.tree_node_id,
            self.evidence_id,
            self.modality,
            self.backend_id,
            self.leg_kind,
            self.index_revision_id,
            self.payload_identity,
            self.filter_identity,
            self.policy_identity,
        )
        plural_values = (
            self.canonical_revision_ids
            + self.canonical_representation_ids
            + self.canonical_asset_ids
            + self.selected_representation_ids
            + self.selected_asset_ids
        )
        return sum(len(value.encode("utf-8")) for value in scalar_values + plural_values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "hit_id": self.hit_id,
            "tenant_id": self.tenant_id,
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "tree_revision_id": self.tree_revision_id,
            "evidence_graph_release_id": self.evidence_graph_release_id,
            "evidence_graph_release_sha256": self.evidence_graph_release_sha256,
            "evidence_binding_id": self.evidence_binding_id,
            "tree_node_id": self.tree_node_id,
            "canonical_evidence_id": self.canonical_evidence_id,
            "derived_evidence_id": self.derived_evidence_id,
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "canonical_representation_ids": list(
                self.canonical_representation_ids
            ),
            "canonical_asset_ids": list(self.canonical_asset_ids),
            "selected_representation_ids": list(
                self.selected_representation_ids
            ),
            "selected_asset_ids": list(self.selected_asset_ids),
            "modality": self.modality,
            "backend_id": self.backend_id,
            "leg_kind": self.leg_kind,
            "index_revision_id": self.index_revision_id,
            "raw_score": self.raw_score,
            "rank": self.rank,
            "payload_identity": self.payload_identity,
            "filter_identity": self.filter_identity,
            "policy_identity": self.policy_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "hit_id",
                "tenant_id",
                "corpus_id",
                "document_id",
                "document_version_id",
                "source_sha256",
                "canonical_artifact_sha256",
                "tree_revision_id",
                "evidence_graph_release_id",
                "evidence_graph_release_sha256",
                "evidence_binding_id",
                "tree_node_id",
                "canonical_evidence_id",
                "derived_evidence_id",
                "canonical_revision_ids",
                "canonical_representation_ids",
                "canonical_asset_ids",
                "selected_representation_ids",
                "selected_asset_ids",
                "modality",
                "backend_id",
                "leg_kind",
                "index_revision_id",
                "raw_score",
                "rank",
                "payload_identity",
                "filter_identity",
                "policy_identity",
            },
            dto="CandidateHit",
        )
        result = cls(
            hit_id=payload.get("hit_id"),  # type: ignore[arg-type]
            tenant_id=payload.get("tenant_id"),  # type: ignore[arg-type]
            corpus_id=payload.get("corpus_id"),  # type: ignore[arg-type]
            document_id=payload.get("document_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            canonical_artifact_sha256=payload.get("canonical_artifact_sha256"),  # type: ignore[arg-type]
            tree_revision_id=payload.get("tree_revision_id"),  # type: ignore[arg-type]
            evidence_graph_release_id=payload.get("evidence_graph_release_id"),  # type: ignore[arg-type]
            evidence_graph_release_sha256=payload.get("evidence_graph_release_sha256"),  # type: ignore[arg-type]
            tree_node_id=payload.get("tree_node_id"),  # type: ignore[arg-type]
            canonical_evidence_id=payload.get("canonical_evidence_id"),  # type: ignore[arg-type]
            derived_evidence_id=payload.get("derived_evidence_id"),  # type: ignore[arg-type]
            canonical_revision_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_revision_ids", ()),
                    "canonical_revision_ids",
                    max_items=MAX_CANONICAL_REVISIONS_PER_HIT,
                )
            ),  # type: ignore[arg-type]
            canonical_representation_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_representation_ids", ()),
                    "canonical_representation_ids",
                    max_items=MAX_MATERIAL_IDS_PER_HIT,
                )
            ),  # type: ignore[arg-type]
            canonical_asset_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_asset_ids", ()),
                    "canonical_asset_ids",
                    max_items=MAX_MATERIAL_IDS_PER_HIT,
                )
            ),  # type: ignore[arg-type]
            selected_representation_ids=tuple(
                _bounded_sequence(
                    payload.get("selected_representation_ids", ()),
                    "selected_representation_ids",
                    max_items=MAX_MATERIAL_IDS_PER_HIT,
                )
            ),  # type: ignore[arg-type]
            selected_asset_ids=tuple(
                _bounded_sequence(
                    payload.get("selected_asset_ids", ()),
                    "selected_asset_ids",
                    max_items=MAX_MATERIAL_IDS_PER_HIT,
                )
            ),  # type: ignore[arg-type]
            modality=payload.get("modality"),  # type: ignore[arg-type]
            backend_id=payload.get("backend_id"),  # type: ignore[arg-type]
            leg_kind=payload.get("leg_kind"),  # type: ignore[arg-type]
            index_revision_id=payload.get("index_revision_id"),  # type: ignore[arg-type]
            raw_score=payload.get("raw_score"),  # type: ignore[arg-type]
            rank=payload.get("rank"),  # type: ignore[arg-type]
            payload_identity=payload.get("payload_identity"),  # type: ignore[arg-type]
            filter_identity=payload.get("filter_identity"),  # type: ignore[arg-type]
            policy_identity=payload.get("policy_identity"),  # type: ignore[arg-type]
        )
        _verify_generated(
            payload, "evidence_binding_id", result.evidence_binding_id
        )
        return result


@dataclass(frozen=True, slots=True)
class PlannedRetrievalLeg(_BoundedJsonDTOMixin):
    """One immutable leg in the exact retrieval plan."""

    leg_id: str
    leg_kind: str
    backend_id: str
    index_revision_id: str
    leg_config_sha256: str
    max_hits: int
    budget_ms: int
    plan_leg_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_PLANNED_RETRIEVAL_LEG_JSON_BYTES

    def __post_init__(self) -> None:
        _bounded_text(self.leg_id, "leg_id")
        _bounded_text(self.leg_kind, "leg_kind", max_utf8_bytes=32)
        if self.leg_kind not in _LEG_KINDS:
            raise ContractValidationError(
                f"leg_kind must be one of {sorted(_LEG_KINDS)}"
            )
        _bounded_text(self.backend_id, "backend_id")
        _bounded_text(self.index_revision_id, "index_revision_id")
        require_sha256(self.leg_config_sha256, "leg_config_sha256")
        max_hits = _positive_int(self.max_hits, "max_hits")
        if max_hits > ABSOLUTE_MAX_HITS_PER_LEG:
            raise ContractValidationError(
                f"max_hits exceeds absolute limit {ABSOLUTE_MAX_HITS_PER_LEG}"
            )
        budget_ms = _positive_int(self.budget_ms, "budget_ms")
        if budget_ms > MAX_LEG_BUDGET_MS:
            raise ContractValidationError(
                f"budget_ms exceeds absolute limit {MAX_LEG_BUDGET_MS}"
            )
        object.__setattr__(
            self,
            "plan_leg_id",
            _stable_id(
                "retrieval_leg",
                self.leg_id,
                self.leg_kind,
                self.backend_id,
                self.index_revision_id,
                self.leg_config_sha256,
                max_hits,
                budget_ms,
            ),
        )

    @property
    def identity_utf8_bytes(self) -> int:
        return sum(
            len(value.encode("utf-8"))
            for value in (
                self.leg_id,
                self.leg_kind,
                self.backend_id,
                self.index_revision_id,
                self.leg_config_sha256,
                self.plan_leg_id,
            )
        )

    @property
    def semantic_signature(self) -> tuple[str, str, str, str]:
        """Backend query identity, excluding roster ID and resource ceilings.

        Different hit/deadline ceilings for the same backend query still
        produce overlapping rankings.  Treating them as separate RRF legs
        would silently weight that backend twice; intentional diversity must
        therefore use a distinct, revision-bound ``leg_config_sha256``.
        """

        return (
            self.leg_kind,
            self.backend_id,
            self.index_revision_id,
            self.leg_config_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_leg_id": self.plan_leg_id,
            "leg_id": self.leg_id,
            "leg_kind": self.leg_kind,
            "backend_id": self.backend_id,
            "index_revision_id": self.index_revision_id,
            "leg_config_sha256": self.leg_config_sha256,
            "max_hits": self.max_hits,
            "budget_ms": self.budget_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "plan_leg_id",
                "leg_id",
                "leg_kind",
                "backend_id",
                "index_revision_id",
                "leg_config_sha256",
                "max_hits",
                "budget_ms",
            },
            dto="PlannedRetrievalLeg",
        )
        result = cls(
            leg_id=payload.get("leg_id"),  # type: ignore[arg-type]
            leg_kind=payload.get("leg_kind"),  # type: ignore[arg-type]
            backend_id=payload.get("backend_id"),  # type: ignore[arg-type]
            index_revision_id=payload.get("index_revision_id"),  # type: ignore[arg-type]
            leg_config_sha256=payload.get("leg_config_sha256"),  # type: ignore[arg-type]
            max_hits=payload.get("max_hits"),  # type: ignore[arg-type]
            budget_ms=payload.get("budget_ms"),  # type: ignore[arg-type]
        )
        _verify_generated(payload, "plan_leg_id", result.plan_leg_id)
        return result


@dataclass(frozen=True, slots=True)
class CandidateQueryScope(_BoundedJsonDTOMixin):
    """Preselected documents and exact retrieval plan authorized for one query."""

    tenant_id: str
    corpus_id: str
    filter_identity: str
    policy_identity: str
    query_sha256: str
    retrieval_config_sha256: str
    document_bindings: tuple[DocumentEvidenceBinding, ...]
    planned_legs: tuple[PlannedRetrievalLeg, ...]
    ready_index_revisions: Mapping[str, str] = field(default_factory=dict)
    retrieval_plan_sha256: str = field(init=False)
    scope_id: str = field(init=False)
    allowed_document_versions: Mapping[str, str] = field(
        init=False, repr=False, compare=False
    )
    document_bindings_by_id: Mapping[str, DocumentEvidenceBinding] = field(
        init=False, repr=False, compare=False
    )
    planned_legs_by_id: Mapping[str, PlannedRetrievalLeg] = field(
        init=False, repr=False, compare=False
    )

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_CANDIDATE_QUERY_SCOPE_JSON_BYTES

    def __post_init__(self) -> None:
        for field_name in (
            "tenant_id",
            "corpus_id",
            "filter_identity",
            "policy_identity",
        ):
            _bounded_text(getattr(self, field_name), field_name)
        require_sha256(self.query_sha256, "query_sha256")
        require_sha256(
            self.retrieval_config_sha256, "retrieval_config_sha256"
        )

        raw_bindings = _bounded_sequence(
            self.document_bindings,
            "document_bindings",
            max_items=MAX_SELECTED_DOCUMENTS,
        )
        if any(not isinstance(item, DocumentEvidenceBinding) for item in raw_bindings):
            raise ContractValidationError(
                "document_bindings must contain DocumentEvidenceBinding values"
            )
        bindings = tuple(sorted(raw_bindings, key=lambda item: item.document_id))
        if len({item.document_id for item in bindings}) != len(bindings):
            raise ContractValidationError(
                "document_bindings must have unique document_id values"
            )

        raw_legs = _bounded_sequence(
            self.planned_legs, "planned_legs", max_items=ABSOLUTE_MAX_LEGS
        )
        if not raw_legs or any(
            not isinstance(item, PlannedRetrievalLeg) for item in raw_legs
        ):
            raise ContractValidationError(
                "planned_legs must contain at least one PlannedRetrievalLeg"
            )
        planned_legs = tuple(sorted(raw_legs, key=lambda item: item.leg_id))
        if len({item.leg_id for item in planned_legs}) != len(planned_legs):
            raise ContractValidationError(
                "planned_legs must have unique leg_id values"
            )
        if len({item.semantic_signature for item in planned_legs}) != len(
            planned_legs
        ):
            raise ContractValidationError(
                "planned_legs must not contain duplicate backend query semantics"
            )
        if any(item.leg_kind in _GRAPH_LEG_KINDS for item in planned_legs):
            if not bindings or any(
                item.evidence_graph_release_id is None for item in bindings
            ):
                raise ContractValidationError(
                    "entity/graph retrieval plans require an adopted evidence "
                    "graph release for every selected document"
                )

        ready = _string_map(
            self.ready_index_revisions,
            "ready_index_revisions",
            max_items=MAX_READY_INDEXES,
        )
        for leg in planned_legs:
            if ready.get(leg.backend_id) != leg.index_revision_id:
                raise ContractValidationError(
                    "every planned leg must name an exactly ready backend/index revision"
                )
        identity_bytes = sum(item.identity_utf8_bytes for item in bindings)
        identity_bytes += sum(item.identity_utf8_bytes for item in planned_legs)
        identity_bytes += sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8"))
            for key, value in ready.items()
        )
        identity_bytes += sum(
            len(value.encode("utf-8"))
            for value in (
                self.tenant_id,
                self.corpus_id,
                self.filter_identity,
                self.policy_identity,
                self.query_sha256,
                self.retrieval_config_sha256,
            )
        )
        if identity_bytes > MAX_SCOPE_IDENTITY_UTF8_BYTES:
            raise ContractValidationError(
                "query scope identity payload exceeds the absolute byte limit"
            )

        binding_map = MappingProxyType(
            {item.document_id: item for item in bindings}
        )
        leg_map = MappingProxyType({item.leg_id: item for item in planned_legs})
        plan_sha = content_sha256(
            {
                "query_sha256": self.query_sha256,
                "retrieval_config_sha256": self.retrieval_config_sha256,
                "planned_legs": [item.to_dict() for item in planned_legs],
            }
        )
        object.__setattr__(self, "document_bindings", bindings)
        object.__setattr__(self, "planned_legs", planned_legs)
        object.__setattr__(self, "ready_index_revisions", ready)
        object.__setattr__(self, "document_bindings_by_id", binding_map)
        object.__setattr__(self, "planned_legs_by_id", leg_map)
        object.__setattr__(
            self,
            "allowed_document_versions",
            MappingProxyType(
                {item.document_id: item.document_version_id for item in bindings}
            ),
        )
        object.__setattr__(self, "retrieval_plan_sha256", plan_sha)
        object.__setattr__(
            self,
            "scope_id",
            _stable_id(
                "retrieval_scope",
                self.tenant_id,
                self.corpus_id,
                self.filter_identity,
                self.policy_identity,
                self.query_sha256,
                self.retrieval_config_sha256,
                plan_sha,
                [item.to_dict() for item in bindings],
                dict(ready),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "scope_id": self.scope_id,
            "retrieval_plan_sha256": self.retrieval_plan_sha256,
            "tenant_id": self.tenant_id,
            "corpus_id": self.corpus_id,
            "filter_identity": self.filter_identity,
            "policy_identity": self.policy_identity,
            "query_sha256": self.query_sha256,
            "retrieval_config_sha256": self.retrieval_config_sha256,
            "document_bindings": [item.to_dict() for item in self.document_bindings],
            "planned_legs": [item.to_dict() for item in self.planned_legs],
            "ready_index_revisions": dict(self.ready_index_revisions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "scope_id",
                "retrieval_plan_sha256",
                "tenant_id",
                "corpus_id",
                "filter_identity",
                "policy_identity",
                "query_sha256",
                "retrieval_config_sha256",
                "document_bindings",
                "planned_legs",
                "ready_index_revisions",
            },
            dto="CandidateQueryScope",
        )
        raw_bindings = _bounded_sequence(
            payload.get("document_bindings", ()),
            "document_bindings",
            max_items=MAX_SELECTED_DOCUMENTS,
        )
        raw_legs = _bounded_sequence(
            payload.get("planned_legs", ()),
            "planned_legs",
            max_items=ABSOLUTE_MAX_LEGS,
        )
        ready_index_revisions = _string_map(
            payload.get("ready_index_revisions", {}),
            "ready_index_revisions",
            max_items=MAX_READY_INDEXES,
        )
        # Enforce the aggregate request-scope budget before any binding turns
        # its potentially large material-ID arrays into normalized tuples.
        _preflight_utf8_payload(
            payload,
            "query scope identity payload",
            max_utf8_bytes=MAX_SCOPE_IDENTITY_UTF8_BYTES,
        )
        result = cls(
            tenant_id=payload.get("tenant_id"),  # type: ignore[arg-type]
            corpus_id=payload.get("corpus_id"),  # type: ignore[arg-type]
            filter_identity=payload.get("filter_identity"),  # type: ignore[arg-type]
            policy_identity=payload.get("policy_identity"),  # type: ignore[arg-type]
            query_sha256=payload.get("query_sha256"),  # type: ignore[arg-type]
            retrieval_config_sha256=payload.get("retrieval_config_sha256"),  # type: ignore[arg-type]
            document_bindings=tuple(
                DocumentEvidenceBinding.from_dict(
                    require_mapping(item, "document_bindings[]")
                )
                for item in raw_bindings
            ),
            planned_legs=tuple(
                PlannedRetrievalLeg.from_dict(
                    require_mapping(item, "planned_legs[]")
                )
                for item in raw_legs
            ),
            ready_index_revisions=ready_index_revisions,
        )
        _verify_generated(
            payload, "retrieval_plan_sha256", result.retrieval_plan_sha256
        )
        _verify_generated(payload, "scope_id", result.scope_id)
        return result


# A short alias is convenient for API users while keeping the longer class name
# unambiguous in serialized/debug output.
QueryScope = CandidateQueryScope


def _filter_reason(hit: CandidateHit, scope: CandidateQueryScope) -> str | None:
    """Return one deterministic rejection reason, or ``None`` when eligible."""

    ready_revision = scope.ready_index_revisions.get(hit.backend_id)
    if ready_revision != hit.index_revision_id:
        return "index_readiness"
    if hit.policy_identity != scope.policy_identity:
        return "policy_identity"
    if hit.filter_identity != scope.filter_identity:
        return "filter_identity"
    binding = scope.document_bindings_by_id.get(hit.document_id)
    if binding is None:
        return "document_access"
    if binding.document_version_id != hit.document_version_id:
        return "document_version"
    if binding.evidence_binding_id != hit.evidence_binding_id:
        return "evidence_binding"
    if hit.evidence_graph_release_id is not None and (
        binding.evidence_graph_release_id != hit.evidence_graph_release_id
        or binding.evidence_graph_release_sha256
        != hit.evidence_graph_release_sha256
    ):
        return "evidence_graph_release"
    if not set(hit.canonical_revision_ids).issubset(binding.canonical_revision_ids):
        return "canonical_material"
    if not set(hit.canonical_representation_ids).issubset(
        binding.canonical_representation_ids
    ):
        return "canonical_material"
    if not set(hit.canonical_asset_ids).issubset(binding.canonical_asset_ids):
        return "canonical_material"
    return None


@dataclass(frozen=True, slots=True)
class CandidateLeg(_BoundedJsonDTOMixin):
    """One ranked backend result list participating in fusion."""

    leg_id: str
    leg_kind: str
    backend_id: str
    index_revision_id: str
    query_sha256: str
    retrieval_config_sha256: str
    leg_config_sha256: str
    hits: tuple[CandidateHit, ...] = ()
    status: str = "success"
    latency_ms: int = 0
    attempt_count: int = 1
    error_code: str | None = None

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_CANDIDATE_LEG_JSON_BYTES

    def __post_init__(self) -> None:
        _bounded_text(self.leg_id, "leg_id")
        _bounded_text(self.leg_kind, "leg_kind", max_utf8_bytes=32)
        if self.leg_kind not in _LEG_KINDS:
            raise ContractValidationError(
                f"leg_kind must be one of {sorted(_LEG_KINDS)}"
            )
        _bounded_text(self.backend_id, "backend_id")
        _bounded_text(self.index_revision_id, "index_revision_id")
        require_sha256(self.query_sha256, "query_sha256")
        require_sha256(
            self.retrieval_config_sha256, "retrieval_config_sha256"
        )
        require_sha256(self.leg_config_sha256, "leg_config_sha256")
        status = _bounded_text(self.status, "status", max_utf8_bytes=32)
        if status not in _LEG_STATUSES:
            raise ContractValidationError(
                f"status must be one of {sorted(_LEG_STATUSES)}"
            )
        latency_ms = _non_negative_int(self.latency_ms, "latency_ms")
        if latency_ms > MAX_RECORDED_LATENCY_MS:
            raise ContractValidationError(
                f"latency_ms exceeds absolute limit {MAX_RECORDED_LATENCY_MS}"
            )
        attempts = _positive_int(self.attempt_count, "attempt_count")
        if attempts > MAX_ATTEMPTS_PER_LEG:
            raise ContractValidationError(
                f"attempt_count exceeds absolute limit {MAX_ATTEMPTS_PER_LEG}"
            )
        error_code = _optional_error_code(self.error_code)
        hits = tuple(
            _bounded_sequence(
                self.hits, "hits", max_items=ABSOLUTE_MAX_HITS_PER_LEG
            )
        )
        if any(not isinstance(hit, CandidateHit) for hit in hits):
            raise ContractValidationError("hits must contain CandidateHit values")
        hits = tuple(sorted(hits, key=lambda hit: (hit.rank, hit.hit_id)))
        expected_ranks = tuple(range(1, len(hits) + 1))
        actual_ranks = tuple(hit.rank for hit in hits)
        if actual_ranks != expected_ranks:
            raise ContractValidationError(
                "CandidateLeg ranks must be unique and contiguous from 1"
            )
        for hit in hits:
            if (
                hit.backend_id != self.backend_id
                or hit.index_revision_id != self.index_revision_id
                or hit.leg_kind != self.leg_kind
            ):
                raise ContractValidationError(
                    "CandidateLeg hit kind/backend/index does not match the leg"
                )
        if status == "success" and not hits:
            raise ContractValidationError("success leg must contain at least one hit")
        if status != "success" and hits:
            raise ContractValidationError(
                "only a success leg may contain candidate hits"
            )
        if status == "error" and error_code is None:
            raise ContractValidationError("error leg requires a privacy-safe error_code")
        if status in {"success", "empty"} and error_code is not None:
            raise ContractValidationError(
                "success/empty legs must not carry an error_code"
            )
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "error_code", error_code)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "leg_id": self.leg_id,
            "leg_kind": self.leg_kind,
            "backend_id": self.backend_id,
            "index_revision_id": self.index_revision_id,
            "query_sha256": self.query_sha256,
            "retrieval_config_sha256": self.retrieval_config_sha256,
            "leg_config_sha256": self.leg_config_sha256,
            "hits": [hit.to_dict() for hit in self.hits],
            "status": self.status,
            "latency_ms": self.latency_ms,
            "attempt_count": self.attempt_count,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "leg_id",
                "leg_kind",
                "backend_id",
                "index_revision_id",
                "query_sha256",
                "retrieval_config_sha256",
                "leg_config_sha256",
                "hits",
                "status",
                "latency_ms",
                "attempt_count",
                "error_code",
            },
            dto="CandidateLeg",
        )
        raw_hits = _bounded_sequence(
            payload.get("hits", ()),
            "hits",
            max_items=ABSOLUTE_MAX_HITS_PER_LEG,
        )
        return cls(
            leg_id=payload.get("leg_id"),  # type: ignore[arg-type]
            leg_kind=payload.get("leg_kind"),  # type: ignore[arg-type]
            backend_id=payload.get("backend_id"),  # type: ignore[arg-type]
            index_revision_id=payload.get("index_revision_id"),  # type: ignore[arg-type]
            query_sha256=payload.get("query_sha256"),  # type: ignore[arg-type]
            retrieval_config_sha256=payload.get("retrieval_config_sha256"),  # type: ignore[arg-type]
            leg_config_sha256=payload.get("leg_config_sha256"),  # type: ignore[arg-type]
            hits=tuple(
                CandidateHit.from_dict(require_mapping(hit, "hits[]"))
                for hit in raw_hits
            ),
            status=payload.get("status"),  # type: ignore[arg-type]
            latency_ms=payload.get("latency_ms"),  # type: ignore[arg-type]
            attempt_count=payload.get("attempt_count"),  # type: ignore[arg-type]
            error_code=payload.get("error_code"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class FusionBounds(_BoundedJsonDTOMixin):
    """Explicit request bounds, additionally constrained by absolute guards."""

    max_legs: int
    max_hits_per_leg: int
    max_output: int

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_FUSION_BOUNDS_JSON_BYTES

    def __post_init__(self) -> None:
        max_legs = _positive_int(self.max_legs, "max_legs")
        max_hits = _positive_int(self.max_hits_per_leg, "max_hits_per_leg")
        max_output = _positive_int(self.max_output, "max_output")
        if max_legs > ABSOLUTE_MAX_LEGS:
            raise ContractValidationError(
                f"max_legs exceeds absolute limit {ABSOLUTE_MAX_LEGS}"
            )
        if max_hits > ABSOLUTE_MAX_HITS_PER_LEG:
            raise ContractValidationError(
                "max_hits_per_leg exceeds absolute limit "
                f"{ABSOLUTE_MAX_HITS_PER_LEG}"
            )
        if max_output > ABSOLUTE_MAX_OUTPUT:
            raise ContractValidationError(
                f"max_output exceeds absolute limit {ABSOLUTE_MAX_OUTPUT}"
            )
        if max_output > max_legs * max_hits:
            raise ContractValidationError(
                "max_output cannot exceed max_legs * max_hits_per_leg"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_legs": self.max_legs,
            "max_hits_per_leg": self.max_hits_per_leg,
            "max_output": self.max_output,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "max_legs",
                "max_hits_per_leg",
                "max_output",
            },
            dto="FusionBounds",
        )
        return cls(
            max_legs=payload.get("max_legs"),  # type: ignore[arg-type]
            max_hits_per_leg=payload.get("max_hits_per_leg"),  # type: ignore[arg-type]
            max_output=payload.get("max_output"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LegRankTrace(_BoundedJsonDTOMixin):
    """Eligible rank trace plus non-sensitive filtered-hit counts for one leg."""

    leg_id: str
    leg_kind: str
    backend_id: str
    index_revision_id: str
    plan_leg_id: str
    status: str
    latency_ms: int
    budget_ms: int
    attempt_count: int
    planned_max_hits: int
    submitted_hit_count: int
    error_code: str | None = None
    eligible_hits: tuple[CandidateHit, ...] = ()
    filtered_counts: Mapping[str, int] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_LEG_RANK_TRACE_JSON_BYTES

    def __post_init__(self) -> None:
        _bounded_text(self.leg_id, "leg_id")
        _bounded_text(self.leg_kind, "leg_kind", max_utf8_bytes=32)
        if self.leg_kind not in _LEG_KINDS:
            raise ContractValidationError(
                f"leg_kind must be one of {sorted(_LEG_KINDS)}"
            )
        _bounded_text(self.backend_id, "backend_id")
        _bounded_text(self.index_revision_id, "index_revision_id")
        _bounded_text(self.plan_leg_id, "plan_leg_id")
        status = _bounded_text(self.status, "status", max_utf8_bytes=32)
        if status not in _LEG_STATUSES:
            raise ContractValidationError(
                f"status must be one of {sorted(_LEG_STATUSES)}"
            )
        latency_ms = _non_negative_int(self.latency_ms, "latency_ms")
        if latency_ms > MAX_RECORDED_LATENCY_MS:
            raise ContractValidationError(
                f"latency_ms exceeds absolute limit {MAX_RECORDED_LATENCY_MS}"
            )
        budget_ms = _positive_int(self.budget_ms, "budget_ms")
        if budget_ms > MAX_LEG_BUDGET_MS:
            raise ContractValidationError(
                f"budget_ms exceeds absolute limit {MAX_LEG_BUDGET_MS}"
            )
        attempts = _positive_int(self.attempt_count, "attempt_count")
        if attempts > MAX_ATTEMPTS_PER_LEG:
            raise ContractValidationError(
                f"attempt_count exceeds absolute limit {MAX_ATTEMPTS_PER_LEG}"
            )
        planned_max_hits = _positive_int(
            self.planned_max_hits, "planned_max_hits"
        )
        if planned_max_hits > ABSOLUTE_MAX_HITS_PER_LEG:
            raise ContractValidationError(
                "planned_max_hits exceeds the absolute hit limit"
            )
        submitted = _non_negative_int(
            self.submitted_hit_count, "submitted_hit_count"
        )
        if submitted > planned_max_hits:
            raise ContractValidationError(
                "submitted_hit_count exceeds planned_max_hits"
            )
        error_code = _optional_error_code(self.error_code)
        hits = tuple(
            _bounded_sequence(
                self.eligible_hits,
                "eligible_hits",
                max_items=ABSOLUTE_MAX_HITS_PER_LEG,
            )
        )
        if any(not isinstance(hit, CandidateHit) for hit in hits):
            raise ContractValidationError(
                "eligible_hits must contain CandidateHit values"
            )
        hits = tuple(sorted(hits, key=lambda hit: (hit.rank, hit.hit_id)))
        if len({hit.rank for hit in hits}) != len(hits):
            raise ContractValidationError(
                "eligible_hits must have unique ranks within a leg"
            )
        if len({hit.hit_id for hit in hits}) != len(hits):
            raise ContractValidationError(
                "eligible_hits must have unique hit_id values within a leg"
            )
        for hit in hits:
            if hit.rank > submitted:
                raise ContractValidationError(
                    "eligible hit rank cannot exceed submitted_hit_count"
                )
            if (
                hit.backend_id != self.backend_id
                or hit.index_revision_id != self.index_revision_id
                or hit.leg_kind != self.leg_kind
            ):
                raise ContractValidationError(
                    "eligible hit backend/index revision does not match its trace"
                )
        counts_input = require_mapping(self.filtered_counts, "filtered_counts")
        if len(counts_input) > len(_FILTER_REASONS):
            raise ContractValidationError(
                "filtered_counts exceeds maximum cardinality"
            )
        if any(not isinstance(reason, str) for reason in counts_input):
            raise ContractValidationError("filtered_counts contains a non-string key")
        counts: dict[str, int] = {}
        for reason in sorted(counts_input):
            if reason not in _FILTER_REASONS:
                raise ContractValidationError(
                    f"filtered_counts has unknown reason {reason!r}"
                )
            count = _non_negative_int(
                counts_input[reason], f"filtered_counts[{reason!r}]"
            )
            if count:
                counts[reason] = count
        if len(hits) + sum(counts.values()) != submitted:
            raise ContractValidationError(
                "eligible_hits and filtered_counts must account for every submitted hit"
            )
        if status == "success" and submitted == 0:
            raise ContractValidationError(
                "success trace must have a submitted candidate"
            )
        if status != "success" and submitted != 0:
            raise ContractValidationError(
                "only a success trace may have submitted candidates"
            )
        if status == "error" and error_code is None:
            raise ContractValidationError("error trace requires an error_code")
        if status in {"success", "empty"} and error_code is not None:
            raise ContractValidationError(
                "success/empty traces must not carry an error_code"
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "budget_ms", budget_ms)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "planned_max_hits", planned_max_hits)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "eligible_hits", hits)
        object.__setattr__(self, "filtered_counts", MappingProxyType(counts))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "leg_id": self.leg_id,
            "leg_kind": self.leg_kind,
            "backend_id": self.backend_id,
            "index_revision_id": self.index_revision_id,
            "plan_leg_id": self.plan_leg_id,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "budget_ms": self.budget_ms,
            "attempt_count": self.attempt_count,
            "planned_max_hits": self.planned_max_hits,
            "submitted_hit_count": self.submitted_hit_count,
            "error_code": self.error_code,
            "eligible_hits": [hit.to_dict() for hit in self.eligible_hits],
            "filtered_counts": dict(self.filtered_counts),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        max_hits: int | None = None,
    ) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "leg_id",
                "leg_kind",
                "backend_id",
                "index_revision_id",
                "plan_leg_id",
                "status",
                "latency_ms",
                "budget_ms",
                "attempt_count",
                "planned_max_hits",
                "submitted_hit_count",
                "error_code",
                "eligible_hits",
                "filtered_counts",
            },
            dto="LegRankTrace",
        )
        planned_max_hits = _positive_int(
            payload.get("planned_max_hits"), "planned_max_hits"
        )
        if planned_max_hits > ABSOLUTE_MAX_HITS_PER_LEG:
            raise ContractValidationError(
                "planned_max_hits exceeds the absolute hit limit"
            )
        submitted_hit_count = _non_negative_int(
            payload.get("submitted_hit_count"), "submitted_hit_count"
        )
        if submitted_hit_count > planned_max_hits:
            raise ContractValidationError(
                "submitted_hit_count exceeds planned_max_hits"
            )
        if max_hits is not None:
            request_max_hits = _positive_int(max_hits, "max_hits")
            if request_max_hits > ABSOLUTE_MAX_HITS_PER_LEG:
                raise ContractValidationError(
                    "max_hits exceeds the absolute hit limit"
                )
            if planned_max_hits > request_max_hits:
                raise ContractValidationError(
                    "planned_max_hits exceeds the request max_hits"
                )
        raw_hits = _bounded_sequence(
            payload.get("eligible_hits", ()),
            "eligible_hits",
            max_items=submitted_hit_count,
        )
        return cls(
            leg_id=payload.get("leg_id"),  # type: ignore[arg-type]
            leg_kind=payload.get("leg_kind"),  # type: ignore[arg-type]
            backend_id=payload.get("backend_id"),  # type: ignore[arg-type]
            index_revision_id=payload.get("index_revision_id"),  # type: ignore[arg-type]
            plan_leg_id=payload.get("plan_leg_id"),  # type: ignore[arg-type]
            status=payload.get("status"),  # type: ignore[arg-type]
            latency_ms=payload.get("latency_ms"),  # type: ignore[arg-type]
            budget_ms=payload.get("budget_ms"),  # type: ignore[arg-type]
            attempt_count=payload.get("attempt_count"),  # type: ignore[arg-type]
            planned_max_hits=planned_max_hits,
            submitted_hit_count=submitted_hit_count,
            error_code=payload.get("error_code"),  # type: ignore[arg-type]
            eligible_hits=tuple(
                CandidateHit.from_dict(require_mapping(hit, "eligible_hits[]"))
                for hit in raw_hits
            ),
            filtered_counts=require_mapping(
                payload.get("filtered_counts", {}), "filtered_counts"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RRFContribution(_BoundedJsonDTOMixin):
    """One leg's counted rank and any same-evidence hits it superseded."""

    leg_id: str
    selected_hit_id: str
    selected_rank: int
    selected_raw_score: float
    rrf_contribution: float
    supporting_hit_ids: tuple[str, ...]

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_RRF_CONTRIBUTION_JSON_BYTES

    def __post_init__(self) -> None:
        _bounded_text(self.leg_id, "leg_id")
        _bounded_text(self.selected_hit_id, "selected_hit_id")
        rank = _positive_int(self.selected_rank, "selected_rank")
        raw_score = _finite_score(self.selected_raw_score, "selected_raw_score")
        contribution = _finite_score(self.rrf_contribution, "rrf_contribution")
        if contribution <= 0.0:
            raise ContractValidationError("rrf_contribution must be greater than zero")
        raw_supporting = _bounded_sequence(
            self.supporting_hit_ids,
            "supporting_hit_ids",
            max_items=MAX_SUPPORTING_HITS_PER_CONTRIBUTION,
        )
        supporting = tuple(
            _bounded_text(value, "supporting_hit_ids[]")
            for value in raw_supporting
        )
        if not supporting or supporting[0] != self.selected_hit_id:
            raise ContractValidationError(
                "supporting_hit_ids must start with selected_hit_id"
            )
        if len(set(supporting)) != len(supporting):
            raise ContractValidationError("supporting_hit_ids must be unique")
        object.__setattr__(self, "selected_rank", rank)
        object.__setattr__(self, "selected_raw_score", raw_score)
        object.__setattr__(self, "rrf_contribution", contribution)
        object.__setattr__(self, "supporting_hit_ids", supporting)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "leg_id": self.leg_id,
            "selected_hit_id": self.selected_hit_id,
            "selected_rank": self.selected_rank,
            "selected_raw_score": self.selected_raw_score,
            "rrf_contribution": self.rrf_contribution,
            "supporting_hit_ids": list(self.supporting_hit_ids),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        max_supporting_hits: int | None = None,
    ) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "leg_id",
                "selected_hit_id",
                "selected_rank",
                "selected_raw_score",
                "rrf_contribution",
                "supporting_hit_ids",
            },
            dto="RRFContribution",
        )
        support_limit = MAX_SUPPORTING_HITS_PER_CONTRIBUTION
        if max_supporting_hits is not None:
            support_limit = _positive_int(
                max_supporting_hits, "max_supporting_hits"
            )
            if support_limit > MAX_SUPPORTING_HITS_PER_CONTRIBUTION:
                raise ContractValidationError(
                    "max_supporting_hits exceeds the absolute hit limit"
                )
        return cls(
            leg_id=payload.get("leg_id"),  # type: ignore[arg-type]
            selected_hit_id=payload.get("selected_hit_id"),  # type: ignore[arg-type]
            selected_rank=payload.get("selected_rank"),  # type: ignore[arg-type]
            selected_raw_score=payload.get("selected_raw_score"),  # type: ignore[arg-type]
            rrf_contribution=payload.get("rrf_contribution"),  # type: ignore[arg-type]
            supporting_hit_ids=tuple(
                _bounded_sequence(
                    payload.get("supporting_hit_ids", ()),
                    "supporting_hit_ids",
                    max_items=support_limit,
                )
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class FusedCandidate(_BoundedJsonDTOMixin):
    """One evidence-deduplicated output with per-leg contribution trace."""

    representative_hit: CandidateHit
    contributions: tuple[RRFContribution, ...]
    fused_score: float
    fused_rank: int
    tree_node_ids: tuple[str, ...] = ()
    canonical_revision_ids: tuple[str, ...] = ()
    canonical_representation_ids: tuple[str, ...] = ()
    canonical_asset_ids: tuple[str, ...] = ()
    selected_representation_ids: tuple[str, ...] = ()
    selected_asset_ids: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    evidence_graph_release_id: str | None = None
    evidence_graph_release_sha256: str | None = None
    fused_candidate_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_FUSED_CANDIDATE_JSON_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.representative_hit, CandidateHit):
            raise ContractValidationError(
                "representative_hit must be a CandidateHit"
            )
        contributions = tuple(
            _bounded_sequence(
                self.contributions,
                "contributions",
                max_items=ABSOLUTE_MAX_LEGS,
            )
        )
        if not contributions or any(
            not isinstance(item, RRFContribution) for item in contributions
        ):
            raise ContractValidationError(
                "contributions must contain at least one RRFContribution"
            )
        contributions = tuple(sorted(contributions, key=lambda item: item.leg_id))
        if len({item.leg_id for item in contributions}) != len(contributions):
            raise ContractValidationError("contributions must have unique leg_id values")
        selected_ids = {item.selected_hit_id for item in contributions}
        if self.representative_hit.hit_id not in selected_ids:
            raise ContractValidationError(
                "representative_hit must be selected by one contribution"
            )
        score = _finite_score(self.fused_score, "fused_score")
        if score <= 0.0:
            raise ContractValidationError("fused_score must be greater than zero")
        rank = _positive_int(self.fused_rank, "fused_rank")
        tree_node_ids = _bounded_ids(
            self.tree_node_ids,
            "tree_node_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            non_empty=True,
        )
        canonical_revision_ids = _bounded_ids(
            self.canonical_revision_ids,
            "canonical_revision_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            non_empty=True,
        )
        canonical_representation_ids = _bounded_ids(
            self.canonical_representation_ids,
            "canonical_representation_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
        )
        canonical_asset_ids = _bounded_ids(
            self.canonical_asset_ids,
            "canonical_asset_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
        )
        selected_representation_ids = _bounded_ids(
            self.selected_representation_ids,
            "selected_representation_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
        )
        selected_asset_ids = _bounded_ids(
            self.selected_asset_ids,
            "selected_asset_ids",
            max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
        )
        raw_modalities = _bounded_sequence(
            self.modalities,
            "modalities",
            max_items=MAX_MODALITIES_PER_FUSED_CANDIDATE,
        )
        modalities = tuple(
            sorted(
                {
                    _bounded_text(
                        value,
                        "modalities[]",
                        max_utf8_bytes=MAX_MODALITY_UTF8_BYTES,
                    )
                    for value in raw_modalities
                }
            )
        )
        if not modalities:
            raise ContractValidationError("modalities must not be empty")
        graph_release_id = _optional_id(
            self.evidence_graph_release_id, "evidence_graph_release_id"
        )
        graph_release_sha = self.evidence_graph_release_sha256
        if (graph_release_id is None) != (graph_release_sha is None):
            raise ContractValidationError(
                "fused evidence graph release ID and SHA-256 must be paired"
            )
        if graph_release_sha is not None:
            require_sha256(
                graph_release_sha, "evidence_graph_release_sha256"
            )
        representative = self.representative_hit
        if representative.tree_node_id not in tree_node_ids:
            raise ContractValidationError(
                "tree_node_ids must contain the representative tree_node_id"
            )
        if not set(representative.canonical_revision_ids).issubset(
            canonical_revision_ids
        ):
            raise ContractValidationError(
                "canonical_revision_ids must contain representative revisions"
            )
        if not set(representative.canonical_representation_ids).issubset(
            canonical_representation_ids
        ):
            raise ContractValidationError(
                "canonical_representation_ids must contain representative materials"
            )
        if not set(representative.canonical_asset_ids).issubset(
            canonical_asset_ids
        ):
            raise ContractValidationError(
                "canonical_asset_ids must contain representative materials"
            )
        if not set(representative.selected_representation_ids).issubset(
            selected_representation_ids
        ):
            raise ContractValidationError(
                "selected_representation_ids must contain representative selections"
            )
        if not set(representative.selected_asset_ids).issubset(selected_asset_ids):
            raise ContractValidationError(
                "selected_asset_ids must contain representative selections"
            )
        if representative.modality not in modalities:
            raise ContractValidationError(
                "modalities must contain the representative modality"
            )
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "fused_score", score)
        object.__setattr__(self, "fused_rank", rank)
        object.__setattr__(self, "tree_node_ids", tree_node_ids)
        object.__setattr__(self, "canonical_revision_ids", canonical_revision_ids)
        object.__setattr__(
            self, "canonical_representation_ids", canonical_representation_ids
        )
        object.__setattr__(self, "canonical_asset_ids", canonical_asset_ids)
        object.__setattr__(
            self, "selected_representation_ids", selected_representation_ids
        )
        object.__setattr__(self, "selected_asset_ids", selected_asset_ids)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "evidence_graph_release_id", graph_release_id)
        object.__setattr__(
            self,
            "fused_candidate_id",
            _stable_id(
                "fused_candidate",
                representative.to_dict(),
                [item.to_dict() for item in contributions],
                score,
                rank,
                tree_node_ids,
                canonical_revision_ids,
                canonical_representation_ids,
                canonical_asset_ids,
                selected_representation_ids,
                selected_asset_ids,
                modalities,
                graph_release_id,
                graph_release_sha,
            ),
        )

    @property
    def evidence_kind(self) -> str:
        return self.representative_hit.evidence_kind

    @property
    def evidence_id(self) -> str:
        return self.representative_hit.evidence_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "fused_candidate_id": self.fused_candidate_id,
            "representative_hit": self.representative_hit.to_dict(),
            "contributions": [item.to_dict() for item in self.contributions],
            "fused_score": self.fused_score,
            "fused_rank": self.fused_rank,
            "tree_node_ids": list(self.tree_node_ids),
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "canonical_representation_ids": list(
                self.canonical_representation_ids
            ),
            "canonical_asset_ids": list(self.canonical_asset_ids),
            "selected_representation_ids": list(
                self.selected_representation_ids
            ),
            "selected_asset_ids": list(self.selected_asset_ids),
            "modalities": list(self.modalities),
            "evidence_graph_release_id": self.evidence_graph_release_id,
            "evidence_graph_release_sha256": self.evidence_graph_release_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        max_legs: int | None = None,
        max_supporting_hits: int | None = None,
    ) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "fused_candidate_id",
                "representative_hit",
                "contributions",
                "fused_score",
                "fused_rank",
                "tree_node_ids",
                "canonical_revision_ids",
                "canonical_representation_ids",
                "canonical_asset_ids",
                "selected_representation_ids",
                "selected_asset_ids",
                "modalities",
                "evidence_graph_release_id",
                "evidence_graph_release_sha256",
            },
            dto="FusedCandidate",
        )
        contribution_limit = ABSOLUTE_MAX_LEGS
        if max_legs is not None:
            contribution_limit = _positive_int(max_legs, "max_legs")
            if contribution_limit > ABSOLUTE_MAX_LEGS:
                raise ContractValidationError(
                    "max_legs exceeds the absolute leg limit"
                )
        raw_contributions = _bounded_sequence(
            payload.get("contributions", ()),
            "contributions",
            max_items=contribution_limit,
        )
        result = cls(
            representative_hit=CandidateHit.from_dict(
                require_mapping(payload.get("representative_hit"), "representative_hit")
            ),
            contributions=tuple(
                RRFContribution.from_dict(
                    require_mapping(item, "contributions[]"),
                    max_supporting_hits=max_supporting_hits,
                )
                for item in raw_contributions
            ),
            fused_score=payload.get("fused_score"),  # type: ignore[arg-type]
            fused_rank=payload.get("fused_rank"),  # type: ignore[arg-type]
            tree_node_ids=tuple(
                _bounded_sequence(
                    payload.get("tree_node_ids", ()),
                    "tree_node_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            canonical_revision_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_revision_ids", ()),
                    "canonical_revision_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            canonical_representation_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_representation_ids", ()),
                    "canonical_representation_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            canonical_asset_ids=tuple(
                _bounded_sequence(
                    payload.get("canonical_asset_ids", ()),
                    "canonical_asset_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            selected_representation_ids=tuple(
                _bounded_sequence(
                    payload.get("selected_representation_ids", ()),
                    "selected_representation_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            selected_asset_ids=tuple(
                _bounded_sequence(
                    payload.get("selected_asset_ids", ()),
                    "selected_asset_ids",
                    max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            modalities=tuple(
                _bounded_sequence(
                    payload.get("modalities", ()),
                    "modalities",
                    max_items=MAX_MODALITIES_PER_FUSED_CANDIDATE,
                )
            ),  # type: ignore[arg-type]
            evidence_graph_release_id=payload.get("evidence_graph_release_id"),  # type: ignore[arg-type]
            evidence_graph_release_sha256=payload.get("evidence_graph_release_sha256"),  # type: ignore[arg-type]
        )
        _verify_generated(
            payload, "fused_candidate_id", result.fused_candidate_id
        )
        return result


def _rrf_score(
    per_leg: Mapping[str, Sequence[CandidateHit]], rrf_k: int
) -> Fraction:
    score = Fraction(0, 1)
    for hits in per_leg.values():
        if not hits:
            raise ContractValidationError("RRF evidence group must not be empty")
        score += Fraction(1, rrf_k + hits[0].rank)
    return score


def _top_rrf_rows(
    grouped: Mapping[
        tuple[str, str, str, str, str, str, str],
        Mapping[str, Sequence[CandidateHit]],
    ],
    *,
    rrf_k: int,
    max_output: int,
) -> list[
    tuple[Fraction, tuple[str, str, str, str, str, str, str]]
]:
    """Return deterministic RRF top-K in ``O(E log K)`` time/``O(K)`` space."""

    rows = (
        (_rrf_score(per_leg, rrf_k), dedup_key)
        for dedup_key, per_leg in grouped.items()
    )
    return heapq.nsmallest(
        max_output,
        rows,
        key=lambda item: (-item[0], item[1]),
    )


def _materialize_fused_candidate(
    *,
    dedup_key: tuple[str, str, str, str, str, str, str],
    per_leg: Mapping[str, Sequence[CandidateHit]],
    exact_score: Fraction,
    fused_rank: int,
    rrf_k: int,
) -> FusedCandidate:
    """Materialize provenance unions only for a selected top-K evidence row."""

    contributions: list[RRFContribution] = []
    selected: list[tuple[int, str, str, CandidateHit]] = []
    tree_node_ids: set[str] = set()
    canonical_revision_ids: set[str] = set()
    canonical_representation_ids: set[str] = set()
    canonical_asset_ids: set[str] = set()
    selected_representation_ids: set[str] = set()
    selected_asset_ids: set[str] = set()
    modalities: set[str] = set()
    graph_releases: set[tuple[str, str]] = set()
    for leg_id in sorted(per_leg):
        support = per_leg[leg_id]
        if not support:
            raise ContractValidationError("fused evidence group must not be empty")
        for hit in support:
            if hit.dedup_key != dedup_key:
                raise ContractValidationError(
                    "fused evidence group contains an inconsistent evidence key"
                )
            _add_bounded_unique_id(
                tree_node_ids,
                hit.tree_node_id,
                "tree_node_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _extend_bounded_unique_ids(
                canonical_revision_ids,
                hit.canonical_revision_ids,
                "canonical_revision_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _extend_bounded_unique_ids(
                canonical_representation_ids,
                hit.canonical_representation_ids,
                "canonical_representation_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _extend_bounded_unique_ids(
                canonical_asset_ids,
                hit.canonical_asset_ids,
                "canonical_asset_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _extend_bounded_unique_ids(
                selected_representation_ids,
                hit.selected_representation_ids,
                "selected_representation_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _extend_bounded_unique_ids(
                selected_asset_ids,
                hit.selected_asset_ids,
                "selected_asset_ids",
                max_items=MAX_BINDINGS_PER_FUSED_CANDIDATE,
            )
            _add_bounded_unique_id(
                modalities,
                hit.modality,
                "modalities",
                max_items=MAX_MODALITIES_PER_FUSED_CANDIDATE,
            )
            if hit.evidence_graph_release_id is not None:
                graph_release_sha = hit.evidence_graph_release_sha256
                assert graph_release_sha is not None
                graph_releases.add(
                    (hit.evidence_graph_release_id, graph_release_sha)
                )
                if len(graph_releases) > 1:
                    raise ContractValidationError(
                        "eligible hits mix evidence graph releases"
                    )
        counted = support[0]
        exact = Fraction(1, rrf_k + counted.rank)
        contributions.append(
            RRFContribution(
                leg_id=leg_id,
                selected_hit_id=counted.hit_id,
                selected_rank=counted.rank,
                selected_raw_score=counted.raw_score,
                rrf_contribution=float(exact),
                supporting_hit_ids=tuple(hit.hit_id for hit in support),
            )
        )
        selected.append((counted.rank, leg_id, counted.hit_id, counted))

    if sum(
        (Fraction(1, rrf_k + item.selected_rank) for item in contributions),
        Fraction(0, 1),
    ) != exact_score:
        raise ContractValidationError("RRF score does not match its evidence group")
    representative = min(selected, key=lambda item: item[:3])[3]
    graph_release_id, graph_release_sha = (
        next(iter(graph_releases)) if graph_releases else (None, None)
    )
    return FusedCandidate(
        representative_hit=representative,
        contributions=tuple(contributions),
        fused_score=float(exact_score),
        fused_rank=fused_rank,
        tree_node_ids=tuple(sorted(tree_node_ids)),
        canonical_revision_ids=tuple(sorted(canonical_revision_ids)),
        canonical_representation_ids=tuple(
            sorted(canonical_representation_ids)
        ),
        canonical_asset_ids=tuple(sorted(canonical_asset_ids)),
        selected_representation_ids=tuple(sorted(selected_representation_ids)),
        selected_asset_ids=tuple(sorted(selected_asset_ids)),
        modalities=tuple(sorted(modalities)),
        evidence_graph_release_id=graph_release_id,
        evidence_graph_release_sha256=graph_release_sha,
    )


def _validate_result_semantics(
    *,
    scope: CandidateQueryScope,
    bounds: FusionBounds,
    rrf_k: int,
    traces: tuple[LegRankTrace, ...],
    candidates: tuple[FusedCandidate, ...],
    eligible_candidate_count: int,
) -> None:
    """Validate fused provenance against the retained eligible-hit trace."""

    trace_by_id = {trace.leg_id: trace for trace in traces}
    planned_ids = set(scope.planned_legs_by_id)
    if set(trace_by_id) != planned_ids:
        missing = sorted(planned_ids - set(trace_by_id))
        unexpected = sorted(set(trace_by_id) - planned_ids)
        raise ContractValidationError(
            "leg traces must cover the exact retrieval plan; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    for leg_id, trace in trace_by_id.items():
        plan = scope.planned_legs_by_id[leg_id]
        if (
            trace.leg_kind != plan.leg_kind
            or trace.backend_id != plan.backend_id
            or trace.index_revision_id != plan.index_revision_id
            or trace.plan_leg_id != plan.plan_leg_id
            or trace.budget_ms != plan.budget_ms
            or trace.planned_max_hits != plan.max_hits
        ):
            raise ContractValidationError(
                "leg trace does not match its exact planned backend/config/budget"
            )

    eligible_hit_ids: set[str] = set()
    grouped: dict[
        tuple[str, str, str, str, str, str, str], dict[str, list[CandidateHit]]
    ] = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        for hit in trace.eligible_hits:
            if hit.tenant_id != scope.tenant_id or hit.corpus_id != scope.corpus_id:
                raise ContractValidationError(
                    "eligible trace contains a hit from a mixed query scope"
                )
            reason = _filter_reason(hit, scope)
            if reason is not None:
                raise ContractValidationError(
                    f"eligible trace contains a hit rejected by {reason}"
                )
            if hit.hit_id in eligible_hit_ids:
                raise ContractValidationError(
                    "eligible traces must not repeat a hit_id across legs"
                )
            eligible_hit_ids.add(hit.hit_id)
            grouped[hit.dedup_key][trace.leg_id].append(hit)

    expected_rows = _top_rrf_rows(
        grouped,
        rrf_k=rrf_k,
        max_output=bounds.max_output,
    )
    if eligible_candidate_count != len(grouped):
        raise ContractValidationError(
            "eligible_candidate_count does not match the eligible evidence trace"
        )
    expected_returned = min(len(grouped), bounds.max_output)
    if len(candidates) != expected_returned:
        raise ContractValidationError(
            "returned candidate count does not match bounded eligible evidence"
        )
    if len({candidate.representative_hit.dedup_key for candidate in candidates}) != len(
        candidates
    ):
        raise ContractValidationError("fused candidates must have unique evidence keys")

    for fused_rank, (candidate, expected) in enumerate(
        zip(candidates, expected_rows), start=1
    ):
        exact_score, dedup_key = expected
        expected_candidate = _materialize_fused_candidate(
            dedup_key=dedup_key,
            per_leg=grouped[dedup_key],
            exact_score=exact_score,
            fused_rank=fused_rank,
            rrf_k=rrf_k,
        )
        if candidate.representative_hit.dedup_key != dedup_key:
            raise ContractValidationError(
                "fused candidate order/evidence key does not match ordinary RRF"
            )
        if candidate.representative_hit != expected_candidate.representative_hit:
            raise ContractValidationError(
                "representative_hit does not match the deterministic rank trace"
            )
        if {item.leg_id for item in candidate.contributions} != {
            item.leg_id for item in expected_candidate.contributions
        }:
            raise ContractValidationError(
                "contribution leg does not exist for the fused evidence"
            )
        expected_by_leg = {
            item.leg_id: item for item in expected_candidate.contributions
        }
        for contribution in candidate.contributions:
            expected_contribution = expected_by_leg[contribution.leg_id]
            if contribution.selected_hit_id != expected_contribution.selected_hit_id:
                raise ContractValidationError(
                    "selected contribution hit does not resolve to the best rank"
                )
            if contribution.supporting_hit_ids != expected_contribution.supporting_hit_ids:
                raise ContractValidationError(
                    "supporting hit IDs do not resolve to the fused evidence key"
                )
            if (
                contribution.selected_rank != expected_contribution.selected_rank
                or contribution.selected_raw_score
                != expected_contribution.selected_raw_score
            ):
                raise ContractValidationError(
                    "contribution rank/raw score does not match its selected hit"
                )
            if contribution.rrf_contribution != expected_contribution.rrf_contribution:
                raise ContractValidationError(
                    "RRF contribution must equal 1 / (rrf_k + selected_rank)"
                )
        if candidate.fused_score != float(exact_score):
            raise ContractValidationError(
                "fused_score must equal the sum of per-leg RRF contributions"
            )
        if candidate.tree_node_ids != expected_candidate.tree_node_ids:
            raise ContractValidationError(
                "tree_node_ids must equal the union of contributing hits"
            )
        if (
            candidate.canonical_revision_ids
            != expected_candidate.canonical_revision_ids
        ):
            raise ContractValidationError(
                "canonical_revision_ids must equal the union of contributing hits"
            )
        if (
            candidate.canonical_representation_ids
            != expected_candidate.canonical_representation_ids
        ):
            raise ContractValidationError(
                "canonical_representation_ids must equal the union of contributing hits"
            )
        if candidate.canonical_asset_ids != expected_candidate.canonical_asset_ids:
            raise ContractValidationError(
                "canonical_asset_ids must equal the union of contributing hits"
            )
        if (
            candidate.selected_representation_ids
            != expected_candidate.selected_representation_ids
        ):
            raise ContractValidationError(
                "selected_representation_ids must equal the union of contributing hits"
            )
        if candidate.selected_asset_ids != expected_candidate.selected_asset_ids:
            raise ContractValidationError(
                "selected_asset_ids must equal the union of contributing hits"
            )
        if (
            candidate.evidence_graph_release_id
            != expected_candidate.evidence_graph_release_id
            or candidate.evidence_graph_release_sha256
            != expected_candidate.evidence_graph_release_sha256
        ):
            raise ContractValidationError(
                "fused evidence graph release must match contributing hits"
            )
        if candidate.modalities != expected_candidate.modalities:
            raise ContractValidationError(
                "modalities must equal the union of contributing hits"
            )


@dataclass(frozen=True, slots=True)
class CandidateFusionResult(_BoundedJsonDTOMixin):
    """Self-contained, bounded RRF output and audit trace."""

    scope: CandidateQueryScope
    bounds: FusionBounds
    rrf_k: int
    leg_traces: tuple[LegRankTrace, ...]
    candidates: tuple[FusedCandidate, ...]
    eligible_candidate_count: int
    fusion_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    MAX_JSON_BYTES: ClassVar[int] = MAX_CANDIDATE_FUSION_RESULT_JSON_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CandidateQueryScope):
            raise ContractValidationError("scope must be a CandidateQueryScope")
        if not isinstance(self.bounds, FusionBounds):
            raise ContractValidationError("bounds must be FusionBounds")
        rrf_k = _positive_int(self.rrf_k, "rrf_k")
        if rrf_k > ABSOLUTE_MAX_RRF_K:
            raise ContractValidationError(
                f"rrf_k exceeds absolute limit {ABSOLUTE_MAX_RRF_K}"
            )
        if len(self.scope.planned_legs) > self.bounds.max_legs:
            raise ContractValidationError("retrieval plan exceeds max_legs")
        if any(
            leg.max_hits > self.bounds.max_hits_per_leg
            for leg in self.scope.planned_legs
        ):
            raise ContractValidationError(
                "retrieval plan max_hits exceeds FusionBounds"
            )
        traces = tuple(
            _bounded_sequence(
                self.leg_traces,
                "leg_traces",
                max_items=ABSOLUTE_MAX_LEGS,
            )
        )
        if any(not isinstance(item, LegRankTrace) for item in traces):
            raise ContractValidationError("leg_traces must contain LegRankTrace values")
        traces = tuple(sorted(traces, key=lambda item: item.leg_id))
        if len(traces) > self.bounds.max_legs:
            raise ContractValidationError("leg_traces exceeds max_legs")
        if len({item.leg_id for item in traces}) != len(traces):
            raise ContractValidationError("leg_traces must have unique leg_id values")
        for trace in traces:
            if trace.submitted_hit_count > self.bounds.max_hits_per_leg:
                raise ContractValidationError(
                    "leg trace submitted_hit_count exceeds max_hits_per_leg"
                )

        candidates = tuple(
            _bounded_sequence(
                self.candidates,
                "candidates",
                max_items=ABSOLUTE_MAX_OUTPUT,
            )
        )
        if any(not isinstance(item, FusedCandidate) for item in candidates):
            raise ContractValidationError(
                "candidates must contain FusedCandidate values"
            )
        candidates = tuple(sorted(candidates, key=lambda item: item.fused_rank))
        if len(candidates) > self.bounds.max_output:
            raise ContractValidationError("candidates exceeds max_output")
        if tuple(item.fused_rank for item in candidates) != tuple(
            range(1, len(candidates) + 1)
        ):
            raise ContractValidationError(
                "fused candidate ranks must be contiguous from 1"
            )
        eligible_count = _non_negative_int(
            self.eligible_candidate_count, "eligible_candidate_count"
        )
        if eligible_count < len(candidates):
            raise ContractValidationError(
                "eligible_candidate_count cannot be smaller than returned candidates"
            )
        identity_bytes = sum(
            hit.identity_utf8_bytes
            for trace in traces
            for hit in trace.eligible_hits
        )
        identity_bytes += sum(
            len(value.encode("utf-8"))
            for candidate in candidates
            for value in (
                (
                    candidate.fused_candidate_id,
                    candidate.evidence_graph_release_id or "",
                    candidate.evidence_graph_release_sha256 or "",
                )
                + candidate.tree_node_ids
                + candidate.canonical_revision_ids
                + candidate.canonical_representation_ids
                + candidate.canonical_asset_ids
                + candidate.selected_representation_ids
                + candidate.selected_asset_ids
                + candidate.modalities
                + tuple(
                    hit_id
                    for contribution in candidate.contributions
                    for hit_id in contribution.supporting_hit_ids
                )
            )
        )
        if identity_bytes > MAX_TRACE_IDENTITY_UTF8_BYTES:
            raise ContractValidationError(
                "fusion trace identity payload exceeds the absolute byte limit"
            )
        _validate_result_semantics(
            scope=self.scope,
            bounds=self.bounds,
            rrf_k=rrf_k,
            traces=traces,
            candidates=candidates,
            eligible_candidate_count=eligible_count,
        )

        object.__setattr__(self, "rrf_k", rrf_k)
        object.__setattr__(self, "leg_traces", traces)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "eligible_candidate_count", eligible_count)
        object.__setattr__(
            self,
            "fusion_id",
            _stable_id(
                "candidate_fusion",
                self.scope.scope_id,
                self.bounds.to_dict(),
                rrf_k,
                [item.to_dict() for item in traces],
                [item.to_dict() for item in candidates],
                eligible_count,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "fusion_id": self.fusion_id,
            "scope": self.scope.to_dict(),
            "bounds": self.bounds.to_dict(),
            "rrf_k": self.rrf_k,
            "leg_traces": [item.to_dict() for item in self.leg_traces],
            "candidates": [item.to_dict() for item in self.candidates],
            "eligible_candidate_count": self.eligible_candidate_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_schema(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "fusion_id",
                "scope",
                "bounds",
                "rrf_k",
                "leg_traces",
                "candidates",
                "eligible_candidate_count",
            },
            dto="CandidateFusionResult",
        )
        bounds = FusionBounds.from_dict(
            require_mapping(payload.get("bounds"), "bounds")
        )
        rrf_k = _positive_int(payload.get("rrf_k"), "rrf_k")
        if rrf_k > ABSOLUTE_MAX_RRF_K:
            raise ContractValidationError(
                f"rrf_k exceeds absolute limit {ABSOLUTE_MAX_RRF_K}"
            )

        raw_traces = _bounded_sequence(
            payload.get("leg_traces", ()),
            "leg_traces",
            max_items=bounds.max_legs,
        )
        raw_candidates = _bounded_sequence(
            payload.get("candidates", ()),
            "candidates",
            max_items=bounds.max_output,
        )
        eligible_candidate_count = _non_negative_int(
            payload.get("eligible_candidate_count"),
            "eligible_candidate_count",
        )
        if eligible_candidate_count > bounds.max_legs * bounds.max_hits_per_leg:
            raise ContractValidationError(
                "eligible_candidate_count exceeds the bounded retrieval request"
            )
        expected_candidate_count = min(
            eligible_candidate_count, bounds.max_output
        )
        if len(raw_candidates) != expected_candidate_count:
            raise ContractValidationError(
                "candidate count does not match eligible_candidate_count and max_output"
            )

        # Bound aggregate trace/candidate identity before constructing any
        # nested scope binding, hit, contribution, or fused candidate.  Cheap
        # outer count inconsistencies above fail before this full payload walk.
        _preflight_utf8_payload(
            (raw_traces, raw_candidates),
            "fusion trace identity payload",
            max_utf8_bytes=MAX_TRACE_IDENTITY_UTF8_BYTES,
        )
        raw_candidate_payloads = tuple(
            require_mapping(item, "candidates[]") for item in raw_candidates
        )
        raw_candidate_ranks = tuple(
            _positive_int(item.get("fused_rank"), "candidates[].fused_rank")
            for item in raw_candidate_payloads
        )
        if tuple(sorted(raw_candidate_ranks)) != tuple(
            range(1, len(raw_candidate_payloads) + 1)
        ):
            raise ContractValidationError(
                "fused candidate ranks must be contiguous from 1"
            )

        scope_payload = require_mapping(payload.get("scope"), "scope")
        raw_scope_legs = _bounded_sequence(
            scope_payload.get("planned_legs", ()),
            "scope.planned_legs",
            max_items=bounds.max_legs,
        )
        raw_plan_payloads = tuple(
            require_mapping(raw_plan, "scope.planned_legs[]")
            for raw_plan in raw_scope_legs
        )
        planned_leg_ids: list[str] = []
        for plan_payload in raw_plan_payloads:
            planned_leg_ids.append(
                _bounded_text(
                    plan_payload.get("leg_id"),
                    "scope.planned_legs[].leg_id",
                )
            )
            planned_max_hits = _positive_int(
                plan_payload.get("max_hits"), "scope.planned_legs[].max_hits"
            )
            if planned_max_hits > bounds.max_hits_per_leg:
                raise ContractValidationError(
                    "retrieval plan max_hits exceeds FusionBounds"
                )
        if len(set(planned_leg_ids)) != len(planned_leg_ids):
            raise ContractValidationError(
                "scope.planned_legs must have unique leg_id values"
            )

        raw_trace_payloads = tuple(
            require_mapping(item, "leg_traces[]") for item in raw_traces
        )
        trace_leg_ids = tuple(
            _bounded_text(item.get("leg_id"), "leg_traces[].leg_id")
            for item in raw_trace_payloads
        )
        if len(set(trace_leg_ids)) != len(trace_leg_ids):
            raise ContractValidationError(
                "leg_traces must have unique leg_id values"
            )
        planned_leg_id_set = set(planned_leg_ids)
        trace_leg_id_set = set(trace_leg_ids)
        if trace_leg_id_set != planned_leg_id_set:
            missing = sorted(planned_leg_id_set - trace_leg_id_set)
            unexpected = sorted(trace_leg_id_set - planned_leg_id_set)
            raise ContractValidationError(
                "leg traces must cover the exact retrieval plan; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        scope = CandidateQueryScope.from_dict(scope_payload)
        max_eligible_candidates = sum(
            plan.max_hits for plan in scope.planned_legs
        )
        if eligible_candidate_count > max_eligible_candidates:
            raise ContractValidationError(
                "eligible_candidate_count exceeds the bounded retrieval plan"
            )

        def parse_trace(trace_payload: Mapping[str, object]) -> LegRankTrace:
            leg_id = _bounded_text(trace_payload.get("leg_id"), "leg_id")
            plan = scope.planned_legs_by_id.get(leg_id)
            if plan is None:
                raise ContractValidationError(
                    "leg trace does not exist in the retrieval plan"
                )
            return LegRankTrace.from_dict(
                trace_payload,
                max_hits=min(bounds.max_hits_per_leg, plan.max_hits),
            )

        result = cls(
            scope=scope,
            bounds=bounds,
            rrf_k=rrf_k,
            leg_traces=tuple(parse_trace(item) for item in raw_trace_payloads),
            candidates=tuple(
                FusedCandidate.from_dict(
                    item,
                    max_legs=bounds.max_legs,
                    max_supporting_hits=bounds.max_hits_per_leg,
                )
                for item in raw_candidate_payloads
            ),
            eligible_candidate_count=eligible_candidate_count,
        )
        _verify_generated(payload, "fusion_id", result.fusion_id)
        return result


def fuse_candidates(
    legs: Sequence[CandidateLeg],
    *,
    scope: CandidateQueryScope,
    bounds: FusionBounds,
    rrf_k: int = 60,
) -> CandidateFusionResult:
    """Fuse bounded ranked legs using deterministic ordinary RRF.

    With ``H`` eligible hits, ``E`` evidence identities, and output bound ``K``,
    ranking is ``O(H + E log K)`` and request memory is ``O(H + E + K)`` because
    the complete audit trace is retained.  Material/provenance unions are
    constructed only for the selected top-K identities.  Inputs must be finite
    sequences: generators and other potentially unbounded iterables are
    rejected before iteration.  Hit IDs are global within one retrieval plan:
    conflicting reuse is rejected, while byte-identical reuse is counted only
    in the lexicographically first planned leg.  Distinct hit IDs for the same
    evidence contribute once per explicitly planned leg, as ordinary RRF
    requires.  Backend-equivalent plans and unplanned clone legs are rejected
    before scoring, even when only their resource ceilings differ, so one
    backend cannot acquire an accidental implicit fusion weight.
    """

    if not isinstance(scope, CandidateQueryScope):
        raise ContractValidationError("scope must be a CandidateQueryScope")
    if not isinstance(bounds, FusionBounds):
        raise ContractValidationError("bounds must be FusionBounds")
    rrf_k = _positive_int(rrf_k, "rrf_k")
    if rrf_k > ABSOLUTE_MAX_RRF_K:
        raise ContractValidationError(
            f"rrf_k exceeds absolute limit {ABSOLUTE_MAX_RRF_K}"
        )
    legs = _bounded_sequence(legs, "legs", max_items=ABSOLUTE_MAX_LEGS)
    if len(legs) > bounds.max_legs:
        raise ContractValidationError(
            f"retrieval supplied {len(legs)} legs; max_legs is {bounds.max_legs}"
        )
    if any(not isinstance(leg, CandidateLeg) for leg in legs):
        raise ContractValidationError("legs must contain CandidateLeg values")

    ordered_legs = tuple(sorted(legs, key=lambda leg: leg.leg_id))
    if len({leg.leg_id for leg in ordered_legs}) != len(ordered_legs):
        raise ContractValidationError("retrieval legs must have unique leg_id values")
    planned_ids = set(scope.planned_legs_by_id)
    supplied_ids = {leg.leg_id for leg in ordered_legs}
    if supplied_ids != planned_ids:
        raise ContractValidationError(
            "retrieval legs must cover the exact planned leg roster; "
            f"missing={sorted(planned_ids - supplied_ids)!r}, "
            f"unexpected={sorted(supplied_ids - planned_ids)!r}"
        )
    for leg in ordered_legs:
        plan = scope.planned_legs_by_id[leg.leg_id]
        if (
            leg.leg_kind != plan.leg_kind
            or leg.backend_id != plan.backend_id
            or leg.index_revision_id != plan.index_revision_id
            or leg.leg_config_sha256 != plan.leg_config_sha256
            or leg.query_sha256 != scope.query_sha256
            or leg.retrieval_config_sha256 != scope.retrieval_config_sha256
        ):
            raise ContractValidationError(
                "retrieval leg backend/query/config identity does not match its plan"
            )
        if len(leg.hits) > min(bounds.max_hits_per_leg, plan.max_hits):
            raise ContractValidationError(
                f"leg {leg.leg_id!r} supplied {len(leg.hits)} hits; "
                "it exceeds its bounded plan"
            )

    # Preflight the full bounded input before filtering.  Mixed tenant/corpus
    # results indicate a backend isolation error, so the entire request fails.
    hit_payloads: dict[str, object] = {}
    hit_owner: dict[str, tuple[str, int]] = {}
    input_identity_bytes = 0
    for leg in ordered_legs:
        input_identity_bytes += sum(
            len(value.encode("utf-8"))
            for value in (
                leg.leg_id,
                leg.leg_kind,
                leg.backend_id,
                leg.index_revision_id,
                leg.query_sha256,
                leg.retrieval_config_sha256,
                leg.leg_config_sha256,
                leg.status,
                leg.error_code or "",
            )
        )
        for hit in leg.hits:
            input_identity_bytes += hit.identity_utf8_bytes
            if input_identity_bytes > MAX_TRACE_IDENTITY_UTF8_BYTES:
                raise ContractValidationError(
                    "retrieval candidate identity payload exceeds the absolute byte limit"
                )
            if hit.tenant_id != scope.tenant_id or hit.corpus_id != scope.corpus_id:
                raise ContractValidationError(
                    "mixed query scope: every candidate must match the exact "
                    "tenant_id and corpus_id"
                )
            payload = json_ready(hit.to_dict())
            previous = hit_payloads.get(hit.hit_id)
            if previous is not None and previous != payload:
                raise ContractValidationError(
                    f"duplicate conflicting hit_id {hit.hit_id!r}"
                )
            if previous is None:
                hit_payloads[hit.hit_id] = payload
                hit_owner[hit.hit_id] = (leg.leg_id, hit.rank)

    traces: list[LegRankTrace] = []
    eligible_by_key: dict[
        tuple[str, str, str, str, str, str, str], dict[str, list[CandidateHit]]
    ] = defaultdict(lambda: defaultdict(list))
    for leg in ordered_legs:
        plan = scope.planned_legs_by_id[leg.leg_id]
        eligible: list[CandidateHit] = []
        filtered: dict[str, int] = defaultdict(int)
        for hit in leg.hits:
            if hit_owner[hit.hit_id] != (leg.leg_id, hit.rank):
                filtered["duplicate_hit_id"] += 1
                continue
            reason = _filter_reason(hit, scope)
            if reason is not None:
                filtered[reason] += 1
                continue
            eligible.append(hit)
            eligible_by_key[hit.dedup_key][leg.leg_id].append(hit)
        traces.append(
            LegRankTrace(
                leg_id=leg.leg_id,
                leg_kind=leg.leg_kind,
                backend_id=leg.backend_id,
                index_revision_id=leg.index_revision_id,
                plan_leg_id=plan.plan_leg_id,
                status=leg.status,
                latency_ms=leg.latency_ms,
                budget_ms=plan.budget_ms,
                attempt_count=leg.attempt_count,
                planned_max_hits=plan.max_hits,
                submitted_hit_count=len(leg.hits),
                error_code=leg.error_code,
                eligible_hits=tuple(eligible),
                filtered_counts=filtered,
            )
        )

    ranked = _top_rrf_rows(
        eligible_by_key,
        rrf_k=rrf_k,
        max_output=bounds.max_output,
    )
    output = tuple(
        _materialize_fused_candidate(
            dedup_key=dedup_key,
            per_leg=eligible_by_key[dedup_key],
            exact_score=exact_score,
            fused_rank=rank,
            rrf_k=rrf_k,
        )
        for rank, (exact_score, dedup_key) in enumerate(ranked, start=1)
    )
    return CandidateFusionResult(
        scope=scope,
        bounds=bounds,
        rrf_k=rrf_k,
        leg_traces=tuple(traces),
        candidates=output,
        eligible_candidate_count=len(eligible_by_key),
    )


# Explicitly named alias for callers that want the algorithm in the API name.
fuse_rrf = fuse_candidates


__all__ = [
    "ABSOLUTE_MAX_HITS_PER_LEG",
    "ABSOLUTE_MAX_LEGS",
    "ABSOLUTE_MAX_OUTPUT",
    "ABSOLUTE_MAX_RRF_K",
    "CandidateFusionResult",
    "CandidateHit",
    "CandidateLeg",
    "CandidateQueryScope",
    "DocumentEvidenceBinding",
    "FusedCandidate",
    "FusionBounds",
    "LegRankTrace",
    "MAX_BINDINGS_PER_FUSED_CANDIDATE",
    "MAX_CANDIDATE_FUSION_RESULT_JSON_BYTES",
    "MAX_CANDIDATE_HIT_JSON_BYTES",
    "MAX_CANDIDATE_LEG_JSON_BYTES",
    "MAX_CANDIDATE_QUERY_SCOPE_JSON_BYTES",
    "MAX_CANONICAL_REVISIONS_PER_HIT",
    "MAX_DOCUMENT_EVIDENCE_BINDING_JSON_BYTES",
    "MAX_FUSED_CANDIDATE_JSON_BYTES",
    "MAX_FUSION_BOUNDS_JSON_BYTES",
    "MAX_IDENTIFIER_UTF8_BYTES",
    "MAX_LEG_RANK_TRACE_JSON_BYTES",
    "MAX_MATERIAL_IDS_PER_HIT",
    "MAX_PLANNED_RETRIEVAL_LEG_JSON_BYTES",
    "MAX_RRF_CONTRIBUTION_JSON_BYTES",
    "MAX_SELECTED_DOCUMENTS",
    "MAX_SUPPORTING_HITS_PER_CONTRIBUTION",
    "MAX_TRACE_IDENTITY_UTF8_BYTES",
    "PlannedRetrievalLeg",
    "QueryScope",
    "RETRIEVAL_CANDIDATE_SCHEMA_VERSION",
    "RRFContribution",
    "fuse_candidates",
    "fuse_rrf",
]
