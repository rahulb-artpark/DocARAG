"""Dependency-light contracts for DocARAG's durable PostgreSQL ledger.

The relational ledger stores orchestration truth, not large document payloads or
embeddings.  Every record is tenant/corpus scoped and document-derived records
remain bound to an immutable document version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from docarag.domain.base import (
    ContractValidationError,
    JsonObject,
    canonical_json_bytes,
    freeze_json,
    require_non_empty,
    require_sha256,
    stable_id,
)
from docarag.domain.evidence import DocumentRef
from docarag.domain.parsing import ParserArtifactReceipt


MAX_LEDGER_ID_CHARACTERS = 512
MAX_LEDGER_NAME_CHARACTERS = 1024
MAX_LEDGER_URI_CHARACTERS = 32 * 1024
MAX_LEDGER_JSON_BYTES = 1024 * 1024


def _bounded_text(value: object, field_name: str, maximum: int) -> str:
    text = require_non_empty(value, field_name)
    if len(text) > maximum or "\x00" in text:
        raise ContractValidationError(
            f"{field_name} must be at most {maximum} characters and contain no NUL"
        )
    return text


def _bounded_id(value: object, field_name: str) -> str:
    return _bounded_text(value, field_name, MAX_LEDGER_ID_CHARACTERS)


def _bounded_json(value: Mapping[str, object], field_name: str) -> JsonObject:
    frozen = freeze_json(value, field=field_name)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guaranteed by input type
        raise ContractValidationError(f"{field_name} must be a JSON object")
    if len(canonical_json_bytes(frozen)) > MAX_LEDGER_JSON_BYTES:
        raise ContractValidationError(
            f"{field_name} exceeds the {MAX_LEDGER_JSON_BYTES}-byte limit"
        )
    return frozen


@dataclass(frozen=True, slots=True)
class CorpusScope:
    tenant_id: str
    corpus_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _bounded_id(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "corpus_id", _bounded_id(self.corpus_id, "corpus_id"))


@dataclass(frozen=True, slots=True)
class DocumentScope:
    tenant_id: str
    corpus_id: str
    document_id: str
    document_version_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "tenant_id",
            "corpus_id",
            "document_id",
            "document_version_id",
        ):
            object.__setattr__(
                self, field_name, _bounded_id(getattr(self, field_name), field_name)
            )

    @property
    def corpus_scope(self) -> CorpusScope:
        return CorpusScope(self.tenant_id, self.corpus_id)


class ArtifactAvailability(StrEnum):
    EMITTED = "emitted"
    NOT_EMITTED = "not_emitted"


@dataclass(frozen=True, slots=True)
class ArtifactReceiptRecord:
    """Immutable artifact outcome stored by the relational source of truth."""

    scope: DocumentScope
    receipt_id: str
    artifact_role: str
    availability: ArtifactAvailability
    artifact_schema: str
    producer_task_id: str
    source_sha256: str
    artifact_key: str = "primary"
    uri: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str = "application/octet-stream"
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DocumentScope):
            raise ContractValidationError("scope must be a DocumentScope")
        for field_name in (
            "receipt_id",
            "artifact_role",
            "artifact_schema",
            "producer_task_id",
            "artifact_key",
            "media_type",
        ):
            object.__setattr__(
                self, field_name, _bounded_id(getattr(self, field_name), field_name)
            )
        require_sha256(self.source_sha256, "source_sha256")
        try:
            availability = ArtifactAvailability(self.availability)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("availability is invalid") from exc
        if availability is ArtifactAvailability.EMITTED:
            if self.uri is None:
                raise ContractValidationError("an emitted artifact requires uri")
            _bounded_text(self.uri, "uri", MAX_LEDGER_URI_CHARACTERS)
            require_sha256(self.sha256, "sha256")
            if (
                isinstance(self.size_bytes, bool)
                or not isinstance(self.size_bytes, int)
                or self.size_bytes < 1
            ):
                raise ContractValidationError(
                    "an emitted artifact requires a positive size_bytes"
                )
        elif any(value is not None for value in (self.uri, self.sha256, self.size_bytes)):
            raise ContractValidationError(
                "a not-emitted artifact cannot have uri, sha256, or size_bytes"
            )
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "metadata", _bounded_json(self.metadata, "metadata"))

    @classmethod
    def from_parser_receipt(
        cls,
        scope: DocumentScope,
        receipt: ParserArtifactReceipt,
    ) -> "ArtifactReceiptRecord":
        if receipt.document_version_id != scope.document_version_id:
            raise ContractValidationError("parser receipt belongs to another document")
        return cls(
            scope=scope,
            receipt_id=receipt.receipt_id,
            artifact_role=receipt.artifact_role,
            availability=ArtifactAvailability(receipt.availability.value),
            artifact_schema=receipt.artifact_schema,
            producer_task_id=receipt.producer_task_id,
            source_sha256=receipt.source_sha256,
            artifact_key=receipt.artifact_key,
            uri=receipt.uri,
            sha256=receipt.sha256,
            size_bytes=receipt.size_bytes,
            media_type=receipt.media_type,
            metadata=receipt.metadata,
        )


class IndexRevisionState(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IndexRevisionSpec:
    """Immutable identity and source bindings for one BookRAG index build."""

    scope: DocumentScope
    configuration_sha256: str
    canonical_artifact_sha256: str
    qdrant_collection: str
    # ``None`` means Qdrant's unnamed/default dense vector.  A name must never
    # be invented because it is part of the index publication binding.
    qdrant_vector_name: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    # An indexer that already names its revision -- a fingerprint over its
    # code, configuration and source -- keeps that name, because the run
    # directory and every artifact in it are addressed by it.  Otherwise one
    # is derived from the bindings above.
    revision_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DocumentScope):
            raise ContractValidationError("scope must be a DocumentScope")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        require_sha256(self.canonical_artifact_sha256, "canonical_artifact_sha256")
        _bounded_id(self.qdrant_collection, "qdrant_collection")
        if self.qdrant_vector_name is not None:
            _bounded_id(self.qdrant_vector_name, "qdrant_vector_name")
        metadata = _bounded_json(self.metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)
        if self.revision_id is not None:
            require_non_empty(self.revision_id, "revision_id")
            return
        object.__setattr__(
            self,
            "revision_id",
            stable_id(
                "index-revision",
                self.scope.tenant_id,
                self.scope.corpus_id,
                self.scope.document_id,
                self.scope.document_version_id,
                self.configuration_sha256,
                self.canonical_artifact_sha256,
                self.qdrant_collection,
                self.qdrant_vector_name,
                metadata,
            ),
        )


@dataclass(frozen=True, slots=True)
class IndexPublication:
    tree_artifact_sha256: str
    graph_artifact_sha256: str
    retrieval_manifest_sha256: str
    qdrant_point_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "tree_artifact_sha256",
            "graph_artifact_sha256",
            "retrieval_manifest_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if (
            isinstance(self.qdrant_point_count, bool)
            or not isinstance(self.qdrant_point_count, int)
            or self.qdrant_point_count < 0
        ):
            raise ContractValidationError("qdrant_point_count must be non-negative")


class StageEventStatus(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class StageEvent:
    scope: DocumentScope
    trace_id: str
    stage_run_id: str
    sequence_no: int
    stage_name: str
    event_type: str
    status: StageEventStatus
    payload: JsonObject = field(default_factory=dict)
    occurred_at: datetime | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DocumentScope):
            raise ContractValidationError("scope must be a DocumentScope")
        for field_name in ("trace_id", "stage_run_id", "stage_name", "event_type"):
            object.__setattr__(
                self, field_name, _bounded_id(getattr(self, field_name), field_name)
            )
        if (
            isinstance(self.sequence_no, bool)
            or not isinstance(self.sequence_no, int)
            or self.sequence_no < 0
        ):
            raise ContractValidationError("sequence_no must be a non-negative integer")
        try:
            status = StageEventStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("stage event status is invalid") from exc
        if self.occurred_at is not None and self.occurred_at.tzinfo is None:
            raise ContractValidationError("occurred_at must be timezone-aware")
        payload = _bounded_json(self.payload, "payload")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "event_id",
            stable_id(
                "stage-event",
                self.scope.tenant_id,
                self.scope.corpus_id,
                self.stage_run_id,
                self.sequence_no,
                self.stage_name,
                self.event_type,
                status.value,
                payload,
            ),
        )


class WorkState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"


@dataclass(frozen=True, slots=True)
class WorkSpec:
    scope: DocumentScope
    kind: str
    idempotency_key: str
    payload: JsonObject = field(default_factory=dict)
    priority: int = 0
    max_attempts: int = 3
    work_item_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DocumentScope):
            raise ContractValidationError("scope must be a DocumentScope")
        _bounded_id(self.kind, "kind")
        _bounded_id(self.idempotency_key, "idempotency_key")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or not -32768 <= self.priority <= 32767
        ):
            raise ContractValidationError("priority must fit a PostgreSQL smallint")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 32767
        ):
            raise ContractValidationError("max_attempts must be between 1 and 32767")
        payload = _bounded_json(self.payload, "payload")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "work_item_id",
            stable_id(
                "work-item",
                self.scope.tenant_id,
                self.scope.corpus_id,
                self.kind,
                self.idempotency_key,
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkLease:
    scope: DocumentScope
    work_item_id: str
    kind: str
    payload: Mapping[str, object]
    worker_id: str
    fencing_token: int
    attempt_count: int
    max_attempts: int
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        for field_name in ("work_item_id", "kind", "worker_id"):
            _bounded_id(getattr(self, field_name), field_name)
        for field_name in ("fencing_token", "attempt_count", "max_attempts"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractValidationError(f"{field_name} must be positive")
        if self.lease_expires_at.tzinfo is None:
            raise ContractValidationError("lease_expires_at must be timezone-aware")
        object.__setattr__(self, "payload", _bounded_json(self.payload, "payload"))


@dataclass(frozen=True, slots=True)
class SchemaStatus:
    present: bool
    current: bool
    schema_sha256: str
    applied_sha256: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "current": self.current,
            "schema_sha256": self.schema_sha256,
            "applied_sha256": self.applied_sha256,
            "errors": list(self.errors),
        }

    def as_dict(self) -> dict[str, object]:
        return self.to_dict()


@dataclass(frozen=True, slots=True)
class PostgresHealth:
    reachable: bool
    endpoint: str
    schema: SchemaStatus
    server_version_num: int | None = None
    database_name: str | None = None
    error_code: str | None = None

    @property
    def ok(self) -> bool:
        return self.reachable and self.schema.current

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reachable": self.reachable,
            "endpoint": self.endpoint,
            "server_version_num": self.server_version_num,
            "database_name": self.database_name,
            "schema": self.schema.to_dict(),
            "error_code": self.error_code,
        }

    def as_dict(self) -> dict[str, object]:
        return self.to_dict()


def scope_for_document(
    corpus_scope: CorpusScope,
    document: DocumentRef,
) -> DocumentScope:
    return DocumentScope(
        tenant_id=corpus_scope.tenant_id,
        corpus_id=corpus_scope.corpus_id,
        document_id=document.document_id,
        document_version_id=document.document_version_id,
    )


# ---------------------------------------------------------------- evidence --
# What one revision's parsing and extraction produced.  These carry no database
# keys: identity inside a revision is a local name the writer resolves to a
# surrogate, so a projection never has to know what the database assigned.


@dataclass(frozen=True, slots=True)
class TextRecord:
    """A piece of text stored once per digest and referenced by everything."""

    body: str

    def __post_init__(self) -> None:
        if not isinstance(self.body, str):
            raise ValueError("text body must be a string")


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """Bytes that live in the content-addressed store."""

    sha256: str
    media_type: str
    byte_size: int
    cas_uri: str
    width: int | None = None
    height: int | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, "asset.sha256")
        require_non_empty(self.media_type, "asset.media_type")
        require_non_empty(self.cas_uri, "asset.cas_uri")
        _require_non_negative(self.byte_size, "asset.byte_size")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(value, f"asset.{name}")


@dataclass(frozen=True, slots=True)
class RepresentationRecord:
    """One text channel of a block: parsed text, OCR, a description, markup."""

    kind: str
    text: str | None = None
    asset_sha256: str | None = None
    lang: str | None = None
    confidence: float | None = None
    producer: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.kind, "representation.kind")
        if self.text is None and self.asset_sha256 is None:
            raise ValueError("representation must carry text or an asset")
        if self.asset_sha256 is not None:
            _require_sha256(self.asset_sha256, "representation.asset_sha256")
        _require_probability(self.confidence, "representation.confidence")


@dataclass(frozen=True, slots=True)
class ElementRecord:
    """A typed element inside a block: a cell, a symbol, a diagram node."""

    local_key: str
    kind: str
    content: str | None = None
    role: str | None = None
    row_no: int | None = None
    col_no: int | None = None
    row_span: int | None = None
    col_span: int | None = None
    geometry: Mapping[str, object] | None = None
    entity_ref: str | None = None
    confidence: float | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.local_key, "element.local_key")
        require_non_empty(self.kind, "element.kind")
        for name in ("row_no", "col_no"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(value, f"element.{name}")
        _require_probability(self.confidence, "element.confidence")


@dataclass(frozen=True, slots=True)
class ElementEdgeRecord:
    """A typed relation between two elements of one structure."""

    src_local_key: str
    dst_local_key: str
    kind: str
    label: str | None = None
    ordinal: int = 0
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.src_local_key, "edge.src_local_key")
        require_non_empty(self.dst_local_key, "edge.dst_local_key")
        require_non_empty(self.kind, "edge.kind")
        if self.src_local_key == self.dst_local_key:
            raise ValueError("edge endpoints must differ")
        _require_non_negative(self.ordinal, "edge.ordinal")


@dataclass(frozen=True, slots=True)
class StructureRecord:
    """One graph inside one block: a table, a formula, a diagram, a chart."""

    kind: str
    producer: str
    elements: tuple[ElementRecord, ...]
    edges: tuple[ElementEdgeRecord, ...] = ()
    availability: str = "complete"
    confidence: float | None = None
    source_uri: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.kind, "structure.kind")
        require_non_empty(self.producer, "structure.producer")
        if self.availability not in {"complete", "partial"}:
            raise ValueError("structure.availability must be complete or partial")
        if not self.elements:
            raise ValueError("structure requires at least one element")
        keys = [item.local_key for item in self.elements]
        if len(keys) != len(set(keys)):
            raise ValueError("structure contains duplicate element local keys")
        known = set(keys)
        for edge in self.edges:
            if edge.src_local_key not in known or edge.dst_local_key not in known:
                raise ValueError("edge endpoint is outside its structure")
        _require_probability(self.confidence, "structure.confidence")


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """A reconciled unit of a document, with its text, assets and structures."""

    uid: str
    kind: str
    block_no: int
    revision_uid: str | None = None
    canonical: Mapping[str, object] | None = None
    page_uid: str | None = None
    page_no: int | None = None
    subtype: str | None = None
    reading_order: float | None = None
    quality: str | None = None
    reconciliation: str | None = None
    content_sha256: str | None = None
    geometry: Mapping[str, object] | None = None
    source_observation_ids: tuple[str, ...] = ()
    representations: tuple[RepresentationRecord, ...] = ()
    asset_roles: tuple[tuple[str, str], ...] = ()
    structures: tuple[StructureRecord, ...] = ()
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.uid, "block.uid")
        require_non_empty(self.kind, "block.kind")
        _require_non_negative(self.block_no, "block.block_no")
        if self.page_no is not None:
            _require_non_negative(self.page_no, "block.page_no")
        for sha, _role in self.asset_roles:
            _require_sha256(sha, "block.asset_roles[].sha256")


@dataclass(frozen=True, slots=True)
class TreeNodeRecord:
    """A node of the document tree, bound to the block it was built from."""

    node_no: int
    kind: str
    parent_no: int | None = None
    sibling_no: int = 0
    block_uid: str | None = None
    label: str | None = None
    summary: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.node_no, "tree_node.node_no")
        require_non_empty(self.kind, "tree_node.kind")
        if self.parent_no is not None:
            _require_non_negative(self.parent_no, "tree_node.parent_no")
            if self.parent_no == self.node_no:
                raise ValueError("tree node cannot be its own parent")
        _require_non_negative(self.sibling_no, "tree_node.sibling_no")


@dataclass(frozen=True, slots=True)
class EntityRecord:
    """A corpus-scoped entity, deduplicated across revisions by identity."""

    ref: str
    kind: str
    canonical: str
    description: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.ref, "entity.ref")
        require_non_empty(self.kind, "entity.kind")
        require_non_empty(self.canonical, "entity.canonical")


@dataclass(frozen=True, slots=True)
class MentionRecord:
    """Where one revision saw an entity, and how it was written there."""

    entity_ref: str
    node_no: int | None = None
    block_uid: str | None = None
    surface: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.entity_ref, "mention.entity_ref")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("mention offsets must be supplied together")
        if self.char_start is not None:
            _require_non_negative(self.char_start, "mention.char_start")
            if self.char_end <= self.char_start:
                raise ValueError("mention.char_end must exceed char_start")
        _require_probability(self.confidence, "mention.confidence")


@dataclass(frozen=True, slots=True)
class StatementRoleRecord:
    entity_ref: str
    role: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        require_non_empty(self.entity_ref, "role.entity_ref")
        require_non_empty(self.role, "role.role")
        _require_non_negative(self.ordinal, "role.ordinal")


@dataclass(frozen=True, slots=True)
class StatementEvidenceRecord:
    node_no: int | None = None
    block_uid: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        if self.node_no is None and self.block_uid is None:
            raise ValueError("statement evidence must locate a node or a block")


@dataclass(frozen=True, slots=True)
class StatementRecord:
    """One asserted fact.  Two roles are a binary edge; more are a hyperedge."""

    predicate: str
    roles: tuple[StatementRoleRecord, ...]
    directed: bool = True
    weight: float | None = None
    confidence: float | None = None
    producer: str | None = None
    description: str | None = None
    evidence: tuple[StatementEvidenceRecord, ...] = ()
    properties: Mapping[str, object] = field(default_factory=dict)
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.predicate, "statement.predicate")
        if len(self.roles) < 2:
            raise ValueError("statement requires at least two roles")
        _require_probability(self.confidence, "statement.confidence")


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    """Everything one revision produced, ready to be written in one pass."""

    revision_id: str
    canonical_document: Mapping[str, object] | None = None
    assets: tuple[AssetRecord, ...] = ()
    blocks: tuple[BlockRecord, ...] = ()
    tree_nodes: tuple[TreeNodeRecord, ...] = ()
    entities: tuple[EntityRecord, ...] = ()
    mentions: tuple[MentionRecord, ...] = ()
    statements: tuple[StatementRecord, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.revision_id, "evidence.revision_id")
        uids = [block.uid for block in self.blocks]
        if len(uids) != len(set(uids)):
            raise ValueError("evidence contains duplicate block uids")
        numbers = [node.node_no for node in self.tree_nodes]
        if len(numbers) != len(set(numbers)):
            raise ValueError("evidence contains duplicate tree node numbers")
        refs = [entity.ref for entity in self.entities]
        if len(refs) != len(set(refs)):
            raise ValueError("evidence contains duplicate entity refs")

    def counts(self) -> dict[str, int]:
        """Return what was written, for the caller to report."""

        return {
            "assets": len(self.assets),
            "blocks": len(self.blocks),
            "representations": sum(len(b.representations) for b in self.blocks),
            "structures": sum(len(b.structures) for b in self.blocks),
            "elements": sum(
                len(s.elements) for b in self.blocks for s in b.structures
            ),
            "tree_nodes": len(self.tree_nodes),
            "entities": len(self.entities),
            "mentions": len(self.mentions),
            "statements": len(self.statements),
        }


def _require_sha256(value: object, field_name: str) -> None:
    require_sha256(value, field_name)


def _require_non_negative(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_probability(value: object, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


__all__ = [
    "ArtifactAvailability",
    "ArtifactReceiptRecord",
    "AssetRecord",
    "BlockRecord",
    "CorpusScope",
    "DocumentEvidence",
    "DocumentScope",
    "ElementEdgeRecord",
    "ElementRecord",
    "EntityRecord",
    "IndexPublication",
    "IndexRevisionSpec",
    "IndexRevisionState",
    "MentionRecord",
    "PostgresHealth",
    "RepresentationRecord",
    "SchemaStatus",
    "StageEvent",
    "StageEventStatus",
    "StatementEvidenceRecord",
    "StatementRecord",
    "StatementRoleRecord",
    "StructureRecord",
    "TextRecord",
    "TreeNodeRecord",
    "WorkLease",
    "WorkSpec",
    "WorkState",
    "scope_for_document",
]
