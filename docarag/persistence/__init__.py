"""Durable relational persistence for DocARAG.

PostgreSQL is the only supported relational backend.  Vector embeddings remain
in Qdrant and large immutable artifacts remain in the configured artifact
store; this package keeps their identities, receipts, work state, and traces.
"""

from docarag.persistence.contracts import (
    ArtifactAvailability,
    ArtifactReceiptRecord,
    CorpusScope,
    DocumentScope,
    IndexPublication,
    IndexRevisionSpec,
    IndexRevisionState,
    PostgresHealth,
    SchemaStatus,
    StageEvent,
    StageEventStatus,
    WorkLease,
    WorkSpec,
    WorkState,
    scope_for_document,
)
from docarag.persistence.postgres import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_DATABASE_URL,
    DEFAULT_DATABASE_URL_ENV,
    ImmutableRecordConflict,
    LeaseLostError,
    PostgresLedger,
    PostgresLedgerError,
    PsycopgUnavailableError,
    SchemaDriftError,
    resolve_database_url,
    safe_database_endpoint,
)
from docarag.persistence.stage_runtime import (
    DurableStage,
    NoopStage,
    StageLeaseLostError,
    StageLeaseUnavailableError,
    StageRuntime,
    StageRuntimeConfigurationError,
    StageRuntimeError,
    durable_stage,
)

__all__ = [
    "ArtifactAvailability",
    "ArtifactReceiptRecord",
    "CorpusScope",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_DATABASE_URL",
    "DEFAULT_DATABASE_URL_ENV",
    "DocumentScope",
    "DurableStage",
    "ImmutableRecordConflict",
    "IndexPublication",
    "IndexRevisionSpec",
    "IndexRevisionState",
    "LeaseLostError",
    "NoopStage",
    "PostgresHealth",
    "PostgresLedger",
    "PostgresLedgerError",
    "PsycopgUnavailableError",
    "SchemaDriftError",
    "SchemaStatus",
    "StageEvent",
    "StageEventStatus",
    "StageLeaseLostError",
    "StageLeaseUnavailableError",
    "StageRuntime",
    "StageRuntimeConfigurationError",
    "StageRuntimeError",
    "WorkLease",
    "WorkSpec",
    "WorkState",
    "resolve_database_url",
    "safe_database_endpoint",
    "scope_for_document",
    "durable_stage",
]
