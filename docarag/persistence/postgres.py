"""PostgreSQL implementation of DocARAG's durable orchestration ledger.

The module imports psycopg only when the default connection factory is used.
Controller commands and unit tests therefore remain dependency-light, while a
pool or test transaction can be injected through ``connection_factory``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import os
from typing import Any, ContextManager, Protocol, TypeVar
from urllib.parse import urlsplit

from docarag.domain.base import (
    canonical_json_bytes,
    content_sha256,
    json_ready,
    require_non_empty,
)
from docarag.domain.evidence import DocumentRef
from docarag.persistence.contracts import (
    ArtifactReceiptRecord,
    AssetRecord,
    BlockRecord,
    CorpusScope,
    DocumentEvidence,
    DocumentScope,
    EntityRecord,
    MentionRecord,
    StatementRecord,
    StructureRecord,
    TreeNodeRecord,
    IndexPublication,
    IndexRevisionSpec,
    IndexRevisionState,
    PostgresHealth,
    SchemaStatus,
    StageEvent,
    WorkLease,
    WorkSpec,
    WorkState,
    MAX_LEDGER_JSON_BYTES,
    MAX_LEDGER_URI_CHARACTERS,
    scope_for_document,
)
from docarag.persistence.schema import EVIDENCE_TABLES, SCHEMA_SHA256, SCHEMA_SQL
from docarag.settings import DEFAULT_DATABASE_URL, DEFAULT_DATABASE_URL_ENV


DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
_SCHEMA_ADVISORY_LOCK = 7_640_227_241_925_119_841



class PostgresLedgerError(RuntimeError):
    """Base class for relational ledger failures."""


class PsycopgUnavailableError(PostgresLedgerError):
    pass


class SchemaDriftError(PostgresLedgerError):
    pass


class ImmutableRecordConflict(PostgresLedgerError):
    pass


class LeaseLostError(PostgresLedgerError):
    pass


class _CursorResult(Protocol):
    def fetchone(self) -> object | None: ...

    def fetchall(self) -> Sequence[object]: ...


class _Connection(Protocol):
    def execute(
        self, query: str, params: Sequence[object] | None = None
    ) -> _CursorResult: ...


_ConnectionResource = _Connection | ContextManager[_Connection]
ConnectionFactory = Callable[[], _ConnectionResource]
T = TypeVar("T")


_SCHEMA_META_SQL = """
CREATE SCHEMA IF NOT EXISTS docarag;
CREATE TABLE IF NOT EXISTS docarag.schema_meta (
    id boolean PRIMARY KEY DEFAULT true,
    schema_sha256 char(64) NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT schema_meta_singleton CHECK (id),
    CONSTRAINT schema_meta_sha CHECK (schema_sha256 ~ '^[0-9a-f]{64}$')
)
""".strip()


def resolve_database_url(
    database_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    env_name: str = DEFAULT_DATABASE_URL_ENV,
) -> str:
    """Resolve an explicit URL, a secret-bearing environment value, or local default."""

    env = os.environ if environment is None else environment
    if database_url is not None:
        value = database_url
    elif env_name in env:
        value = env[env_name]
    else:
        value = DEFAULT_DATABASE_URL
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PostgreSQL database URL must be a non-empty URI")
    parsed = urlsplit(value)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("PostgreSQL database URL must use postgres:// or postgresql://")
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("PostgreSQL database URL has an invalid host or port") from exc
    if host is None or not parsed.path or parsed.path == "/":
        raise ValueError("PostgreSQL database URL requires a host and database name")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("PostgreSQL database URL port is invalid")
    if parsed.fragment:
        raise ValueError("PostgreSQL database URL must not contain a fragment")
    return value


def safe_database_endpoint(database_url: str) -> str:
    """Return a log-safe endpoint with credentials and query parameters removed."""

    parsed = urlsplit(resolve_database_url(database_url, environment={}))
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return f"postgresql://{netloc}{parsed.path}"


def _default_connection_factory(
    database_url: str,
    *,
    connect_timeout_seconds: int,
) -> ConnectionFactory:
    def connect() -> _Connection:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise PsycopgUnavailableError(
                "PostgreSQL support requires psycopg 3; install the project's "
                "controller dependencies"
            ) from exc
        return psycopg.connect(
            database_url,
            application_name="docarag",
            connect_timeout=connect_timeout_seconds,
            row_factory=dict_row,
        )

    return connect


def _json_text(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("ledger JSON payload must be an object")
    rendered = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(rendered.encode("utf-8")) > MAX_LEDGER_JSON_BYTES:
        raise ValueError(
            f"ledger JSON payload exceeds the {MAX_LEDGER_JSON_BYTES}-byte limit"
        )
    return rendered


def _json_array(value: Sequence[object]) -> str:
    rendered = json.dumps(
        [json_ready(item) if isinstance(item, Mapping) else item for item in value],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(rendered.encode("utf-8")) > MAX_LEDGER_JSON_BYTES:
        raise ValueError(
            f"ledger JSON array exceeds the {MAX_LEDGER_JSON_BYTES}-byte limit"
        )
    return rendered


def _json_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise PostgresLedgerError(f"PostgreSQL returned invalid {field_name}")
    return value


def _mapping(row: object | None, *, operation: str) -> Mapping[str, Any] | None:
    if row is None:
        return None
    if not isinstance(row, Mapping):
        raise PostgresLedgerError(
            f"{operation} requires psycopg dict_row or mapping-compatible rows"
        )
    return row


def _bounded_seconds(value: float, *, field_name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number of seconds")
    minimum = 0 if allow_zero else 0.001
    if not minimum <= value <= 86_400:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier} and at most 86400")
    return float(value)


class PostgresLedger:
    """Transactional PostgreSQL source of truth for DocARAG indexing.

    A new connection is acquired per public operation.  Inject a
    ``psycopg_pool.ConnectionPool.connection`` method in a long-lived service,
    or a context-managed test connection in unit/integration tests.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        env_name: str = DEFAULT_DATABASE_URL_ENV,
        connection_factory: ConnectionFactory | None = None,
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(connect_timeout_seconds, bool)
            or not isinstance(connect_timeout_seconds, int)
            or not 1 <= connect_timeout_seconds <= 60
        ):
            raise ValueError("connect_timeout_seconds must be between 1 and 60")
        resolved_url = resolve_database_url(
            database_url,
            environment=environment,
            env_name=env_name,
        )
        self._safe_endpoint = safe_database_endpoint(resolved_url)
        self._connection_factory = connection_factory or _default_connection_factory(
            resolved_url,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    @property
    def endpoint(self) -> str:
        return self._safe_endpoint

    def __repr__(self) -> str:
        return f"PostgresLedger(endpoint={self._safe_endpoint!r})"

    @contextmanager
    def _connection(self):
        resource = self._connection_factory()
        enter = getattr(resource, "__enter__", None)
        exit_ = getattr(resource, "__exit__", None)
        if callable(enter) and callable(exit_):
            with resource as connection:  # type: ignore[union-attr]
                yield connection
            return

        connection = resource
        try:
            yield connection
        except BaseException:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback):
                rollback()
            raise
        else:
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        finally:
            close = getattr(connection, "close", None)
            if callable(close):
                close()

    def ensure_schema(self) -> SchemaStatus:
        """Create the declarative schema if absent; never drop existing objects."""

        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_SCHEMA_ADVISORY_LOCK,),
            )
            connection.execute(_SCHEMA_META_SQL)
            if self._applied_schema_sha256(connection) is None:
                connection.execute(SCHEMA_SQL)
                connection.execute(
                    "INSERT INTO docarag.schema_meta (schema_sha256) VALUES (%s)",
                    (SCHEMA_SHA256,),
                )
        return self.schema_status()

    def reset_schema(self) -> SchemaStatus:
        """Drop the entire schema and recreate it from the current definition."""

        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_SCHEMA_ADVISORY_LOCK,),
            )
            connection.execute("DROP SCHEMA IF EXISTS docarag CASCADE")
            connection.execute(_SCHEMA_META_SQL)
            connection.execute(SCHEMA_SQL)
            connection.execute(
                "INSERT INTO docarag.schema_meta (schema_sha256) VALUES (%s)",
                (SCHEMA_SHA256,),
            )
        return self.schema_status()

    def schema_status(self) -> SchemaStatus:
        with self._connection() as connection:
            return self._schema_status_on_connection(connection)

    def _schema_status_on_connection(self, connection: _Connection) -> SchemaStatus:
        applied = self._applied_schema_sha256(connection)
        if applied is None:
            return SchemaStatus(
                present=False,
                current=False,
                schema_sha256=SCHEMA_SHA256,
                applied_sha256=None,
                errors=("schema not initialized; run `docarag storage init`",),
            )
        if applied != SCHEMA_SHA256:
            return SchemaStatus(
                present=True,
                current=False,
                schema_sha256=SCHEMA_SHA256,
                applied_sha256=applied,
                errors=(
                    "schema definition changed since it was applied; "
                    "run `docarag storage reset`",
                ),
            )
        return SchemaStatus(
            present=True,
            current=True,
            schema_sha256=SCHEMA_SHA256,
            applied_sha256=applied,
        )

    @staticmethod
    def _applied_schema_sha256(connection: _Connection) -> str | None:
        relation = _mapping(
            connection.execute(
                "SELECT to_regclass('docarag.schema_meta') AS relation"
            ).fetchone(),
            operation="schema discovery",
        )
        if relation is None or relation.get("relation") is None:
            return None
        row = _mapping(
            connection.execute(
                "SELECT schema_sha256 FROM docarag.schema_meta WHERE id"
            ).fetchone(),
            operation="schema status",
        )
        if row is None:
            return None
        applied = row.get("schema_sha256")
        return str(applied) if isinstance(applied, str) else None

    def health(self) -> PostgresHealth:
        unavailable_schema = SchemaStatus(
            present=False,
            current=False,
            schema_sha256=SCHEMA_SHA256,
            applied_sha256=None,
            errors=("database_unreachable",),
        )
        try:
            with self._connection() as connection:
                row = _mapping(
                    connection.execute(
                        "SELECT current_setting('server_version_num')::integer "
                        "AS server_version_num, current_database() AS database_name"
                    ).fetchone(),
                    operation="PostgreSQL health check",
                )
                if row is None:
                    raise PostgresLedgerError("PostgreSQL health check returned no row")
                schema = self._schema_status_on_connection(connection)
                version = row.get("server_version_num")
                return PostgresHealth(
                    reachable=True,
                    endpoint=self.endpoint,
                    schema=schema,
                    server_version_num=(
                        version
                        if isinstance(version, int) and not isinstance(version, bool)
                        else None
                    ),
                    database_name=(
                        str(row["database_name"])
                        if row.get("database_name") is not None
                        else None
                    ),
                )
        except Exception as exc:
            return PostgresHealth(
                reachable=False,
                endpoint=self.endpoint,
                schema=unavailable_schema,
                error_code=type(exc).__name__,
            )

    def ensure_scope(
        self,
        scope: CorpusScope,
        *,
        tenant_display_name: str | None = None,
        corpus_display_name: str | None = None,
    ) -> None:
        tenant_name = tenant_display_name or scope.tenant_id
        corpus_name = corpus_display_name or scope.corpus_id
        self._validate_display_name(tenant_name, "tenant_display_name")
        self._validate_display_name(corpus_name, "corpus_display_name")
        with self._connection() as connection:
            self._ensure_scope_on_connection(
                connection,
                scope,
                tenant_display_name=tenant_name,
                corpus_display_name=corpus_name,
            )

    @staticmethod
    def _validate_display_name(value: str, field_name: str) -> None:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 1024
            or "\x00" in value
        ):
            raise ValueError(f"{field_name} must contain 1 to 1024 characters")

    @staticmethod
    def _ensure_scope_on_connection(
        connection: _Connection,
        scope: CorpusScope,
        *,
        tenant_display_name: str,
        corpus_display_name: str,
    ) -> None:
        connection.execute(
            "INSERT INTO docarag.tenants (tenant_id, display_name) "
            "VALUES (%s, %s) "
            "ON CONFLICT (tenant_id) DO UPDATE SET "
            "display_name = EXCLUDED.display_name, updated_at = clock_timestamp()",
            (scope.tenant_id, tenant_display_name),
        )
        connection.execute(
            "INSERT INTO docarag.corpora (tenant_id, corpus_id, display_name) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (tenant_id, corpus_id) DO UPDATE SET "
            "display_name = EXCLUDED.display_name, updated_at = clock_timestamp()",
            (scope.tenant_id, scope.corpus_id, corpus_display_name),
        )

    def record_document(
        self,
        corpus_scope: CorpusScope,
        document: DocumentRef,
        *,
        tenant_display_name: str | None = None,
        corpus_display_name: str | None = None,
    ) -> DocumentScope:
        """Idempotently admit a content-bound document version.

        Reusing a document-version ID with different source facts is rejected;
        the database trigger also forbids later mutation or deletion.
        """

        if not isinstance(document, DocumentRef):
            raise TypeError("document must be a DocumentRef")
        tenant_name = tenant_display_name or corpus_scope.tenant_id
        corpus_name = corpus_display_name or corpus_scope.corpus_id
        self._validate_display_name(tenant_name, "tenant_display_name")
        self._validate_display_name(corpus_name, "corpus_display_name")
        if len(document.source_uri) > MAX_LEDGER_URI_CHARACTERS:
            raise ValueError(
                f"document source_uri exceeds {MAX_LEDGER_URI_CHARACTERS} characters"
            )
        if len(document.mime_type) > 512:
            raise ValueError("document mime_type exceeds 512 characters")
        if document.display_name is not None:
            self._validate_display_name(document.display_name, "document.display_name")
        scope = scope_for_document(corpus_scope, document)
        metadata_text = _json_text(document.metadata)
        with self._connection() as connection:
            self._ensure_scope_on_connection(
                connection,
                corpus_scope,
                tenant_display_name=tenant_name,
                corpus_display_name=corpus_name,
            )
            connection.execute(
                "INSERT INTO docarag.documents "
                "(tenant_id, corpus_id, document_id, display_name) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, corpus_id, document_id) DO UPDATE SET "
                "display_name = COALESCE(EXCLUDED.display_name, "
                "docarag.documents.display_name), updated_at = clock_timestamp()",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    document.display_name,
                ),
            )
            connection.execute(
                "INSERT INTO docarag.document_versions "
                "(tenant_id, corpus_id, document_id, document_version_id, "
                "source_sha256, source_uri, mime_type, page_count, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (tenant_id, corpus_id, document_id, document_version_id) "
                "DO NOTHING",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    document.content_sha256,
                    document.source_uri,
                    document.mime_type,
                    document.page_count,
                    metadata_text,
                ),
            )
            stored = _mapping(
                connection.execute(
                    "SELECT source_sha256, source_uri, mime_type, page_count, metadata "
                    "FROM docarag.document_versions WHERE tenant_id = %s "
                    "AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                    ),
                ).fetchone(),
                operation="document version admission",
            )
            if stored is None or not self._document_version_matches(stored, document):
                raise ImmutableRecordConflict(
                    "document_version_id already exists with different source facts"
                )
        return scope

    @staticmethod
    def _document_version_matches(
        stored: Mapping[str, object], document: DocumentRef
    ) -> bool:
        try:
            metadata = _json_mapping(stored.get("metadata"), field_name="metadata")
        except (PostgresLedgerError, json.JSONDecodeError):
            return False
        return (
            stored.get("source_sha256") == document.content_sha256
            and stored.get("source_uri") == document.source_uri
            and stored.get("mime_type") == document.mime_type
            and stored.get("page_count") == document.page_count
            and canonical_json_bytes(metadata) == canonical_json_bytes(document.metadata)
        )

    def record_artifact(self, receipt: ArtifactReceiptRecord) -> None:
        """Append one immutable artifact outcome, accepting exact retries only."""

        metadata_text = _json_text(receipt.metadata)
        scope = receipt.scope
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO docarag.artifact_receipts "
                "(tenant_id, corpus_id, document_id, document_version_id, receipt_id, "
                "artifact_role, availability, artifact_schema, producer_task_id, "
                "source_sha256, artifact_key, uri, sha256, size_bytes, media_type, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s::jsonb) ON CONFLICT DO NOTHING",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    receipt.receipt_id,
                    receipt.artifact_role,
                    receipt.availability.value,
                    receipt.artifact_schema,
                    receipt.producer_task_id,
                    receipt.source_sha256,
                    receipt.artifact_key,
                    receipt.uri,
                    receipt.sha256,
                    receipt.size_bytes,
                    receipt.media_type,
                    metadata_text,
                ),
            )
            stored = _mapping(
                connection.execute(
                    "SELECT receipt_id, availability, artifact_schema, source_sha256, "
                    "uri, sha256, size_bytes, media_type, metadata "
                    "FROM docarag.artifact_receipts WHERE tenant_id = %s "
                    "AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s AND producer_task_id = %s "
                    "AND artifact_role = %s AND artifact_key = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                        receipt.producer_task_id,
                        receipt.artifact_role,
                        receipt.artifact_key,
                    ),
                ).fetchone(),
                operation="artifact receipt admission",
            )
            if stored is None or not self._artifact_matches(stored, receipt):
                raise ImmutableRecordConflict(
                    "artifact receipt identity already exists with different facts"
                )

    @staticmethod
    def _artifact_matches(
        stored: Mapping[str, object], receipt: ArtifactReceiptRecord
    ) -> bool:
        try:
            metadata = _json_mapping(stored.get("metadata"), field_name="metadata")
        except (PostgresLedgerError, json.JSONDecodeError):
            return False
        expected = (
            receipt.receipt_id,
            receipt.availability.value,
            receipt.artifact_schema,
            receipt.source_sha256,
            receipt.uri,
            receipt.sha256,
            receipt.size_bytes,
            receipt.media_type,
        )
        actual = tuple(
            stored.get(field)
            for field in (
                "receipt_id",
                "availability",
                "artifact_schema",
                "source_sha256",
                "uri",
                "sha256",
                "size_bytes",
                "media_type",
            )
        )
        return actual == expected and canonical_json_bytes(metadata) == canonical_json_bytes(
            receipt.metadata
        )

    def create_index_revision(self, spec: IndexRevisionSpec) -> IndexRevisionState:
        scope = spec.scope
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO docarag.index_revisions "
                "(tenant_id, corpus_id, document_id, document_version_id, revision_id, "
                "configuration_sha256, canonical_artifact_sha256, qdrant_collection, "
                "qdrant_vector_name, metadata) VALUES "
                "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT DO NOTHING",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    spec.revision_id,
                    spec.configuration_sha256,
                    spec.canonical_artifact_sha256,
                    spec.qdrant_collection,
                    spec.qdrant_vector_name,
                    _json_text(spec.metadata),
                ),
            )
            stored = self._load_index_revision(connection, spec)
            if stored is None or not self._index_revision_matches(stored, spec):
                raise ImmutableRecordConflict(
                    "index revision ID already exists with different bindings"
                )
            return IndexRevisionState(stored["state"])

    @staticmethod
    def _load_index_revision(
        connection: _Connection, spec: IndexRevisionSpec
    ) -> Mapping[str, Any] | None:
        scope = spec.scope
        return _mapping(
            connection.execute(
                "SELECT configuration_sha256, canonical_artifact_sha256, "
                "qdrant_collection, qdrant_vector_name, state, metadata, "
                "tree_artifact_sha256, graph_artifact_sha256, "
                "retrieval_manifest_sha256, qdrant_point_count, failure_code, "
                "failure_message FROM docarag.index_revisions "
                "WHERE tenant_id = %s AND corpus_id = %s AND document_id = %s "
                "AND document_version_id = %s AND revision_id = %s",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    spec.revision_id,
                ),
            ).fetchone(),
            operation="index revision lookup",
        )

    @staticmethod
    def _index_revision_matches(
        stored: Mapping[str, object], spec: IndexRevisionSpec
    ) -> bool:
        try:
            metadata = _json_mapping(stored.get("metadata"), field_name="metadata")
        except (PostgresLedgerError, json.JSONDecodeError):
            return False
        return (
            stored.get("configuration_sha256") == spec.configuration_sha256
            and stored.get("canonical_artifact_sha256")
            == spec.canonical_artifact_sha256
            and stored.get("qdrant_collection") == spec.qdrant_collection
            and stored.get("qdrant_vector_name") == spec.qdrant_vector_name
            and canonical_json_bytes(metadata) == canonical_json_bytes(spec.metadata)
        )

    def publish_index_revision(
        self, spec: IndexRevisionSpec, publication: IndexPublication
    ) -> None:
        scope = spec.scope
        with self._connection() as connection:
            updated = _mapping(
                connection.execute(
                    "UPDATE docarag.index_revisions SET state = 'ready', "
                    "tree_artifact_sha256 = %s, graph_artifact_sha256 = %s, "
                    "retrieval_manifest_sha256 = %s, qdrant_point_count = %s, "
                    "finished_at = clock_timestamp() WHERE tenant_id = %s "
                    "AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s AND revision_id = %s "
                    "AND state = 'building' RETURNING revision_id",
                    (
                        publication.tree_artifact_sha256,
                        publication.graph_artifact_sha256,
                        publication.retrieval_manifest_sha256,
                        publication.qdrant_point_count,
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                        spec.revision_id,
                    ),
                ).fetchone(),
                operation="index revision publication",
            )
            if updated is not None:
                return
            stored = self._load_index_revision(connection, spec)
            if stored is None or not self._publication_matches(stored, publication):
                raise ImmutableRecordConflict(
                    "index revision is missing, failed, or already has another publication"
                )

    @staticmethod
    def _publication_matches(
        stored: Mapping[str, object], publication: IndexPublication
    ) -> bool:
        return stored.get("state") == IndexRevisionState.READY.value and tuple(
            stored.get(field)
            for field in (
                "tree_artifact_sha256",
                "graph_artifact_sha256",
                "retrieval_manifest_sha256",
                "qdrant_point_count",
            )
        ) == (
            publication.tree_artifact_sha256,
            publication.graph_artifact_sha256,
            publication.retrieval_manifest_sha256,
            publication.qdrant_point_count,
        )

    def fail_index_revision(
        self,
        spec: IndexRevisionSpec,
        *,
        failure_code: str,
        failure_message: str,
    ) -> None:
        if not failure_code.strip() or len(failure_code) > 512:
            raise ValueError("failure_code must contain 1 to 512 characters")
        if not failure_message.strip() or len(failure_message) > 16_384:
            raise ValueError("failure_message must contain 1 to 16384 characters")
        scope = spec.scope
        with self._connection() as connection:
            updated = _mapping(
                connection.execute(
                    "UPDATE docarag.index_revisions SET state = 'failed', "
                    "failure_code = %s, failure_message = %s, "
                    "finished_at = clock_timestamp() WHERE tenant_id = %s "
                    "AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s AND revision_id = %s "
                    "AND state = 'building' RETURNING revision_id",
                    (
                        failure_code,
                        failure_message,
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                        spec.revision_id,
                    ),
                ).fetchone(),
                operation="index revision failure",
            )
            if updated is not None:
                return
            stored = self._load_index_revision(connection, spec)
            if not (
                stored
                and stored.get("state") == IndexRevisionState.FAILED.value
                and stored.get("failure_code") == failure_code
                and stored.get("failure_message") == failure_message
            ):
                raise ImmutableRecordConflict(
                    "index revision is missing, ready, or failed differently"
                )

    def append_stage_event(self, event: StageEvent) -> None:
        scope = event.scope
        payload_text = _json_text(event.payload)
        payload_sha256 = content_sha256(event.payload)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO docarag.stage_events "
                "(tenant_id, corpus_id, document_id, document_version_id, event_id, "
                "trace_id, stage_run_id, sequence_no, stage_name, event_type, status, "
                "payload, payload_sha256, occurred_at) VALUES "
                "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, "
                "COALESCE(%s, clock_timestamp())) ON CONFLICT DO NOTHING",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    event.event_id,
                    event.trace_id,
                    event.stage_run_id,
                    event.sequence_no,
                    event.stage_name,
                    event.event_type,
                    event.status.value,
                    payload_text,
                    payload_sha256,
                    event.occurred_at,
                ),
            )
            stored = _mapping(
                connection.execute(
                    "SELECT event_id, trace_id, stage_name, event_type, status, "
                    "payload_sha256 FROM docarag.stage_events WHERE tenant_id = %s "
                    "AND corpus_id = %s AND stage_run_id = %s AND sequence_no = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        event.stage_run_id,
                        event.sequence_no,
                    ),
                ).fetchone(),
                operation="stage event append",
            )
            if stored is None or tuple(
                stored.get(field)
                for field in (
                    "event_id",
                    "trace_id",
                    "stage_name",
                    "event_type",
                    "status",
                    "payload_sha256",
                )
            ) != (
                event.event_id,
                event.trace_id,
                event.stage_name,
                event.event_type,
                event.status.value,
                payload_sha256,
            ):
                raise ImmutableRecordConflict(
                    "stage_run_id/sequence_no already contains another event"
                )

    def enqueue_work(self, spec: WorkSpec) -> str:
        scope = spec.scope
        payload_text = _json_text(spec.payload)
        payload_sha256 = content_sha256(spec.payload)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO docarag.work_items "
                "(tenant_id, corpus_id, document_id, document_version_id, work_item_id, "
                "kind, idempotency_key, payload, payload_sha256, priority, max_attempts) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s) "
                "ON CONFLICT (tenant_id, corpus_id, kind, idempotency_key) DO NOTHING",
                (
                    scope.tenant_id,
                    scope.corpus_id,
                    scope.document_id,
                    scope.document_version_id,
                    spec.work_item_id,
                    spec.kind,
                    spec.idempotency_key,
                    payload_text,
                    payload_sha256,
                    spec.priority,
                    spec.max_attempts,
                ),
            )
            stored = _mapping(
                connection.execute(
                    "SELECT work_item_id, document_id, document_version_id, payload_sha256, "
                    "priority, max_attempts FROM docarag.work_items "
                    "WHERE tenant_id = %s AND corpus_id = %s AND kind = %s "
                    "AND idempotency_key = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        spec.kind,
                        spec.idempotency_key,
                    ),
                ).fetchone(),
                operation="work enqueue",
            )
            if stored is None or tuple(
                stored.get(field)
                for field in (
                    "work_item_id",
                    "document_id",
                    "document_version_id",
                    "payload_sha256",
                    "priority",
                    "max_attempts",
                )
            ) != (
                spec.work_item_id,
                scope.document_id,
                scope.document_version_id,
                payload_sha256,
                spec.priority,
                spec.max_attempts,
            ):
                raise ImmutableRecordConflict(
                    "work idempotency key already exists with different work"
                )
        return spec.work_item_id

    def get_work_state(
        self,
        scope: DocumentScope,
        work_item_id: str,
    ) -> WorkState | None:
        """Return the exact document-scoped state of a durable work item.

        Stage runners use this read after an idempotent enqueue when no lease is
        available.  It distinguishes a safely replayed, already-succeeded stage
        from work that is pending, leased by another worker, or terminally dead.
        The full document scope is intentionally required so callers cannot use
        a corpus-wide identifier as an authority boundary.
        """

        if not isinstance(scope, DocumentScope):
            raise TypeError("scope must be a DocumentScope")
        if not isinstance(work_item_id, str) or not work_item_id.strip():
            raise ValueError("work_item_id must be a non-empty string")
        if len(work_item_id) > 512:
            raise ValueError("work_item_id must be at most 512 characters")
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    "SELECT state FROM docarag.work_items WHERE tenant_id = %s "
                    "AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s AND work_item_id = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                        work_item_id,
                    ),
                ).fetchone(),
                operation="work state lookup",
            )
        return None if row is None else WorkState(str(row["state"]))

    def lease_work(
        self,
        corpus_scope: CorpusScope,
        *,
        worker_id: str,
        lease_seconds: float = 60,
        kinds: Sequence[str] | None = None,
    ) -> WorkLease | None:
        if not worker_id.strip() or len(worker_id) > 512:
            raise ValueError("worker_id must contain 1 to 512 characters")
        seconds = _bounded_seconds(lease_seconds, field_name="lease_seconds")
        accepted_kinds = tuple(kinds or ())
        if isinstance(kinds, (str, bytes, bytearray)):
            raise ValueError("kinds must be a sequence of work-kind strings")
        if any(not kind.strip() or len(kind) > 512 for kind in accepted_kinds):
            raise ValueError("kinds must contain non-empty values up to 512 characters")
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    "WITH candidate AS ("
                    " SELECT tenant_id, corpus_id, work_item_id"
                    " FROM docarag.work_items"
                    " WHERE tenant_id = %s AND corpus_id = %s"
                    " AND (%s::boolean OR kind = ANY(%s::text[]))"
                    " AND attempt_count < max_attempts"
                    " AND ((state = 'pending' AND available_at <= clock_timestamp())"
                    "   OR (state = 'leased' AND lease_expires_at <= clock_timestamp()))"
                    " ORDER BY priority DESC, available_at, created_at, work_item_id"
                    " FOR UPDATE SKIP LOCKED LIMIT 1"
                    ") UPDATE docarag.work_items AS work SET"
                    " state = 'leased', lease_owner = %s,"
                    " lease_expires_at = clock_timestamp() + make_interval(secs => %s),"
                    " fencing_token = work.fencing_token + 1,"
                    " attempt_count = work.attempt_count + 1,"
                    " error = NULL, updated_at = clock_timestamp()"
                    " FROM candidate WHERE work.tenant_id = candidate.tenant_id"
                    " AND work.corpus_id = candidate.corpus_id"
                    " AND work.work_item_id = candidate.work_item_id"
                    " RETURNING work.tenant_id, work.corpus_id, work.document_id,"
                    " work.document_version_id, work.work_item_id, work.kind, work.payload,"
                    " work.lease_owner, work.fencing_token, work.attempt_count,"
                    " work.max_attempts, work.lease_expires_at",
                    (
                        corpus_scope.tenant_id,
                        corpus_scope.corpus_id,
                        not accepted_kinds,
                        list(accepted_kinds),
                        worker_id,
                        seconds,
                    ),
                ).fetchone(),
                operation="work leasing",
            )
        return None if row is None else self._lease_from_row(row)

    @staticmethod
    def _lease_from_row(row: Mapping[str, object]) -> WorkLease:
        expires = row.get("lease_expires_at")
        if not isinstance(expires, datetime):
            raise PostgresLedgerError("PostgreSQL returned an invalid lease expiry")
        return WorkLease(
            scope=DocumentScope(
                tenant_id=str(row["tenant_id"]),
                corpus_id=str(row["corpus_id"]),
                document_id=str(row["document_id"]),
                document_version_id=str(row["document_version_id"]),
            ),
            work_item_id=str(row["work_item_id"]),
            kind=str(row["kind"]),
            payload=_json_mapping(row.get("payload"), field_name="work payload"),
            worker_id=str(row["lease_owner"]),
            fencing_token=int(row["fencing_token"]),
            attempt_count=int(row["attempt_count"]),
            max_attempts=int(row["max_attempts"]),
            lease_expires_at=expires,
        )

    def heartbeat_work(
        self,
        lease: WorkLease,
        *,
        extension_seconds: float = 60,
    ) -> WorkLease:
        seconds = _bounded_seconds(extension_seconds, field_name="extension_seconds")
        scope = lease.scope
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    "UPDATE docarag.work_items SET "
                    "lease_expires_at = clock_timestamp() + make_interval(secs => %s), "
                    "updated_at = clock_timestamp() WHERE tenant_id = %s "
                    "AND corpus_id = %s AND work_item_id = %s AND state = 'leased' "
                    "AND lease_owner = %s AND fencing_token = %s "
                    "AND lease_expires_at > clock_timestamp() RETURNING tenant_id, "
                    "corpus_id, document_id, document_version_id, work_item_id, kind, "
                    "payload, lease_owner, fencing_token, attempt_count, max_attempts, "
                    "lease_expires_at",
                    (
                        seconds,
                        scope.tenant_id,
                        scope.corpus_id,
                        lease.work_item_id,
                        lease.worker_id,
                        lease.fencing_token,
                    ),
                ).fetchone(),
                operation="work heartbeat",
            )
        if row is None:
            raise LeaseLostError("work lease is stale, expired, or owned by another worker")
        return self._lease_from_row(row)

    def complete_work(
        self,
        lease: WorkLease,
        *,
        result: Mapping[str, object] | None = None,
    ) -> None:
        self._finish_work(
            lease,
            sql=(
                "UPDATE docarag.work_items SET state = 'succeeded', result = %s::jsonb, "
                "error = NULL, lease_owner = NULL, lease_expires_at = NULL, "
                "finished_at = clock_timestamp(), updated_at = clock_timestamp() "
                "WHERE tenant_id = %s AND corpus_id = %s AND work_item_id = %s "
                "AND state = 'leased' AND lease_owner = %s AND fencing_token = %s "
                "AND lease_expires_at > clock_timestamp() RETURNING work_item_id"
            ),
            leading_params=(_json_text(result or {}),),
        )

    def _finish_work(
        self,
        lease: WorkLease,
        *,
        sql: str,
        leading_params: tuple[object, ...],
    ) -> None:
        scope = lease.scope
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    sql,
                    (
                        *leading_params,
                        scope.tenant_id,
                        scope.corpus_id,
                        lease.work_item_id,
                        lease.worker_id,
                        lease.fencing_token,
                    ),
                ).fetchone(),
                operation="work completion",
            )
        if row is None:
            raise LeaseLostError("work lease is stale, expired, or owned by another worker")

    def fail_work(
        self,
        lease: WorkLease,
        *,
        error: Mapping[str, object],
        retry_delay_seconds: float = 0,
    ) -> WorkState:
        delay = _bounded_seconds(
            retry_delay_seconds,
            field_name="retry_delay_seconds",
            allow_zero=True,
        )
        scope = lease.scope
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    "UPDATE docarag.work_items SET "
                    "state = CASE WHEN attempt_count >= max_attempts "
                    "THEN 'dead' ELSE 'pending' END, "
                    "available_at = CASE WHEN attempt_count >= max_attempts "
                    "THEN available_at ELSE clock_timestamp() + make_interval(secs => %s) END, "
                    "error = %s::jsonb, result = NULL, lease_owner = NULL, "
                    "lease_expires_at = NULL, "
                    "finished_at = CASE WHEN attempt_count >= max_attempts "
                    "THEN clock_timestamp() ELSE NULL END, updated_at = clock_timestamp() "
                    "WHERE tenant_id = %s AND corpus_id = %s AND work_item_id = %s "
                    "AND state = 'leased' AND lease_owner = %s AND fencing_token = %s "
                    "AND lease_expires_at > clock_timestamp() RETURNING state",
                    (
                        delay,
                        _json_text(error),
                        scope.tenant_id,
                        scope.corpus_id,
                        lease.work_item_id,
                        lease.worker_id,
                        lease.fencing_token,
                    ),
                ).fetchone(),
                operation="work failure",
            )
        if row is None:
            raise LeaseLostError("work lease is stale, expired, or owned by another worker")
        return WorkState(str(row["state"]))

    # ------------------------------------------------------------ evidence --

    def revision_key(self, scope: DocumentScope, revision_id: str) -> int:
        """Resolve the surrogate the evidence tables are keyed by."""

        require_non_empty(revision_id, "revision_id")
        with self._connection() as connection:
            row = _mapping(
                connection.execute(
                    "SELECT revision_key FROM docarag.index_revisions "
                    "WHERE tenant_id = %s AND corpus_id = %s AND document_id = %s "
                    "AND document_version_id = %s AND revision_id = %s",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        scope.document_id,
                        scope.document_version_id,
                        revision_id,
                    ),
                ).fetchone(),
                operation="revision lookup",
            )
        if row is None:
            raise PostgresLedgerError(
                f"index revision {revision_id} has not been admitted to the ledger"
            )
        return int(row["revision_key"])

    def clear_revision_evidence(self, revision_key: int) -> dict[str, int]:
        """Remove one revision's derived rows.

        The ledger is untouched: it records that the document was admitted and
        what ran over it, and is protected from deletion by database triggers.
        Assets and text blobs are content addressed and shared, so they stay.
        """

        removed: dict[str, int] = {}
        with self._connection() as connection:
            for table in EVIDENCE_TABLES:
                cursor = connection.execute(
                    f"DELETE FROM docarag.{table} WHERE revision_key = %s",
                    (revision_key,),
                )
                count = max(getattr(cursor, "rowcount", 0) or 0, 0)
                if count:
                    removed[table] = count
        return removed

    def persist_document_evidence(
        self, scope: DocumentScope, evidence: DocumentEvidence
    ) -> dict[str, int]:
        """Replace one revision's evidence, then seal it for retrieval.

        Local names in the evidence -- a block's uid, a tree node's number, an
        entity's ref -- are resolved to surrogates here, so a projection never
        has to know what the database assigned.
        """

        revision_key = self.revision_key(scope, evidence.revision_id)
        with self._connection() as connection:
            for table in EVIDENCE_TABLES:
                connection.execute(
                    f"DELETE FROM docarag.{table} WHERE revision_key = %s",
                    (revision_key,),
                )

            if evidence.canonical_document is not None:
                connection.execute(
                    "INSERT INTO docarag.revision_document "
                    "(revision_key, canonical_document) VALUES (%s, %s::jsonb)",
                    (
                        revision_key,
                        json.dumps(evidence.canonical_document, ensure_ascii=False),
                    ),
                )
            asset_keys = self._intern_assets(connection, evidence.assets)
            block_keys = self._write_blocks(
                connection, revision_key, evidence.blocks, asset_keys
            )
            entity_keys = self._intern_entities(connection, scope, evidence.entities)
            node_keys = self._write_tree(
                connection, revision_key, evidence.tree_nodes, block_keys
            )
            self._write_mentions(
                connection, revision_key, evidence.mentions, entity_keys, node_keys,
                block_keys,
            )
            self._write_statements(
                connection, revision_key, evidence.statements, entity_keys, node_keys,
                block_keys,
            )
            self._seal_revision(connection, revision_key)
        return evidence.counts()

    # -- writers ------------------------------------------------------------

    def _intern_text(self, connection: _Connection, body: str) -> int:
        digest = hashlib.sha256(body.encode("utf-8")).digest()
        row = _mapping(
            connection.execute(
                "INSERT INTO docarag.text_blob (sha256, byte_size, body) "
                "VALUES (%s, %s, %s) ON CONFLICT (sha256) DO UPDATE "
                "SET byte_size = EXCLUDED.byte_size RETURNING blob_key",
                (digest, len(body.encode("utf-8")), body),
            ).fetchone(),
            operation="text interning",
        )
        return int(row["blob_key"])

    def _intern_assets(
        self, connection: _Connection, assets: Sequence[AssetRecord]
    ) -> dict[str, int]:
        keys: dict[str, int] = {}
        for asset in assets:
            row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.asset "
                    "(sha256, media_type, byte_size, width, height, cas_uri, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (sha256) DO UPDATE SET cas_uri = EXCLUDED.cas_uri "
                    "RETURNING asset_key",
                    (
                        bytes.fromhex(asset.sha256),
                        asset.media_type,
                        asset.byte_size,
                        asset.width,
                        asset.height,
                        asset.cas_uri,
                        _json_text(asset.attrs),
                    ),
                ).fetchone(),
                operation="asset interning",
            )
            keys[asset.sha256] = int(row["asset_key"])
        return keys

    def _write_blocks(
        self,
        connection: _Connection,
        revision_key: int,
        blocks: Sequence[BlockRecord],
        asset_keys: Mapping[str, int],
    ) -> dict[str, int]:
        block_keys: dict[str, int] = {}
        for block in blocks:
            row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.block (revision_key, block_uid, canonical_blob_key, block_revision_uid, "
                    "page_uid, page_no, block_no, block_kind, subtype, reading_order, quality, "
                    "reconciliation, content_sha256, geometry, "
                    "source_observation_ids, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, "
                    "%s::jsonb, %s::jsonb) RETURNING block_key",
                    (
                        revision_key,
                        block.uid,
                        self._intern_text(
                            connection,
                            json.dumps(
                                block.canonical, ensure_ascii=False, sort_keys=True
                            ),
                        )
                        if block.canonical is not None
                        else None,
                        block.revision_uid,
                        block.page_uid,
                        block.page_no,
                        block.block_no,
                        block.kind,
                        block.subtype,
                        block.reading_order,
                        block.quality,
                        block.reconciliation,
                        block.content_sha256,
                        _json_text(block.geometry) if block.geometry else None,
                        _json_array(block.source_observation_ids),
                        _json_text(block.attrs),
                    ),
                ).fetchone(),
                operation="block write",
            )
            block_key = int(row["block_key"])
            block_keys[block.uid] = block_key

            for rep_no, rep in enumerate(block.representations):
                connection.execute(
                    "INSERT INTO docarag.block_representation "
                    "(revision_key, block_key, rep_no, rep_kind, blob_key, "
                    "asset_key, lang, confidence, producer, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    (
                        revision_key,
                        block_key,
                        rep_no,
                        rep.kind,
                        self._intern_text(connection, rep.text)
                        if rep.text is not None
                        else None,
                        asset_keys.get(rep.asset_sha256) if rep.asset_sha256 else None,
                        rep.lang,
                        rep.confidence,
                        rep.producer,
                        _json_text(rep.attrs),
                    ),
                )

            for sha, role in block.asset_roles:
                asset_key = asset_keys.get(sha)
                if asset_key is None:
                    continue
                connection.execute(
                    "INSERT INTO docarag.block_asset "
                    "(revision_key, block_key, asset_key, role) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (revision_key, block_key, asset_key, role),
                )

            for structure in block.structures:
                self._write_structure(connection, revision_key, block_key, structure)
        return block_keys

    def _write_structure(
        self,
        connection: _Connection,
        revision_key: int,
        block_key: int,
        structure: StructureRecord,
    ) -> None:
        row = _mapping(
            connection.execute(
                "INSERT INTO docarag.block_structure (revision_key, block_key, "
                "structure_kind, producer, availability, confidence, source_uri, attrs) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) RETURNING structure_key",
                (
                    revision_key,
                    block_key,
                    structure.kind,
                    structure.producer,
                    structure.availability,
                    structure.confidence,
                    structure.source_uri,
                    _json_text(structure.attrs),
                ),
            ).fetchone(),
            operation="structure write",
        )
        structure_key = int(row["structure_key"])
        element_keys: dict[str, int] = {}
        for element in structure.elements:
            element_row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.block_element (revision_key, structure_key, "
                    "local_key, element_kind, content, role, row_no, col_no, "
                    "row_span, col_span, geometry, confidence, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, "
                    "%s::jsonb) RETURNING element_key",
                    (
                        revision_key,
                        structure_key,
                        element.local_key,
                        element.kind,
                        element.content,
                        element.role,
                        element.row_no,
                        element.col_no,
                        element.row_span,
                        element.col_span,
                        _json_text(element.geometry) if element.geometry else None,
                        element.confidence,
                        _json_text(element.attrs),
                    ),
                ).fetchone(),
                operation="element write",
            )
            element_keys[element.local_key] = int(element_row["element_key"])

        for edge in structure.edges:
            connection.execute(
                "INSERT INTO docarag.block_element_edge (revision_key, structure_key, "
                "src_key, dst_key, edge_kind, ordinal, label, attrs) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) ON CONFLICT DO NOTHING",
                (
                    revision_key,
                    structure_key,
                    element_keys[edge.src_local_key],
                    element_keys[edge.dst_local_key],
                    edge.kind,
                    edge.ordinal,
                    edge.label,
                    _json_text(edge.attrs),
                ),
            )

    def _write_tree(
        self,
        connection: _Connection,
        revision_key: int,
        nodes: Sequence[TreeNodeRecord],
        block_keys: Mapping[str, int],
    ) -> dict[int, int]:
        """Insert nodes, then bind parents once every node has a key."""

        node_keys: dict[int, int] = {}
        for node in nodes:
            row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.tree_node (revision_key, node_no, "
                    "sibling_no, node_kind, block_key, label, summary_blob_key, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) RETURNING node_key",
                    (
                        revision_key,
                        node.node_no,
                        node.sibling_no,
                        node.kind,
                        block_keys.get(node.block_uid) if node.block_uid else None,
                        node.label,
                        self._intern_text(connection, node.summary)
                        if node.summary
                        else None,
                        _json_text(node.attrs),
                    ),
                ).fetchone(),
                operation="tree node write",
            )
            node_keys[node.node_no] = int(row["node_key"])

        for node in nodes:
            if node.parent_no is None:
                continue
            parent_key = node_keys.get(node.parent_no)
            if parent_key is None:
                continue
            connection.execute(
                "UPDATE docarag.tree_node SET parent_key = %s "
                "WHERE revision_key = %s AND node_key = %s",
                (parent_key, revision_key, node_keys[node.node_no]),
            )
        return node_keys

    def _intern_entities(
        self,
        connection: _Connection,
        scope: DocumentScope,
        entities: Sequence[EntityRecord],
    ) -> dict[str, int]:
        """Resolve corpus-scoped entities, reusing any this corpus already knows."""

        keys: dict[str, int] = {}
        for entity in entities:
            row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.entity (tenant_id, corpus_id, entity_kind, "
                    "canonical, description_blob_key, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (tenant_id, corpus_id, entity_kind, canonical) "
                    "DO UPDATE SET attrs = docarag.entity.attrs || EXCLUDED.attrs "
                    "RETURNING entity_key",
                    (
                        scope.tenant_id,
                        scope.corpus_id,
                        entity.kind,
                        entity.canonical,
                        self._intern_text(connection, entity.description)
                        if entity.description
                        else None,
                        _json_text(entity.attrs),
                    ),
                ).fetchone(),
                operation="entity interning",
            )
            keys[entity.ref] = int(row["entity_key"])
        return keys

    def _write_mentions(
        self,
        connection: _Connection,
        revision_key: int,
        mentions: Sequence[MentionRecord],
        entity_keys: Mapping[str, int],
        node_keys: Mapping[int, int],
        block_keys: Mapping[str, int],
    ) -> None:
        for mention in mentions:
            entity_key = entity_keys.get(mention.entity_ref)
            if entity_key is None:
                continue
            connection.execute(
                "INSERT INTO docarag.entity_mention (revision_key, entity_key, "
                "surface, node_key, block_key, char_start, char_end, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    revision_key,
                    entity_key,
                    mention.surface,
                    node_keys.get(mention.node_no) if mention.node_no is not None else None,
                    block_keys.get(mention.block_uid) if mention.block_uid else None,
                    mention.char_start,
                    mention.char_end,
                    mention.confidence,
                ),
            )

    def _write_statements(
        self,
        connection: _Connection,
        revision_key: int,
        statements: Sequence[StatementRecord],
        entity_keys: Mapping[str, int],
        node_keys: Mapping[int, int],
        block_keys: Mapping[str, int],
    ) -> None:
        for statement in statements:
            roles = [
                (role, entity_keys[role.entity_ref])
                for role in statement.roles
                if role.entity_ref in entity_keys
            ]
            if len(roles) < 2:
                continue
            row = _mapping(
                connection.execute(
                    "INSERT INTO docarag.statement (revision_key, predicate, directed, "
                    "weight, confidence, producer, description_blob_key, attrs) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) RETURNING stmt_key",
                    (
                        revision_key,
                        statement.predicate,
                        statement.directed,
                        statement.weight,
                        statement.confidence,
                        statement.producer,
                        self._intern_text(connection, statement.description)
                        if statement.description
                        else None,
                        _json_text(statement.attrs),
                    ),
                ).fetchone(),
                operation="statement write",
            )
            stmt_key = int(row["stmt_key"])
            for role, entity_key in roles:
                connection.execute(
                    "INSERT INTO docarag.statement_role (revision_key, stmt_key, role, "
                    "entity_key, ordinal) VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (revision_key, stmt_key, role.role, entity_key, role.ordinal),
                )
            for name, value in (statement.properties or {}).items():
                connection.execute(
                    "INSERT INTO docarag.statement_property (revision_key, stmt_key, "
                    "name, text_val, num_val) VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (
                        revision_key,
                        stmt_key,
                        str(name),
                        None if isinstance(value, (int, float)) else str(value),
                        value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                    ),
                )
            for item in statement.evidence:
                connection.execute(
                    "INSERT INTO docarag.statement_evidence (revision_key, stmt_key, "
                    "node_key, block_key, element_key, char_start, char_end) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        revision_key,
                        stmt_key,
                        node_keys.get(item.node_no) if item.node_no is not None else None,
                        block_keys.get(item.block_uid) if item.block_uid else None,
                        None,
                        item.char_start,
                        item.char_end,
                    ),
                )

    def _seal_revision(self, connection: _Connection, revision_key: int) -> None:
        """Derive every tree path, then build the reranker's feed.

        A path depends on the final ordering of every ancestor, so it can only
        be known once the last node is written.  A sealed revision is immutable,
        so neither the path nor the feed ever has to be maintained.
        """

        connection.execute(_SEAL_TREE_PATHS_SQL, (revision_key, revision_key))
        connection.execute(_BUILD_NODE_RETRIEVAL_SQL, (revision_key,))


_SEAL_TREE_PATHS_SQL = """
WITH RECURSIVE walk AS (
    SELECT node_key, node_no::text::public.ltree AS path, 0 AS depth
    FROM docarag.tree_node
    WHERE revision_key = %s AND parent_key IS NULL
    UNION ALL
    SELECT child.node_key,
           parent.path || child.node_no::text::public.ltree,
           parent.depth + 1
    FROM docarag.tree_node child
    JOIN walk parent ON child.parent_key = parent.node_key
)
UPDATE docarag.tree_node AS target
SET path = walk.path, depth = walk.depth
FROM walk
WHERE target.revision_key = %s AND target.node_key = walk.node_key
"""

# One row per node holding the single string a reranker scores, chosen by a
# fixed precedence so the same node always presents the same text.
_BUILD_NODE_RETRIEVAL_SQL = """
INSERT INTO docarag.node_retrieval
    (revision_key, node_key, node_kind, page_no, text_len, retrieval_text)
SELECT node.revision_key, node.node_key, node.node_kind, blk.page_no,
       char_length(chosen.body), chosen.body
FROM docarag.tree_node node
LEFT JOIN docarag.block blk
    ON blk.revision_key = node.revision_key AND blk.block_key = node.block_key
CROSS JOIN LATERAL (
    SELECT coalesce(
        (
            SELECT txt.body
            FROM docarag.block_representation rep
            JOIN docarag.text_blob txt ON txt.blob_key = rep.blob_key
            WHERE rep.revision_key = node.revision_key
              AND rep.block_key = node.block_key
            ORDER BY array_position(
                ARRAY['text', 'html', 'ocr_text', 'visual_description',
                      'caption', 'vlm_latex', 'latex', 'markdown'],
                rep.rep_kind
            ) NULLS LAST, rep.rep_no
            LIMIT 1
        ),
        node.label,
        ''
    ) AS body
) AS chosen
WHERE node.revision_key = %s AND chosen.body <> ''
ON CONFLICT (revision_key, node_key) DO UPDATE
SET retrieval_text = EXCLUDED.retrieval_text, text_len = EXCLUDED.text_len
"""


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_DATABASE_URL",
    "DEFAULT_DATABASE_URL_ENV",
    "ImmutableRecordConflict",
    "LeaseLostError",
    "PostgresLedger",
    "PostgresLedgerError",
    "PsycopgUnavailableError",
    "SchemaDriftError",
    "resolve_database_url",
    "safe_database_endpoint",
]
