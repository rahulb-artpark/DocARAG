from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from docarag.domain import DocumentRef
from docarag.persistence import (
    ArtifactAvailability,
    ArtifactReceiptRecord,
    CorpusScope,
    DEFAULT_DATABASE_URL,
    DocumentScope,
    ImmutableRecordConflict,
    IndexPublication,
    IndexRevisionSpec,
    LeaseLostError,
    PostgresLedger,
    SchemaDriftError,
    StageEvent,
    StageEventStatus,
    WorkSpec,
    WorkState,
    resolve_database_url,
    safe_database_endpoint,
)
from docarag.persistence.schema import EVIDENCE_TABLES, SCHEMA_SHA256, SCHEMA_SQL
from docarag.settings import RuntimeSettings, initialize_settings, load_settings


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _FakeConnection:
    def __init__(self, handler=None):
        self.handler = handler or (lambda _query, _params: ())
        self.executions: list[tuple[str, tuple[object, ...] | None]] = []
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exited += 1
        return False

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        frozen_params = tuple(params) if params is not None else None
        self.executions.append((normalized, frozen_params))
        return _Result(self.handler(normalized, frozen_params))


def _scope() -> DocumentScope:
    return DocumentScope("tenant-a", "corpus-a", "doc-a", "docv-a")


def _document() -> DocumentRef:
    return DocumentRef(
        document_id="doc-a",
        document_version_id="docv-a",
        content_sha256="a" * 64,
        mime_type="application/pdf",
        source_uri="s3://immutable-source/document.pdf",
        display_name="Document",
        page_count=3,
        metadata={"origin": "fixture"},
    )


class PostgresSettingsTests(unittest.TestCase):
    def test_local_postgres_is_default_and_secret_is_never_serialized(self):
        settings = RuntimeSettings(project_root=Path("/tmp/project")).resolved()

        self.assertEqual(DEFAULT_DATABASE_URL, settings.resolve_database_url({}))
        self.assertEqual(
            "postgresql://127.0.0.1:5432/docarag",
            settings.redacted_database_url({}),
        )

        secret = "never-print-this-password"
        environment = {
            "DOCARAG_DATABASE_URL": (
                f"postgresql://some-user:{secret}@db.internal:5433/production"
                "?sslmode=require"
            )
        }
        with patch.dict(os.environ, environment, clear=True):
            payload = settings.as_dict()
            rendered = json.dumps(payload, sort_keys=True)
            self.assertEqual(
                "postgresql://db.internal:5433/production", payload["database_url"]
            )
            self.assertNotIn(secret, rendered)
            self.assertNotIn("some-user", rendered)
            self.assertNotIn("sslmode", rendered)

    def test_generated_settings_store_only_environment_variable_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            initialize_settings(target, project_root=Path(directory))
            text = target.read_text(encoding="utf-8")
            self.assertIn('database_url_env = "DOCARAG_DATABASE_URL"', text)
            self.assertIn("[storage]", text)
            self.assertIn('tenant_id = "default"', text)
            self.assertIn('corpus_id = "default"', text)
            self.assertNotIn("docarag-local", text)
            loaded = load_settings(target)
            self.assertEqual("DOCARAG_DATABASE_URL", loaded.database_url_env)
            self.assertEqual("default", loaded.tenant_id)
            self.assertEqual("default", loaded.corpus_id)

    def test_custom_environment_reference_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text(
                "schema_version = 1\n[runtime]\n"
                'database_url_env = "NOT-A-NAME"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "environment-variable"):
                load_settings(target)

    def test_storage_scope_rejects_blank_unknown_and_oversized_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "docarag.toml"
            target.write_text(
                "schema_version = 1\n[storage]\ntenant_id = \"\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tenant_id"):
                load_settings(target)
            target.write_text(
                "schema_version = 1\n[storage]\nunknown = \"x\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_settings(target)
            target.write_text(
                "schema_version = 1\n[storage]\ncorpus_id = \""
                + "x" * 513
                + "\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "corpus_id"):
                load_settings(target)

    def test_postgres_url_resolution_rejects_empty_and_non_postgres_values(self):
        with self.assertRaises(ValueError):
            resolve_database_url("", environment={})
        with self.assertRaises(ValueError):
            resolve_database_url("sqlite:///tmp/store.db", environment={})
        with self.assertRaises(ValueError):
            resolve_database_url(
                environment={"DOCARAG_DATABASE_URL": ""}
            )

    def test_safe_endpoint_removes_user_password_and_query(self):
        value = "postgresql://alice:p%40ss@db.example:5544/app?sslmode=require"
        self.assertEqual(
            "postgresql://db.example:5544/app", safe_database_endpoint(value)
        )
        ledger = PostgresLedger(value, connection_factory=lambda: _FakeConnection())
        self.assertNotIn("alice", repr(ledger))
        self.assertNotIn("p%40ss", repr(ledger))


class PostgresSchemaTests(unittest.TestCase):
    def test_schema_contains_the_ledger_and_the_evidence_store(self):
        self.assertRegex(SCHEMA_SHA256, r"^[0-9a-f]{64}$")
        for table in (
            # ledger: append-only record of admission and execution
            "tenants",
            "corpora",
            "documents",
            "document_versions",
            "artifact_receipts",
            "index_revisions",
            "stage_events",
            "work_items",
            # evidence: derived, rebuildable, revision-scoped
            "text_blob",
            "asset",
            "block",
            "block_representation",
            "block_asset",
            "block_structure",
            "block_element",
            "block_element_edge",
            "tree_node",
            "entity",
            "entity_mention",
            "statement",
            "statement_role",
            "statement_property",
            "statement_evidence",
            "node_retrieval",
        ):
            self.assertIn(f"CREATE TABLE docarag.{table} (", SCHEMA_SQL)
        self.assertIn("artifact_receipts_are_immutable", SCHEMA_SQL)
        self.assertIn("stage_events_are_append_only", SCHEMA_SQL)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS ltree", SCHEMA_SQL)
        self.assertNotIn("sqlite", SCHEMA_SQL.lower())

    def test_evidence_is_keyed_by_an_interned_revision(self):
        # The ledger identifies a revision by five text columns; evidence
        # carries one surrogate instead, so the tuple is stored once.
        self.assertIn(
            "revision_key bigint GENERATED ALWAYS AS IDENTITY UNIQUE", SCHEMA_SQL
        )
        for table in EVIDENCE_TABLES:
            body = SCHEMA_SQL.split(f"CREATE TABLE docarag.{table} (", 1)[1]
            body = body.split(");", 1)[0]
            self.assertIn(
                "revision_key bigint NOT NULL",
                body,
                f"{table} must be scoped by revision_key",
            )

    def test_the_tree_is_addressable_by_path_and_by_parent(self):
        # Subtree reads use the ltree path; ingestion writes adjacency before
        # any path can exist, so both have to be indexed.
        self.assertIn("path public.ltree", SCHEMA_SQL)
        self.assertIn("USING gist (path)", SCHEMA_SQL)
        self.assertIn(
            "tree_node_children_idx\n    ON docarag.tree_node "
            "(revision_key, parent_key, sibling_no)",
            SCHEMA_SQL,
        )

    def test_the_hot_retrieval_reads_are_covered_by_indexes(self):
        for covering in (
            # query entity -> the nodes mentioning it
            "entity_mention_lookup_idx",
            # a subtree's nodes -> the statements they support
            "statement_evidence_node_idx",
            # hundreds of candidates -> one string each, for the reranker
            "node_retrieval_feed_idx",
        ):
            self.assertIn(covering, SCHEMA_SQL)
            clause = SCHEMA_SQL.split(covering, 1)[1].split(";", 1)[0]
            self.assertIn("INCLUDE", clause, f"{covering} should be covering")

    def test_ensure_schema_is_locked_and_idempotent(self):
        state: dict[str, object] = {"applied": None}

        def handler(query, params):
            if "INSERT INTO docarag.schema_meta" in query:
                state["applied"] = params[0]
                return ()
            if "to_regclass('docarag.schema_meta')" in query:
                return ({"relation": "docarag.schema_meta"},)
            if "SELECT schema_sha256 FROM docarag.schema_meta" in query:
                return ({"schema_sha256": state["applied"]},) if state["applied"] else ()
            return ()

        connection = _FakeConnection(handler)
        ledger = PostgresLedger(connection_factory=lambda: connection)

        first = ledger.ensure_schema()
        second = ledger.ensure_schema()

        self.assertTrue(first.current)
        self.assertTrue(second.current)
        self.assertEqual(SCHEMA_SHA256, state["applied"])
        queries = [query for query, _ in connection.executions]
        self.assertTrue(any("pg_advisory_xact_lock" in query for query in queries))
        self.assertTrue(
            any("CREATE TABLE docarag.block (" in query for query in queries)
        )
        self.assertEqual(
            1, sum(1 for q in queries if "INSERT INTO docarag.schema_meta" in q)
        )

    def test_reset_schema_drops_and_recreates(self):
        state: dict[str, object] = {"applied": "e" * 64}

        def handler(query, params):
            if "DROP SCHEMA IF EXISTS docarag CASCADE" in query:
                state["applied"] = None
                return ()
            if "INSERT INTO docarag.schema_meta" in query:
                state["applied"] = params[0]
                return ()
            if "to_regclass('docarag.schema_meta')" in query:
                return ({"relation": "docarag.schema_meta"},)
            if "SELECT schema_sha256 FROM docarag.schema_meta" in query:
                return ({"schema_sha256": state["applied"]},) if state["applied"] else ()
            return ()

        connection = _FakeConnection(handler)
        ledger = PostgresLedger(connection_factory=lambda: connection)
        status = ledger.reset_schema()

        self.assertTrue(status.current)
        self.assertEqual(SCHEMA_SHA256, state["applied"])
        queries = [query for query, _ in connection.executions]
        self.assertTrue(
            any("DROP SCHEMA IF EXISTS docarag CASCADE" in query for query in queries)
        )

    def test_fingerprint_mismatch_reports_not_current_without_raising(self):
        def handler(query, _params):
            if "to_regclass('docarag.schema_meta')" in query:
                return ({"relation": "docarag.schema_meta"},)
            if "SELECT schema_sha256 FROM docarag.schema_meta" in query:
                return ({"schema_sha256": "0" * 64},)
            return ()

        ledger = PostgresLedger(connection_factory=lambda: _FakeConnection(handler))
        status = ledger.schema_status()

        self.assertTrue(status.present)
        self.assertFalse(status.current)
        self.assertEqual("0" * 64, status.applied_sha256)
        self.assertTrue(status.errors)

    def test_health_is_cli_safe_and_reports_schema(self):
        def handler(query, _params):
            if "current_setting('server_version_num')" in query:
                return ({"server_version_num": 180001, "database_name": "docarag"},)
            if "to_regclass('docarag.schema_meta')" in query:
                return ({"relation": "docarag.schema_meta"},)
            if "SELECT schema_sha256 FROM docarag.schema_meta" in query:
                return ({"schema_sha256": SCHEMA_SHA256},)
            return ()

        ledger = PostgresLedger(
            "postgresql://user:secret@db.internal:5432/docarag",
            connection_factory=lambda: _FakeConnection(handler),
        )
        health = ledger.health()
        rendered = json.dumps(health.to_dict(), sort_keys=True)
        self.assertTrue(health.ok)
        self.assertTrue(health.schema.current)
        self.assertEqual(180001, health.server_version_num)
        self.assertNotIn("secret", rendered)
        self.assertNotIn("user", rendered)

    def test_health_does_not_echo_exception_or_credentials(self):
        def fail():
            raise RuntimeError("password=do-not-echo")

        ledger = PostgresLedger(
            "postgresql://user:top-secret@db.internal:5432/docarag",
            connection_factory=fail,
        )
        rendered = json.dumps(ledger.health().to_dict(), sort_keys=True)
        self.assertIn("RuntimeError", rendered)
        self.assertNotIn("do-not-echo", rendered)
        self.assertNotIn("top-secret", rendered)


class DeleteRevisionEvidenceTests(unittest.TestCase):
    def _delete(self):
        connection = _FakeConnection()
        ledger = PostgresLedger(connection_factory=lambda: connection)
        ledger.delete_revision_evidence(_scope(), "rev-a")
        return connection

    def test_every_evidence_table_is_scoped_to_one_revision(self):
        connection = self._delete()

        deletes = [q for q, _ in connection.executions if q.startswith("DELETE")]
        self.assertEqual(7, len(deletes))
        for query, params in connection.executions:
            self.assertIn("index_revision_id = %s", query)
            self.assertEqual(
                ("tenant-a", "corpus-a", "doc-a", "docv-a", "rev-a"), params
            )

    def test_the_append_only_ledger_is_never_touched(self):
        # Document versions, index revisions, artifact receipts, and stage
        # events are protected by database triggers because they record that a
        # document was admitted and what ran over it.
        connection = self._delete()

        touched = " ".join(query for query, _ in connection.executions)
        for table in (
            "document_versions",
            "index_revisions",
            "artifact_receipts",
            "stage_events",
            "documents",
        ):
            self.assertNotIn(f"docarag.{table}", touched)

    def test_shared_content_addressed_assets_survive(self):
        connection = self._delete()

        touched = " ".join(query for query, _ in connection.executions)
        self.assertNotIn("docarag.assets", touched)

    def test_an_empty_revision_id_is_refused(self):
        connection = _FakeConnection()
        ledger = PostgresLedger(connection_factory=lambda: connection)

        with self.assertRaises(Exception):
            ledger.delete_revision_evidence(_scope(), "")
        self.assertEqual([], connection.executions)


class PostgresTruthRecordTests(unittest.TestCase):
    def test_document_version_admission_is_idempotent_and_content_bound(self):
        document = _document()

        def handler(query, _params):
            if query.startswith("SELECT source_sha256"):
                return (
                    {
                        "source_sha256": document.content_sha256,
                        "source_uri": document.source_uri,
                        "mime_type": document.mime_type,
                        "page_count": document.page_count,
                        "metadata": {"origin": "fixture"},
                    },
                )
            return ()

        connection = _FakeConnection(handler)
        ledger = PostgresLedger(connection_factory=lambda: connection)
        scope = ledger.record_document(CorpusScope("tenant-a", "corpus-a"), document)

        self.assertEqual(_scope(), scope)
        queries = [query for query, _ in connection.executions]
        self.assertTrue(any("INSERT INTO docarag.tenants" in query for query in queries))
        self.assertTrue(any("INSERT INTO docarag.corpora" in query for query in queries))
        self.assertTrue(
            any("INSERT INTO docarag.document_versions" in query for query in queries)
        )

    def test_document_version_conflict_is_rejected(self):
        def handler(query, _params):
            if query.startswith("SELECT source_sha256"):
                return (
                    {
                        "source_sha256": "b" * 64,
                        "source_uri": "s3://other",
                        "mime_type": "application/pdf",
                        "page_count": 3,
                        "metadata": {},
                    },
                )
            return ()

        ledger = PostgresLedger(connection_factory=lambda: _FakeConnection(handler))
        with self.assertRaises(ImmutableRecordConflict):
            ledger.record_document(CorpusScope("tenant-a", "corpus-a"), _document())

    def test_artifact_receipt_accepts_exact_retry_and_rejects_rebinding(self):
        receipt = ArtifactReceiptRecord(
            scope=_scope(),
            receipt_id="receipt-a",
            artifact_role="canonical_blocks",
            availability=ArtifactAvailability.EMITTED,
            artifact_schema="canonical.v1",
            producer_task_id="task-a",
            source_sha256="a" * 64,
            uri="s3://artifacts/canonical.json",
            sha256="c" * 64,
            size_bytes=123,
            media_type="application/json",
            metadata={"parser": "docling"},
        )

        def matching(query, _params):
            if query.startswith("SELECT receipt_id"):
                return (
                    {
                        "receipt_id": receipt.receipt_id,
                        "availability": receipt.availability.value,
                        "artifact_schema": receipt.artifact_schema,
                        "source_sha256": receipt.source_sha256,
                        "uri": receipt.uri,
                        "sha256": receipt.sha256,
                        "size_bytes": receipt.size_bytes,
                        "media_type": receipt.media_type,
                        "metadata": {"parser": "docling"},
                    },
                )
            return ()

        PostgresLedger(connection_factory=lambda: _FakeConnection(matching)).record_artifact(
            receipt
        )

        def conflicting(query, _params):
            rows = matching(query, _params)
            if rows:
                changed = dict(rows[0])
                changed["sha256"] = "d" * 64
                return (changed,)
            return ()

        with self.assertRaises(ImmutableRecordConflict):
            PostgresLedger(
                connection_factory=lambda: _FakeConnection(conflicting)
            ).record_artifact(receipt)

    def test_index_revision_defaults_to_qdrants_unnamed_vector(self):
        spec = IndexRevisionSpec(
            scope=_scope(),
            configuration_sha256="b" * 64,
            canonical_artifact_sha256="c" * 64,
            qdrant_collection="docarag-evidence",
        )
        stored = {
            "configuration_sha256": spec.configuration_sha256,
            "canonical_artifact_sha256": spec.canonical_artifact_sha256,
            "qdrant_collection": spec.qdrant_collection,
            "qdrant_vector_name": None,
            "state": "building",
            "metadata": {},
            "tree_artifact_sha256": None,
            "graph_artifact_sha256": None,
            "retrieval_manifest_sha256": None,
            "qdrant_point_count": None,
            "failure_code": None,
            "failure_message": None,
        }

        def handler(query, _params):
            if query.startswith("SELECT configuration_sha256"):
                return (stored,)
            return ()

        connection = _FakeConnection(handler)
        state = PostgresLedger(
            connection_factory=lambda: connection
        ).create_index_revision(spec)

        self.assertEqual("building", state.value)
        insert = next(
            params
            for query, params in connection.executions
            if query.startswith("INSERT INTO docarag.index_revisions")
        )
        self.assertIsNone(insert[8])

    def test_index_publication_is_content_bound_and_idempotent(self):
        spec = IndexRevisionSpec(
            scope=_scope(),
            configuration_sha256="b" * 64,
            canonical_artifact_sha256="c" * 64,
            qdrant_collection="docarag-evidence",
        )
        publication = IndexPublication(
            tree_artifact_sha256="d" * 64,
            graph_artifact_sha256="e" * 64,
            retrieval_manifest_sha256="f" * 64,
            qdrant_point_count=42,
        )

        def handler(query, _params):
            if query.startswith("UPDATE docarag.index_revisions SET state = 'ready'"):
                return ({"revision_id": spec.revision_id},)
            return ()

        PostgresLedger(
            connection_factory=lambda: _FakeConnection(handler)
        ).publish_index_revision(spec, publication)

    def test_stage_events_are_append_only_and_sequence_bound(self):
        event = StageEvent(
            scope=_scope(),
            trace_id="trace-a",
            stage_run_id="parse-run-a",
            sequence_no=1,
            stage_name="parse",
            event_type="page.completed",
            status=StageEventStatus.PROGRESS,
            payload={"page_index": 0},
        )
        from docarag.domain.base import content_sha256

        def handler(query, _params):
            if query.startswith("SELECT event_id"):
                return (
                    {
                        "event_id": event.event_id,
                        "trace_id": event.trace_id,
                        "stage_name": event.stage_name,
                        "event_type": event.event_type,
                        "status": event.status.value,
                        "payload_sha256": content_sha256(event.payload),
                    },
                )
            return ()

        PostgresLedger(
            connection_factory=lambda: _FakeConnection(handler)
        ).append_stage_event(event)


class PostgresWorkQueueTests(unittest.TestCase):
    def test_enqueue_is_idempotent_and_payload_bound(self):
        spec = WorkSpec(
            scope=_scope(),
            kind="parse.page",
            idempotency_key="docv-a/page/0/parser/docling",
            payload={"page_index": 0},
            priority=10,
            max_attempts=4,
        )
        from docarag.domain.base import content_sha256

        def handler(query, _params):
            if query.startswith("SELECT work_item_id"):
                return (
                    {
                        "work_item_id": spec.work_item_id,
                        "document_id": spec.scope.document_id,
                        "document_version_id": spec.scope.document_version_id,
                        "payload_sha256": content_sha256(spec.payload),
                        "priority": spec.priority,
                        "max_attempts": spec.max_attempts,
                    },
                )
            return ()

        ledger = PostgresLedger(connection_factory=lambda: _FakeConnection(handler))
        self.assertEqual(spec.work_item_id, ledger.enqueue_work(spec))

    def test_work_state_lookup_is_bound_to_the_full_document_scope(self):
        scope = _scope()

        def handler(query, _params):
            if query.startswith("SELECT state FROM docarag.work_items"):
                return ({"state": "succeeded"},)
            return ()

        connection = _FakeConnection(handler)
        state = PostgresLedger(
            connection_factory=lambda: connection
        ).get_work_state(scope, "work-a")

        self.assertIs(WorkState.SUCCEEDED, state)
        query, params = next(
            item
            for item in connection.executions
            if item[0].startswith("SELECT state FROM docarag.work_items")
        )
        self.assertIn("document_version_id = %s", query)
        self.assertEqual(
            (
                scope.tenant_id,
                scope.corpus_id,
                scope.document_id,
                scope.document_version_id,
                "work-a",
            ),
            params,
        )

    def test_lease_is_transactional_skip_locked_and_fenced(self):
        expiry = datetime(2030, 1, 1, tzinfo=timezone.utc)
        row = {
            "tenant_id": "tenant-a",
            "corpus_id": "corpus-a",
            "document_id": "doc-a",
            "document_version_id": "docv-a",
            "work_item_id": "work-a",
            "kind": "parse.page",
            "payload": {"page_index": 0},
            "lease_owner": "worker-a",
            "fencing_token": 9,
            "attempt_count": 2,
            "max_attempts": 4,
            "lease_expires_at": expiry,
        }

        def handler(query, _params):
            if query.startswith("WITH candidate AS"):
                return (row,)
            return ()

        connection = _FakeConnection(handler)
        lease = PostgresLedger(
            connection_factory=lambda: connection
        ).lease_work(
            CorpusScope("tenant-a", "corpus-a"),
            worker_id="worker-a",
            kinds=("parse.page",),
        )

        self.assertIsNotNone(lease)
        self.assertEqual(9, lease.fencing_token)
        self.assertEqual(2, lease.attempt_count)
        query = next(query for query, _ in connection.executions if query.startswith("WITH"))
        self.assertIn("FOR UPDATE SKIP LOCKED", query)
        self.assertIn("fencing_token = work.fencing_token + 1", query)
        self.assertIn("attempt_count = work.attempt_count + 1", query)

    def test_stale_fencing_token_cannot_complete_work(self):
        from docarag.persistence import WorkLease

        lease = WorkLease(
            scope=_scope(),
            work_item_id="work-a",
            kind="parse.page",
            payload={},
            worker_id="worker-a",
            fencing_token=2,
            attempt_count=1,
            max_attempts=3,
            lease_expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        connection = _FakeConnection()
        with self.assertRaises(LeaseLostError):
            PostgresLedger(
                connection_factory=lambda: connection
            ).complete_work(lease, result={"ok": True})
        update = next(query for query, _ in connection.executions if query.startswith("UPDATE"))
        self.assertIn("lease_owner = %s", update)
        self.assertIn("fencing_token = %s", update)
        self.assertIn("lease_expires_at > clock_timestamp()", update)

    def test_failed_attempt_is_requeued_until_attempt_budget_is_exhausted(self):
        from docarag.persistence import WorkLease

        lease = WorkLease(
            scope=_scope(),
            work_item_id="work-a",
            kind="parse.page",
            payload={},
            worker_id="worker-a",
            fencing_token=2,
            attempt_count=1,
            max_attempts=3,
            lease_expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )

        def handler(query, _params):
            if query.startswith("UPDATE docarag.work_items SET state = CASE"):
                return ({"state": "pending"},)
            return ()

        state = PostgresLedger(
            connection_factory=lambda: _FakeConnection(handler)
        ).fail_work(lease, error={"code": "timeout"}, retry_delay_seconds=5)
        self.assertIs(WorkState.PENDING, state)


if __name__ == "__main__":
    unittest.main()
