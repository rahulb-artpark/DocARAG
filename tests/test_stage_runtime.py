from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from Core.pipelines.doc_tree_builder import _stable_local_artifact_receipt
from docarag.persistence import (
    ArtifactAvailability,
    DocumentScope,
    StageEventStatus,
    StageLeaseUnavailableError,
    StageRuntimeConfigurationError,
    WorkLease,
    WorkState,
    durable_stage,
)
from docarag.persistence.contracts import IndexRevisionState


def _environment() -> dict[str, str]:
    return {
        "DOCARAG_DATABASE_URL": "postgresql://ignored:ignored@db/docarag",
        "DOCARAG_TENANT_ID": "tenant-a",
        "DOCARAG_CORPUS_ID": "corpus-a",
        "DOCARAG_DOCUMENT_ID": "document-a",
        "DOCARAG_DOCUMENT_VERSION_ID": "document-version-a",
        "DOCARAG_INDEX_REVISION_ID": "revision-a",
        "DOCARAG_TRACE_ID": "trace-a",
        "DOCARAG_SOURCE_SHA256": "a" * 64,
        "DOCARAG_STAGE_LEASE_SECONDS": "30",
        "DOCARAG_STAGE_RETRY_DELAY_SECONDS": "0",
    }


class _MemoryLedger:
    def __init__(self) -> None:
        self.specs = {}
        self.states = {}
        self.events = []
        self.receipts = []
        self.completed = []
        self.failed = []
        self.lease_enabled = True

    def enqueue_work(self, spec):
        previous = self.specs.get(spec.work_item_id)
        if previous is not None and previous != spec:
            raise AssertionError("work changed across retry")
        self.specs[spec.work_item_id] = spec
        self.states.setdefault(spec.work_item_id, WorkState.PENDING)
        return spec.work_item_id

    def get_work_state(self, scope, work_item_id):
        self._assert_scope(scope)
        return self.states.get(work_item_id)

    def lease_work(self, corpus_scope, *, worker_id, lease_seconds=60, kinds=None):
        if not self.lease_enabled:
            return None
        for work_item_id, spec in self.specs.items():
            if kinds and spec.kind not in kinds:
                continue
            if self.states[work_item_id] is not WorkState.PENDING:
                continue
            self.states[work_item_id] = WorkState.LEASED
            return WorkLease(
                scope=spec.scope,
                work_item_id=work_item_id,
                kind=spec.kind,
                payload=spec.payload,
                worker_id=worker_id,
                fencing_token=1,
                attempt_count=1,
                max_attempts=spec.max_attempts,
                lease_expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=lease_seconds),
            )
        return None

    def heartbeat_work(self, lease, *, extension_seconds=60):
        return WorkLease(
            scope=lease.scope,
            work_item_id=lease.work_item_id,
            kind=lease.kind,
            payload=lease.payload,
            worker_id=lease.worker_id,
            fencing_token=lease.fencing_token,
            attempt_count=lease.attempt_count,
            max_attempts=lease.max_attempts,
            lease_expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=extension_seconds),
        )

    def complete_work(self, lease, *, result=None):
        self.states[lease.work_item_id] = WorkState.SUCCEEDED
        self.completed.append((lease, dict(result or {})))

    def fail_work(self, lease, *, error, retry_delay_seconds=0):
        self.states[lease.work_item_id] = WorkState.PENDING
        self.failed.append((lease, dict(error), retry_delay_seconds))
        return WorkState.PENDING

    def append_stage_event(self, event):
        self.events.append(event)

    def record_artifact(self, receipt):
        self.receipts.append(receipt)

    def create_index_revision(self, spec):
        self.index_revision_specs = getattr(self, "index_revision_specs", [])
        self.index_revision_specs.append(spec)
        return IndexRevisionState.BUILDING

    def publish_index_revision(self, spec, publication):
        self.published = getattr(self, "published", [])
        self.published.append((spec, publication))

    @staticmethod
    def _assert_scope(scope):
        if scope != DocumentScope(
            "tenant-a", "corpus-a", "document-a", "document-version-a"
        ):
            raise AssertionError("wrong document scope")


class DurableStageRuntimeTests(unittest.TestCase):
    def test_absent_database_environment_is_a_safe_noop(self):
        with durable_stage("parse", environment={}) as stage:
            self.assertFalse(stage.enabled)
            stage.checkpoint("ignored")
            stage.record_artifact("/does/not/exist", artifact_role="ignored")

    def test_explicit_database_with_partial_scope_fails_closed(self):
        with self.assertRaisesRegex(
            StageRuntimeConfigurationError, "environment is incomplete"
        ):
            durable_stage(
                "parse",
                environment={"DOCARAG_DATABASE_URL": _environment()["DOCARAG_DATABASE_URL"]},
                ledger=_MemoryLedger(),
            )

    def test_success_is_leased_traced_checkpointed_and_receipted(self):
        ledger = _MemoryLedger()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "canonical.json"
            artifact.write_text('{"blocks":1}\n', encoding="utf-8")
            with durable_stage(
                "parse",
                payload={"parser_profile": "balanced-v1"},
                environment=_environment(),
                ledger=ledger,
            ) as stage:
                self.assertTrue(stage.enabled)
                self.assertFalse(stage.replayed)
                stage.checkpoint("pages.complete", {"page_count": 1})
                stage.record_artifact(
                    artifact,
                    artifact_role="canonical-evidence",
                    artifact_schema="docarag.canonical-evidence.v1",
                    media_type="application/json",
                )

        self.assertEqual(WorkState.SUCCEEDED, next(iter(ledger.states.values())))
        self.assertEqual(1, len(ledger.completed))
        self.assertEqual(1, len(ledger.receipts))
        receipt = ledger.receipts[0]
        self.assertIs(ArtifactAvailability.EMITTED, receipt.availability)
        self.assertEqual("a" * 64, receipt.source_sha256)
        self.assertEqual(
            [
                StageEventStatus.STARTED,
                StageEventStatus.PROGRESS,
                StageEventStatus.PROGRESS,
                StageEventStatus.SUCCEEDED,
            ],
            [event.status for event in ledger.events],
        )
        result = ledger.completed[0][1]
        self.assertEqual(2, result["checkpoint_count"])
        self.assertEqual([receipt.receipt_id], result["receipt_ids"])

    def test_publish_index_revision_uses_the_pipelines_own_revision_id(self):
        # DOCARAG_INDEX_REVISION_ID in _environment() is "revision-a" -- the
        # same identity every other stage's work_item_id/artifact receipt in
        # this run is keyed by. Leaving IndexRevisionSpec.revision_id unset
        # here previously let it derive a different, content-hash identity,
        # so the row this call marked 'ready' was never the row
        # persist_document_evidence wrote blocks/entities/statements into.
        ledger = _MemoryLedger()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree_path = root / "tree.pkl"
            graph_path = root / "graph_data.json"
            manifest_path = root / "attestation.json"
            for path in (tree_path, graph_path, manifest_path):
                path.write_text("{}", encoding="utf-8")
            with durable_stage(
                "vector", environment=_environment(), ledger=ledger
            ) as stage:
                stage.publish_index_revision(
                    configuration_sha256="b" * 64,
                    canonical_artifact_sha256="c" * 64,
                    tree_artifact_path=tree_path,
                    graph_artifact_path=graph_path,
                    retrieval_manifest_path=manifest_path,
                    qdrant_collection="TreeVDB",
                    qdrant_point_count=40,
                    metadata={"entity_collection": "TreeVDB_entity"},
                )

        self.assertEqual(1, len(ledger.index_revision_specs))
        spec = ledger.index_revision_specs[0]
        self.assertEqual("revision-a", spec.revision_id)
        self.assertEqual(1, len(ledger.published))
        published_spec, publication = ledger.published[0]
        self.assertEqual("revision-a", published_spec.revision_id)
        self.assertEqual(40, publication.qdrant_point_count)
        # No redundant duplicate of the identity spec.revision_id already is.
        self.assertNotIn("shell_index_revision_id", spec.metadata)

    def test_publish_index_revision_spec_metadata_is_always_empty(self):
        # docarag/evidence_persist.py's _admit_revision always admits this
        # same revision_id first (before any stage runs), with metadata={}
        # (IndexRevisionSpec's default -- it passes none). create_index_revision
        # rejects a later call for the same identity whose metadata differs
        # byte-for-byte (ImmutableRecordConflict) -- so if this call ever put
        # caller-supplied, publish-time-only details (entity_collection,
        # retrieval_identity_sha256, ...) into IndexRevisionSpec.metadata,
        # every single publish would permanently fail against the row
        # admission already created. Regression test for exactly that: the
        # metadata this call sends to create_index_revision must stay empty
        # no matter what the caller passes, and the caller's details must
        # still be observable -- just from the checkpoint event, which has
        # no identity-matching contract to violate.
        ledger = _MemoryLedger()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree_path = root / "tree.pkl"
            graph_path = root / "graph_data.json"
            manifest_path = root / "attestation.json"
            for path in (tree_path, graph_path, manifest_path):
                path.write_text("{}", encoding="utf-8")
            with durable_stage(
                "vector", environment=_environment(), ledger=ledger
            ) as stage:
                stage.publish_index_revision(
                    configuration_sha256="b" * 64,
                    canonical_artifact_sha256="c" * 64,
                    tree_artifact_path=tree_path,
                    graph_artifact_path=graph_path,
                    retrieval_manifest_path=manifest_path,
                    qdrant_collection="TreeVDB",
                    qdrant_point_count=40,
                    metadata={
                        "entity_collection": "TreeVDB_entity",
                        "retrieval_identity_sha256": "d" * 64,
                    },
                )

        spec = ledger.index_revision_specs[0]
        self.assertEqual({}, dict(spec.metadata))
        published_events = [
            event for event in ledger.events
            if event.event_type == "stage.checkpoint"
            and event.payload.get("checkpoint") == "index-revision.published"
        ]
        self.assertEqual(1, len(published_events))
        self.assertEqual(
            "TreeVDB_entity", published_events[0].payload.get("entity_collection")
        )
        self.assertEqual(
            "d" * 64, published_events[0].payload.get("retrieval_identity_sha256")
        )

    def test_failed_stage_is_requeued_and_failure_is_observable(self):
        ledger = _MemoryLedger()
        with self.assertRaisesRegex(RuntimeError, "parser crashed"):
            with durable_stage(
                "parse", environment=_environment(), ledger=ledger
            ):
                raise RuntimeError("parser crashed")

        self.assertEqual(1, len(ledger.failed))
        self.assertEqual(WorkState.PENDING, next(iter(ledger.states.values())))
        self.assertEqual(StageEventStatus.FAILED, ledger.events[-1].status)
        self.assertEqual("RuntimeError", ledger.events[-1].payload["error_type"])

    def test_succeeded_work_is_replayed_without_another_lease(self):
        ledger = _MemoryLedger()
        with durable_stage(
            "tree", environment=_environment(), ledger=ledger
        ) as first:
            work_item_id = first.work_item_id

        ledger.lease_enabled = False
        with durable_stage(
            "tree", environment=_environment(), ledger=ledger
        ) as replay:
            self.assertTrue(replay.replayed)
            self.assertEqual(work_item_id, replay.work_item_id)

        self.assertEqual(1, len(ledger.completed))
        self.assertEqual(StageEventStatus.SKIPPED, ledger.events[-1].status)
        self.assertEqual("already_succeeded", ledger.events[-1].payload["reason"])

    def test_nonterminal_work_without_lease_is_not_run_concurrently(self):
        ledger = _MemoryLedger()
        ledger.lease_enabled = False
        with self.assertRaises(StageLeaseUnavailableError):
            with durable_stage(
                "graph", environment=_environment(), ledger=ledger
            ):
                self.fail("stage body must not execute without a lease")
        self.assertEqual(StageEventStatus.SKIPPED, ledger.events[-1].status)

    def test_core_artifact_receipt_accepts_an_unchanged_regular_file(self):
        """Regression: before/after fingerprint tuples must have identical shape."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.json"
            payload = b'{"canonical_blocks":[]}\n'
            path.write_bytes(payload)

            resolved, sha256, size = _stable_local_artifact_receipt(
                path, label="canonical evidence"
            )

        self.assertEqual(path.resolve(), resolved)
        self.assertEqual(len(payload), size)
        self.assertEqual(
            "17b8539ec50dfc87059d92c797c34fdab4fa03f709e56f21f1e10d7f5172ddc5",
            sha256,
        )


if __name__ == "__main__":
    unittest.main()
