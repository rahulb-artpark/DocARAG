"""Durable, dependency-light stage execution for the BookRAG compatibility core.

The ML pipeline imports this module directly, so importing it must not import a
model framework or connect to PostgreSQL.  :func:`durable_stage` is a no-op
unless an explicit PostgreSQL URL and a complete document/revision scope are in
the environment.  Once enabled, failures are deliberately fail-closed: every
stage is represented by idempotent work, a fenced renewable lease, append-only
events, retries, checkpoints, and immutable artifact receipts.
"""

from __future__ import annotations

import json

from collections.abc import Mapping
import hashlib
import mimetypes
import os
from pathlib import Path
import re
import socket
import stat
from threading import Event, Lock, Thread
import time
from types import TracebackType
from typing import Protocol, Self
import uuid

from docarag.domain.base import JsonObject, require_sha256, stable_id
from docarag.persistence.contracts import (
    ArtifactAvailability,
    ArtifactReceiptRecord,
    CorpusScope,
    DocumentScope,
    IndexPublication,
    IndexRevisionSpec,
    IndexRevisionState,
    StageEvent,
    StageEventStatus,
    WorkLease,
    WorkSpec,
    WorkState,
)


DATABASE_URL_ENV = "DOCARAG_DATABASE_URL"
TENANT_ID_ENV = "DOCARAG_TENANT_ID"
CORPUS_ID_ENV = "DOCARAG_CORPUS_ID"
DOCUMENT_ID_ENV = "DOCARAG_DOCUMENT_ID"
DOCUMENT_VERSION_ID_ENV = "DOCARAG_DOCUMENT_VERSION_ID"
INDEX_REVISION_ID_ENV = "DOCARAG_INDEX_REVISION_ID"
TRACE_ID_ENV = "DOCARAG_TRACE_ID"
SOURCE_SHA256_ENV = "DOCARAG_SOURCE_SHA256"
SYSTEM_CONFIG_SHA256_ENV = "DOCARAG_SYSTEM_CONFIG_SHA256"
WORKER_ID_ENV = "DOCARAG_WORKER_ID"
LEASE_SECONDS_ENV = "DOCARAG_STAGE_LEASE_SECONDS"
MAX_ATTEMPTS_ENV = "DOCARAG_STAGE_MAX_ATTEMPTS"
RETRY_DELAY_SECONDS_ENV = "DOCARAG_STAGE_RETRY_DELAY_SECONDS"

DEFAULT_LEASE_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 5.0
_STAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")


class StageRuntimeError(RuntimeError):
    """Base error for enabled durable stage execution."""


class StageRuntimeConfigurationError(StageRuntimeError):
    pass


class StageLeaseUnavailableError(StageRuntimeError):
    pass


class StageLeaseLostError(StageRuntimeError):
    pass


class StageLedger(Protocol):
    """The narrow PostgreSQL API used by the stage runtime and its test fakes."""

    def enqueue_work(self, spec: WorkSpec) -> str: ...

    def get_work_state(
        self, scope: DocumentScope, work_item_id: str
    ) -> WorkState | None: ...

    def lease_work(
        self,
        corpus_scope: CorpusScope,
        *,
        worker_id: str,
        lease_seconds: float = 60,
        kinds: tuple[str, ...] | None = None,
    ) -> WorkLease | None: ...

    def heartbeat_work(
        self, lease: WorkLease, *, extension_seconds: float = 60
    ) -> WorkLease: ...

    def complete_work(
        self, lease: WorkLease, *, result: Mapping[str, object] | None = None
    ) -> None: ...

    def fail_work(
        self,
        lease: WorkLease,
        *,
        error: Mapping[str, object],
        retry_delay_seconds: float = 0,
    ) -> WorkState: ...

    def append_stage_event(self, event: StageEvent) -> None: ...

    def record_artifact(self, receipt: ArtifactReceiptRecord) -> None: ...

    def create_index_revision(
        self, spec: IndexRevisionSpec
    ) -> IndexRevisionState: ...

    def publish_index_revision(
        self, spec: IndexRevisionSpec, publication: IndexPublication
    ) -> None: ...


def _number(
    environment: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise StageRuntimeConfigurationError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise StageRuntimeConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return value


def _integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise StageRuntimeConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise StageRuntimeConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _stable_file_receipt(path_value: str | os.PathLike[str]) -> tuple[Path, str, int]:
    """Hash one non-empty regular file and reject replacement during the read."""

    if not isinstance(path_value, (str, os.PathLike)):
        raise TypeError("artifact path must be a filesystem path")
    candidate = Path(os.fspath(path_value)).expanduser()
    try:
        path = candidate.resolve(strict=True)
        if candidate.is_symlink() or path.is_symlink():
            raise StageRuntimeError("artifact path may not be a symbolic link")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, StageRuntimeError):
            raise
        raise StageRuntimeError(f"artifact is unavailable: {candidate}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise StageRuntimeError("artifact must be a non-empty regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise StageRuntimeError(f"artifact could not be read: {path}") from exc
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or after_identity != path_identity
        or size != before.st_size
    ):
        raise StageRuntimeError("artifact changed while its receipt was created")
    return path, digest.hexdigest(), size


class NoopStage:
    """Drop-in context used by direct Core calls without stage-ledger env."""

    enabled = False
    replayed = False
    work_item_id: str | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def checkpoint(
        self, _name: str, _payload: Mapping[str, object] | None = None
    ) -> None:
        return None

    def record_artifact(self, _path: str | os.PathLike[str], **_kwargs: object) -> None:
        return None

    def record_not_emitted(self, **_kwargs: object) -> None:
        return None

    def publish_index_revision(self, **_kwargs: object) -> None:
        return None


class StageRuntime:
    """Process-local factory for document-scoped durable stages."""

    def __init__(
        self,
        *,
        ledger: StageLedger,
        scope: DocumentScope,
        index_revision_id: str,
        trace_id: str,
        worker_id: str,
        source_sha256: str | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        if not isinstance(scope, DocumentScope):
            raise TypeError("scope must be a DocumentScope")
        for value, field in (
            (index_revision_id, "index_revision_id"),
            (trace_id, "trace_id"),
            (worker_id, "worker_id"),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise StageRuntimeConfigurationError(
                    f"{field} must contain 1 to 512 characters"
                )
        if source_sha256 is not None:
            require_sha256(source_sha256, "source_sha256")
        if not 0.3 <= lease_seconds <= 86_400:
            raise StageRuntimeConfigurationError(
                "lease_seconds must be between 0.3 and 86400"
            )
        if not 1 <= max_attempts <= 32_767:
            raise StageRuntimeConfigurationError(
                "max_attempts must be between 1 and 32767"
            )
        if not 0 <= retry_delay_seconds <= 86_400:
            raise StageRuntimeConfigurationError(
                "retry_delay_seconds must be between 0 and 86400"
            )
        self.ledger = ledger
        self.scope = scope
        self.index_revision_id = index_revision_id
        self.trace_id = trace_id
        self.worker_id = worker_id
        self.source_sha256 = source_sha256
        self.lease_seconds = float(lease_seconds)
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        ledger: StageLedger | None = None,
    ) -> StageRuntime | None:
        """Build an enabled runtime only from an explicit complete stage scope."""

        env = os.environ if environment is None else environment
        if not env.get(DATABASE_URL_ENV):
            return None
        required = (
            TENANT_ID_ENV,
            CORPUS_ID_ENV,
            DOCUMENT_ID_ENV,
            DOCUMENT_VERSION_ID_ENV,
            INDEX_REVISION_ID_ENV,
        )
        missing = tuple(name for name in required if not env.get(name))
        if missing:
            raise StageRuntimeConfigurationError(
                "durable stage environment is incomplete: " + ", ".join(missing)
            )
        scope = DocumentScope(
            env[TENANT_ID_ENV],
            env[CORPUS_ID_ENV],
            env[DOCUMENT_ID_ENV],
            env[DOCUMENT_VERSION_ID_ENV],
        )
        revision = env[INDEX_REVISION_ID_ENV]
        trace_id = env.get(TRACE_ID_ENV) or stable_id(
            "index-trace",
            scope.tenant_id,
            scope.corpus_id,
            scope.document_id,
            scope.document_version_id,
            revision,
        )
        worker_id = env.get(WORKER_ID_ENV) or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        )
        source_sha256 = env.get(SOURCE_SHA256_ENV) or None
        if ledger is None:
            # psycopg remains a lazy optional import inside PostgresLedger.
            from docarag.persistence.postgres import PostgresLedger

            ledger = PostgresLedger(database_url=env[DATABASE_URL_ENV])
        return cls(
            ledger=ledger,
            scope=scope,
            index_revision_id=revision,
            trace_id=trace_id,
            worker_id=worker_id,
            source_sha256=source_sha256,
            lease_seconds=_number(
                env,
                LEASE_SECONDS_ENV,
                DEFAULT_LEASE_SECONDS,
                minimum=0.3,
                maximum=86_400,
            ),
            max_attempts=_integer(
                env,
                MAX_ATTEMPTS_ENV,
                DEFAULT_MAX_ATTEMPTS,
                minimum=1,
                maximum=32_767,
            ),
            retry_delay_seconds=_number(
                env,
                RETRY_DELAY_SECONDS_ENV,
                DEFAULT_RETRY_DELAY_SECONDS,
                minimum=0,
                maximum=86_400,
            ),
        )

    def stage(
        self,
        stage_name: str,
        *,
        payload: Mapping[str, object] | None = None,
        source_sha256: str | None = None,
    ) -> DurableStage:
        return DurableStage(
            self,
            stage_name,
            payload=payload,
            source_sha256=source_sha256,
        )


class DurableStage:
    """One leased stage attempt with renewable fencing and append-only trace."""

    enabled = True

    def __init__(
        self,
        runtime: StageRuntime,
        stage_name: str,
        *,
        payload: Mapping[str, object] | None,
        source_sha256: str | None,
    ) -> None:
        if not isinstance(stage_name, str) or _STAGE_NAME.fullmatch(stage_name) is None:
            raise StageRuntimeConfigurationError(
                "stage_name must match [A-Za-z][A-Za-z0-9_.-]{0,127}"
            )
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("stage payload must be a mapping")
        if source_sha256 is not None:
            require_sha256(source_sha256, "source_sha256")
        self.runtime = runtime
        self.stage_name = stage_name
        self.source_sha256 = source_sha256 or runtime.source_sha256
        self.stage_run_id = f"stage_{uuid.uuid4().hex}"
        kind_suffix = stable_id(
            "stage-kind", runtime.scope.document_version_id, runtime.index_revision_id, stage_name
        ).rsplit("_", 1)[-1]
        self.kind = f"index-stage.{stage_name}.{kind_suffix}"
        self.spec = WorkSpec(
            scope=runtime.scope,
            kind=self.kind,
            idempotency_key=stable_id(
                "index-stage-idempotency",
                runtime.scope.document_version_id,
                runtime.index_revision_id,
                stage_name,
                # The same stage against different inputs is different work: a
                # replay of identical work stays idempotent, and a legitimate
                # rebuild -- new seal, new configuration -- is admitted rather
                # than colliding with history.
                json.dumps(dict(payload or {}), sort_keys=True, default=str),
            ),

            payload={
                "index_revision_id": runtime.index_revision_id,
                "stage_name": stage_name,
                "inputs": dict(payload or {}),
            },
            max_attempts=runtime.max_attempts,
        )
        self.work_item_id = self.spec.work_item_id
        self.replayed = False
        self._lease: WorkLease | None = None
        self._lease_lock = Lock()
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None
        self._heartbeat_error: BaseException | None = None
        self._sequence = 0
        self._started_at = 0.0
        self._receipt_ids: list[str] = []
        self._checkpoint_count = 0

    def _event(
        self,
        event_type: str,
        status: StageEventStatus,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        event = StageEvent(
            scope=self.runtime.scope,
            trace_id=self.runtime.trace_id,
            stage_run_id=self.stage_run_id,
            sequence_no=self._sequence,
            stage_name=self.stage_name,
            event_type=event_type,
            status=status,
            payload=dict(payload or {}),
        )
        self.runtime.ledger.append_stage_event(event)
        self._sequence += 1

    def __enter__(self) -> Self:
        self._started_at = time.monotonic()
        enqueued_id = self.runtime.ledger.enqueue_work(self.spec)
        if enqueued_id != self.work_item_id:
            raise StageRuntimeError("ledger returned another work item identity")
        lease = self.runtime.ledger.lease_work(
            self.runtime.scope.corpus_scope,
            worker_id=self.runtime.worker_id,
            lease_seconds=self.runtime.lease_seconds,
            kinds=(self.kind,),
        )
        if lease is None:
            state = self.runtime.ledger.get_work_state(
                self.runtime.scope, self.work_item_id
            )
            if state is WorkState.SUCCEEDED:
                self.replayed = True
                self._event(
                    "stage.replayed",
                    StageEventStatus.SKIPPED,
                    {
                        "index_revision_id": self.runtime.index_revision_id,
                        "work_item_id": self.work_item_id,
                        "reason": "already_succeeded",
                    },
                )
                return self
            self._event(
                "stage.lease_unavailable",
                StageEventStatus.SKIPPED,
                {
                    "index_revision_id": self.runtime.index_revision_id,
                    "work_item_id": self.work_item_id,
                    "work_state": state.value if state is not None else "missing",
                },
            )
            raise StageLeaseUnavailableError(
                f"durable stage {self.stage_name!r} could not acquire its work lease "
                f"(state={state.value if state is not None else 'missing'})"
            )
        if lease.work_item_id != self.work_item_id or lease.kind != self.kind:
            raise StageRuntimeError("ledger leased another stage work item")
        with self._lease_lock:
            self._lease = lease
        try:
            self._event(
                "stage.started",
                StageEventStatus.STARTED,
                {
                    "attempt_count": lease.attempt_count,
                    "fencing_token": lease.fencing_token,
                    "index_revision_id": self.runtime.index_revision_id,
                    "work_item_id": lease.work_item_id,
                    "worker_id": lease.worker_id,
                },
            )
        except BaseException as exc:
            try:
                self.runtime.ledger.fail_work(
                    lease,
                    error=self._error_payload(exc),
                    retry_delay_seconds=self.runtime.retry_delay_seconds,
                )
            except BaseException as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "DocARAG could not release the stage lease after its "
                        f"start event failed: {type(cleanup_error).__name__}: "
                        f"{cleanup_error}"
                    )
            raise
        self._heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            name=f"docarag-{self.stage_name}-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        interval = max(0.1, self.runtime.lease_seconds / 3.0)
        while not self._heartbeat_stop.wait(interval):
            with self._lease_lock:
                lease = self._lease
            if lease is None:
                return
            try:
                renewed = self.runtime.ledger.heartbeat_work(
                    lease,
                    extension_seconds=self.runtime.lease_seconds,
                )
            except BaseException as exc:  # surfaced by __exit__ in the caller thread
                self._heartbeat_error = exc
                self._heartbeat_stop.set()
                return
            with self._lease_lock:
                self._lease = renewed

    def _stop_heartbeat(self) -> WorkLease | None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=min(5.0, self.runtime.lease_seconds))
            if thread.is_alive():
                raise StageLeaseLostError("stage heartbeat thread did not stop")
        with self._lease_lock:
            return self._lease

    @staticmethod
    def _error_payload(error: BaseException) -> dict[str, object]:
        message = str(error).replace("\x00", " ")[:2048]
        return {
            "error_type": type(error).__name__,
            "message": message,
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if self.replayed:
            return False
        lease = self._stop_heartbeat()
        heartbeat_error = self._heartbeat_error
        if exc is None and heartbeat_error is not None:
            exc = StageLeaseLostError(
                f"durable stage {self.stage_name!r} lost its heartbeat lease: "
                f"{type(heartbeat_error).__name__}"
            )
        if lease is None:
            raise StageLeaseLostError("durable stage has no active lease")
        elapsed = round(max(0.0, time.monotonic() - self._started_at), 6)
        if exc is None:
            self.runtime.ledger.complete_work(
                lease,
                result={
                    "checkpoint_count": self._checkpoint_count,
                    "elapsed_seconds": elapsed,
                    "index_revision_id": self.runtime.index_revision_id,
                    "receipt_ids": list(self._receipt_ids),
                },
            )
            self._event(
                "stage.succeeded",
                StageEventStatus.SUCCEEDED,
                {
                    "checkpoint_count": self._checkpoint_count,
                    "elapsed_seconds": elapsed,
                    "receipt_count": len(self._receipt_ids),
                    "work_item_id": self.work_item_id,
                },
            )
            return False

        error_payload = self._error_payload(exc)
        try:
            state = self.runtime.ledger.fail_work(
                lease,
                error=error_payload,
                retry_delay_seconds=self.runtime.retry_delay_seconds,
            )
            self._event(
                "stage.failed",
                StageEventStatus.FAILED,
                {
                    **error_payload,
                    "elapsed_seconds": elapsed,
                    "work_item_id": self.work_item_id,
                    "work_state": state.value,
                },
            )
        except BaseException as ledger_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "DocARAG could not persist the stage failure: "
                    f"{type(ledger_error).__name__}: {ledger_error}"
                )
        if heartbeat_error is not None and exc is not heartbeat_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "The durable stage heartbeat failed first: "
                    f"{type(heartbeat_error).__name__}: {heartbeat_error}"
                )
        if exc_type is None:
            raise exc
        return False

    def checkpoint(
        self,
        name: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(name, str) or _STAGE_NAME.fullmatch(name) is None:
            raise StageRuntimeConfigurationError(
                "checkpoint name must match [A-Za-z][A-Za-z0-9_.-]{0,127}"
            )
        self._checkpoint_count += 1
        self._event(
            "stage.checkpoint",
            StageEventStatus.PROGRESS,
            {**dict(payload or {}), "checkpoint": name},
        )

    def _artifact_source_sha256(self, value: str | None) -> str:
        source_sha256 = value or self.source_sha256
        require_sha256(source_sha256, "source_sha256")
        return source_sha256  # type: ignore[return-value]

    def record_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        artifact_role: str,
        artifact_schema: str,
        artifact_key: str = "primary",
        media_type: str | None = None,
        source_sha256: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        artifact_path, artifact_sha256, size_bytes = _stable_file_receipt(path)
        resolved_media_type = media_type or mimetypes.guess_type(artifact_path.name)[0]
        resolved_media_type = resolved_media_type or "application/octet-stream"
        stable_metadata: JsonObject = {
            **dict(metadata or {}),
            "index_revision_id": self.runtime.index_revision_id,
            "stage_name": self.stage_name,
        }
        receipt = ArtifactReceiptRecord(
            scope=self.runtime.scope,
            receipt_id=stable_id(
                "stage-artifact-receipt",
                self.work_item_id,
                artifact_role,
                artifact_key,
                artifact_sha256,
                size_bytes,
            ),
            artifact_role=artifact_role,
            availability=ArtifactAvailability.EMITTED,
            artifact_schema=artifact_schema,
            producer_task_id=self.work_item_id,
            source_sha256=self._artifact_source_sha256(source_sha256),
            artifact_key=artifact_key,
            uri=artifact_path.as_uri(),
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            media_type=resolved_media_type,
            metadata=stable_metadata,
        )
        self.runtime.ledger.record_artifact(receipt)
        self._receipt_ids.append(receipt.receipt_id)
        self.checkpoint(
            "artifact.recorded",
            {
                "artifact_key": artifact_key,
                "artifact_role": artifact_role,
                "receipt_id": receipt.receipt_id,
                "sha256": artifact_sha256,
                "size_bytes": size_bytes,
            },
        )

    def record_not_emitted(
        self,
        *,
        artifact_role: str,
        artifact_schema: str,
        artifact_key: str = "primary",
        media_type: str = "application/octet-stream",
        source_sha256: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        stable_metadata: JsonObject = {
            **dict(metadata or {}),
            "index_revision_id": self.runtime.index_revision_id,
            "stage_name": self.stage_name,
        }
        receipt = ArtifactReceiptRecord(
            scope=self.runtime.scope,
            receipt_id=stable_id(
                "stage-artifact-not-emitted",
                self.work_item_id,
                artifact_role,
                artifact_key,
            ),
            artifact_role=artifact_role,
            availability=ArtifactAvailability.NOT_EMITTED,
            artifact_schema=artifact_schema,
            producer_task_id=self.work_item_id,
            source_sha256=self._artifact_source_sha256(source_sha256),
            artifact_key=artifact_key,
            media_type=media_type,
            metadata=stable_metadata,
        )
        self.runtime.ledger.record_artifact(receipt)
        self._receipt_ids.append(receipt.receipt_id)
        self.checkpoint(
            "artifact.not_emitted",
            {
                "artifact_key": artifact_key,
                "artifact_role": artifact_role,
                "receipt_id": receipt.receipt_id,
            },
        )

    def publish_index_revision(
        self,
        *,
        configuration_sha256: str | None,
        canonical_artifact_sha256: str | None,
        tree_artifact_path: str | os.PathLike[str],
        graph_artifact_path: str | os.PathLike[str],
        retrieval_manifest_path: str | os.PathLike[str],
        qdrant_collection: str,
        qdrant_point_count: int,
        qdrant_vector_name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Publish the final relational index revision after verified vectors.

        The relational revision has the contract-derived identity; the shell's
        independently derived revision remains an explicit immutable metadata
        binding rather than being forced into a mismatched identity field.
        """

        configuration_digest = require_sha256(
            configuration_sha256, "configuration_sha256"
        )
        canonical_digest = require_sha256(
            canonical_artifact_sha256,
            "canonical_artifact_sha256",
        )
        if (
            isinstance(qdrant_point_count, bool)
            or not isinstance(qdrant_point_count, int)
            or qdrant_point_count < 0
        ):
            raise StageRuntimeConfigurationError(
                "qdrant_point_count must be a non-negative integer"
            )
        _tree_path, tree_sha256, _tree_size = _stable_file_receipt(
            tree_artifact_path
        )
        _graph_path, graph_sha256, _graph_size = _stable_file_receipt(
            graph_artifact_path
        )
        _manifest_path, manifest_sha256, _manifest_size = _stable_file_receipt(
            retrieval_manifest_path
        )
        spec = IndexRevisionSpec(
            scope=self.runtime.scope,
            configuration_sha256=configuration_digest,
            canonical_artifact_sha256=canonical_digest,
            qdrant_collection=qdrant_collection,
            qdrant_vector_name=qdrant_vector_name,
            # This pipeline already names its revision (self.runtime.index_revision_id,
            # the same identity every other stage's work_item_id/artifact receipt
            # is keyed by) -- IndexRevisionSpec's own contract is to keep that
            # name rather than derive a different one. Leaving revision_id
            # unset here previously left create_index_revision to derive its
            # own content-hash identity instead, so the row this call marks
            # 'ready' was never the same row persist_document_evidence wrote
            # blocks/entities/statements into: two disconnected revisions for
            # one document, one with data and never ready, one ready and
            # empty.
            revision_id=self.runtime.index_revision_id,
            # No metadata here, ever: with an explicit revision_id,
            # create_index_revision's ON CONFLICT DO NOTHING + verify-match
            # check compares metadata byte-for-byte against whatever row
            # already exists -- and the ledger's own admission path
            # (evidence_persist.py's _admit_revision) always creates this
            # same row first, with empty metadata, well before any stage
            # reaches this call. Passing anything caller-supplied here would
            # make this call permanently raise ImmutableRecordConflict
            # instead of publishing. Publish-time details belong in the
            # checkpoint below, an append-only event log with no
            # identity-matching contract, not in this identity-bearing spec.
        )
        state = self.runtime.ledger.create_index_revision(spec)
        if state is IndexRevisionState.FAILED:
            raise StageRuntimeError(
                "the relational index revision is already terminally failed"
            )
        publication = IndexPublication(
            tree_artifact_sha256=tree_sha256,
            graph_artifact_sha256=graph_sha256,
            retrieval_manifest_sha256=manifest_sha256,
            qdrant_point_count=qdrant_point_count,
        )
        self.runtime.ledger.publish_index_revision(spec, publication)
        self.checkpoint(
            "index-revision.published",
            {
                **dict(metadata or {}),
                "qdrant_point_count": qdrant_point_count,
                "relational_revision_id": spec.revision_id,
                "retrieval_manifest_sha256": manifest_sha256,
            },
        )


def durable_stage(
    stage_name: str,
    *,
    payload: Mapping[str, object] | None = None,
    source_sha256: str | None = None,
    environment: Mapping[str, str] | None = None,
    ledger: StageLedger | None = None,
) -> DurableStage | NoopStage:
    """Return an enabled durable stage or a safe no-op for direct Core use."""

    runtime = StageRuntime.from_environment(environment, ledger=ledger)
    if runtime is None:
        return NoopStage()
    return runtime.stage(
        stage_name,
        payload=payload,
        source_sha256=source_sha256,
    )


__all__ = [
    "DATABASE_URL_ENV",
    "DurableStage",
    "INDEX_REVISION_ID_ENV",
    "NoopStage",
    "SOURCE_SHA256_ENV",
    "SYSTEM_CONFIG_SHA256_ENV",
    "StageLeaseLostError",
    "StageLeaseUnavailableError",
    "StageRuntime",
    "StageRuntimeConfigurationError",
    "StageRuntimeError",
    "durable_stage",
]
