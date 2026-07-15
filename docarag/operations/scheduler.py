"""Bounded multi-worker execution over PostgreSQL fencing leases."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from threading import Event, Thread
import time
from typing import Protocol

from docarag.persistence import CorpusScope, LeaseLostError, WorkLease, WorkState


class LeaseLedger(Protocol):
    def lease_work(
        self,
        corpus_scope: CorpusScope,
        *,
        worker_id: str,
        lease_seconds: float,
        kinds: Sequence[str] | None = None,
    ) -> WorkLease | None: ...

    def heartbeat_work(self, lease: WorkLease, *, extension_seconds: float) -> WorkLease: ...
    def complete_work(self, lease: WorkLease, *, result: Mapping[str, object] | None = None) -> None: ...
    def fail_work(
        self,
        lease: WorkLease,
        *,
        error: Mapping[str, object],
        retry_delay_seconds: float = 0,
    ) -> WorkState: ...


class DurableWorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkOutcome:
    claimed: bool
    work_item_id: str | None = None
    kind: str | None = None
    state: WorkState | None = None
    attempt_count: int | None = None


class DurableLeaseWorker:
    """Claim one fenced job at a time and heartbeat it during execution.

    Horizontal scaling is achieved by starting multiple workers with distinct
    IDs against the same PostgreSQL corpus queue.  ``FOR UPDATE SKIP LOCKED``
    and fencing remain in :class:`PostgresLedger`; this class adds bounded
    handler dispatch, heartbeat ownership, and deterministic failure receipts.
    """

    def __init__(
        self,
        ledger: LeaseLedger,
        corpus_scope: CorpusScope,
        *,
        worker_id: str,
        handlers: Mapping[str, Callable[[WorkLease], Mapping[str, object] | None]],
        lease_seconds: float = 120.0,
        retry_delay_seconds: float = 5.0,
    ) -> None:
        if not worker_id.strip() or len(worker_id) > 512:
            raise ValueError("worker_id must contain 1 to 512 characters")
        if not handlers or any(not key.strip() for key in handlers):
            raise ValueError("at least one non-empty work handler is required")
        if not 3 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 3 and 86400")
        if not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("retry_delay_seconds is invalid")
        self.ledger = ledger
        self.corpus_scope = corpus_scope
        self.worker_id = worker_id
        self.handlers = dict(handlers)
        self.lease_seconds = float(lease_seconds)
        self.retry_delay_seconds = float(retry_delay_seconds)

    def run_once(self) -> WorkOutcome:
        lease = self.ledger.lease_work(
            self.corpus_scope,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            kinds=tuple(sorted(self.handlers)),
        )
        if lease is None:
            return WorkOutcome(claimed=False)
        handler = self.handlers.get(lease.kind)
        if handler is None:  # Defensive; the lease query was capability-filtered.
            state = self.ledger.fail_work(
                lease,
                error={"code": "unsupported_work_kind", "retryable": False},
                retry_delay_seconds=0,
            )
            return WorkOutcome(True, lease.work_item_id, lease.kind, state, lease.attempt_count)

        stop = Event()
        lost = Event()
        current = [lease]

        def heartbeat() -> None:
            interval = max(1.0, self.lease_seconds / 3.0)
            while not stop.wait(interval):
                try:
                    current[0] = self.ledger.heartbeat_work(
                        current[0], extension_seconds=self.lease_seconds
                    )
                except Exception:
                    lost.set()
                    return

        thread = Thread(target=heartbeat, name=f"docarag-lease-{lease.work_item_id}", daemon=True)
        thread.start()
        try:
            result = handler(lease)
            if result is not None and not isinstance(result, Mapping):
                raise TypeError("work handler result must be a mapping or None")
        except BaseException as exc:
            stop.set()
            thread.join(timeout=min(5.0, self.lease_seconds))
            if lost.is_set():
                raise DurableWorkerError("work lease was lost during failed execution") from exc
            error = _safe_error(exc)
            state = self.ledger.fail_work(
                current[0],
                error=error,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            return WorkOutcome(True, lease.work_item_id, lease.kind, state, lease.attempt_count)
        stop.set()
        thread.join(timeout=min(5.0, self.lease_seconds))
        if lost.is_set():
            raise DurableWorkerError("work lease was lost; result was not committed")
        self.ledger.complete_work(current[0], result=result or {})
        return WorkOutcome(
            True,
            lease.work_item_id,
            lease.kind,
            WorkState.SUCCEEDED,
            lease.attempt_count,
        )

    def run_until_idle(self, *, max_items: int = 1000) -> tuple[WorkOutcome, ...]:
        if not 1 <= max_items <= 1_000_000:
            raise ValueError("max_items must be between 1 and 1000000")
        outcomes: list[WorkOutcome] = []
        for _ in range(max_items):
            outcome = self.run_once()
            if not outcome.claimed:
                break
            outcomes.append(outcome)
        return tuple(outcomes)


def _safe_error(exc: BaseException) -> dict[str, object]:
    kind = type(exc).__name__[:256]
    # Hash the raw diagnostic for correlation without putting source text,
    # prompts, paths, or credentials into durable queue state.
    digest = hashlib.sha256(str(exc).encode("utf-8", errors="replace")).hexdigest()
    return {
        "code": "handler_failed",
        "exception_type": kind,
        "diagnostic_sha256": digest,
        "retryable": not isinstance(exc, (TypeError, ValueError)),
    }


__all__ = [
    "DurableLeaseWorker",
    "DurableWorkerError",
    "LeaseLedger",
    "WorkOutcome",
]
