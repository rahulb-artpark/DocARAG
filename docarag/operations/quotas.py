"""Atomic tenant quota contracts with a safe single-process implementation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol


class QuotaExceededError(RuntimeError):
    def __init__(self, dimension: str) -> None:
        self.dimension = dimension
        super().__init__(f"tenant quota exceeded: {dimension}")


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    max_documents: int
    max_source_bytes: int
    max_artifact_bytes: int
    max_active_jobs: int
    max_requests_per_window: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class QuotaDelta:
    documents: int = 0
    source_bytes: int = 0
    artifact_bytes: int = 0
    active_jobs: int = 0
    requests: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            for name in self.__dataclass_fields__
        ):
            raise ValueError("quota deltas must be integers")

    def plus(self, other: "QuotaDelta") -> "QuotaDelta":
        return QuotaDelta(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


class QuotaStore(Protocol):
    def reserve(self, tenant_id: str, delta: QuotaDelta, limits: QuotaLimits) -> QuotaDelta: ...
    def release(self, tenant_id: str, delta: QuotaDelta) -> QuotaDelta: ...


class InMemoryQuotaStore:
    """Thread-safe local-pilot store; not a multi-host coordination authority.

    A production API must implement the same port using one PostgreSQL
    transaction and row lock.  Keeping that distinction explicit prevents a
    process-local counter from being presented as a distributed quota.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._usage: dict[str, QuotaDelta] = {}

    def reserve(self, tenant_id: str, delta: QuotaDelta, limits: QuotaLimits) -> QuotaDelta:
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        if any(getattr(delta, name) < 0 for name in delta.__dataclass_fields__):
            raise ValueError("quota reservation cannot contain negative values")
        with self._lock:
            candidate = self._usage.get(tenant_id, QuotaDelta()).plus(delta)
            checks = {
                "documents": limits.max_documents,
                "source_bytes": limits.max_source_bytes,
                "artifact_bytes": limits.max_artifact_bytes,
                "active_jobs": limits.max_active_jobs,
                "requests": limits.max_requests_per_window,
            }
            for name, limit in checks.items():
                if getattr(candidate, name) > limit:
                    raise QuotaExceededError(name)
            self._usage[tenant_id] = candidate
            return candidate

    def release(self, tenant_id: str, delta: QuotaDelta) -> QuotaDelta:
        if any(getattr(delta, name) < 0 for name in delta.__dataclass_fields__):
            raise ValueError("quota release cannot contain negative values")
        with self._lock:
            current = self._usage.get(tenant_id, QuotaDelta())
            values = {
                name: getattr(current, name) - getattr(delta, name)
                for name in delta.__dataclass_fields__
            }
            if any(value < 0 for value in values.values()):
                raise ValueError("quota release exceeds current usage")
            updated = QuotaDelta(**values)
            self._usage[tenant_id] = updated
            return updated

    def usage(self, tenant_id: str) -> QuotaDelta:
        with self._lock:
            return self._usage.get(tenant_id, QuotaDelta())


__all__ = [
    "InMemoryQuotaStore",
    "QuotaDelta",
    "QuotaExceededError",
    "QuotaLimits",
    "QuotaStore",
]
