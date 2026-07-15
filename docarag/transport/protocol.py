"""Versioned JSON protocol shared by parser-worker clients and servers.

The transport version is intentionally independent of the domain contract
version.  A transport implementation can evolve without silently accepting a
different ``ParseRequest`` or ``ParseResult`` schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from docarag.domain.base import ContractValidationError, require_non_empty


PARSER_WORKER_PROTOCOL_VERSION = "docarag.parser-worker-http.v1"
HEALTH_PATH = "/v1/health"
CAPABILITIES_PATH = "/v1/capabilities"
PARSE_PATH = "/v1/parse"


class WorkerState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    CLOSED = "closed"


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field} must be a boolean")
    return value


def require_protocol_version(payload: Mapping[str, object]) -> None:
    actual = payload.get("protocol_version")
    if actual != PARSER_WORKER_PROTOCOL_VERSION:
        raise ContractValidationError(
            "unsupported parser-worker protocol_version "
            f"{actual!r}; expected {PARSER_WORKER_PROTOCOL_VERSION!r}"
        )


def reject_unknown_fields(
    payload: Mapping[str, object], allowed: frozenset[str], *, envelope: str
) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ContractValidationError(
            f"{envelope} contains unknown fields: {', '.join(sorted(unknown))}"
        )


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    """Bounded health snapshot; it never contains vendor exception details."""

    worker_instance_id: str
    state: WorkerState
    accepting_requests: bool
    in_flight: int
    max_in_flight: int
    adapter_name: str
    adapter_version: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        require_non_empty(self.worker_instance_id, "worker_instance_id")
        try:
            state = WorkerState(self.state)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("health.state is invalid") from exc
        _require_bool(self.accepting_requests, "health.accepting_requests")
        _require_int(self.in_flight, "health.in_flight")
        _require_int(self.max_in_flight, "health.max_in_flight", minimum=1)
        if self.in_flight > self.max_in_flight:
            raise ContractValidationError("health.in_flight exceeds max_in_flight")
        require_non_empty(self.adapter_name, "health.adapter_name")
        require_non_empty(self.adapter_version, "health.adapter_version")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ContractValidationError(
                "health.manifest_sha256 must be a lowercase SHA-256 digest"
            )
        if self.accepting_requests != (state == WorkerState.READY):
            raise ContractValidationError(
                "health.accepting_requests must be true exactly when state is ready"
            )
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PARSER_WORKER_PROTOCOL_VERSION,
            "worker_instance_id": self.worker_instance_id,
            "state": self.state.value,
            "accepting_requests": self.accepting_requests,
            "in_flight": self.in_flight,
            "max_in_flight": self.max_in_flight,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WorkerHealth":
        require_protocol_version(payload)
        reject_unknown_fields(
            payload,
            frozenset(
                {
                    "protocol_version",
                    "worker_instance_id",
                    "state",
                    "accepting_requests",
                    "in_flight",
                    "max_in_flight",
                    "adapter_name",
                    "adapter_version",
                    "manifest_sha256",
                }
            ),
            envelope="health response",
        )
        return cls(
            worker_instance_id=payload.get("worker_instance_id"),  # type: ignore[arg-type]
            state=payload.get("state"),  # type: ignore[arg-type]
            accepting_requests=payload.get("accepting_requests"),  # type: ignore[arg-type]
            in_flight=payload.get("in_flight"),  # type: ignore[arg-type]
            max_in_flight=payload.get("max_in_flight"),  # type: ignore[arg-type]
            adapter_name=payload.get("adapter_name"),  # type: ignore[arg-type]
            adapter_version=payload.get("adapter_version"),  # type: ignore[arg-type]
            manifest_sha256=payload.get("manifest_sha256"),  # type: ignore[arg-type]
        )


EMPTY_HEADERS: Mapping[str, str] = MappingProxyType({})

