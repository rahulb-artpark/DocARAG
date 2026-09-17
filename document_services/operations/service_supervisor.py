"""Owned-service lifecycle with readiness and GPU-capacity admission.

Only resources declared ``owned`` may be spawned or signalled.  Borrowed and
shared resources are health-checked but are never terminated.  Commands are
argument vectors, secrets are inherited only through named environment
variables, and PID reuse is rejected using Linux process start ticks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from types import MappingProxyType
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from document_services.domain.base import canonical_json_bytes
from document_services.operations.gpu_admission import (
    GpuAdmissionController,
    GpuAdmissionError,
    process_start_ticks,
)


SERVICE_REGISTRY_SCHEMA_VERSION = "docarag.service-registry.v1"
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MUTABLE_RELEASE = re.compile(r"(?:^|[:@/])(latest|main|master|head|nightly)(?:$|[-:/])", re.I)
_SECRET_FLAG = re.compile(r"(?:api[-_]?key|password|secret|token)", re.I)


class ServiceSupervisorError(RuntimeError):
    pass


class ServiceOwnership(StrEnum):
    OWNED = "owned"
    BORROWED = "borrowed"
    SHARED = "shared"


class ServiceState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    EXTERNAL_READY = "external_ready"
    EXTERNAL_UNHEALTHY = "external_unhealthy"


def _health_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("health_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("health_url must not contain credentials or a fragment")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("service supervisor health checks must use loopback")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    name: str
    ownership: ServiceOwnership
    runtime_identity: str
    health_url: str
    argv: tuple[str, ...] = ()
    working_directory: Path | None = None
    environment_from: Mapping[str, str] = field(default_factory=dict, repr=False)
    readiness_timeout_seconds: float = 120.0
    shutdown_timeout_seconds: float = 30.0
    gpu_memory_mib: int = 0
    gpu_index: int | None = None
    exclusive_gpu_family: bool = True
    health_bearer_token_env: str | None = field(default=None, repr=False)
    spec_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("service name must be a bounded lowercase identifier")
        try:
            ownership = ServiceOwnership(self.ownership)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid service ownership") from exc
        if not self.runtime_identity.strip() or len(self.runtime_identity) > 512:
            raise ValueError("runtime_identity must be a bounded immutable release")
        if _MUTABLE_RELEASE.search(self.runtime_identity):
            raise ValueError("runtime_identity must not use a mutable release label")
        url = _health_url(self.health_url)
        argv = tuple(self.argv)
        if ownership is ServiceOwnership.OWNED:
            if not argv or not Path(argv[0]).is_absolute():
                raise ValueError("owned service argv[0] must be an absolute executable")
            for index, argument in enumerate(argv):
                if not isinstance(argument, str) or not argument or "\x00" in argument:
                    raise ValueError("service argv contains an invalid argument")
                if _SECRET_FLAG.search(argument) and (
                    "=" in argument or (index and _SECRET_FLAG.search(argv[index - 1]))
                ):
                    raise ValueError("literal secrets are forbidden in service argv")
        elif argv:
            raise ValueError("borrowed/shared services must not declare spawn argv")
        root = self.working_directory
        if root is not None:
            root = root.expanduser().resolve()
            if not root.is_absolute():
                raise ValueError("working_directory must be absolute")
        env = dict(self.environment_from)
        if any(_ENV.fullmatch(key) is None or _ENV.fullmatch(value) is None for key, value in env.items()):
            raise ValueError("environment_from maps child names to environment-variable names")
        if self.health_bearer_token_env is not None and _ENV.fullmatch(self.health_bearer_token_env) is None:
            raise ValueError("health_bearer_token_env must name an environment variable")
        if not 0.1 <= self.readiness_timeout_seconds <= 3600:
            raise ValueError("readiness_timeout_seconds is out of range")
        if not 0.1 <= self.shutdown_timeout_seconds <= 300:
            raise ValueError("shutdown_timeout_seconds is out of range")
        if isinstance(self.gpu_memory_mib, bool) or not 0 <= self.gpu_memory_mib <= 1024 * 1024:
            raise ValueError("gpu_memory_mib is invalid")
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError("gpu_index is invalid")
        if not isinstance(self.exclusive_gpu_family, bool):
            raise ValueError("exclusive_gpu_family must be boolean")
        safe = {
            "name": self.name,
            "ownership": ownership.value,
            "runtime_identity": self.runtime_identity,
            "health_url": url,
            "argv": list(argv),
            "working_directory": None if root is None else str(root),
            "environment_from": env,
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "gpu_memory_mib": self.gpu_memory_mib,
            "gpu_index": self.gpu_index,
            "exclusive_gpu_family": self.exclusive_gpu_family,
            "health_bearer_token_env": self.health_bearer_token_env,
        }
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(self, "health_url", url)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "working_directory", root)
        object.__setattr__(self, "environment_from", MappingProxyType(env))
        object.__setattr__(self, "spec_sha256", hashlib.sha256(canonical_json_bytes(safe)).hexdigest())


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    name: str
    ownership: ServiceOwnership
    state: ServiceState
    runtime_identity: str
    spec_sha256: str
    pid: int | None = None
    gpu_lease_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ownership": self.ownership.value,
            "state": self.state.value,
            "runtime_identity": self.runtime_identity,
            "spec_sha256": self.spec_sha256,
            "pid": self.pid,
            "gpu_lease_id": self.gpu_lease_id,
            "detail": self.detail,
        }


def _default_health_probe(url: str, bearer_token: str | None, timeout: float) -> bool:
    headers = {"Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


class ManagedServiceSupervisor:
    """Start, stop, and inspect an explicit set of service specifications."""

    def __init__(
        self,
        state_root: str | os.PathLike[str],
        *,
        gpu_admission: GpuAdmissionController | None = None,
        environment: Mapping[str, str] | None = None,
        health_probe: Callable[[str, str | None, float], bool] = _default_health_probe,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        process_identity: Callable[[int], int | None] = process_start_ticks,
    ) -> None:
        root = Path(state_root).expanduser()
        if not root.is_absolute() or root == Path("/"):
            raise ValueError("service state_root must be an absolute non-root path")
        self.state_root = root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.logs_root = self.state_root / "logs"
        self.logs_root.mkdir(exist_ok=True, mode=0o700)
        self.registry_path = self.state_root / "services.json"
        self.lock_path = self.state_root / "services.lock"
        self.gpu_admission = gpu_admission
        self.environment = dict(os.environ if environment is None else environment)
        self.health_probe = health_probe
        self.popen = popen
        self.clock = clock
        self.sleeper = sleeper
        self.process_identity = process_identity

    def start(self, spec: ServiceSpec) -> ServiceStatus:
        with self._operation_lock():
            return self._start_locked(spec)

    def _start_locked(self, spec: ServiceSpec) -> ServiceStatus:
        if spec.ownership is not ServiceOwnership.OWNED:
            return self._external_status(spec)
        records = self._read_registry()
        existing = records.get(spec.name)
        if existing is not None and self._record_process_alive(existing):
            if not hmac.compare_digest(str(existing.get("spec_sha256", "")), spec.spec_sha256):
                raise ServiceSupervisorError(
                    f"owned service {spec.name} is running with another immutable spec"
                )
            healthy = self._probe(spec)
            return self._status_from_record(
                spec, existing, ServiceState.READY if healthy else ServiceState.UNHEALTHY
            )
        if existing is not None:
            self._release_record_gpu(existing)
            records.pop(spec.name, None)
            self._write_registry(records)

        executable = Path(spec.argv[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ServiceSupervisorError(
                f"owned service executable is missing or not executable: {executable}"
            )
        if spec.working_directory is not None and not spec.working_directory.is_dir():
            raise ServiceSupervisorError("owned service working_directory is unavailable")

        reservation = None
        if spec.gpu_memory_mib:
            if self.gpu_admission is None:
                raise ServiceSupervisorError("GPU service requires configured GPU admission")
            reservation = self.gpu_admission.reserve(
                owner=spec.name,
                memory_mib=spec.gpu_memory_mib,
                gpu_index=spec.gpu_index,
                exclusive_family=spec.exclusive_gpu_family,
            )
        child_environment = dict(self.environment)
        for child_name, source_name in spec.environment_from.items():
            if source_name not in self.environment or not self.environment[source_name]:
                if reservation is not None:
                    self.gpu_admission.release(reservation.lease_id)
                raise ServiceSupervisorError(
                    f"required environment variable {source_name} is not configured"
                )
            child_environment[child_name] = self.environment[source_name]
        if reservation is not None:
            child_environment["CUDA_VISIBLE_DEVICES"] = str(reservation.gpu_index)

        log_path = self.logs_root / f"{spec.name}.log"
        try:
            log = log_path.open("ab", buffering=0)
            process = self.popen(
                list(spec.argv),
                cwd=None if spec.working_directory is None else str(spec.working_directory),
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException:
            if reservation is not None:
                self.gpu_admission.release(reservation.lease_id)
            raise
        finally:
            if "log" in locals():
                log.close()
        start_ticks = self.process_identity(process.pid)
        if start_ticks is None:
            if reservation is not None:
                self.gpu_admission.release(reservation.lease_id)
            raise ServiceSupervisorError("spawned service exited before identity binding")
        bound = None
        if reservation is not None:
            try:
                bound = self.gpu_admission.bind(reservation.lease_id, pid=process.pid)
            except BaseException:
                self._terminate_exact(process.pid, start_ticks, spec.shutdown_timeout_seconds)
                self.gpu_admission.release(reservation.lease_id)
                raise
        record: dict[str, object] = {
            "name": spec.name,
            "runtime_identity": spec.runtime_identity,
            "spec_sha256": spec.spec_sha256,
            "pid": process.pid,
            "process_start_ticks": start_ticks,
            "gpu_lease_id": None if bound is None else bound.lease_id,
            "started_unix": time.time(),
        }
        records[spec.name] = record
        self._write_registry(records)
        deadline = self.clock() + spec.readiness_timeout_seconds
        while self.clock() < deadline:
            if self.process_identity(process.pid) != start_ticks:
                self._remove_record(spec.name, record)
                raise ServiceSupervisorError(f"owned service {spec.name} exited before ready")
            if self._probe(spec):
                return self._status_from_record(spec, record, ServiceState.READY)
            self.sleeper(min(0.25, max(0.01, deadline - self.clock())))
        self._terminate_exact(process.pid, start_ticks, spec.shutdown_timeout_seconds)
        self._remove_record(spec.name, record)
        raise ServiceSupervisorError(f"owned service {spec.name} did not become ready")

    def stop(self, spec: ServiceSpec) -> ServiceStatus:
        with self._operation_lock():
            return self._stop_locked(spec)

    def _stop_locked(self, spec: ServiceSpec) -> ServiceStatus:
        if spec.ownership is not ServiceOwnership.OWNED:
            # This branch is an important safety property: even a healthy
            # borrowed/shared endpoint is never signalled.
            return self._external_status(spec)
        records = self._read_registry()
        record = records.get(spec.name)
        if record is None:
            return ServiceStatus(
                spec.name,
                spec.ownership,
                ServiceState.STOPPED,
                spec.runtime_identity,
                spec.spec_sha256,
            )
        if not hmac.compare_digest(str(record.get("spec_sha256", "")), spec.spec_sha256):
            raise ServiceSupervisorError(
                f"refusing to stop {spec.name}: active immutable spec differs"
            )
        pid = int(record["pid"])
        start_ticks = int(record["process_start_ticks"])
        if self.process_identity(pid) == start_ticks:
            self._terminate_exact(pid, start_ticks, spec.shutdown_timeout_seconds)
        self._remove_record(spec.name, record)
        return ServiceStatus(
            spec.name,
            spec.ownership,
            ServiceState.STOPPED,
            spec.runtime_identity,
            spec.spec_sha256,
        )

    def status(self, spec: ServiceSpec) -> ServiceStatus:
        with self._operation_lock():
            return self._status_locked(spec)

    def _status_locked(self, spec: ServiceSpec) -> ServiceStatus:
        if spec.ownership is not ServiceOwnership.OWNED:
            return self._external_status(spec)
        record = self._read_registry().get(spec.name)
        if record is None or not self._record_process_alive(record):
            if record is not None:
                self._remove_record(spec.name, record)
            return ServiceStatus(
                spec.name,
                spec.ownership,
                ServiceState.STOPPED,
                spec.runtime_identity,
                spec.spec_sha256,
            )
        state = ServiceState.READY if self._probe(spec) else ServiceState.UNHEALTHY
        return self._status_from_record(spec, record, state)

    def _probe(self, spec: ServiceSpec) -> bool:
        token = None
        if spec.health_bearer_token_env is not None:
            token = self.environment.get(spec.health_bearer_token_env)
            if not token:
                return False
        return self.health_probe(spec.health_url, token, 2.0)

    def _external_status(self, spec: ServiceSpec) -> ServiceStatus:
        ready = self._probe(spec)
        return ServiceStatus(
            spec.name,
            spec.ownership,
            ServiceState.EXTERNAL_READY if ready else ServiceState.EXTERNAL_UNHEALTHY,
            spec.runtime_identity,
            spec.spec_sha256,
            detail="lifecycle is operator-owned; no signal was sent",
        )

    def _status_from_record(
        self, spec: ServiceSpec, record: Mapping[str, object], state: ServiceState
    ) -> ServiceStatus:
        return ServiceStatus(
            spec.name,
            spec.ownership,
            state,
            spec.runtime_identity,
            spec.spec_sha256,
            pid=int(record["pid"]),
            gpu_lease_id=(
                None
                if record.get("gpu_lease_id") is None
                else str(record["gpu_lease_id"])
            ),
        )

    def _record_process_alive(self, record: Mapping[str, object]) -> bool:
        try:
            return self.process_identity(int(record["pid"])) == int(
                record["process_start_ticks"]
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _terminate_exact(self, pid: int, start_ticks: int, timeout: float) -> None:
        if self.process_identity(pid) != start_ticks:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            if self.process_identity(pid) != start_ticks:
                return
            self.sleeper(min(0.1, max(0.01, deadline - self.clock())))
        if self.process_identity(pid) == start_ticks:
            os.killpg(pid, signal.SIGKILL)

    def _remove_record(self, name: str, expected: Mapping[str, object]) -> None:
        records = self._read_registry()
        current = records.get(name)
        if current is not None and current.get("pid") == expected.get("pid"):
            self._release_record_gpu(current)
            records.pop(name, None)
            self._write_registry(records)

    def _release_record_gpu(self, record: Mapping[str, object]) -> None:
        lease_id = record.get("gpu_lease_id")
        if lease_id is not None and self.gpu_admission is not None:
            try:
                self.gpu_admission.release(
                    str(lease_id), expected_pid=int(record["pid"])
                )
            except GpuAdmissionError:
                # A mismatched lease is not ours to release.  Surface state via
                # GPU status without risking another service's reservation.
                pass

    def _read_registry(self) -> dict[str, dict[str, object]]:
        if not self.registry_path.exists():
            return {}
        try:
            if self.registry_path.stat().st_size > 1024 * 1024:
                raise ServiceSupervisorError("service registry is oversized")
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SERVICE_REGISTRY_SCHEMA_VERSION:
                raise ServiceSupervisorError("unsupported service registry schema")
            records = payload.get("services")
            if not isinstance(records, dict):
                raise ServiceSupervisorError("service registry is malformed")
            return {str(key): dict(value) for key, value in records.items()}
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ServiceSupervisorError("service registry is unreadable") from exc

    def _write_registry(self, records: Mapping[str, Mapping[str, object]]) -> None:
        payload = {
            "schema_version": SERVICE_REGISTRY_SCHEMA_VERSION,
            "services": {key: dict(records[key]) for key in sorted(records)},
        }
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.registry_path)

    class _OperationLock:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.handle = None

        def __enter__(self) -> None:
            self.handle = self.path.open("a+b")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

        def __exit__(self, exc_type, exc, traceback) -> None:
            assert self.handle is not None
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _operation_lock(self) -> "ManagedServiceSupervisor._OperationLock":
        return self._OperationLock(self.lock_path)


__all__ = [
    "ManagedServiceSupervisor",
    "SERVICE_REGISTRY_SCHEMA_VERSION",
    "ServiceOwnership",
    "ServiceSpec",
    "ServiceState",
    "ServiceStatus",
    "ServiceSupervisorError",
]
