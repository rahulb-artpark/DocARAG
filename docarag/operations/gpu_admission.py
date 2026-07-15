"""Cross-process, capacity-aware NVIDIA GPU admission.

The lease registry is coordination state, while ``nvidia-smi`` remains the
authority for physical free memory.  Admission requires *both* enough observed
free memory and enough configured allocation budget.  Stale leases are removed
only after the recorded process identity no longer matches, which prevents PID
reuse from silently preserving or stealing capacity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Protocol
import uuid


GPU_LEASE_SCHEMA_VERSION = "docarag.gpu-leases.v1"
DEFAULT_ALLOCATION_LIMIT_MIB = 42 * 1024
DEFAULT_HEADROOM_MIB = 6 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class GpuAdmissionError(RuntimeError):
    """A GPU request is invalid, unsafe, or cannot currently be admitted."""


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    uuid: str
    total_mib: int
    free_mib: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.total_mib < 1 or not 0 <= self.free_mib <= self.total_mib:
            raise GpuAdmissionError("invalid GPU inventory record")
        if not self.uuid.strip():
            raise GpuAdmissionError("GPU inventory requires a stable UUID")


class GpuInventory(Protocol):
    def devices(self) -> tuple[GpuDevice, ...]: ...


class NvidiaSmiInventory:
    """Strict bounded ``nvidia-smi`` inventory adapter."""

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        if not 0.1 <= timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 0.1 and 60")
        self.timeout_seconds = float(timeout_seconds)

    def devices(self) -> tuple[GpuDevice, ...]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GpuAdmissionError("nvidia-smi inventory is unavailable") from exc
        if completed.returncode != 0:
            raise GpuAdmissionError("nvidia-smi inventory failed")
        if len(completed.stdout.encode("utf-8")) > 64 * 1024:
            raise GpuAdmissionError("nvidia-smi returned an oversized inventory")
        devices: list[GpuDevice] = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise GpuAdmissionError("nvidia-smi returned an invalid inventory row")
            try:
                devices.append(
                    GpuDevice(
                        index=int(fields[0]),
                        uuid=fields[1],
                        total_mib=int(fields[2]),
                        free_mib=int(fields[3]),
                    )
                )
            except ValueError as exc:
                raise GpuAdmissionError(
                    "nvidia-smi returned non-integer memory fields"
                ) from exc
        if not devices:
            raise GpuAdmissionError("no NVIDIA GPU is visible")
        return tuple(devices)


@dataclass(frozen=True, slots=True)
class GpuLeaseRecord:
    lease_id: str
    owner: str
    gpu_uuid: str
    gpu_index: int
    memory_mib: int
    exclusive_family: bool
    pid: int | None
    process_start_ticks: int | None
    created_unix: float
    reservation_expires_unix: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "owner": self.owner,
            "gpu_uuid": self.gpu_uuid,
            "gpu_index": self.gpu_index,
            "memory_mib": self.memory_mib,
            "exclusive_family": self.exclusive_family,
            "pid": self.pid,
            "process_start_ticks": self.process_start_ticks,
            "created_unix": self.created_unix,
            "reservation_expires_unix": self.reservation_expires_unix,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GpuLeaseRecord":
        try:
            return cls(
                lease_id=str(value["lease_id"]),
                owner=str(value["owner"]),
                gpu_uuid=str(value["gpu_uuid"]),
                gpu_index=int(value["gpu_index"]),
                memory_mib=int(value["memory_mib"]),
                exclusive_family=bool(value["exclusive_family"]),
                pid=None if value.get("pid") is None else int(value["pid"]),
                process_start_ticks=(
                    None
                    if value.get("process_start_ticks") is None
                    else int(value["process_start_ticks"])
                ),
                created_unix=float(value["created_unix"]),
                reservation_expires_unix=(
                    None
                    if value.get("reservation_expires_unix") is None
                    else float(value["reservation_expires_unix"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuAdmissionError("GPU lease registry is invalid") from exc


def process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, or ``None`` for a missing process."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None
    try:
        # The executable name in parentheses can contain spaces.  Fields after
        # the final ')' begin with proc field 3; starttime is field 22.
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = payload[payload.rfind(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


class GpuAdmissionController:
    """Atomic lease registry backed by one deployment-private directory."""

    def __init__(
        self,
        state_root: str | os.PathLike[str],
        *,
        inventory: GpuInventory | None = None,
        allocation_limit_mib: int = DEFAULT_ALLOCATION_LIMIT_MIB,
        headroom_mib: int = DEFAULT_HEADROOM_MIB,
        clock: Callable[[], float] = time.time,
        process_identity: Callable[[int], int | None] = process_start_ticks,
    ) -> None:
        root = Path(state_root).expanduser()
        if not root.is_absolute() or root == Path("/"):
            raise ValueError("GPU state_root must be an absolute non-root path")
        if not 1 <= allocation_limit_mib <= 1024 * 1024:
            raise ValueError("allocation_limit_mib is invalid")
        if not 0 <= headroom_mib < allocation_limit_mib:
            raise ValueError("headroom_mib must be below allocation_limit_mib")
        self.state_root = root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.registry_path = self.state_root / "gpu-leases.json"
        self.lock_path = self.state_root / "gpu-leases.lock"
        self.inventory = inventory or NvidiaSmiInventory()
        self.allocation_limit_mib = allocation_limit_mib
        self.headroom_mib = headroom_mib
        self.clock = clock
        self.process_identity = process_identity

    def reserve(
        self,
        *,
        owner: str,
        memory_mib: int,
        gpu_index: int | None = None,
        exclusive_family: bool = True,
        reservation_seconds: float = 30.0,
    ) -> GpuLeaseRecord:
        if _TOKEN.fullmatch(owner) is None:
            raise ValueError("owner must be a bounded deployment identifier")
        if isinstance(memory_mib, bool) or not 1 <= memory_mib <= self.allocation_limit_mib:
            raise ValueError("memory_mib must fit the configured allocation limit")
        if not 1 <= reservation_seconds <= 300:
            raise ValueError("reservation_seconds must be between 1 and 300")
        if not isinstance(exclusive_family, bool):
            raise ValueError("exclusive_family must be boolean")
        devices = self.inventory.devices()
        selected = self._select_device(devices, gpu_index, memory_mib)
        now = self.clock()
        with self._locked_registry() as leases:
            active = self._reap_stale(leases, now)
            same_gpu = [item for item in active if item.gpu_uuid == selected.uuid]
            if exclusive_family and same_gpu:
                raise GpuAdmissionError(
                    f"GPU {selected.index} already has a managed model-family lease"
                )
            if any(item.exclusive_family for item in same_gpu):
                raise GpuAdmissionError(
                    f"GPU {selected.index} is reserved by an exclusive model family"
                )
            managed_mib = sum(item.memory_mib for item in same_gpu)
            if managed_mib + memory_mib > self.allocation_limit_mib:
                raise GpuAdmissionError("configured managed GPU allocation limit exceeded")
            if selected.free_mib < memory_mib + self.headroom_mib:
                raise GpuAdmissionError(
                    f"GPU {selected.index} needs {memory_mib + self.headroom_mib} MiB "
                    f"free including headroom; observed {selected.free_mib} MiB"
                )
            lease = GpuLeaseRecord(
                lease_id=f"gpu_{uuid.uuid4().hex}",
                owner=owner,
                gpu_uuid=selected.uuid,
                gpu_index=selected.index,
                memory_mib=memory_mib,
                exclusive_family=exclusive_family,
                pid=None,
                process_start_ticks=None,
                created_unix=now,
                reservation_expires_unix=now + reservation_seconds,
            )
            active.append(lease)
            self._write_registry(active)
            return lease

    def bind(self, lease_id: str, *, pid: int) -> GpuLeaseRecord:
        start_ticks = self.process_identity(pid)
        if start_ticks is None:
            raise GpuAdmissionError("cannot bind a GPU lease to a missing process")
        with self._locked_registry() as leases:
            active = self._reap_stale(leases, self.clock())
            matched = next((item for item in active if item.lease_id == lease_id), None)
            if matched is None:
                raise GpuAdmissionError("GPU reservation expired or does not exist")
            bound = GpuLeaseRecord(
                lease_id=matched.lease_id,
                owner=matched.owner,
                gpu_uuid=matched.gpu_uuid,
                gpu_index=matched.gpu_index,
                memory_mib=matched.memory_mib,
                exclusive_family=matched.exclusive_family,
                pid=pid,
                process_start_ticks=start_ticks,
                created_unix=matched.created_unix,
                reservation_expires_unix=None,
            )
            self._write_registry(
                [bound if item.lease_id == lease_id else item for item in active]
            )
            return bound

    def release(self, lease_id: str, *, expected_pid: int | None = None) -> bool:
        with self._locked_registry() as leases:
            active = self._reap_stale(leases, self.clock())
            matched = next((item for item in active if item.lease_id == lease_id), None)
            if matched is None:
                self._write_registry(active)
                return False
            if expected_pid is not None and matched.pid != expected_pid:
                raise GpuAdmissionError("GPU lease belongs to another process")
            self._write_registry([item for item in active if item.lease_id != lease_id])
            return True

    def status(self) -> tuple[GpuLeaseRecord, ...]:
        with self._locked_registry() as leases:
            active = self._reap_stale(leases, self.clock())
            self._write_registry(active)
            return tuple(active)

    def _select_device(
        self,
        devices: Sequence[GpuDevice],
        requested_index: int | None,
        memory_mib: int,
    ) -> GpuDevice:
        candidates = [
            item for item in devices if requested_index is None or item.index == requested_index
        ]
        if not candidates:
            raise GpuAdmissionError("requested NVIDIA GPU is not visible")
        eligible = [
            item for item in candidates if item.free_mib >= memory_mib + self.headroom_mib
        ]
        return max(eligible or candidates, key=lambda item: (item.free_mib, -item.index))

    def _reap_stale(
        self, leases: Sequence[GpuLeaseRecord], now: float
    ) -> list[GpuLeaseRecord]:
        active: list[GpuLeaseRecord] = []
        for lease in leases:
            if lease.pid is None:
                if lease.reservation_expires_unix is not None and now < lease.reservation_expires_unix:
                    active.append(lease)
                continue
            if self.process_identity(lease.pid) == lease.process_start_ticks:
                active.append(lease)
        return active

    class _RegistryContext:
        def __init__(self, parent: "GpuAdmissionController") -> None:
            self.parent = parent
            self.handle = None

        def __enter__(self) -> list[GpuLeaseRecord]:
            self.handle = self.parent.lock_path.open("a+b")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            return self.parent._read_registry()

        def __exit__(self, exc_type, exc, traceback) -> None:
            assert self.handle is not None
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _locked_registry(self) -> "GpuAdmissionController._RegistryContext":
        return self._RegistryContext(self)

    def _read_registry(self) -> list[GpuLeaseRecord]:
        if not self.registry_path.exists():
            return []
        try:
            if self.registry_path.stat().st_size > 1024 * 1024:
                raise GpuAdmissionError("GPU lease registry is oversized")
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != GPU_LEASE_SCHEMA_VERSION:
                raise GpuAdmissionError("unsupported GPU lease registry schema")
            values = payload.get("leases")
            if not isinstance(values, list):
                raise GpuAdmissionError("GPU lease registry has no lease array")
            return [GpuLeaseRecord.from_dict(item) for item in values]
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            raise GpuAdmissionError("GPU lease registry is unreadable") from exc

    def _write_registry(self, leases: Sequence[GpuLeaseRecord]) -> None:
        payload = {
            "schema_version": GPU_LEASE_SCHEMA_VERSION,
            "allocation_limit_mib": self.allocation_limit_mib,
            "headroom_mib": self.headroom_mib,
            "leases": [item.to_dict() for item in sorted(leases, key=lambda v: v.lease_id)],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=".gpu-leases-", dir=self.state_root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.registry_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


__all__ = [
    "DEFAULT_ALLOCATION_LIMIT_MIB",
    "DEFAULT_HEADROOM_MIB",
    "GPU_LEASE_SCHEMA_VERSION",
    "GpuAdmissionController",
    "GpuAdmissionError",
    "GpuDevice",
    "GpuInventory",
    "GpuLeaseRecord",
    "NvidiaSmiInventory",
    "process_start_ticks",
]
