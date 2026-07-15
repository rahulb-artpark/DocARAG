"""Atomic local attestations and process locks for remote vector indexes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

VECTOR_INDEX_ATTESTATION_NAME = "docarag-qdrant-index.ready.json"
MAX_VECTOR_INDEX_ATTESTATION_BYTES = 64 * 1024


class VectorIndexAttestationError(RuntimeError):
    """A vector index could not establish or publish local attestation."""


class VectorIndexAttestationReadLease:
    """An idempotently releasable shared lease held for a query lifetime."""

    def __init__(self, directory: str | Path):
        self._context = vector_index_lock(directory, exclusive=False)
        self._context.__enter__()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.__exit__(None, None, None)

    def __enter__(self) -> "VectorIndexAttestationReadLease":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _attestation_path(directory: str | Path) -> Path:
    return Path(directory) / VECTOR_INDEX_ATTESTATION_NAME


def read_vector_index_attestation(directory: str | Path) -> dict[str, Any] | None:
    """Read a bounded, regular, non-writable-by-others attestation."""

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        return None
    try:
        entries = list(root.iterdir())
    except OSError:
        return None
    if any(entry.name != VECTOR_INDEX_ATTESTATION_NAME for entry in entries):
        return None
    path = _attestation_path(root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        if file_stat.st_mode & 0o022:
            return None
        if file_stat.st_size <= 0 or file_stat.st_size > MAX_VECTOR_INDEX_ATTESTATION_BYTES:
            return None
        raw = bytearray()
        while len(raw) <= MAX_VECTOR_INDEX_ATTESTATION_BYTES:
            remaining = MAX_VECTOR_INDEX_ATTESTATION_BYTES + 1 - len(raw)
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != file_stat.st_size:
            return None
    finally:
        os.close(descriptor)

    try:
        payload = json.loads(
            bytes(raw).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def invalidate_vector_index_attestation(directory: str | Path) -> None:
    """Durably remove readiness before mutating any collection records."""

    root = Path(directory)
    path = _attestation_path(root)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(root)


def publish_vector_index_attestation(
    directory: str | Path,
    attestation: Mapping[str, Any],
) -> None:
    """Durably replace the content binding with canonical owner-only JSON."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(attestation),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_VECTOR_INDEX_ATTESTATION_BYTES:
        raise VectorIndexAttestationError("vector attestation exceeds size limit")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{VECTOR_INDEX_ATTESTATION_NAME}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, _attestation_path(root))
        _fsync_directory(root)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def vector_index_attestation_matches(
    directory: str | Path,
    expected: Mapping[str, Any],
    *,
    observed_point_count: int | None,
) -> bool:
    """Require exact local identity and a matching live scoped point count."""

    observed = read_vector_index_attestation(directory)
    if observed != dict(expected):
        return False
    point_count = expected.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        return False
    if isinstance(observed_point_count, bool) or not isinstance(
        observed_point_count, int
    ):
        return False
    return observed_point_count == point_count


@contextmanager
def vector_index_lock(
    directory: str | Path,
    *,
    exclusive: bool,
) -> Iterator[None]:
    """Serialize remote-scope rebuilds outside the attestation-only directory."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - production target is Linux
        raise VectorIndexAttestationError("vector index locking requires fcntl") from exc

    root = Path(directory)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.build.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise VectorIndexAttestationError("vector index lock is not a regular file")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
