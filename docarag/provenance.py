"""Dependency-light validation for immutable DocARAG run artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable


@dataclass(frozen=True)
class ArtifactVerification:
    """Result of validating one published artifact manifest."""

    valid: bool
    manifest_path: Path
    artifact_set_sha256: str | None = None
    file_count: int = 0
    total_bytes: int = 0
    errors: tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_QUERY_HASH_ROLES = frozenset(
    {
        "tree_pickle",
        "tree_json",
        "graph",
        "run_config",
        "canonical_evidence",
        "entity_attestation",
        "retrieval_attestation",
        "lexical_store",
    }
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_attempt_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and value not in {".", ".."}
        and _ATTEMPT_ID_PATTERN.fullmatch(value) is not None
    )


def environment_fingerprint_sha256(payload: dict[str, object]) -> str:
    """Recompute the canonical runtime identity emitted by ``runtime_probe``."""

    identity = {
        "python": payload.get("python"),
        "platform": payload.get("platform"),
        "packages": payload.get("packages"),
        "gpus": payload.get("gpus"),
        "environment": payload.get("environment"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _file_stat_key(path: Path) -> tuple[int, int, int, int, int]:
    stat_result = path.stat()
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


@lru_cache(maxsize=1024)
def _sha256_for_stat(
    path: str,
    _device: int,
    _inode: int,
    _size: int,
    _mtime_ns: int,
    _ctime_ns: int,
) -> str:
    return sha256_file(Path(path))


def _cached_sha256_file(path: Path) -> str:
    """Hash once per file identity and reject a file changed during hashing."""

    for _ in range(2):
        before = _file_stat_key(path)
        digest = _sha256_for_stat(str(path), *before)
        if _file_stat_key(path) == before:
            return digest
    raise OSError(f"file changed while it was being hashed: {path}")


def _hash_file(path: Path, *, deep: bool) -> str:
    return sha256_file(path) if deep else _cached_sha256_file(path)


@lru_cache(maxsize=256)
def _json_for_stat(
    path: str,
    _device: int,
    _inode: int,
    _size: int,
    _mtime_ns: int,
    _ctime_ns: int,
) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cached_json(path: Path) -> object:
    key = _file_stat_key(path)
    payload = _json_for_stat(str(path), *key)
    if _file_stat_key(path) != key:
        raise OSError(f"file changed while it was being parsed: {path}")
    return payload


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _atomic_write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _vector_attestation_descriptor(
    path: Path,
    *,
    revision_id: str,
    document_version_id: str,
    source_sha256: str,
    required_index_kind: str | None,
) -> dict[str, object]:
    """Validate and project a secret-free Qdrant content attestation."""

    if path.is_symlink() or not path.is_file() or not (0 < path.stat().st_size <= 64 * 1024):
        raise ValueError(f"Qdrant attestation is missing, unsafe, or oversized: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qdrant attestation is invalid: {path}") from exc
    if not isinstance(payload, dict) or payload.get("state") != "content_attested":
        raise ValueError(f"Qdrant attestation is not content-attested: {path}")
    scope = payload.get("scope_filter")
    required_scope_keys = {
        "tenant_id",
        "corpus_id",
        "document_version_id",
        "index_revision_id",
        "source_sha256",
        "index_kind",
    }
    if not isinstance(scope, dict) or set(scope) != required_scope_keys:
        raise ValueError(f"Qdrant attestation scope is invalid: {path}")
    if scope.get("document_version_id") != document_version_id:
        raise ValueError("Qdrant attestation belongs to another document version")
    if scope.get("index_revision_id") != revision_id:
        raise ValueError("Qdrant attestation belongs to another index revision")
    if payload.get("document_version_id") != document_version_id:
        raise ValueError("Qdrant attestation document-version binding differs")
    if payload.get("index_revision_id") != revision_id:
        raise ValueError("Qdrant attestation index-revision binding differs")
    if scope.get("source_sha256") != source_sha256:
        raise ValueError("Qdrant attestation belongs to another source")
    if required_index_kind is not None and scope.get("index_kind") != required_index_kind:
        raise ValueError(
            f"Qdrant attestation index kind must be {required_index_kind!r}"
        )
    if any(
        not isinstance(scope.get(name), str)
        or not scope[name]
        or scope[name] != scope[name].strip()
        for name in required_scope_keys
    ):
        raise ValueError(f"Qdrant attestation scope values are invalid: {path}")
    point_count = payload.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
        raise ValueError(f"Qdrant attestation point count is invalid: {path}")
    collection = payload.get("collection_name")
    if not isinstance(collection, str) or not collection or collection != collection.strip():
        raise ValueError(f"Qdrant attestation collection is invalid: {path}")
    endpoint_sha256 = payload.get("qdrant_endpoint_sha256")
    identity_sha256 = payload.get("identity_sha256")
    if not _is_sha256(endpoint_sha256) or not _is_sha256(identity_sha256):
        raise ValueError(f"Qdrant attestation identity is invalid: {path}")
    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"identity_sha256", "state"}
    }
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != identity_sha256:
        raise ValueError(f"Qdrant attestation self-digest is invalid: {path}")
    return {
        "collection_name": collection,
        "scope_filter": scope,
        "point_count": point_count,
        "qdrant_endpoint_sha256": endpoint_sha256,
        "attestation_sha256": sha256_file(path),
        "identity_sha256": identity_sha256,
    }


def publish_artifact_manifest(
    run_dir: Path,
    *,
    revision_id: str,
    attempt_id: str,
    retrieval_attestation: Path | None,
    graph_artifact: Path,
    entity_attestation: Path,
    lexical_store: Path | None = None,
) -> dict[str, object]:
    """Hash and atomically publish the authoritative outputs of one attempt."""

    run = run_dir.expanduser().resolve()
    if not revision_id:
        raise ValueError("revision_id must not be empty")
    if run.name != revision_id:
        raise ValueError("publication revision ID differs from run directory")
    if not _is_attempt_id(attempt_id):
        raise ValueError(f"attempt_id is unsafe: {attempt_id!r}")

    run_manifest = run / "run-manifest.json"
    if run_manifest.is_symlink() or not run_manifest.is_file() or not run_manifest.stat().st_size:
        raise ValueError(f"required run manifest is missing or unsafe: {run_manifest}")
    try:
        run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run-manifest.json is invalid: {exc}") from exc
    if not isinstance(run_payload, dict) or run_payload.get("schema_version") != 1:
        raise ValueError("run-manifest.json has an unsupported schema")
    if run_payload.get("revision_id") != revision_id:
        raise ValueError("run and publication revision IDs differ")

    current_state = run / "index-state.json"
    if current_state.is_symlink() or not current_state.is_file():
        raise ValueError("index-state.json is missing or unsafe during publication")
    try:
        state_payload = json.loads(current_state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"index-state.json is invalid: {exc}") from exc
    if not isinstance(state_payload, dict) or state_payload.get("schema_version") != 1:
        raise ValueError("index-state.json has an unsupported schema")
    if state_payload.get("revision_id") != revision_id:
        raise ValueError("index state and publication revision IDs differ")
    if state_payload.get("attempt_id") != attempt_id:
        raise ValueError("index state and publication attempt IDs differ")
    if state_payload.get("overall_status") not in {"running", "succeeded"}:
        # A running attempt publishes as it finishes; a sealed run republishes
        # when one of its stages is legitimately re-run.  A failed run has no
        # coherent artifact set to describe.
        raise ValueError("artifacts may only be published by the active attempt")
    if state_payload.get("current_stage") not in {None, ""}:
        raise ValueError("cannot publish while an index stage is still active")
    stages = state_payload.get("stages")
    if not isinstance(stages, dict) or not all(
        isinstance(stages.get(stage), dict)
        and stages[stage].get("status") == "succeeded"
        for stage in ("tree", "graph", "vdb")
    ):
        raise ValueError("cannot publish before every index stage succeeds")

    retrieval = (
        retrieval_attestation.expanduser().resolve()
        if retrieval_attestation is not None
        else None
    )
    graph = graph_artifact.expanduser().resolve()
    entity = entity_attestation.expanduser().resolve()
    lexical = lexical_store.expanduser().resolve() if lexical_store is not None else None
    selected_roles = [
        ("graph artifact", graph),
        ("entity attestation", entity),
    ]
    if retrieval is not None:
        selected_roles.append(("retrieval attestation", retrieval))
    if lexical is not None:
        selected_roles.append(("lexical store", lexical))
    if (retrieval is None) == (lexical is None):
        raise ValueError(
            "publish exactly one retrieval attestation or BM25 lexical store"
        )
    for role_name, role_path in selected_roles:
        if not _is_below(role_path, run):
            raise ValueError(f"published {role_name} escapes the run directory: {role_path}")
        if role_path.is_symlink() or role_path != role_path.absolute():
            raise ValueError(f"published {role_name} is not a direct run artifact: {role_path}")
    if not graph.is_file() or graph.stat().st_size == 0:
        raise ValueError(f"selected graph artifact is missing or empty: {graph}")
    for role_name, role_path in selected_roles:
        if role_name == "graph artifact":
            continue
        if not role_path.is_file() or role_path.stat().st_size == 0:
            raise ValueError(f"published {role_name} is not a non-empty file: {role_path}")

    required = (run / "tree.pkl", run / "tree.json", run / "run_config.yaml")
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required published artifact is missing or empty: {path}")

    roots: set[Path] = {
        *required,
        *(path for path in run.glob("graph_data*.json") if path.is_file()),
    }
    for optional_name in (
        "indexing_stats.json",
        "summary-checkpoints.sqlite3",
        "canonical_blocks.json",
        "auto",
        "vlm",
        "kg_extractor_res",
    ):
        optional = run / optional_name
        if optional.exists():
            roots.add(optional)
    roots.add(entity)
    if retrieval is not None:
        roots.add(retrieval)
    if lexical is not None:
        roots.add(lexical)

    try:
        tree_payload = json.loads((run / "tree.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tree.json for artifact publication: {exc}") from exc

    def collect_references(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                strict_keys = {"img_path"}
                if key in strict_keys | {"image_path", "crop_path", "page_path"} and isinstance(item, str):
                    candidate = Path(item).expanduser()
                    if not candidate.is_absolute():
                        candidate = run / candidate
                    try:
                        resolved = candidate.resolve()
                    except (OSError, RuntimeError) as exc:
                        if key in strict_keys:
                            raise ValueError(
                                f"cannot resolve referenced evidence {item}: {exc}"
                            ) from exc
                        resolved = None
                    if resolved and _is_below(resolved, run) and resolved.is_file():
                        roots.add(resolved)
                    elif key in strict_keys and item:
                        raise ValueError(
                            f"referenced evidence is missing or outside the run: {item}"
                        )
                collect_references(item)
        elif isinstance(value, list):
            for item in value:
                collect_references(item)

    collect_references(tree_payload)

    files: set[Path] = set()
    for root in roots:
        try:
            resolved_root = root.resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot resolve artifact path {root}: {exc}") from exc
        if not _is_below(resolved_root, run):
            raise ValueError(f"published artifact escapes the run directory: {root}")
        if root.is_symlink():
            raise ValueError(f"published artifacts may not be symbolic links: {root}")
        if resolved_root.is_file():
            files.add(resolved_root)
            continue
        if not resolved_root.is_dir():
            continue
        for candidate in resolved_root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(
                    f"published artifacts may not contain symbolic links: {candidate}"
                )
            if candidate.is_file():
                resolved = candidate.resolve()
                if not _is_below(resolved, run):
                    raise ValueError(
                        f"published artifact escapes the run directory: {candidate}"
                    )
                files.add(resolved)

    if graph not in files:
        raise ValueError(f"selected graph artifact was not published: {graph}")

    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(files, key=lambda item: item.relative_to(run).as_posix()):
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(run).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )

    content_digest = hashlib.sha256()
    for entry in entries:
        encoded_entry = json.dumps(
            entry, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        content_digest.update(len(encoded_entry).to_bytes(8, "big"))
        content_digest.update(encoded_entry)

    runtime_manifest = run / "attempts" / attempt_id / "runtime.json"
    try:
        resolved_runtime = runtime_manifest.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"attempt runtime manifest is missing: {runtime_manifest}") from exc
    expected_attempt_dir = run / "attempts" / attempt_id
    if (
        runtime_manifest.is_symlink()
        or resolved_runtime != runtime_manifest.absolute()
        or resolved_runtime.parent != expected_attempt_dir.absolute()
        or not resolved_runtime.is_file()
        or resolved_runtime.stat().st_size == 0
    ):
        raise ValueError(f"attempt runtime manifest is missing or empty: {runtime_manifest}")
    try:
        runtime_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"attempt runtime manifest is invalid: {exc}") from exc
    if not isinstance(runtime_payload, dict):
        raise ValueError("attempt runtime manifest is not an object")
    if runtime_payload.get("schema_version") != 1:
        raise ValueError("attempt runtime manifest has an unsupported schema")
    if runtime_payload.get("revision_id") != revision_id:
        raise ValueError("runtime and publication revision IDs differ")
    if runtime_payload.get("attempt_id") != attempt_id:
        raise ValueError("runtime and publication attempt IDs differ")
    if not isinstance(runtime_payload.get("python"), dict):
        raise ValueError("runtime Python identity is missing")
    if not isinstance(runtime_payload.get("platform"), dict):
        raise ValueError("runtime platform identity is missing")
    if not isinstance(runtime_payload.get("packages"), dict):
        raise ValueError("runtime package identity is missing")
    if not isinstance(runtime_payload.get("gpus"), list):
        raise ValueError("runtime GPU identity is missing")
    if not isinstance(runtime_payload.get("environment"), dict):
        raise ValueError("runtime process environment identity is missing")
    recorded_fingerprint = runtime_payload.get("environment_fingerprint_sha256")
    expected_fingerprint = environment_fingerprint_sha256(runtime_payload)
    if not _is_sha256(recorded_fingerprint) or recorded_fingerprint != expected_fingerprint:
        raise ValueError("runtime environment fingerprint is invalid")

    source_payload = run_payload.get("source")
    source_sha256 = (
        source_payload.get("sha256") if isinstance(source_payload, dict) else None
    )
    if not _is_sha256(source_sha256):
        raise ValueError("run manifest source SHA-256 is invalid")
    document_version_id = run_payload.get("document_version_id")
    if not isinstance(document_version_id, str) or not document_version_id.strip():
        raise ValueError("run manifest document-version ID is invalid")
    entity_descriptor = _vector_attestation_descriptor(
        entity,
        revision_id=revision_id,
        document_version_id=document_version_id,
        source_sha256=source_sha256,
        required_index_kind="entity",
    )
    vector_indexes: dict[str, object] = {"entity": entity_descriptor}
    roles = {
        "tree_pickle": "tree.pkl",
        "tree_json": "tree.json",
        "graph": graph.relative_to(run).as_posix(),
        "entity_attestation": entity.relative_to(run).as_posix(),
        "run_config": "run_config.yaml",
    }
    if retrieval is not None:
        retrieval_descriptor = _vector_attestation_descriptor(
            retrieval,
            revision_id=revision_id,
            document_version_id=document_version_id,
            source_sha256=source_sha256,
            required_index_kind=None,
        )
        if retrieval_descriptor["scope_filter"].get("index_kind") == "entity":
            raise ValueError("retrieval attestation may not use the entity index kind")
        roles["retrieval_attestation"] = retrieval.relative_to(run).as_posix()
        vector_indexes["retrieval"] = retrieval_descriptor
    else:
        roles["lexical_store"] = lexical.relative_to(run).as_posix()
    canonical_evidence = run / "canonical_blocks.json"
    if canonical_evidence.exists():
        if not canonical_evidence.is_file() or canonical_evidence.stat().st_size == 0:
            raise ValueError(
                "canonical_blocks.json exists but is not a non-empty regular file"
            )
        roles["canonical_evidence"] = "canonical_blocks.json"

    payload: dict[str, object] = {
        "schema_version": 2,
        "revision_id": revision_id,
        "document_version_id": document_version_id,
        "attempt_id": attempt_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hash_algorithm": "sha256",
        "run_manifest": {
            "path": "run-manifest.json",
            "sha256": sha256_file(run_manifest),
        },
        "runtime_manifest": {
            "path": runtime_manifest.relative_to(run).as_posix(),
            "sha256": sha256_file(runtime_manifest),
            "environment_fingerprint_sha256": runtime_payload.get(
                "environment_fingerprint_sha256"
            ),
        },
        "roles": roles,
        "vector_indexes": vector_indexes,
        "artifact_set_sha256": content_digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    encoded_payload = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(run / "artifact-manifest.json", encoded_payload, 0o400)
    _atomic_write(
        run / "attempts" / attempt_id / "artifact-manifest.json",
        encoded_payload,
        0o400,
    )
    return payload


def verify_artifact_manifest(
    run_dir: Path,
    *,
    deep: bool = False,
    hash_roles: Iterable[str] = (),
    hash_bm25: bool = False,
) -> ArtifactVerification:
    """Validate a run-local artifact manifest without importing ML libraries.

    The normal path validates identity, containment, file existence, and sizes.
    ``hash_roles`` selectively hashes files that a latency-sensitive caller is
    about to parse or deserialize; these digests are cached by immutable stat
    identity. ``deep=True`` hashes every published artifact and is intended for
    deduplication and audits. ``hash_bm25`` covers a published BM25 pickle
    without hashing an entire vector-store directory.
    """

    run = run_dir.expanduser().resolve()
    manifest_path = run / "artifact-manifest.json"
    errors: list[str] = []

    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("artifact-manifest.json is missing or is a symbolic link")
        payload = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if deep
            else _cached_json(manifest_path)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return ArtifactVerification(False, manifest_path, errors=(str(exc),))

    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        return ArtifactVerification(
            False,
            manifest_path,
            errors=("artifact manifest has an unsupported schema",),
        )

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("artifact manifest has no files")
        entries = []

    seen: set[str] = set()
    entry_by_path: dict[str, dict[str, object]] = {}
    resolved_by_path: dict[str, Path] = {}
    canonical_entries: list[dict[str, object]] = []
    measured_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("artifact manifest contains a non-object file entry")
            continue
        relative = entry.get("path")
        size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or not isinstance(size, int)
            or size < 0
            or not _is_sha256(expected_sha)
        ):
            errors.append(f"invalid artifact entry: {relative!r}")
            continue
        seen.add(relative)
        entry_by_path[relative] = entry
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"artifact path is not run-relative: {relative}")
            continue
        candidate = run / relative_path
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(run)
        except (OSError, RuntimeError, ValueError):
            errors.append(f"artifact path escapes the run or is missing: {relative}")
            continue
        if candidate.is_symlink() or resolved != candidate.absolute() or not resolved.is_file():
            errors.append(f"artifact path is not a direct regular file: {relative}")
            continue
        try:
            actual_size = resolved.stat().st_size
        except OSError as exc:
            errors.append(f"cannot stat artifact {relative}: {exc}")
            continue
        if actual_size != size:
            errors.append(
                f"artifact size differs for {relative}: expected {size}, got {actual_size}"
            )
            continue
        resolved_by_path[relative] = resolved
        if deep:
            try:
                actual_sha = sha256_file(resolved)
            except OSError as exc:
                errors.append(f"cannot hash artifact {relative}: {exc}")
                continue
            if actual_sha != expected_sha:
                errors.append(f"artifact digest differs for {relative}")
                continue
        measured_bytes += actual_size
        canonical_entries.append(
            {"path": relative, "size_bytes": size, "sha256": expected_sha}
        )

    expected_count = payload.get("file_count")
    expected_bytes = payload.get("total_bytes")
    if expected_count != len(entries):
        errors.append("artifact manifest file_count is inconsistent")
    if expected_bytes != measured_bytes:
        errors.append("artifact manifest total_bytes is inconsistent")

    roles = payload.get("roles")
    required_file_roles = {
        "tree_pickle",
        "tree_json",
        "graph",
        "run_config",
        "entity_attestation",
    }
    required_directory_roles: set[str] = set()
    optional_file_roles = {"canonical_evidence"}
    if not isinstance(roles, dict):
        errors.append("artifact manifest has no role mapping")
        roles = {}
    has_retrieval_attestation = isinstance(roles.get("retrieval_attestation"), str)
    has_lexical_store = isinstance(roles.get("lexical_store"), str)
    if has_retrieval_attestation == has_lexical_store:
        errors.append(
            "artifact roles require exactly one retrieval_attestation or lexical_store"
        )
    elif has_retrieval_attestation:
        required_file_roles.add("retrieval_attestation")
    else:
        required_file_roles.add("lexical_store")
    for role in sorted(required_file_roles | required_directory_roles):
        value = roles.get(role)
        if not isinstance(value, str) or not value:
            errors.append(f"artifact role is missing: {role}")
            continue
        role_path = Path(value)
        if role_path.is_absolute() or ".." in role_path.parts:
            errors.append(f"artifact role is not run-relative: {role}")
            continue
        if role in required_file_roles:
            if value not in seen:
                errors.append(f"artifact role does not reference a published file: {role}")
        else:
            prefix = value.rstrip("/") + "/"
            if not any(path.startswith(prefix) for path in seen):
                errors.append(f"artifact role has no published files: {role}")
    for role in sorted(optional_file_roles & set(roles)):
        value = roles.get(role)
        if not isinstance(value, str) or not value:
            errors.append(f"optional artifact role is invalid: {role}")
            continue
        role_path = Path(value)
        if role_path.is_absolute() or ".." in role_path.parts:
            errors.append(f"artifact role is not run-relative: {role}")
        elif value not in seen:
            errors.append(f"artifact role does not reference a published file: {role}")

    vector_indexes = payload.get("vector_indexes")
    expected_vector_names = {"entity"}
    if has_retrieval_attestation:
        expected_vector_names.add("retrieval")
    if not isinstance(vector_indexes, dict) or set(vector_indexes) != expected_vector_names:
        errors.append("artifact manifest vector index descriptors are inconsistent")
        vector_indexes = {}
    for index_name, role_name in (
        ("entity", "entity_attestation"),
        ("retrieval", "retrieval_attestation"),
    ):
        if index_name not in expected_vector_names:
            continue
        descriptor = vector_indexes.get(index_name)
        role_value = roles.get(role_name)
        entry = entry_by_path.get(role_value) if isinstance(role_value, str) else None
        if not isinstance(descriptor, dict):
            errors.append(f"vector index descriptor is invalid: {index_name}")
            continue
        if entry is None or descriptor.get("attestation_sha256") != entry.get("sha256"):
            errors.append(f"vector attestation digest differs: {index_name}")
        if not _is_sha256(descriptor.get("qdrant_endpoint_sha256")):
            errors.append(f"vector endpoint fingerprint is invalid: {index_name}")
        if not _is_sha256(descriptor.get("identity_sha256")):
            errors.append(f"vector identity digest is invalid: {index_name}")
        point_count = descriptor.get("point_count")
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            errors.append(f"vector point count is invalid: {index_name}")
        scope = descriptor.get("scope_filter")
        if not isinstance(scope, dict) or set(scope) != {
            "tenant_id",
            "corpus_id",
            "document_version_id",
            "index_revision_id",
            "source_sha256",
            "index_kind",
        }:
            errors.append(f"vector scope is invalid: {index_name}")

    if not deep:
        selected_paths: set[str] = set()
        for role in set(hash_roles):
            role_value = roles.get(role)
            if isinstance(role_value, str) and role_value:
                selected_paths.add(role_value)
        if hash_bm25:
            lexical_value = roles.get("lexical_store")
            if isinstance(lexical_value, str) and lexical_value:
                selected_paths.add(lexical_value)
        for relative in sorted(selected_paths):
            entry = entry_by_path.get(relative)
            resolved = resolved_by_path.get(relative)
            if entry is None or resolved is None:
                errors.append(f"selected query artifact is unavailable: {relative}")
                continue
            try:
                actual_sha = _cached_sha256_file(resolved)
            except OSError as exc:
                errors.append(f"cannot hash selected query artifact {relative}: {exc}")
                continue
            if actual_sha != entry.get("sha256"):
                errors.append(f"artifact digest differs for {relative}")

    set_digest = hashlib.sha256()
    for entry in canonical_entries:
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        set_digest.update(len(encoded).to_bytes(8, "big"))
        set_digest.update(encoded)
    artifact_set_sha256 = payload.get("artifact_set_sha256")
    if artifact_set_sha256 != set_digest.hexdigest():
        errors.append("artifact_set_sha256 is inconsistent")

    artifact_revision = payload.get("revision_id")
    artifact_document_version = payload.get("document_version_id")
    artifact_attempt = payload.get("attempt_id")
    if not isinstance(artifact_revision, str) or not artifact_revision:
        errors.append("artifact manifest revision ID is invalid")
    elif run.name != artifact_revision:
        errors.append("artifact manifest revision ID differs from run directory")
    if (
        not isinstance(artifact_document_version, str)
        or not artifact_document_version
    ):
        errors.append("artifact manifest document-version ID is invalid")
    if not _is_attempt_id(artifact_attempt):
        errors.append("artifact manifest attempt ID is invalid")

    run_manifest = payload.get("run_manifest")
    source_manifest = run / "run-manifest.json"
    run_payload: dict[str, object] = {}
    if not isinstance(run_manifest, dict):
        errors.append("artifact manifest has no run manifest identity")
    elif run_manifest.get("path") != "run-manifest.json":
        errors.append("run manifest path is invalid")
    elif not _is_sha256(run_manifest.get("sha256")):
        errors.append("run manifest digest is invalid")
    elif source_manifest.is_symlink() or not source_manifest.is_file():
        errors.append("run-manifest.json is missing or unsafe")
    else:
        try:
            actual_manifest_sha = _hash_file(source_manifest, deep=deep)
        except OSError as exc:
            errors.append(f"cannot hash run-manifest.json: {exc}")
        else:
            if actual_manifest_sha != run_manifest.get("sha256"):
                errors.append("run-manifest.json digest differs")
        try:
            loaded_run = (
                json.loads(source_manifest.read_text(encoding="utf-8"))
                if deep
                else _cached_json(source_manifest)
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append("run-manifest.json cannot be parsed")
        else:
            if not isinstance(loaded_run, dict):
                errors.append("run-manifest.json is not an object")
            else:
                run_payload = loaded_run

    if run_payload:
        if run_payload.get("schema_version") != 1:
            errors.append("run-manifest.json has an unsupported schema")
        if run_payload.get("revision_id") != artifact_revision:
            errors.append("run and artifact manifest revision IDs differ")
        if run_payload.get("document_version_id") != artifact_document_version:
            errors.append("run and artifact document-version IDs differ")
        source = run_payload.get("source")
        if not isinstance(source, dict):
            errors.append("run manifest has no source identity")
        else:
            snapshot_value = source.get("snapshot_path")
            source_sha = source.get("sha256")
            source_size = source.get("size_bytes")
            if (
                not isinstance(snapshot_value, str)
                or not _is_sha256(source_sha)
                or not isinstance(source_size, int)
                or source_size < 0
            ):
                errors.append("run manifest source snapshot identity is incomplete")
            else:
                for index_name, descriptor in vector_indexes.items():
                    scope = descriptor.get("scope_filter") if isinstance(descriptor, dict) else None
                    if not isinstance(scope, dict):
                        continue
                    if scope.get("document_version_id") != artifact_document_version:
                        errors.append(
                            f"vector scope document version differs: {index_name}"
                        )
                    if scope.get("index_revision_id") != artifact_revision:
                        errors.append(
                            f"vector scope index revision differs: {index_name}"
                        )
                    if scope.get("tenant_id") != run_payload.get("tenant_id"):
                        errors.append(f"vector scope tenant differs: {index_name}")
                    if scope.get("corpus_id") != run_payload.get("corpus_id"):
                        errors.append(f"vector scope corpus differs: {index_name}")
                    if scope.get("source_sha256") != source_sha:
                        errors.append(f"vector scope source differs: {index_name}")
                    if index_name == "entity" and scope.get("index_kind") != "entity":
                        errors.append("entity vector scope has the wrong index kind")
                    if index_name == "retrieval" and scope.get("index_kind") == "entity":
                        errors.append("retrieval vector scope uses the entity index kind")
                snapshot = Path(snapshot_value).expanduser()
                try:
                    resolved_snapshot = snapshot.resolve(strict=True)
                except (OSError, RuntimeError):
                    errors.append("immutable source snapshot is missing")
                else:
                    expected_name = f"{source_sha}.pdf"
                    expected_snapshot_root = run.parent.parent / "_sources" / "sha256"
                    try:
                        resolved_snapshot_root = expected_snapshot_root.resolve(
                            strict=True
                        )
                    except (OSError, RuntimeError):
                        resolved_snapshot_root = None
                    if (
                        snapshot.is_symlink()
                        or resolved_snapshot != snapshot.absolute()
                        or not resolved_snapshot.is_file()
                        or resolved_snapshot.name != expected_name
                        or resolved_snapshot_root is None
                        or resolved_snapshot.parent != resolved_snapshot_root
                    ):
                        errors.append("immutable source snapshot path is unsafe")
                    else:
                        try:
                            actual_source_size = resolved_snapshot.stat().st_size
                        except OSError as exc:
                            errors.append(f"cannot stat source snapshot: {exc}")
                        else:
                            if actual_source_size != source_size:
                                errors.append("immutable source snapshot size differs")
                            elif deep:
                                try:
                                    actual_source_sha = sha256_file(resolved_snapshot)
                                except OSError as exc:
                                    errors.append(f"cannot hash source snapshot: {exc}")
                                else:
                                    if actual_source_sha != source_sha:
                                        errors.append(
                                            "immutable source snapshot digest differs"
                                        )

    runtime_manifest = payload.get("runtime_manifest")
    runtime_payload: dict[str, object] = {}
    if not isinstance(runtime_manifest, dict):
        errors.append("artifact manifest has no runtime manifest identity")
    else:
        runtime_relative = runtime_manifest.get("path")
        runtime_sha = runtime_manifest.get("sha256")
        expected_runtime_relative = (
            f"attempts/{artifact_attempt}/runtime.json"
            if _is_attempt_id(artifact_attempt)
            else None
        )
        if runtime_relative != expected_runtime_relative:
            errors.append("runtime manifest path is invalid")
        elif not _is_sha256(runtime_sha):
            errors.append("runtime manifest digest is invalid")
        elif not _is_sha256(runtime_manifest.get("environment_fingerprint_sha256")):
            errors.append("runtime environment fingerprint reference is invalid")
        else:
            relative_path = Path(runtime_relative)
            runtime_path = run / relative_path
            try:
                resolved_runtime = runtime_path.resolve(strict=True)
                resolved_runtime.relative_to(run)
            except (OSError, RuntimeError, ValueError):
                errors.append("runtime manifest escapes the run or is missing")
            else:
                if (
                    relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or runtime_path.is_symlink()
                    or resolved_runtime != runtime_path.absolute()
                    or not resolved_runtime.is_file()
                ):
                    errors.append("runtime manifest path is unsafe")
                else:
                    try:
                        actual_runtime_sha = _hash_file(resolved_runtime, deep=deep)
                    except OSError as exc:
                        errors.append(f"cannot hash runtime manifest: {exc}")
                    else:
                        if actual_runtime_sha != runtime_sha:
                            errors.append("runtime manifest digest differs")
                    try:
                        loaded_runtime = (
                            json.loads(resolved_runtime.read_text(encoding="utf-8"))
                            if deep
                            else _cached_json(resolved_runtime)
                        )
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        errors.append("runtime manifest cannot be parsed")
                    else:
                        if not isinstance(loaded_runtime, dict):
                            errors.append("runtime manifest is not an object")
                        else:
                            runtime_payload = loaded_runtime

    if runtime_payload:
        if runtime_payload.get("schema_version") != 1:
            errors.append("runtime manifest has an unsupported schema")
        if runtime_payload.get("revision_id") != artifact_revision:
            errors.append("runtime and artifact manifest revision IDs differ")
        if runtime_payload.get("attempt_id") != artifact_attempt:
            errors.append("runtime and artifact manifest attempt IDs differ")
        if not isinstance(runtime_payload.get("python"), dict):
            errors.append("runtime Python identity is missing")
        if not isinstance(runtime_payload.get("platform"), dict):
            errors.append("runtime platform identity is missing")
        if not isinstance(runtime_payload.get("packages"), dict):
            errors.append("runtime package identity is missing")
        if not isinstance(runtime_payload.get("gpus"), list):
            errors.append("runtime GPU identity is missing")
        if not isinstance(runtime_payload.get("environment"), dict):
            errors.append("runtime process environment identity is missing")
        recorded_fingerprint = runtime_payload.get("environment_fingerprint_sha256")
        computed_fingerprint = environment_fingerprint_sha256(runtime_payload)
        if not _is_sha256(recorded_fingerprint) or recorded_fingerprint != computed_fingerprint:
            errors.append("runtime environment fingerprint is inconsistent")
        if isinstance(runtime_manifest, dict) and (
            runtime_manifest.get("environment_fingerprint_sha256")
            != recorded_fingerprint
        ):
            errors.append("artifact and runtime environment fingerprints differ")

    try:
        root_manifest_sha = _hash_file(manifest_path, deep=deep)
    except OSError as exc:
        errors.append(f"cannot hash artifact-manifest.json: {exc}")
        root_manifest_sha = None

    if _is_attempt_id(artifact_attempt):
        attempt_manifest_path = (
            run / "attempts" / str(artifact_attempt) / "artifact-manifest.json"
        )
        try:
            resolved_attempt_manifest = attempt_manifest_path.resolve(strict=True)
            resolved_attempt_manifest.relative_to(run)
        except (OSError, RuntimeError, ValueError):
            errors.append("attempt-local artifact manifest is missing")
        else:
            if (
                attempt_manifest_path.is_symlink()
                or resolved_attempt_manifest != attempt_manifest_path.absolute()
                or not resolved_attempt_manifest.is_file()
            ):
                errors.append("attempt-local artifact manifest is unsafe")
            elif root_manifest_sha is not None:
                try:
                    attempt_manifest_sha = _hash_file(
                        resolved_attempt_manifest, deep=deep
                    )
                except OSError as exc:
                    errors.append(f"cannot hash attempt-local artifact manifest: {exc}")
                else:
                    if attempt_manifest_sha != root_manifest_sha:
                        errors.append("root and attempt-local artifact manifests differ")

    def validate_terminal_state(state_path: Path, label: str) -> None:
        try:
            resolved_state = state_path.resolve(strict=True)
            resolved_state.relative_to(run)
        except (OSError, RuntimeError, ValueError):
            errors.append(f"{label} is missing")
            return
        if (
            state_path.is_symlink()
            or resolved_state != state_path.absolute()
            or not resolved_state.is_file()
        ):
            errors.append(f"{label} is unsafe")
            return
        try:
            loaded_state = (
                json.loads(resolved_state.read_text(encoding="utf-8"))
                if deep
                else _cached_json(resolved_state)
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{label} cannot be parsed")
            return
        if not isinstance(loaded_state, dict):
            errors.append(f"{label} is not an object")
            return
        if loaded_state.get("schema_version") != 1:
            errors.append(f"{label} has an unsupported schema")
        if loaded_state.get("revision_id") != artifact_revision:
            errors.append(f"{label} revision ID differs")
        if loaded_state.get("attempt_id") != artifact_attempt:
            errors.append(f"{label} attempt ID differs")
        if loaded_state.get("overall_status") != "succeeded":
            errors.append(f"{label} is not a successful terminal state")
        state_stages = loaded_state.get("stages")
        if not isinstance(state_stages, dict) or not all(
            isinstance(state_stages.get(stage), dict)
            and state_stages[stage].get("status") == "succeeded"
            for stage in ("tree", "graph", "vdb")
        ):
            errors.append(f"{label} stage results are inconsistent")
        if loaded_state.get("artifact_manifest") != "artifact-manifest.json":
            errors.append(f"{label} artifact manifest reference differs")
        if root_manifest_sha is not None and (
            loaded_state.get("artifact_manifest_sha256") != root_manifest_sha
        ):
            errors.append(f"{label} artifact manifest digest differs")
        if loaded_state.get("artifact_set_sha256") != artifact_set_sha256:
            errors.append(f"{label} artifact set identity differs")

    validate_terminal_state(run / "index-state.json", "terminal index state")
    if _is_attempt_id(artifact_attempt):
        validate_terminal_state(
            run / "attempts" / str(artifact_attempt) / "state.json",
            "terminal attempt state",
        )

    return ArtifactVerification(
        not errors,
        manifest_path,
        artifact_set_sha256=(
            artifact_set_sha256 if isinstance(artifact_set_sha256, str) else None
        ),
        file_count=len(entries),
        total_bytes=measured_bytes,
        errors=tuple(errors),
    )


def verify_query_artifacts(run_dir: Path) -> ArtifactVerification:
    """Verify only the published files that query startup will parse/load.

    Tree, graph, configuration, vector attestations, and any BM25 pickle digest
    are enforced. Live Qdrant state is checked separately at readiness time.
    """

    return verify_artifact_manifest(
        run_dir,
        deep=False,
        hash_roles=_QUERY_HASH_ROLES,
        hash_bm25=True,
    )


def query_readiness_stat_key(run_dir: Path) -> tuple[object, ...]:
    """Return the file identity that makes cached ask readiness valid.

    This is deliberately bounded to manifests, terminal states, files loaded at
    query startup, and the selected store sentinels. It does not walk or hash a
    vector database on every question.
    """

    run = run_dir.expanduser().resolve()
    paths: set[Path] = {
        run / "index-state.json",
        run / "artifact-manifest.json",
        run / "run-manifest.json",
        run / "tree.pkl",
        run / "tree.json",
        run / "run_config.yaml",
    }
    manifest_path = run / "artifact-manifest.json"
    try:
        loaded = _cached_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        loaded = None
    if isinstance(loaded, dict):
        attempt_id = loaded.get("attempt_id")
        if _is_attempt_id(attempt_id):
            attempt_dir = run / "attempts" / str(attempt_id)
            paths.update(
                {
                    attempt_dir / "runtime.json",
                    attempt_dir / "state.json",
                    attempt_dir / "artifact-manifest.json",
                }
            )
        roles = loaded.get("roles")
        if isinstance(roles, dict):
            for role in _QUERY_HASH_ROLES:
                value = roles.get(role)
                if isinstance(value, str) and value:
                    paths.add(run / value)

    identity: list[object] = []
    for path in sorted(paths, key=str):
        try:
            key: object = _file_stat_key(path)
        except OSError:
            key = None
        identity.append((str(path), key))
    return tuple(identity)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocARAG artifact provenance helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run", type=Path, required=True)
    publish.add_argument("--revision-id", required=True)
    publish.add_argument("--attempt-id", required=True)
    retrieval = publish.add_mutually_exclusive_group(required=True)
    retrieval.add_argument("--retrieval-attestation", type=Path)
    retrieval.add_argument("--lexical-store", type=Path)
    publish.add_argument("--graph-artifact", type=Path, required=True)
    publish.add_argument("--entity-attestation", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "publish":
        try:
            publish_artifact_manifest(
                args.run,
                revision_id=args.revision_id,
                attempt_id=args.attempt_id,
                retrieval_attestation=args.retrieval_attestation,
                graph_artifact=args.graph_artifact,
                entity_attestation=args.entity_attestation,
                lexical_store=args.lexical_store,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
