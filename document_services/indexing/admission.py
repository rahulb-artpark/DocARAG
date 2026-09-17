"""Admit a PDF: derive its revision identity and create its run directory.

A faithful port of the admission logic in ``Scripts/index_pdf.sh`` (revision
identity derivation, the content-addressed source snapshot, the immutable
run manifest, and index-state.json transitions) so a run created here is
byte-identical in shape to one the bash driver would have created, and the
two remain interchangeable: ``--finalize-run`` and every other bash tool
that reads these files keeps working unchanged.

What this intentionally does not replicate: detached process management
(setsid/nohup/trap-based driver supervision) and the local GPU capacity
queue. A run admitted here is driven synchronously by ``docarag stage``,
which the rest of this session's work already runs standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

STAGE_NAMES = ("tree", "graph", "vdb")
_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileIdentity:
    sha256: str
    size: int
    mtime_ns: int
    modified_at: str


def file_identity(path: Path) -> FileIdentity:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
            size += len(chunk)
            digest.update(chunk)
    stat = path.stat()
    modified = (
        datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return FileIdentity(digest.hexdigest(), size, stat.st_mtime_ns, modified)


def compute_code_fingerprint(repo: Path, controller: Path) -> tuple[str, int]:
    """Hash the indexing code and dependency manifests deterministically.

    Same scope as the bash version -- retrieval_engine/**, document_services/**, main.py, the
    controller script, and root dependency manifests -- so a revision ID
    derived here means the same thing as one bash would have derived for
    identical files.
    """

    repo = repo.resolve()
    controller = controller.resolve()
    files: set[Path] = set()

    for root_name in ("retrieval_engine", "document_services"):
        root = repo / root_name
        if root.is_dir():
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                if "__pycache__" in candidate.parts or candidate.suffix in (
                    ".pyc",
                    ".pyo",
                ):
                    continue
                files.add(candidate.resolve())

    for candidate in (repo / "main.py", controller):
        if candidate.is_file():
            files.add(candidate.resolve())

    dependency_patterns = (
        "pyproject.toml",
        "uv.lock",
        "setup.py",
        "setup.cfg",
        "requirements*.txt",
        "environment*.yaml",
        "environment*.yml",
    )
    for pattern in dependency_patterns:
        for candidate in repo.glob(pattern):
            if candidate.is_file():
                files.add(candidate.resolve())

    fingerprint = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(repo).as_posix()):
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        content_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
                size += len(chunk)
                content_hash.update(chunk)
        fingerprint.update(len(relative).to_bytes(8, "big"))
        fingerprint.update(relative)
        fingerprint.update(size.to_bytes(8, "big"))
        fingerprint.update(content_hash.digest())

    return fingerprint.hexdigest(), len(files)


def atomic_immutable_write(path: Path, payload: bytes, mode: int = 0o400) -> None:
    """Write once; a later write of identical bytes is a no-op, not a race.

    Hardlinking the temp file into place means a concurrent writer racing to
    create the same immutable file either wins outright or discovers the
    winner's bytes already match -- there is no window where a reader sees a
    partially written file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable run input differs from existing file: {path}")
        os.chmod(path, mode)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, mode)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"immutable run input differs from existing file: {path}"
                )
            os.chmod(path, mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def create_source_snapshot(
    source_path: Path, snapshot_path: Path, expected_sha256: str, expected_size: int
) -> None:
    """Materialize the content-addressed, read-only copy every stage reads."""

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        identity = file_identity(snapshot_path)
        if (identity.sha256, identity.size) != (expected_sha256, expected_size):
            raise RuntimeError(f"content-addressed source is corrupt: {snapshot_path}")
        os.chmod(snapshot_path, 0o400)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{snapshot_path.name}.", dir=snapshot_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination, source_path.open("rb") as source:
            while True:
                chunk = source.read(_HASH_CHUNK)
                if not chunk:
                    break
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary_path, 0o400)
        try:
            os.link(temporary_path, snapshot_path)
        except FileExistsError:
            identity = file_identity(snapshot_path)
            if (identity.sha256, identity.size) != (expected_sha256, expected_size):
                raise RuntimeError(
                    f"content-addressed source is corrupt: {snapshot_path}"
                )
            os.chmod(snapshot_path, 0o400)
        directory_fd = os.open(snapshot_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def derive_document_identity(pdf_path: Path, source_sha256: str) -> tuple[str, str, str]:
    """Return (source_uri, document_id, document_version_id)."""

    from document_services.domain.evidence import DocumentRef

    source_uri = pdf_path.resolve().as_uri()
    document = DocumentRef.identified(
        content_sha256=source_sha256,
        mime_type="application/pdf",
        source_uri=source_uri,
        display_name=pdf_path.name,
    )
    return source_uri, document.document_id, document.document_version_id


def derive_revision_id(
    *,
    source_uri: str,
    source_sha256: str,
    config_sha256: str,
    code_sha256: str,
    tenant_id: str,
    corpus_id: str,
    document_version_id: str,
    routed_profile: dict | None = None,
) -> str:
    """The exact schema-2 identity payload bash derives, hashed the same way."""

    payload = {
        "identity_schema_version": 2,
        "source_uri": source_uri,
        "source_sha256": source_sha256,
        "system_config_sha256": config_sha256,
        "controller_code_sha256": code_sha256,
        "tenant_id": tenant_id,
        "corpus_id": corpus_id,
        "document_version_id": document_version_id,
        "routed_profile": routed_profile,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataset_name(pdf_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf_path.stem).strip("_")
    return name or "document"


def _new_attempt_id() -> str:
    """Same shape as bash's: a sortable UTC timestamp plus a uuid4 for entropy."""

    import uuid

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def write_run_manifest(
    *,
    run_dir: Path,
    original_source: Path,
    source_snapshot: Path,
    pdf_identity: FileIdentity,
    system_config: Path,
    config_snapshot: Path,
    config_identity: FileIdentity,
    controller: Path,
    controller_sha256: str,
    code_sha256: str,
    code_file_count: int,
    git_commit: str | None,
    git_dirty: bool,
    revision_id: str,
    document_id: str,
    document_version_id: str,
    tenant_id: str,
    corpus_id: str,
    dataset_name: str,
    dataset_json: Path,
    dataset_config: Path,
    state_path: Path,
) -> None:
    """Write the config snapshot, dataset files and the immutable manifest.

    Idempotent: admitting the same revision twice writes the identical bytes
    a second time, which ``atomic_immutable_write`` accepts as a no-op.
    """

    config_payload = system_config.read_bytes()
    if len(config_payload) != config_identity.size or (
        hashlib.sha256(config_payload).hexdigest() != config_identity.sha256
    ):
        raise RuntimeError(
            "system config changed while the revision was being initialized; "
            "rerun admission"
        )
    atomic_immutable_write(config_snapshot, config_payload, mode=0o400)

    dataset_payload = [
        {
            "question": "",
            "answer": "",
            "doc_uuid": revision_id,
            "doc_path": str(source_snapshot),
            "source_uri": original_source.resolve().as_uri(),
            "source_name": original_source.name,
            "source_sha256": pdf_identity.sha256,
            "tenant_id": tenant_id,
            "corpus_id": corpus_id,
            "document_id": document_id,
            "document_version_id": document_version_id,
            "index_revision_id": revision_id,
        }
    ]
    atomic_immutable_write(
        dataset_json,
        (json.dumps(dataset_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    dataset_config_payload = (
        f"dataset_path: {json.dumps(str(dataset_json), ensure_ascii=False)}\n"
        f"working_dir: {json.dumps(str(run_dir.parent), ensure_ascii=False)}\n"
        f"dataset_name: {json.dumps(dataset_name, ensure_ascii=False)}\n"
    ).encode("utf-8")
    atomic_immutable_write(dataset_config, dataset_config_payload)

    manifest_path = run_dir / "run-manifest.json"
    identity_fields = {
        "schema_version": 1,
        "revision_id": revision_id,
        "document_id": document_id,
        "document_version_id": document_version_id,
        "tenant_id": tenant_id,
        "corpus_id": corpus_id,
        "source_path": str(original_source),
        "source_uri": original_source.resolve().as_uri(),
        "source_sha256": pdf_identity.sha256,
        "system_config_sha256": config_identity.sha256,
        "code_sha256": code_sha256,
        "routed_profile_sha256": None,
        "routed_profile_configuration_sha256": None,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {
            "schema_version": existing.get("schema_version"),
            "revision_id": existing.get("revision_id"),
            "document_id": existing.get("document_id"),
            "document_version_id": existing.get("document_version_id"),
            "tenant_id": existing.get("tenant_id"),
            "corpus_id": existing.get("corpus_id"),
            "source_path": existing.get("source", {}).get("path"),
            "source_uri": existing.get("source", {}).get("uri"),
            "source_sha256": existing.get("source", {}).get("sha256"),
            "system_config_sha256": existing.get("system_config", {}).get("sha256"),
            "code_sha256": existing.get("controller", {})
            .get("code_fingerprint", {})
            .get("sha256"),
            "routed_profile_sha256": (existing.get("routed_profile") or {}).get(
                "source_sha256"
            ),
            "routed_profile_configuration_sha256": (
                existing.get("routed_profile") or {}
            ).get("configuration_sha256"),
        }
        if actual != identity_fields:
            raise RuntimeError(f"immutable run manifest identity mismatch: {manifest_path}")
        return

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "revision_id": revision_id,
        "document_id": document_id,
        "document_version_id": document_version_id,
        "tenant_id": tenant_id,
        "corpus_id": corpus_id,
        "created_at": now,
        "source": {
            "path": str(original_source),
            "uri": original_source.resolve().as_uri(),
            "size_bytes": pdf_identity.size,
            "sha256": pdf_identity.sha256,
            "modified_at": pdf_identity.modified_at,
            "snapshot_path": str(source_snapshot),
        },
        "system_config": {
            "path": str(system_config),
            "size_bytes": config_identity.size,
            "sha256": config_identity.sha256,
            "modified_at": config_identity.modified_at,
            "snapshot_path": str(config_snapshot),
        },
        "controller": {
            "path": str(controller),
            "sha256": controller_sha256,
            "code_fingerprint": {
                "algorithm": "sha256",
                "sha256": code_sha256,
                "file_count": code_file_count,
                "scope": [
                    "main.py",
                    "retrieval_engine/**",
                    "document_services/**",
                    "Scripts/index_pdf.sh",
                    "root dependency manifests",
                ],
            },
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "routed_profile": None,
        "inputs": {
            "dataset_json": str(dataset_json),
            "dataset_config": str(dataset_config),
        },
        "outputs": {
            "run_directory": str(run_dir),
            "index_state": str(state_path),
        },
        "stages": list(STAGE_NAMES),
    }
    atomic_immutable_write(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _pending_stage() -> dict:
    return {"status": "pending", "started_at": None, "ended_at": None, "exit_code": None, "error": None}


def update_index_state(
    state_path: Path,
    revision_id: str,
    pid: int,
    event: str,
    *,
    stage: str | None = None,
    exit_code: int | None = None,
    error: str | None = None,
    attempt_id: str | None = None,
) -> None:
    """Faithful port of bash's ``update_index_state``: same events, same shape.

    Kept byte-compatible with the bash version so ``--finalize-run`` and every
    other tool that reads index-state.json keeps working unchanged regardless
    of whether a run was admitted and driven by bash or by this module.
    """

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if event == "initialize":
        attempt_number = 1
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            attempt_number = int(previous.get("attempt_number") or 0) + 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        state = {
            "schema_version": 1,
            "revision_id": revision_id,
            "pid": pid,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "overall_status": "running",
            "phase": "initializing",
            "current_stage": None,
            "started_at": now,
            "ended_at": None,
            "updated_at": now,
            "exit_code": None,
            "error": None,
            "stages": {name: _pending_stage() for name in STAGE_NAMES},
        }
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("revision_id") != revision_id:
            raise RuntimeError("index state identity does not match this run")
        if event == "reconcile_abandoned":
            state["reconciled_by_pid"] = pid
        else:
            state["pid"] = pid
        state["updated_at"] = now

        if event == "stage_started":
            if stage not in STAGE_NAMES:
                raise RuntimeError(f"unknown index stage: {stage}")
            state["overall_status"] = "running"
            state["phase"] = "indexing"
            state["current_stage"] = stage
            state["stages"][stage] = {
                "status": "running",
                "started_at": now,
                "ended_at": None,
                "exit_code": None,
                "error": None,
            }
        elif event == "stage_succeeded":
            if stage not in STAGE_NAMES:
                raise RuntimeError(f"unknown index stage: {stage}")
            item = state["stages"][stage]
            if state.get("current_stage") != stage or item.get("status") != "running":
                raise RuntimeError(f"cannot succeed stage that is not running: {stage}")
            item.update({"status": "succeeded", "ended_at": now, "exit_code": 0, "error": None})
            state["current_stage"] = None
        elif event == "stage_failed":
            if stage not in STAGE_NAMES:
                raise RuntimeError(f"unknown index stage: {stage}")
            state["stages"][stage].update(
                {"status": "failed", "ended_at": now, "exit_code": exit_code, "error": error}
            )
            state.update(
                {
                    "overall_status": "failed",
                    "phase": "finished",
                    "current_stage": stage,
                    "ended_at": now,
                    "exit_code": exit_code,
                    "error": error,
                }
            )
        elif event == "run_succeeded":
            if not all(
                isinstance(state["stages"].get(name), dict)
                and state["stages"][name].get("status") == "succeeded"
                for name in STAGE_NAMES
            ):
                raise RuntimeError("cannot publish a run before every index stage succeeds")
            artifact_manifest = state_path.parent / "artifact-manifest.json"
            if not artifact_manifest.is_file() or artifact_manifest.stat().st_size == 0:
                raise RuntimeError("cannot publish a run without artifact-manifest.json")
            artifact_manifest_bytes = artifact_manifest.read_bytes()
            artifact_payload = json.loads(artifact_manifest_bytes)
            if not isinstance(artifact_payload, dict):
                raise RuntimeError("cannot bind a non-object artifact manifest")
            if artifact_payload.get("schema_version") != 2:
                raise RuntimeError("cannot bind an unsupported artifact manifest")
            if artifact_payload.get("revision_id") != revision_id:
                raise RuntimeError("artifact manifest revision does not match index state")
            if artifact_payload.get("attempt_id") != attempt_id:
                raise RuntimeError("artifact manifest attempt does not match index state")
            artifact_set_sha256 = artifact_payload.get("artifact_set_sha256")
            if not isinstance(artifact_set_sha256, str) or len(artifact_set_sha256) != 64:
                raise RuntimeError("artifact manifest has no valid artifact set identity")
            state.update(
                {
                    "overall_status": "succeeded",
                    "phase": "finished",
                    "current_stage": None,
                    "ended_at": now,
                    "exit_code": 0,
                    "error": None,
                    "artifact_manifest": "artifact-manifest.json",
                    "artifact_manifest_sha256": hashlib.sha256(
                        artifact_manifest_bytes
                    ).hexdigest(),
                    "artifact_set_sha256": artifact_set_sha256,
                }
            )
        elif event == "run_failed":
            if stage in STAGE_NAMES and state["stages"][stage]["status"] == "running":
                state["stages"][stage].update(
                    {"status": "failed", "ended_at": now, "exit_code": exit_code, "error": error}
                )
            state.update(
                {
                    "overall_status": "failed",
                    "phase": "finished",
                    "current_stage": stage,
                    "ended_at": now,
                    "exit_code": exit_code,
                    "error": error,
                }
            )
        elif event == "waiting_for_capacity":
            state.update({"overall_status": "running", "phase": "waiting_for_capacity", "current_stage": None})
        elif event == "capacity_acquired":
            state.update({"overall_status": "running", "phase": "indexing", "current_stage": None})
        elif event == "reconcile_abandoned":
            if state.get("overall_status") != "running":
                raise RuntimeError("only a running attempt can be reconciled as abandoned")
            interrupted_stage = state.get("current_stage")
            if interrupted_stage in STAGE_NAMES:
                item = state["stages"].get(interrupted_stage)
                if isinstance(item, dict) and item.get("status") == "running":
                    item.update(
                        {"status": "failed", "ended_at": now, "exit_code": 143, "error": error}
                    )
            state.update(
                {
                    "overall_status": "failed",
                    "phase": "finished",
                    "current_stage": interrupted_stage,
                    "ended_at": now,
                    "exit_code": 143,
                    "error": error,
                    "termination_reason": "driver_lock_lost",
                }
            )
        else:
            raise RuntimeError(f"unknown state event: {event}")

    def atomic_state_write(destination: Path, mode: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".index-state.", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(state, output, indent=2, ensure_ascii=False, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    terminal = state.get("overall_status") in {"succeeded", "failed"}
    if attempt_id:
        attempt_state = state_path.parent / "attempts" / attempt_id / "state.json"
        atomic_state_write(attempt_state, 0o400 if terminal else 0o600)
    atomic_state_write(state_path, 0o600)


def admit(
    pdf_path: str | Path,
    system_config: str | Path,
    *,
    tenant_id: str = "default",
    corpus_id: str = "default",
    runs_dir: str | Path | None = None,
) -> Path:
    """Admit a PDF and return its run directory, creating it if new.

    Idempotent by construction: admitting the same PDF under the same
    config and code produces the same revision_id, and every write here
    goes through ``atomic_immutable_write`` -- a second admission finds its
    own bytes already in place and proceeds straight to returning the path.
    """

    repo = Path(__file__).resolve().parents[2]
    controller = repo / "Scripts" / "index_pdf.sh"

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise RuntimeError(f"PDF not found: {pdf}")
    config_path = Path(system_config).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"system config not found: {config_path}")

    pdf_identity = file_identity(pdf)
    config_identity = file_identity(config_path)
    code_sha256, code_file_count = compute_code_fingerprint(repo, controller)
    controller_sha256 = file_identity(controller).sha256 if controller.is_file() else ""

    source_uri, document_id, document_version_id = derive_document_identity(
        pdf, pdf_identity.sha256
    )
    revision_id = derive_revision_id(
        source_uri=source_uri,
        source_sha256=pdf_identity.sha256,
        config_sha256=config_identity.sha256,
        code_sha256=code_sha256,
        tenant_id=tenant_id,
        corpus_id=corpus_id,
        document_version_id=document_version_id,
    )

    workdir = Path(runs_dir).expanduser().resolve() if runs_dir else repo / "runs"
    run_dir = workdir / document_id / revision_id
    input_dir = run_dir / "inputs"
    source_store = workdir / "_sources" / "sha256"
    source_snapshot = source_store / f"{pdf_identity.sha256}.pdf"
    dataset_json = input_dir / f"dataset-{document_id}-{revision_id}.json"
    dataset_config = input_dir / f"dataset-{document_id}-{revision_id}.yaml"
    config_snapshot = input_dir / f"system-config-{config_identity.sha256}.yaml"
    state_path = run_dir / "index-state.json"

    input_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    source_store.mkdir(parents=True, exist_ok=True)
    create_source_snapshot(pdf, source_snapshot, pdf_identity.sha256, pdf_identity.size)

    try:
        git_commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        git_commit = None
    try:
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no",
             "--", "main.py", "retrieval_engine", "document_services", "Scripts/index_pdf.sh"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        git_dirty = bool(dirty)
    except (subprocess.CalledProcessError, OSError):
        git_dirty = False

    write_run_manifest(
        run_dir=run_dir,
        original_source=pdf,
        source_snapshot=source_snapshot,
        pdf_identity=pdf_identity,
        system_config=config_path,
        config_snapshot=config_snapshot,
        config_identity=config_identity,
        controller=controller,
        controller_sha256=controller_sha256,
        code_sha256=code_sha256,
        code_file_count=code_file_count,
        git_commit=git_commit,
        git_dirty=git_dirty,
        revision_id=revision_id,
        document_id=document_id,
        document_version_id=document_version_id,
        tenant_id=tenant_id,
        corpus_id=corpus_id,
        dataset_name=f"{_dataset_name(pdf)}-{revision_id[:12]}",
        dataset_json=dataset_json,
        dataset_config=dataset_config,
        state_path=state_path,
    )

    if not state_path.exists():
        attempt_id = _new_attempt_id()
        update_index_state(
            state_path, revision_id, os.getpid(), "initialize", attempt_id=attempt_id
        )

    return run_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Admit a PDF and create its run directory")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--corpus-id", default="default")
    parser.add_argument("--runs-dir")
    args = parser.parse_args(argv)
    try:
        run_dir = admit(
            args.pdf, args.config,
            tenant_id=args.tenant_id, corpus_id=args.corpus_id, runs_dir=args.runs_dir,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
