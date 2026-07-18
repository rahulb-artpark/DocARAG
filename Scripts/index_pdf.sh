#!/usr/bin/env bash
# "Upload" a PDF: build a BookRAG index (tree -> graph -> vdb) for any PDF,
# then view it with Scripts/view.sh.
#
#   bash Scripts/index_pdf.sh /abs/path/to/document.pdf
#   bash Scripts/index_pdf.sh                # prompts for the path
#
# The index runs DETACHED (survives closing the terminal). Tree is usually
# ready in a few minutes; the knowledge graph can take 1-2h on a local LLM.
# Override the defaults with DOCARAG_CONFIG, DOCARAG_RUNS_DIR,
# DOCARAG_PYTHON, or DOCARAG_ENV.
set -euo pipefail
umask 077

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$0")/.."
REPO="$(pwd)"

file_identity() {
  python3 - "$1" <<'PY'
from datetime import datetime, timezone
import hashlib
import os
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
stat = os.stat(path)
modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
print(digest.hexdigest(), stat.st_size, stat.st_mtime_ns, modified)
PY
}

compute_code_fingerprint() {
  local repo="$1"
  local controller="$2"

  python3 - "$repo" "$controller" <<'PY'
import hashlib
from pathlib import Path
import json
import sys

repo = Path(sys.argv[1]).resolve()
controller = Path(sys.argv[2]).resolve()
files = set()

for root_name in ("Core", "docarag"):
    root = repo / root_name
    if root.is_dir():
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
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
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            content_hash.update(chunk)
    fingerprint.update(len(relative).to_bytes(8, "big"))
    fingerprint.update(relative)
    fingerprint.update(size.to_bytes(8, "big"))
    fingerprint.update(content_hash.digest())

print(fingerprint.hexdigest(), len(files))
PY
}

create_source_snapshot() {
  local source_path="$1"
  local snapshot_path="$2"
  local expected_sha256="$3"
  local expected_size="$4"

  python3 - "$source_path" "$snapshot_path" "$expected_sha256" "$expected_size" <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

source_path = Path(sys.argv[1])
snapshot_path = Path(sys.argv[2])
expected_sha256 = sys.argv[3]
expected_size = int(sys.argv[4])
snapshot_path.parent.mkdir(parents=True, exist_ok=True)

def identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size

if snapshot_path.exists():
    actual = identity(snapshot_path)
    if actual != (expected_sha256, expected_size):
        raise SystemExit(f"content-addressed source is corrupt: {snapshot_path}")
    os.chmod(snapshot_path, 0o400)
    raise SystemExit(0)

descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{snapshot_path.name}.", dir=snapshot_path.parent
)
temporary_path = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as destination, source_path.open("rb") as source:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    actual = identity(temporary_path)
    if actual != (expected_sha256, expected_size):
        raise SystemExit(
            "source changed while its immutable snapshot was being created; rerun indexing"
        )
    os.chmod(temporary_path, 0o400)
    try:
        os.link(temporary_path, snapshot_path)
    except FileExistsError:
        if identity(snapshot_path) != (expected_sha256, expected_size):
            raise SystemExit(f"content-addressed source is corrupt: {snapshot_path}")
    directory_fd = os.open(snapshot_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary_path.unlink(missing_ok=True)
PY
}

write_run_inputs() {
  python3 - "$@" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

(
    run_dir_raw,
    original_source_raw,
    source_snapshot_raw,
    pdf_sha256,
    pdf_size_raw,
    pdf_modified_at,
    system_config_raw,
    config_snapshot_raw,
    config_sha256,
    config_size_raw,
    config_modified_at,
    code_sha256,
    code_file_count_raw,
    controller_raw,
    controller_sha256,
    git_commit,
    git_dirty_raw,
    revision_id,
    document_id,
    document_version_id,
    tenant_id,
    corpus_id,
    dataset_name,
    dataset_json_raw,
    dataset_config_raw,
    state_path_raw,
    dataset_working_dir_raw,
    routed_profile_raw,
    routed_profile_snapshot_raw,
    routed_profile_id,
    routed_profile_sha256,
    routed_profile_configuration_sha256,
    routed_profile_size_raw,
    routed_profile_modified_at,
) = sys.argv[1:]

run_dir = Path(run_dir_raw)
original_source = Path(original_source_raw)
source_snapshot = Path(source_snapshot_raw)
system_config = Path(system_config_raw)
config_snapshot = Path(config_snapshot_raw)
controller = Path(controller_raw)
dataset_json = Path(dataset_json_raw)
dataset_config = Path(dataset_config_raw)
state_path = Path(state_path_raw)
routed_profile = Path(routed_profile_raw) if routed_profile_raw else None
routed_profile_snapshot = (
    Path(routed_profile_snapshot_raw) if routed_profile_snapshot_raw else None
)
pdf_size = int(pdf_size_raw)
config_size = int(config_size_raw)
code_file_count = int(code_file_count_raw)
git_dirty = git_dirty_raw == "true"
routed_profile_size = int(routed_profile_size_raw) if routed_profile_size_raw else None

def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def atomic_immutable_write(path, payload, mode=0o400):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"immutable run input differs from existing file: {path}")
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
                raise SystemExit(f"immutable run input differs from existing file: {path}")
            os.chmod(path, mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)

config_payload = system_config.read_bytes()
if len(config_payload) != config_size or sha256_bytes(config_payload) != config_sha256:
    raise SystemExit("system config changed while the revision was being initialized; rerun indexing")
atomic_immutable_write(config_snapshot, config_payload, mode=0o400)

dataset_payload = [
    {
        "question": "",
        "answer": "",
        "doc_uuid": revision_id,
        "doc_path": str(source_snapshot),
        "source_uri": original_source.resolve().as_uri(),
        "source_name": original_source.name,
        "source_sha256": pdf_sha256,
        "tenant_id": tenant_id,
        "corpus_id": corpus_id,
        "document_id": document_id,
        "document_version_id": document_version_id,
        "index_revision_id": revision_id,
    }
]
dataset_json_payload = (
    json.dumps(dataset_payload, indent=2, ensure_ascii=False) + "\n"
).encode("utf-8")
atomic_immutable_write(dataset_json, dataset_json_payload)

# JSON-quoted strings are valid YAML scalars and safely preserve spaces, quotes,
# colons, hashes, and non-ASCII characters in generated paths and names.
dataset_config_payload = (
    f"dataset_path: {json.dumps(str(dataset_json), ensure_ascii=False)}\n"
    f"working_dir: {json.dumps(dataset_working_dir_raw, ensure_ascii=False)}\n"
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
    "source_sha256": pdf_sha256,
    "system_config_sha256": config_sha256,
    "code_sha256": code_sha256,
    "routed_profile_sha256": routed_profile_sha256 or None,
    "routed_profile_configuration_sha256": (
        routed_profile_configuration_sha256 or None
    ),
}

if manifest_path.exists():
    with manifest_path.open(encoding="utf-8") as manifest_file:
        existing = json.load(manifest_file)
    actual_identity = {
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
        "code_sha256": existing.get("controller", {}).get("code_fingerprint", {}).get("sha256"),
        "routed_profile_sha256": (
            (existing.get("routed_profile") or {}).get("source_sha256")
        ),
        "routed_profile_configuration_sha256": (
            (existing.get("routed_profile") or {}).get("configuration_sha256")
        ),
    }
    if actual_identity != identity_fields:
        raise SystemExit(f"immutable run manifest identity mismatch: {manifest_path}")
else:
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
            "size_bytes": pdf_size,
            "sha256": pdf_sha256,
            "modified_at": pdf_modified_at,
            "snapshot_path": str(source_snapshot),
        },
        "system_config": {
            "path": str(system_config),
            "size_bytes": config_size,
            "sha256": config_sha256,
            "modified_at": config_modified_at,
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
                    "Core/**",
                    "docarag/**",
                    "Scripts/index_pdf.sh",
                    "root dependency manifests",
                ],
            },
            "git_commit": None if git_commit == "unknown" else git_commit,
            "git_dirty": git_dirty,
        },
        "routed_profile": (
            None
            if routed_profile is None
            else {
                "profile_id": routed_profile_id,
                "path": str(routed_profile),
                "snapshot_path": str(routed_profile_snapshot),
                "size_bytes": routed_profile_size,
                "source_sha256": routed_profile_sha256,
                "configuration_sha256": routed_profile_configuration_sha256,
                "modified_at": routed_profile_modified_at,
            }
        ),
        "inputs": {
            "dataset_json": str(dataset_json),
            "dataset_config": str(dataset_config),
            **(
                {}
                if routed_profile_snapshot is None
                else {"routed_profile": str(routed_profile_snapshot)}
            ),
        },
        "outputs": {
            "run_directory": str(run_dir),
            "index_state": str(state_path),
        },
        "stages": ["tree", "graph", "vdb"],
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_immutable_write(manifest_path, manifest_payload)
PY
}

update_index_state() {
  local state_path="$1"
  local revision_id="$2"
  local pid="$3"
  local event="$4"
  local stage="${5:-}"
  local exit_code="${6:-}"
  local error="${7:-}"
  local attempt_id="${DRIVER_ATTEMPT_ID:-}"

  python3 - "$state_path" "$revision_id" "$pid" "$event" "$stage" "$exit_code" "$error" "$attempt_id" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

state_path = Path(sys.argv[1])
revision_id = sys.argv[2]
pid = int(sys.argv[3])
event = sys.argv[4]
stage = sys.argv[5] or None
exit_code = int(sys.argv[6]) if sys.argv[6] else None
error = sys.argv[7] or None
attempt_id = sys.argv[8] or None
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
stage_names = ("tree", "graph", "vdb")

def pending_stage():
    return {
        "status": "pending",
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "error": None,
    }

if event == "initialize":
    attempt_number = 1
    try:
        with state_path.open(encoding="utf-8") as previous_file:
            previous = json.load(previous_file)
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
        "stages": {name: pending_stage() for name in stage_names},
    }
else:
    with state_path.open(encoding="utf-8") as state_file:
        state = json.load(state_file)
    if state.get("schema_version") != 1 or state.get("revision_id") != revision_id:
        raise SystemExit("index state identity does not match this run")
    if event == "reconcile_abandoned":
        state["reconciled_by_pid"] = pid
    else:
        state["pid"] = pid
    state["updated_at"] = now

    if event == "stage_started":
        if stage not in stage_names:
            raise SystemExit(f"unknown index stage: {stage}")
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
        if stage not in stage_names:
            raise SystemExit(f"unknown index stage: {stage}")
        item = state["stages"][stage]
        if state.get("current_stage") != stage or item.get("status") != "running":
            raise SystemExit(f"cannot succeed stage that is not running: {stage}")
        item.update({"status": "succeeded", "ended_at": now, "exit_code": 0, "error": None})
        state["current_stage"] = None
    elif event == "stage_failed":
        if stage not in stage_names:
            raise SystemExit(f"unknown index stage: {stage}")
        item = state["stages"][stage]
        item.update({"status": "failed", "ended_at": now, "exit_code": exit_code, "error": error})
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
            for name in stage_names
        ):
            raise SystemExit("cannot publish a run before every index stage succeeds")
        artifact_manifest = state_path.parent / "artifact-manifest.json"
        if not artifact_manifest.is_file() or artifact_manifest.stat().st_size == 0:
            raise SystemExit("cannot publish a run without artifact-manifest.json")
        try:
            artifact_manifest_bytes = artifact_manifest.read_bytes()
            artifact_payload = json.loads(artifact_manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot bind an invalid artifact manifest: {exc}")
        if not isinstance(artifact_payload, dict):
            raise SystemExit("cannot bind a non-object artifact manifest")
        if artifact_payload.get("schema_version") != 2:
            raise SystemExit("cannot bind an unsupported artifact manifest")
        if artifact_payload.get("revision_id") != revision_id:
            raise SystemExit("artifact manifest revision does not match index state")
        if artifact_payload.get("attempt_id") != attempt_id:
            raise SystemExit("artifact manifest attempt does not match index state")
        artifact_set_sha256 = artifact_payload.get("artifact_set_sha256")
        if not isinstance(artifact_set_sha256, str) or len(artifact_set_sha256) != 64:
            raise SystemExit("artifact manifest has no valid artifact set identity")
        artifact_manifest_sha256 = hashlib.sha256(artifact_manifest_bytes).hexdigest()
        state.update(
            {
                "overall_status": "succeeded",
                "phase": "finished",
                "current_stage": None,
                "ended_at": now,
                "exit_code": 0,
                "error": None,
                "artifact_manifest": "artifact-manifest.json",
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "artifact_set_sha256": artifact_set_sha256,
            }
        )
    elif event == "run_failed":
        if stage in stage_names and state["stages"][stage]["status"] == "running":
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
        state["overall_status"] = "running"
        state["phase"] = "waiting_for_capacity"
        state["current_stage"] = None
    elif event == "capacity_acquired":
        state["overall_status"] = "running"
        state["phase"] = "indexing"
        state["current_stage"] = None
    elif event == "reconcile_abandoned":
        if state.get("overall_status") != "running":
            raise SystemExit("only a running attempt can be reconciled as abandoned")
        interrupted_stage = state.get("current_stage")
        if interrupted_stage in stage_names:
            item = state["stages"].get(interrupted_stage)
            if isinstance(item, dict) and item.get("status") == "running":
                item.update(
                    {
                        "status": "failed",
                        "ended_at": now,
                        "exit_code": 143,
                        "error": error,
                    }
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
        raise SystemExit(f"unknown state event: {event}")

def atomic_state_write(destination, mode):
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


# Attempt-local state is the durable history; the run-root file is the latest
# projection used by status/readiness tools. Publishing the attempt copy first
# prevents a hard crash from losing the newest transition from history.
terminal = state.get("overall_status") in {"succeeded", "failed"}
if attempt_id:
    attempt_state = state_path.parent / "attempts" / attempt_id / "state.json"
    atomic_state_write(attempt_state, 0o400 if terminal else 0o600)
atomic_state_write(state_path, 0o600)
PY
}

verify_revision_inputs() {
  local repo="$1"
  local controller="$2"
  local source_snapshot="$3"
  local system_config="$4"
  local expected_pdf_sha="$5"
  local expected_config_sha="$6"
  local expected_code_sha="$7"
  local routed_profile_snapshot="${8:-}"
  local expected_routed_profile_sha="${9:-}"
  local actual_pdf
  local actual_config
  local actual_code
  local actual_routed_profile
  local unused

  # Empty expected hashes are supported only for compatibility with direct use
  # of the old, internal --detached-driver form.
  if [[ -z "$expected_pdf_sha" || -z "$expected_config_sha" || -z "$expected_code_sha" ]]; then
    return 0
  fi

  read -r actual_pdf unused unused unused < <(file_identity "$source_snapshot")
  read -r actual_config unused unused unused < <(file_identity "$system_config")
  read -r actual_code unused < <(compute_code_fingerprint "$repo" "$controller")

  [[ "$actual_pdf" == "$expected_pdf_sha" ]] || {
    echo "[driver] immutable PDF snapshot hash changed: $source_snapshot" >&2
    return 1
  }
  [[ "$actual_config" == "$expected_config_sha" ]] || {
    echo "[driver] immutable system config snapshot hash changed: $system_config" >&2
    return 1
  }
  [[ "$actual_code" == "$expected_code_sha" ]] || {
    echo "[driver] controller/code fingerprint changed after this revision was created" >&2
    return 1
  }
  if [[ -n "$expected_routed_profile_sha" ]]; then
    [[ -n "$routed_profile_snapshot" && -f "$routed_profile_snapshot" ]] || {
      echo "[driver] immutable routed profile snapshot is missing" >&2
      return 1
    }
    read -r actual_routed_profile unused unused unused < <(file_identity "$routed_profile_snapshot")
    [[ "$actual_routed_profile" == "$expected_routed_profile_sha" ]] || {
      echo "[driver] immutable routed profile snapshot hash changed: $routed_profile_snapshot" >&2
      return 1
    }
  fi
}

completed_run_is_valid() {
  local run_dir="$1"
  python3 - "$run_dir" <<'PY'
from pathlib import Path
import json
import sys

from docarag.compat import inspect_run_readiness
from docarag.provenance import verify_artifact_manifest

run_dir = Path(sys.argv[1]).resolve()
config = run_dir / "run_config.yaml"
state_path = run_dir / "index-state.json"
try:
    state = json.loads(state_path.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
status = state.get("overall_status") or state.get("status")
stages = state.get("stages") or {}
terminal_success = status == "succeeded" and all(
    isinstance(stages.get(stage), dict)
    and stages[stage].get("status") == "succeeded"
    for stage in ("tree", "graph", "vdb")
)
readiness = inspect_run_readiness(run_dir, purpose="ask")
artifacts = verify_artifact_manifest(run_dir, deep=True)
raise SystemExit(
    0
    if terminal_success
    and readiness.ready
    and artifacts.valid
    and config.is_file()
    and config.stat().st_size
    else 1
)
PY
}

DRIVER_FINALIZED=1
DRIVER_STATE_FILE=""
DRIVER_REVISION_ID=""
DRIVER_ACTIVE_STAGE=""
DRIVER_ATTEMPT_ID=""
DRIVER_ATTEMPT_DIR=""
DRIVER_CHILD_PID=""
DRIVER_CHILD_OPERATION=""
DRIVER_ABORT_REQUESTED=0
DRIVER_FAILURE_CONTEXT=""

archive_attempt_state() {
  local state_path="$1"
  local attempt_dir="$2"
  [[ -n "$attempt_dir" && -s "$state_path" ]] || return 0
  python3 - "$state_path" "$attempt_dir/state.json" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import tempfile

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() and destination.read_bytes() == source.read_bytes():
    os.chmod(destination, 0o400)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    raise SystemExit(0)
descriptor, name = tempfile.mkstemp(prefix=".state.", dir=destination.parent)
temporary = Path(name)
try:
    with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
        shutil.copyfileobj(input_file, output)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o400)
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
PY
}

archive_previous_attempt_state() {
  local state_path="$1"
  local run_dir="$2"
  local previous_attempt_id

  [[ -s "$state_path" ]] || return 0
  previous_attempt_id="$(python3 - "$state_path" <<'PY'
import json
import re
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as state_file:
        state = json.load(state_file)
except (OSError, ValueError, TypeError):
    raise SystemExit(0)

attempt_id = state.get("attempt_id")
if isinstance(attempt_id, str) and re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id):
    if attempt_id not in {".", ".."}:
        print(attempt_id)
PY
  )"
  if [[ -n "$previous_attempt_id" ]]; then
    if [[ -s "$run_dir/attempts/$previous_attempt_id/state.json" ]]; then
      chmod 0400 "$run_dir/attempts/$previous_attempt_id/state.json"
    else
      archive_attempt_state "$state_path" "$run_dir/attempts/$previous_attempt_id" || true
    fi
  fi
}

reconcile_previous_attempt_state() {
  local state_path="$1"
  local revision_id="$2"
  local previous_attempt_id
  [[ -s "$state_path" ]] || return 0
  if previous_attempt_id="$(python3 - "$state_path" <<'PY'
import json
import re
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as state_file:
        state = json.load(state_file)
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
if state.get("overall_status") != "running":
    raise SystemExit(1)
attempt_id = state.get("attempt_id")
if isinstance(attempt_id, str) and re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id):
    if attempt_id not in {".", ".."}:
        print(attempt_id)
PY
)"; then
    # update_index_state mirrors through DRIVER_ATTEMPT_ID. Point it at the
    # stale attempt while reconciling, before allocating the replacement ID.
    DRIVER_ATTEMPT_ID="$previous_attempt_id"
    update_index_state \
      "$state_path" "$revision_id" "$$" reconcile_abandoned "" 143 \
      "previous indexing driver no longer owns the revision lock; attempt was abandoned/interrupted"
    DRIVER_ATTEMPT_ID=""
  fi
}

run_tracked() {
  local operation="$1"
  shift
  local status
  if [[ "$DRIVER_ABORT_REQUESTED" -ne 0 ]]; then
    return 143
  fi
  DRIVER_CHILD_OPERATION="$operation"
  setsid --wait "$@" &
  DRIVER_CHILD_PID=$!
  if wait "$DRIVER_CHILD_PID"; then
    status=0
  else
    status=$?
  fi
  DRIVER_CHILD_PID=""
  DRIVER_CHILD_OPERATION=""
  return "$status"
}

driver_exit_trap() {
  local status=$?
  local failure_status="$status"
  local failure_error
  trap - EXIT INT TERM
  if [[ "$DRIVER_FINALIZED" -eq 0 && -n "$DRIVER_STATE_FILE" ]]; then
    if [[ "$failure_status" -eq 0 ]]; then
      failure_status=1
    fi
    failure_error="${DRIVER_FAILURE_CONTEXT:-index driver exited unexpectedly with status $failure_status}"
    update_index_state \
      "$DRIVER_STATE_FILE" "$DRIVER_REVISION_ID" "$$" run_failed \
      "$DRIVER_ACTIVE_STAGE" "$failure_status" \
      "$failure_error" || true
    archive_attempt_state "$DRIVER_STATE_FILE" "$DRIVER_ATTEMPT_DIR" || true
  fi
}

driver_signal_trap() {
  local signal_name="$1"
  local exit_code="$2"
  local grace_seconds="${DOCARAG_TERM_GRACE_SECONDS:-3}"
  local remaining
  DRIVER_ABORT_REQUESTED=1
  DRIVER_FAILURE_CONTEXT="index driver received $signal_name during ${DRIVER_CHILD_OPERATION:-controller work}"
  trap - INT TERM
  if [[ ! "$grace_seconds" =~ ^[0-9]+$ ]]; then
    grace_seconds=3
  elif (( grace_seconds > 60 )); then
    grace_seconds=60
  fi
  child_group_alive() {
    kill -0 -- "-$DRIVER_CHILD_PID" 2>/dev/null \
      || kill -0 "$DRIVER_CHILD_PID" 2>/dev/null
  }
  if [[ -n "$DRIVER_CHILD_PID" ]] && child_group_alive; then
    kill -s "$signal_name" -- "-$DRIVER_CHILD_PID" 2>/dev/null \
      || kill -s "$signal_name" "$DRIVER_CHILD_PID" 2>/dev/null \
      || true
    remaining=$((grace_seconds * 10))
    while (( remaining > 0 )) && child_group_alive; do
      sleep 0.1
      remaining=$((remaining - 1))
    done
    if child_group_alive; then
      kill -KILL -- "-$DRIVER_CHILD_PID" 2>/dev/null \
        || kill -KILL "$DRIVER_CHILD_PID" 2>/dev/null \
        || true
    fi
    wait "$DRIVER_CHILD_PID" 2>/dev/null || true
  fi
  DRIVER_CHILD_PID=""
  exit "$exit_code"
}

run_detached_driver() {
  local repo="$1"
  local dataset_config="$2"
  local run_dir="$3"
  local system_config="$4"
  local revision_id="${5:-$(basename "$run_dir")}"
  local expected_pdf_sha="${6:-}"
  local expected_config_sha="${7:-}"
  local expected_code_sha="${8:-}"
  local source_snapshot="${9:-}"
  local inherited_lock_fd="${10:-}"
  local routed_profile_snapshot="${11:-}"
  local expected_routed_profile_sha="${12:-}"
  local controller="$repo/Scripts/index_pdf.sh"
  local state_file="$run_dir/index-state.json"
  local expected_retrieval_artifact
  local retrieval_index_kind
  local index_type
  local stage
  local stage_marker
  local status
  local error
  local lock_fd
  local admission_lock_fd
  local admission_root
  local attempt_log
  local role_manifest
  local expected_graph_artifact
  local expected_entity_attestation
  local -a resolved_roles
  local published_graph_artifact=""
  local published_entity_attestation=""
  local -a python_runner
  local -a publication_retrieval_args

  mkdir -p "$run_dir/logs"
  if [[ -n "$inherited_lock_fd" ]]; then
    if [[ ! "$inherited_lock_fd" =~ ^[0-9]+$ ]] || ! flock -n "$inherited_lock_fd"; then
      echo "[driver] another index driver owns this revision lock: $run_dir/.index.lock" >&2
      return 75
    fi
    lock_fd="$inherited_lock_fd"
  else
    exec {lock_fd}> "$run_dir/.index.lock"
    if ! flock -n "$lock_fd"; then
      echo "[driver] another index driver owns this revision lock: $run_dir/.index.lock" >&2
      return 75
    fi
  fi

  cd "$repo"
  DRIVER_STATE_FILE="$state_file"
  DRIVER_REVISION_ID="$revision_id"
  DRIVER_ACTIVE_STAGE=""
  reconcile_previous_attempt_state "$state_file" "$revision_id"
  archive_previous_attempt_state "$state_file" "$run_dir"
  DRIVER_ATTEMPT_ID="$(python3 - <<'PY'
from datetime import datetime, timezone
import uuid

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
print(f"{timestamp}-{uuid.uuid4().hex}")
PY
)"
  DRIVER_ATTEMPT_DIR="$run_dir/attempts/$DRIVER_ATTEMPT_ID"
  mkdir -p "$run_dir/attempts"
  mkdir "$DRIVER_ATTEMPT_DIR"
  DRIVER_FINALIZED=0
  trap driver_exit_trap EXIT
  trap 'driver_signal_trap INT 130' INT
  trap 'driver_signal_trap TERM 143' TERM

  update_index_state "$state_file" "$revision_id" "$$" initialize

  if [[ "${DOCARAG_INDEX_ADMISSION:-gpu-lock}" != "none" ]]; then
    admission_root="${DOCARAG_RUNS_DIR:-$(dirname "$(dirname "$run_dir")")}"
    mkdir -p "$admission_root"
    exec {admission_lock_fd}> "$admission_root/.gpu-index.lock"
    update_index_state "$state_file" "$revision_id" "$$" waiting_for_capacity
    echo "[driver] $(date) waiting for the DocARAG indexing/GPU admission lock"
    if run_tracked "GPU admission" flock "$admission_lock_fd"; then
      :
    else
      status=$?
      error="indexing/GPU admission wait failed with status $status"
      update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
      archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
      DRIVER_FINALIZED=1
      return "$status"
    fi
    update_index_state "$state_file" "$revision_id" "$$" capacity_acquired
    echo "[driver] $(date) acquired the DocARAG indexing/GPU admission lock"
  fi

  if [[ -n "${DOCARAG_PYTHON:-}" ]]; then
    python_runner=("$DOCARAG_PYTHON")
  else
    python_runner=(conda run --no-capture-output -n "${DOCARAG_ENV:-gbc-rag}" python)
  fi

  if run_tracked "runtime fingerprint" "${python_runner[@]}" "$repo/docarag/runtime_probe.py" \
    --output "$DRIVER_ATTEMPT_DIR/runtime.json" \
    --revision-id "$revision_id" --attempt-id "$DRIVER_ATTEMPT_ID"; then
    :
  else
    status=$?
    error="could not record the indexing runtime fingerprint"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi

  role_manifest="$DRIVER_ATTEMPT_DIR/expected-artifact-roles.json"
  if run_tracked "configured artifact role resolution" \
    "${python_runner[@]}" -m docarag.artifact_roles \
    --config "$system_config" --run "$run_dir" --output "$role_manifest"; then
    mapfile -t resolved_roles < <(
      python3 - "$role_manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    roles = json.load(source)
retrieval = roles.get("retrieval_attestation") or roles.get("lexical_store")
retrieval_scope = roles.get("retrieval_scope") or {}
for value in (
    roles["graph"],
    roles["entity_attestation"],
    retrieval,
    roles["index_type"],
    retrieval_scope.get("index_kind", "lexical"),
):
    if not isinstance(value, str) or not value:
        raise SystemExit("configured artifact role is missing")
    print(value)
PY
    )
    if [[ "${#resolved_roles[@]}" -ne 5 ]]; then
      status=1
      error="configured artifact role resolver returned an incomplete result"
      update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
      archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
      DRIVER_FINALIZED=1
      return "$status"
    fi
    expected_graph_artifact="${resolved_roles[0]}"
    expected_entity_attestation="${resolved_roles[1]}"
    expected_retrieval_artifact="${resolved_roles[2]}"
    index_type="${resolved_roles[3]}"
    retrieval_index_kind="${resolved_roles[4]}"
  else
    status=$?
    error="could not resolve the configured graph/entity/retrieval artifact roles"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi

  if run_tracked "artifact recovery assessment" \
    python3 -m docarag.recovery \
    --run "$run_dir" --attempt-dir "$DRIVER_ATTEMPT_DIR" \
    --retrieval-artifact "$expected_retrieval_artifact"; then
    :
  else
    status=$?
    error="published artifact recovery assessment failed"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi

  verify_stage() {
    local stage_to_verify="$1"
    local stage_started_marker="$2"
    local graph_artifact

    case "$stage_to_verify" in
      tree)
        [[ -s "$run_dir/tree.pkl" ]] || {
          echo "[driver] missing or empty tree artifact: $run_dir/tree.pkl" >&2
          return 1
        }
        run_tracked "tree pickle/JSON integrity verification" \
          "${python_runner[@]}" -m docarag.tree_integrity --run "$run_dir"
        ;;
      graph)
        graph_artifact="$expected_graph_artifact"
        [[ -s "$graph_artifact" && "$graph_artifact" -nt "$stage_started_marker" ]] || {
          echo "[driver] graph stage did not create or refresh configured artifact: $graph_artifact" >&2
          return 1
        }
        python3 - "$graph_artifact" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as artifact:
    payload = json.load(artifact)
graph = payload.get("graph")
if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
    raise SystemExit("graph artifact does not contain a node-link graph")
if not isinstance(payload.get("tree2kg"), dict):
    raise SystemExit("graph artifact does not contain a tree2kg mapping")
PY
        [[ -s "$expected_entity_attestation" && "$expected_entity_attestation" -nt "$stage_started_marker" ]] || {
          echo "[driver] graph stage did not publish configured entity attestation: $expected_entity_attestation" >&2
          return 1
        }
        if ! run_tracked "entity Qdrant readiness verification" \
          "${python_runner[@]}" -m docarag.qdrant_readiness \
          --config "$run_dir/run_config.yaml" \
          --attestation "$expected_entity_attestation" \
          --revision-id "$revision_id" \
          --expected-index-kind entity; then
          return 1
        fi
        published_graph_artifact="$graph_artifact"
        published_entity_attestation="$expected_entity_attestation"
        ;;
      vdb)
        if [[ "$index_type" == "bm25" ]]; then
          [[ -s "$expected_retrieval_artifact" ]] || {
            echo "[driver] missing or empty BM25 index: $expected_retrieval_artifact" >&2
            return 1
          }
          return 0
        fi
        [[ -s "$expected_retrieval_artifact" && "$expected_retrieval_artifact" -nt "$stage_started_marker" ]] || {
          echo "[driver] missing or stale Qdrant retrieval attestation: $expected_retrieval_artifact" >&2
          return 1
        }
        run_tracked "retrieval Qdrant readiness verification" \
          "${python_runner[@]}" -m docarag.qdrant_readiness \
          --config "$run_dir/run_config.yaml" \
          --attestation "$expected_retrieval_artifact" \
          --revision-id "$revision_id" \
          --expected-index-kind "$retrieval_index_kind"
        ;;
      *)
        echo "[driver] unknown stage for verification: $stage_to_verify" >&2
        return 1
        ;;
    esac
  }

  for stage in tree graph vdb; do
    DRIVER_ACTIVE_STAGE="$stage"
    update_index_state "$state_file" "$revision_id" "$$" stage_started "$stage"

    if ! run_tracked "immutable input verification before $stage" \
      bash "$controller" --verify-revision-inputs \
      "$repo" "$controller" "$source_snapshot" "$system_config" \
      "$expected_pdf_sha" "$expected_config_sha" "$expected_code_sha" \
      "$routed_profile_snapshot" "$expected_routed_profile_sha"; then
      status=1
      error="immutable input verification failed before $stage; see $run_dir/logs/index-driver.log"
      update_index_state "$state_file" "$revision_id" "$$" stage_failed "$stage" "$status" "$error"
      archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
      DRIVER_FINALIZED=1
      return "$status"
    fi

    if [[ "$stage" == "tree" && ( \
      -e "$run_dir/tree.pkl" || -L "$run_dir/tree.pkl" || \
      -e "$run_dir/tree.json" || -L "$run_dir/tree.json" \
    ) ]]; then
      if run_tracked "existing tree integrity verification" \
        "${python_runner[@]}" -m docarag.tree_integrity --run "$run_dir"; then
        :
      else
        echo "[driver] existing tree pair is inconsistent; quarantining tree and dependent outputs" >&2
        if run_tracked "tree recovery quarantine" \
          python3 -m docarag.recovery \
          --run "$run_dir" --attempt-dir "$DRIVER_ATTEMPT_DIR" \
          --retrieval-artifact "$expected_retrieval_artifact" --from-stage tree; then
          :
        else
          status=$?
          error="could not quarantine an inconsistent existing tree"
          update_index_state "$state_file" "$revision_id" "$$" stage_failed "$stage" "$status" "$error"
          archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
          DRIVER_FINALIZED=1
          return "$status"
        fi
      fi
    fi

    stage_marker="$run_dir/logs/.index-$stage.started"
    : > "$stage_marker"
    attempt_log="$DRIVER_ATTEMPT_DIR/index-$stage.log"
    ln -sfn "../attempts/$DRIVER_ATTEMPT_ID/index-$stage.log" "$run_dir/logs/index-$stage.log"
    echo "[driver] $(date) starting $stage"
    if run_tracked "$stage index stage" "${python_runner[@]}" main.py \
      -c "$system_config" -d "$dataset_config" --num 1 --nsplit 1 \
      index --stage "$stage" > "$attempt_log" 2>&1; then
      :
    else
      status=$?
      error="$stage command exited with status $status; see $run_dir/logs/index-$stage.log"
      echo "[driver] $(date) $stage FAILED (exit $status); see $run_dir/logs/index-$stage.log" >&2
      update_index_state "$state_file" "$revision_id" "$$" stage_failed "$stage" "$status" "$error"
      archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
      DRIVER_FINALIZED=1
      return "$status"
    fi

    if ! verify_stage "$stage" "$stage_marker"; then
      status=1
      error="$stage artifact verification failed; see $run_dir/logs/index-driver.log"
      echo "[driver] $(date) $stage FAILED artifact verification; stopping" >&2
      update_index_state "$state_file" "$revision_id" "$$" stage_failed "$stage" "$status" "$error"
      archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
      DRIVER_FINALIZED=1
      return "$status"
    fi
    update_index_state "$state_file" "$revision_id" "$$" stage_succeeded "$stage" 0
    echo "[driver] $(date) $stage done (exit 0)"
  done

  DRIVER_ACTIVE_STAGE=""
  if ! run_tracked "immutable input verification before publication" \
    bash "$controller" --verify-revision-inputs \
    "$repo" "$controller" "$source_snapshot" "$system_config" \
    "$expected_pdf_sha" "$expected_config_sha" "$expected_code_sha" \
    "$routed_profile_snapshot" "$expected_routed_profile_sha"; then
    status=1
    error="immutable input verification failed after vdb; this revision is not publishable"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi
  if run_tracked "final tree integrity verification" \
    "${python_runner[@]}" -m docarag.tree_integrity --run "$run_dir"; then
    :
  else
    status=$?
    error="tree pickle/JSON consistency or source identity failed before publication"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi

  if [[ ! -s "$run_dir/run_config.yaml" ]]; then
    status=1
    error="index stages completed without a run configuration snapshot"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi
  chmod 0400 "$run_dir/run_config.yaml"
  if [[ "$index_type" == "bm25" ]]; then
    publication_retrieval_args=(--lexical-store "$expected_retrieval_artifact")
  else
    publication_retrieval_args=(--retrieval-attestation "$expected_retrieval_artifact")
  fi
  if run_tracked "artifact manifest publication" \
    "${python_runner[@]}" -m docarag.provenance publish \
    --run "$run_dir" \
    --revision-id "$revision_id" \
    --attempt-id "$DRIVER_ATTEMPT_ID" \
    "${publication_retrieval_args[@]}" \
    --graph-artifact "$published_graph_artifact" \
    --entity-attestation "$published_entity_attestation"; then
    :
  else
    status=$?
    error="could not publish the artifact integrity manifest"
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi
  # Retrieval reads the tree and the graph from the store, so a run that does
  # not persist its evidence has not produced a usable index.
  if "${python_runner[@]}" -m docarag.evidence_persist --run "$run_dir"; then
    echo "[driver] $(date) evidence persisted to PostgreSQL"
  else
    status=$?
    error="evidence persistence failed; the index cannot be queried"
    echo "[driver] $(date) $error" >&2
    update_index_state "$state_file" "$revision_id" "$$" run_failed "" "$status" "$error"
    archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
    DRIVER_FINALIZED=1
    return "$status"
  fi
  update_index_state "$state_file" "$revision_id" "$$" run_succeeded "" 0
  archive_attempt_state "$state_file" "$DRIVER_ATTEMPT_DIR" || true
  DRIVER_FINALIZED=1
  trap - EXIT INT TERM
  echo "[driver] $(date) ALL DONE"
}

if [[ "${1:-}" == "--verify-revision-inputs" ]]; then
  shift
  verify_revision_inputs "$@"
  exit $?
fi

if [[ "${1:-}" == "--detached-driver" ]]; then
  shift
  run_detached_driver "$@"
  exit $?
fi

for required_command in python3 flock setsid nohup; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "[error] required command is not installed: $required_command" >&2
    exit 1
  fi
done

SYSTEM_CONFIG="${DOCARAG_CONFIG:-config/ollama-gbc.yaml}"
if [[ "$SYSTEM_CONFIG" != /* ]]; then
  SYSTEM_CONFIG="$REPO/$SYSTEM_CONFIG"
fi
if [[ ! -f "$SYSTEM_CONFIG" ]]; then
  echo "[error] DocARAG config not found: $SYSTEM_CONFIG" >&2
  exit 1
fi
SYSTEM_CONFIG="$(readlink -f "$SYSTEM_CONFIG")"

ROUTED_PROFILE="${DOCARAG_ROUTED_PROFILE:-}"
ROUTED_PROFILE_ID=""
ROUTED_PROFILE_SHA256=""
ROUTED_PROFILE_CONFIGURATION_SHA256=""
ROUTED_PROFILE_SIZE=""
ROUTED_PROFILE_MODIFIED_AT=""
if [[ -n "$ROUTED_PROFILE" ]]; then
  ROUTED_PROFILE="$(readlink -f "$ROUTED_PROFILE")" || true
  if [[ ! -f "$ROUTED_PROFILE" ]]; then
    echo "[error] routed parser profile not found: $ROUTED_PROFILE" >&2
    exit 1
  fi
  mapfile -t ROUTED_PROFILE_BINDING < <(python3 - "$ROUTED_PROFILE" <<'PY'
import sys

from docarag.routed_profile import load_routed_runtime_profile

profile = load_routed_runtime_profile(sys.argv[1])
issues = profile.configuration_issues()
if issues:
    details = "; ".join(
        f"{item.runtime_id}.{item.field}: {item.message}" for item in issues
    )
    raise SystemExit("routed parser profile is not configuration-ready: " + details)
print(profile.profile_id)
print(profile.source_sha256)
print(profile.configuration_sha256)
PY
  )
  if [[ "${#ROUTED_PROFILE_BINDING[@]}" -ne 3 ]]; then
    echo "[error] could not derive the routed profile revision binding" >&2
    exit 1
  fi
  ROUTED_PROFILE_ID="${ROUTED_PROFILE_BINDING[0]}"
  ROUTED_PROFILE_SHA256="${ROUTED_PROFILE_BINDING[1]}"
  ROUTED_PROFILE_CONFIGURATION_SHA256="${ROUTED_PROFILE_BINDING[2]}"
  read -r ACTUAL_ROUTED_PROFILE_SHA ROUTED_PROFILE_SIZE _ ROUTED_PROFILE_MODIFIED_AT < <(file_identity "$ROUTED_PROFILE")
  if [[ "$ACTUAL_ROUTED_PROFILE_SHA" != "$ROUTED_PROFILE_SHA256" ]]; then
    echo "[error] routed profile changed while its revision binding was derived" >&2
    exit 1
  fi
  for supplied_name in \
    DOCARAG_ROUTED_PROFILE_ID \
    DOCARAG_ROUTED_PROFILE_SOURCE_SHA256 \
    DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256; do
    supplied_value="${!supplied_name:-}"
    case "$supplied_name" in
      DOCARAG_ROUTED_PROFILE_ID) actual_value="$ROUTED_PROFILE_ID" ;;
      DOCARAG_ROUTED_PROFILE_SOURCE_SHA256) actual_value="$ROUTED_PROFILE_SHA256" ;;
      DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256) actual_value="$ROUTED_PROFILE_CONFIGURATION_SHA256" ;;
    esac
    if [[ -n "$supplied_value" && "$supplied_value" != "$actual_value" ]]; then
      echo "[error] $supplied_name does not match the selected routed profile" >&2
      exit 1
    fi
  done
  export DOCARAG_ROUTED_PROFILE_ID="$ROUTED_PROFILE_ID"
  export DOCARAG_ROUTED_PROFILE_SOURCE_SHA256="$ROUTED_PROFILE_SHA256"
  export DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256="$ROUTED_PROFILE_CONFIGURATION_SHA256"
fi

PDF="${1:-}"
if [[ -z "$PDF" ]]; then
  read -rp "Path to PDF: " PDF
fi
PDF="$(readlink -f "$PDF")" || true
if [[ ! -f "$PDF" ]]; then
  echo "[error] PDF not found: $PDF" >&2
  exit 1
fi

read -r PDF_SHA256 PDF_SIZE PDF_MTIME_NS PDF_MODIFIED_AT < <(file_identity "$PDF")
read -r CONFIG_SHA256 CONFIG_SIZE CONFIG_MTIME_NS CONFIG_MODIFIED_AT < <(file_identity "$SYSTEM_CONFIG")
read -r CODE_SHA256 CODE_FILE_COUNT < <(compute_code_fingerprint "$REPO" "$SCRIPT_PATH")
read -r CONTROLLER_SHA256 _ _ _ < <(file_identity "$SCRIPT_PATH")

mapfile -t DERIVED_DOCUMENT_IDENTITY < <(python3 - "$PDF" "$PDF_SHA256" <<'PY'
from pathlib import Path
import sys

from docarag.domain.evidence import DocumentRef

source_uri = Path(sys.argv[1]).resolve().as_uri()
document = DocumentRef.identified(
    content_sha256=sys.argv[2],
    mime_type="application/pdf",
    source_uri=source_uri,
    display_name=Path(sys.argv[1]).name,
)
print(source_uri)
print(document.document_id)
print(document.document_version_id)
PY
)
if [[ "${#DERIVED_DOCUMENT_IDENTITY[@]}" -ne 3 ]]; then
  echo "[error] could not derive the canonical document identity" >&2
  exit 1
fi
SOURCE_URI="${DERIVED_DOCUMENT_IDENTITY[0]}"
DERIVED_DOCUMENT_ID="${DERIVED_DOCUMENT_IDENTITY[1]}"
DERIVED_DOCUMENT_VERSION_ID="${DERIVED_DOCUMENT_IDENTITY[2]}"
DOCUMENT_ID="${DOCARAG_DOCUMENT_ID:-$DERIVED_DOCUMENT_ID}"
DOCUMENT_VERSION_ID="${DOCARAG_DOCUMENT_VERSION_ID:-$DERIVED_DOCUMENT_VERSION_ID}"
if [[ "$DOCUMENT_ID" != "$DERIVED_DOCUMENT_ID" ]]; then
  echo "[error] DOCARAG_DOCUMENT_ID does not match the canonical file URI" >&2
  exit 1
fi
if [[ "$DOCUMENT_VERSION_ID" != "$DERIVED_DOCUMENT_VERSION_ID" ]]; then
  echo "[error] DOCARAG_DOCUMENT_VERSION_ID does not match the source bytes" >&2
  exit 1
fi
TENANT_ID="${DOCARAG_TENANT_ID:-default}"
CORPUS_ID="${DOCARAG_CORPUS_ID:-default}"
NAME="$(python3 - "$PDF" <<'PY'
from pathlib import Path
import re
import sys
name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(sys.argv[1]).stem).strip("_")
print(name or "document")
PY
)"
REVISION_ID="$(python3 - "$SOURCE_URI" "$PDF_SHA256" "$CONFIG_SHA256" "$CODE_SHA256" \
  "$TENANT_ID" "$CORPUS_ID" "$DOCUMENT_VERSION_ID" "$ROUTED_PROFILE_ID" \
  "$ROUTED_PROFILE_SHA256" "$ROUTED_PROFILE_CONFIGURATION_SHA256" <<'PY'
import hashlib
import json
import sys
payload = {
    "identity_schema_version": 2,
    "source_uri": sys.argv[1],
    "source_sha256": sys.argv[2],
    "system_config_sha256": sys.argv[3],
    "controller_code_sha256": sys.argv[4],
    "tenant_id": sys.argv[5],
    "corpus_id": sys.argv[6],
    "document_version_id": sys.argv[7],
    "routed_profile": (
        None
        if not sys.argv[8]
        else {
            "profile_id": sys.argv[8],
            "source_sha256": sys.argv[9],
            "configuration_sha256": sys.argv[10],
        }
    ),
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
)"
if [[ -n "${DOCARAG_INDEX_REVISION_ID:-}" && "$DOCARAG_INDEX_REVISION_ID" != "$REVISION_ID" ]]; then
  echo "[error] DOCARAG_INDEX_REVISION_ID does not match the derived index revision" >&2
  exit 1
fi
export DOCARAG_TENANT_ID="$TENANT_ID"
export DOCARAG_CORPUS_ID="$CORPUS_ID"
export DOCARAG_DOCUMENT_ID="$DOCUMENT_ID"
export DOCARAG_DOCUMENT_VERSION_ID="$DOCUMENT_VERSION_ID"
export DOCARAG_INDEX_REVISION_ID="$REVISION_ID"

WORKDIR="${DOCARAG_RUNS_DIR:-$REPO/runs}"
if [[ "$WORKDIR" != /* ]]; then
  WORKDIR="$REPO/$WORKDIR"
fi
WORKDIR="$(readlink -m "$WORKDIR")"
export DOCARAG_RUNS_DIR="$WORKDIR"
DOCUMENT_RUNS_DIR="$WORKDIR/$DOCUMENT_ID"
RUN_DIR="$DOCUMENT_RUNS_DIR/$REVISION_ID"
INPUT_DIR="$RUN_DIR/inputs"
LOG_DIR="$RUN_DIR/logs"
SOURCE_STORE="$WORKDIR/_sources/sha256"
SOURCE_SNAPSHOT="$SOURCE_STORE/$PDF_SHA256.pdf"
DATASET_JSON="$INPUT_DIR/dataset-$DOCUMENT_ID-$REVISION_ID.json"
DATASET_CONFIG="$INPUT_DIR/dataset-$DOCUMENT_ID-$REVISION_ID.yaml"
CONFIG_SNAPSHOT="$INPUT_DIR/system-config-$CONFIG_SHA256.yaml"
ROUTED_PROFILE_SNAPSHOT=""
if [[ -n "$ROUTED_PROFILE" ]]; then
  ROUTED_PROFILE_SNAPSHOT="$INPUT_DIR/routed-profile-$ROUTED_PROFILE_SHA256.toml"
fi
STATE_FILE="$RUN_DIR/index-state.json"
DRIVER_LOG="$LOG_DIR/index-driver.log"
LOCK_FILE="$RUN_DIR/.index.lock"
DATASET_NAME="$NAME-${REVISION_ID:0:12}"

mkdir -p "$INPUT_DIR" "$LOG_DIR" "$SOURCE_STORE"
create_source_snapshot "$PDF" "$SOURCE_SNAPSHOT" "$PDF_SHA256" "$PDF_SIZE"
if [[ -n "$ROUTED_PROFILE" ]]; then
  create_source_snapshot \
    "$ROUTED_PROFILE" "$ROUTED_PROFILE_SNAPSHOT" \
    "$ROUTED_PROFILE_SHA256" "$ROUTED_PROFILE_SIZE"
  export DOCARAG_ROUTED_PROFILE="$ROUTED_PROFILE_SNAPSHOT"
fi

if GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; then
  :
else
  GIT_COMMIT="unknown"
fi
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no -- main.py Core docarag Scripts/index_pdf.sh 2>/dev/null || true)" ]]; then
  GIT_DIRTY=true
else
  GIT_DIRTY=false
fi

# Serialize run initialization as well as execution. Without taking the lock
# here, two first-time launchers could race while creating the immutable
# manifest with different creation timestamps.
exec {LAUNCH_LOCK_FD}> "$LOCK_FILE"
if ! flock -n "$LAUNCH_LOCK_FD"; then
  echo "Indexing is already running for this exact revision:"
  echo "  run dir : $RUN_DIR"
  echo "  state   : $STATE_FILE"
  exit 0
fi

write_run_inputs \
  "$RUN_DIR" "$PDF" "$SOURCE_SNAPSHOT" "$PDF_SHA256" "$PDF_SIZE" "$PDF_MODIFIED_AT" \
  "$SYSTEM_CONFIG" "$CONFIG_SNAPSHOT" "$CONFIG_SHA256" "$CONFIG_SIZE" "$CONFIG_MODIFIED_AT" \
  "$CODE_SHA256" "$CODE_FILE_COUNT" "$SCRIPT_PATH" "$CONTROLLER_SHA256" \
  "$GIT_COMMIT" "$GIT_DIRTY" "$REVISION_ID" "$DOCUMENT_ID" \
  "$DOCUMENT_VERSION_ID" "$TENANT_ID" "$CORPUS_ID" "$DATASET_NAME" \
  "$DATASET_JSON" "$DATASET_CONFIG" "$STATE_FILE" "$DOCUMENT_RUNS_DIR" \
  "$ROUTED_PROFILE" "$ROUTED_PROFILE_SNAPSHOT" "$ROUTED_PROFILE_ID" \
  "$ROUTED_PROFILE_SHA256" "$ROUTED_PROFILE_CONFIGURATION_SHA256" \
  "$ROUTED_PROFILE_SIZE" "$ROUTED_PROFILE_MODIFIED_AT"

if completed_run_is_valid "$RUN_DIR"; then
  exec {LAUNCH_LOCK_FD}>&-
  echo "This exact document/config/code revision is already indexed:"
  echo "  run dir     : $RUN_DIR"
  echo "  document id : $DOCUMENT_ID"
  echo "  revision id : $REVISION_ID"
  echo "  state       : $STATE_FILE"
  exit 0
fi

# Let the detached driver inherit the already-held run lock. This closes the
# fork race: a second invocation cannot start a driver or overwrite its state.
printf '[launcher] %s submitting revision %s\n' "$(date)" "$REVISION_ID" >> "$DRIVER_LOG"
setsid nohup bash "$SCRIPT_PATH" \
  --detached-driver "$REPO" "$DATASET_CONFIG" "$RUN_DIR" "$CONFIG_SNAPSHOT" \
  "$REVISION_ID" "$PDF_SHA256" "$CONFIG_SHA256" "$CODE_SHA256" \
  "$SOURCE_SNAPSHOT" "$LAUNCH_LOCK_FD" \
  "$ROUTED_PROFILE_SNAPSHOT" "$ROUTED_PROFILE_SHA256" \
  >> "$DRIVER_LOG" 2>&1 < /dev/null &
DRIVER_PID=$!
disown "$DRIVER_PID" 2>/dev/null || true
exec {LAUNCH_LOCK_FD}>&-

echo "Indexing started (detached) for: $PDF"
echo "  run dir     : $RUN_DIR"
echo "  document id : $DOCUMENT_ID"
echo "  revision id : $REVISION_ID"
echo "  driver pid  : $DRIVER_PID"
echo "  state       : $STATE_FILE"
echo
echo "Watch progress : tail -f $DRIVER_LOG"
echo "                 tail -f $LOG_DIR/index-tree.log   (per-stage detail)"
echo "View when ready: bash Scripts/view.sh $RUN_DIR"
echo "  (tree is viewable in a few minutes; full graph takes longer)"
