"""Forensic, stage-aware recovery of mutable BookRAG index outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

from docarag.provenance import sha256_file, verify_artifact_manifest


STAGE_NUMBER = {"tree": 0, "graph": 1, "vdb": 2}


class RecoveryError(ValueError):
    """Raised when recovery cannot be performed without losing provenance."""


@dataclass(frozen=True)
class RecoveryPlan:
    from_stage: str | None
    reason: str


def _safe_relative(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        return None
    return path.as_posix()


def _is_role_path(path: str, role: str | None) -> bool:
    return bool(role) and (path == role or path.startswith(role.rstrip("/") + "/"))


def _entry_mismatches(run: Path, entry: object) -> tuple[str | None, bool]:
    if not isinstance(entry, dict):
        return None, True
    relative = _safe_relative(entry.get("path"))
    size = entry.get("size_bytes")
    digest = entry.get("sha256")
    if relative is None or not isinstance(size, int) or not isinstance(digest, str):
        return relative, True
    candidate = run / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run)
        valid = (
            not candidate.is_symlink()
            and resolved == candidate.absolute()
            and resolved.is_file()
            and resolved.stat().st_size == size
            and sha256_file(resolved) == digest
        )
    except (OSError, RuntimeError, ValueError):
        valid = False
    return relative, not valid


def plan_recovery(run_dir: Path) -> RecoveryPlan:
    """Find the earliest output stage whose published bytes no longer match."""

    run = run_dir.expanduser().resolve()
    manifest_path = run / "artifact-manifest.json"
    if manifest_path.is_symlink():
        return RecoveryPlan("tree", "artifact manifest is unsafe")
    if not manifest_path.exists():
        return RecoveryPlan(None, "no published artifact manifest")
    if not manifest_path.is_file():
        return RecoveryPlan("tree", "artifact manifest is unsafe")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return RecoveryPlan("tree", "artifact manifest cannot be parsed")
    if not isinstance(payload, dict):
        return RecoveryPlan("tree", "artifact manifest is not an object")

    verification = verify_artifact_manifest(run, deep=True)
    if verification.valid:
        return RecoveryPlan(None, "published artifact set is valid")

    errors = "\n".join(verification.errors)
    if any(
        marker in errors
        for marker in (
            "run-manifest.json",
            "run manifest has no source identity",
            "run manifest source snapshot identity is incomplete",
            "immutable source snapshot",
        )
    ):
        raise RecoveryError(
            "run/source manifest verification failed; refusing to rebind mutable "
            "outputs to modified or missing source-of-truth inputs"
        )

    roles_payload = payload.get("roles")
    roles = roles_payload if isinstance(roles_payload, dict) else {}
    tree_pickle = _safe_relative(roles.get("tree_pickle"))
    tree_json = _safe_relative(roles.get("tree_json"))
    graph = _safe_relative(roles.get("graph"))
    entity = _safe_relative(roles.get("entity_attestation"))
    retrieval = _safe_relative(
        roles.get("retrieval_attestation") or roles.get("lexical_store")
    )
    run_config = _safe_relative(roles.get("run_config"))

    mismatches: list[str] = []
    malformed_entry = False
    entries = payload.get("files")
    if not isinstance(entries, list):
        entries = []
        malformed_entry = True
    for entry in entries:
        relative, mismatched = _entry_mismatches(run, entry)
        if mismatched:
            if relative is None:
                malformed_entry = True
            else:
                mismatches.append(relative)

    if run_config and run_config in mismatches:
        raise RecoveryError(
            "published run configuration changed; refusing to reconstruct with ambiguous settings"
        )

    earliest = 3
    for path in mismatches:
        if path in {tree_pickle, tree_json}:
            earliest = min(earliest, STAGE_NUMBER["tree"])
        elif _is_role_path(path, graph) or _is_role_path(path, entity):
            earliest = min(earliest, STAGE_NUMBER["graph"])
        elif _is_role_path(path, retrieval):
            earliest = min(earliest, STAGE_NUMBER["vdb"])
        else:
            # Parser evidence and other published files feed the tree, so an
            # unknown mismatch must conservatively invalidate the full chain.
            earliest = min(earliest, STAGE_NUMBER["tree"])

    if malformed_entry or not isinstance(roles_payload, dict):
        earliest = STAGE_NUMBER["tree"]
    if earliest == 3:
        # Manifest bookkeeping corruption must not be silently re-blessed.
        # Reconstructing the outputs creates a fresh independent artifact set.
        earliest = STAGE_NUMBER["tree"]
    stage = ("tree", "graph", "vdb")[earliest]
    return RecoveryPlan(stage, "; ".join(verification.errors))


def _role_stage(path: str, roles: dict[str, str | None]) -> str:
    if path in {roles.get("tree_pickle"), roles.get("tree_json")}:
        return "tree"
    if (
        _is_role_path(path, roles.get("graph"))
        or _is_role_path(path, roles.get("entity_attestation"))
        or path.startswith("graph_data")
        or path.startswith("entity_attestation")
        or path.startswith("kg_extractor_res")
    ):
        return "graph"
    if _is_role_path(path, roles.get("retrieval_attestation")) or _is_role_path(
        path, roles.get("lexical_store")
    ):
        return "vdb"
    if path == roles.get("run_config"):
        return "protected"
    if path == "indexing_stats.json":
        return "vdb"
    return "tree"


def quarantine_outputs(
    run_dir: Path,
    attempt_dir: Path,
    *,
    from_stage: str,
    retrieval_artifact: Path | None = None,
) -> list[str]:
    """Atomically move affected outputs aside for audit, never delete/re-bless."""

    if from_stage not in STAGE_NUMBER:
        raise RecoveryError(f"unknown recovery stage: {from_stage}")
    run = run_dir.expanduser().resolve()
    attempt = attempt_dir.expanduser().resolve()
    try:
        attempt.relative_to(run)
    except ValueError as exc:
        raise RecoveryError("attempt recovery directory must be inside the run") from exc
    quarantine = attempt / "recovery"
    quarantine.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    manifest_path = run / "artifact-manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    raw_roles = payload.get("roles")
    raw_roles = raw_roles if isinstance(raw_roles, dict) else {}
    roles = {
        key: _safe_relative(raw_roles.get(key))
        for key in (
            "tree_pickle",
            "tree_json",
            "graph",
            "entity_attestation",
            "retrieval_attestation",
            "lexical_store",
            "run_config",
        )
    }

    candidates: set[Path] = set()
    entries = payload.get("files")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            relative = _safe_relative(entry.get("path"))
            if relative is None:
                continue
            stage = _role_stage(relative, roles)
            if stage != "protected" and STAGE_NUMBER[stage] >= STAGE_NUMBER[from_stage]:
                candidates.add(run / Path(relative).parts[0])

    if STAGE_NUMBER[from_stage] <= STAGE_NUMBER["tree"]:
        candidates.update((run / "tree.pkl", run / "tree.json"))
    if STAGE_NUMBER[from_stage] <= STAGE_NUMBER["graph"]:
        candidates.update(run.glob("graph_data*.json"))
        candidates.update(run.glob("entity_attestation*"))
        candidates.add(run / "kg_extractor_res")
    if retrieval_artifact is not None and STAGE_NUMBER[from_stage] <= STAGE_NUMBER["vdb"]:
        resolved_retrieval = retrieval_artifact.expanduser().resolve()
        try:
            relative_retrieval = resolved_retrieval.relative_to(run)
        except ValueError as exc:
            raise RecoveryError("retrieval artifact escapes the run directory") from exc
        if relative_retrieval == Path("."):
            raise RecoveryError("retrieval artifact cannot be the run directory")
        candidates.add(run / relative_retrieval.parts[0])

    protected = {
        run / "attempts",
        run / "inputs",
        run / "logs",
        run / "run-manifest.json",
        run / "run_config.yaml",
        run / "index-state.json",
        run / ".index.lock",
    }
    moved: list[str] = []
    for candidate in sorted(candidates, key=lambda item: item.name):
        if candidate in protected:
            continue
        if candidate.is_symlink():
            raise RecoveryError(f"refusing to quarantine symbolic-link artifact: {candidate}")
        if not candidate.exists():
            continue
        destination = quarantine / candidate.name
        if destination.exists():
            raise RecoveryError(f"recovery destination already exists: {destination}")
        os.replace(candidate, destination)
        moved.append(candidate.relative_to(run).as_posix())

    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise RecoveryError("refusing to move a symbolic-link artifact manifest")
        destination = quarantine / "artifact-manifest.json"
        if destination.exists():
            raise RecoveryError(f"recovery destination already exists: {destination}")
        os.replace(manifest_path, destination)
        moved.append("artifact-manifest.json")

    directory_fd = os.open(quarantine, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    run_fd = os.open(run, os.O_RDONLY)
    try:
        os.fsync(run_fd)
    finally:
        os.close(run_fd)
    return moved


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair invalid BookRAG stage outputs")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--retrieval-artifact", type=Path)
    parser.add_argument("--from-stage", choices=tuple(STAGE_NUMBER))
    args = parser.parse_args(argv)
    try:
        plan = (
            RecoveryPlan(args.from_stage, "explicit tree-pair/stage invalidation")
            if args.from_stage
            else plan_recovery(args.run)
        )
        moved: list[str] = []
        if plan.from_stage:
            moved = quarantine_outputs(
                args.run,
                args.attempt_dir,
                from_stage=plan.from_stage,
                retrieval_artifact=args.retrieval_artifact,
            )
        print(
            json.dumps(
                {
                    "from_stage": plan.from_stage,
                    "reason": plan.reason,
                    "quarantined": moved,
                },
                sort_keys=True,
            )
        )
    except (OSError, RecoveryError) as exc:
        print(f"recovery error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
