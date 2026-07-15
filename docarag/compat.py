"""Compatibility bridge from the stable CLI to the current BookRAG runtime.

Heavy imports are deliberately local to :func:`ask`. Merely importing the CLI
or running diagnostics must not load ML frameworks.  This module also owns the
security boundary around legacy run artifacts: BookRAG indexes contain pickle
files, so callers may only select runs beneath the configured runs directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from typing import Iterable
import yaml

from docarag.provenance import verify_query_artifacts
from docarag.run_logging import RunLogInfo, RunLogScope
from Core.utils.artifact_paths import (
    QdrantProtocolError,
    QdrantUnavailableError,
    qdrant_collection_state,
    qdrant_endpoint_sha256,
)


_READY_MARKERS = (".ask-ready", ".docarag-ready", "READY")
_STATE_FILES = (
    "readiness.json",
    "run_state.json",
    "run-state.json",
    "index_state.json",
    "index-state.json",
    "state.json",
)
_READY_STATES = {"complete", "completed", "ready", "searchable", "succeeded"}


@dataclass(frozen=True)
class RunReadiness:
    """Dependency-light readiness result for a compatibility command."""

    ready: bool
    purpose: str
    source: str
    missing: tuple[str, ...] = ()


class CompatibilityAskError(RuntimeError):
    """Friendly query failure with a machine-readable trace envelope."""

    def __init__(
        self,
        message: str,
        details: dict[str, object],
        *,
        exit_code: int = 2,
    ):
        super().__init__(message)
        self.details = details
        self.exit_code = exit_code


def _is_descendant(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return relative != Path(".")


def _safe_nonempty_file(run_dir: Path, path: Path) -> bool:
    """Return true only for a non-empty file that cannot escape ``run_dir``."""

    try:
        resolved = path.resolve()
        return (
            resolved.is_relative_to(run_dir)
            and resolved.is_file()
            and resolved.stat().st_size > 0
        )
    except (OSError, RuntimeError):
        return False


def _valid_tree_json(run_dir: Path) -> bool:
    path = run_dir / "tree.json"
    if not _safe_nonempty_file(run_dir, path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("nodes"), list) and bool(
        payload["nodes"]
    )


def _valid_graph_json(run_dir: Path, path: Path) -> bool:
    if not _safe_nonempty_file(run_dir, path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("tree2kg"), dict):
        return False
    graph = payload.get("graph")
    return isinstance(graph, dict) and isinstance(graph.get("nodes"), list) and bool(
        graph["nodes"]
    )


def _state_readiness_signal(run_dir: Path) -> tuple[str, bool] | None:
    """Return ``(source, authoritative)`` for a run-local readiness signal.

    Signals are informative only in this compatibility runtime. They never
    replace live artifact validation; a future external backend must publish a
    versioned capability manifest with its own verifier before it can do so.
    """

    for name in _READY_MARKERS:
        marker = run_dir / name
        try:
            if marker.resolve().is_relative_to(run_dir) and marker.is_file():
                return name, False
        except (OSError, RuntimeError):
            continue

    for name in _STATE_FILES:
        state_path = run_dir / name
        if not _safe_nonempty_file(run_dir, state_path):
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ask_ready") is True or payload.get("searchable") is True:
            return name, False
        status = str(
            payload.get("overall_status") or payload.get("status") or ""
        ).strip().lower()
        capabilities = payload.get("capabilities", ())
        has_ask_capability = isinstance(capabilities, list) and "ask" in capabilities
        stages = payload.get("stages", {})
        verified_legacy_pipeline = (
            name in {"index-state.json", "index_state.json"}
            and status == "succeeded"
            and isinstance(stages, dict)
            and all(
                isinstance(stages.get(stage), dict)
                and stages[stage].get("status") == "succeeded"
                for stage in ("tree", "graph", "vdb")
            )
        )
        if verified_legacy_pipeline:
            return name, False
        if status in _READY_STATES and (
            payload.get("ready") is True or has_ask_capability
        ):
            return name, False
    return None


def _current_index_state(
    run_dir: Path,
) -> tuple[str, str, bool, dict[str, object]] | None:
    """Return source, status, stage consistency, and the current driver state."""

    for name in ("index-state.json", "index_state.json"):
        path = run_dir / name
        if not _safe_nonempty_file(run_dir, path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        status = str(
            payload.get("overall_status") or payload.get("status") or ""
        ).strip().lower()
        stages = payload.get("stages")
        stages_succeeded = isinstance(stages, dict) and all(
            isinstance(stages.get(stage), dict)
            and stages[stage].get("status") == "succeeded"
            for stage in ("tree", "graph", "vdb")
        )
        return name, status, stages_succeeded, payload
    return None


def _published_roles(run_dir: Path) -> dict[str, str] | None:
    """Load verified artifact roles for a newly published immutable run."""

    path = run_dir / "artifact-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    roles = payload.get("roles") if isinstance(payload, dict) else None
    if not isinstance(roles, dict):
        return None
    return {
        key: value
        for key, value in roles.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _role_path(run_dir: Path, roles: dict[str, str], name: str) -> Path | None:
    relative = roles.get(name)
    if not relative:
        return None
    candidate = run_dir / relative
    try:
        resolved = candidate.resolve()
        resolved.relative_to(run_dir)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _published_store_readiness(
    run_dir: Path,
    roles: dict[str, str],
) -> tuple[bool, bool, tuple[str, ...]]:
    """Validate published local bindings against exact live Qdrant scopes."""

    errors: list[str] = []
    lexical = _role_path(run_dir, roles, "lexical_store")
    retrieval_ready = bool(
        lexical is not None and _safe_nonempty_file(run_dir, lexical)
    )

    try:
        artifact_payload = json.loads(
            (run_dir / "artifact-manifest.json").read_text(encoding="utf-8")
        )
        config_payload = yaml.safe_load(
            (run_dir / "run_config.yaml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError):
        return False, False, ("Qdrant runtime configuration cannot be loaded",)
    vector_indexes = (
        artifact_payload.get("vector_indexes")
        if isinstance(artifact_payload, dict)
        else None
    )
    vdb_config = config_payload.get("vdb") if isinstance(config_payload, dict) else None
    if not isinstance(vector_indexes, dict) or not isinstance(vdb_config, dict):
        return False, False, ("Qdrant publication metadata is missing",)
    qdrant_url_env = vdb_config.get(
        "qdrant_url_env", "DOCARAG_QDRANT_URL"
    )
    if (
        not isinstance(qdrant_url_env, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", qdrant_url_env) is None
    ):
        return False, False, (
            "Qdrant runtime endpoint environment reference is invalid",
        )
    qdrant_url = os.environ.get(
        qdrant_url_env, vdb_config.get("qdrant_url")
    )
    api_key_env = vdb_config.get("api_key_env", "QDRANT_API_KEY")
    timeout_seconds = vdb_config.get("timeout_seconds", 10.0)
    try:
        endpoint_sha256 = qdrant_endpoint_sha256(qdrant_url)
    except (TypeError, ValueError):
        return False, False, ("Qdrant runtime endpoint configuration is invalid",)

    readiness: dict[str, bool] = {"entity": False, "retrieval": retrieval_ready}
    required = ["entity"]
    if "retrieval_attestation" in roles:
        required.append("retrieval")
    for index_name in required:
        descriptor = vector_indexes.get(index_name)
        if not isinstance(descriptor, dict):
            errors.append(f"Qdrant {index_name} attestation is missing")
            continue
        if descriptor.get("qdrant_endpoint_sha256") != endpoint_sha256:
            errors.append(f"Qdrant {index_name} endpoint binding mismatches")
            continue
        collection = descriptor.get("collection_name")
        scope_filter = descriptor.get("scope_filter")
        expected_count = descriptor.get("point_count")
        if (
            not isinstance(collection, str)
            or not isinstance(scope_filter, dict)
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
        ):
            errors.append(f"Qdrant {index_name} attestation is invalid")
            continue
        try:
            state = qdrant_collection_state(
                qdrant_url=qdrant_url,
                collection_name=collection,
                scope_filter=scope_filter,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
            )
        except QdrantUnavailableError:
            errors.append(f"Qdrant unavailable while checking {index_name} scope")
            continue
        except (QdrantProtocolError, TypeError, ValueError):
            errors.append(f"Qdrant rejected the {index_name} readiness check")
            continue
        if not state.exists or state.point_count != expected_count:
            errors.append(f"Qdrant {index_name} scoped point count mismatches")
            continue
        readiness[index_name] = True
    return readiness["retrieval"], readiness["entity"], tuple(errors)


def _unpublished_store_readiness(run_dir: Path) -> tuple[bool, bool, tuple[str, ...]]:
    """Intentional schema break: unbound vector directories are never queryable."""

    try:
        lexical = any(
            _safe_nonempty_file(run_dir, path)
            for path in run_dir.glob("**/bm25_index.pkl")
        )
    except OSError:
        lexical = False
    return lexical, False, ("published Qdrant attestations are required",)


def inspect_run_readiness(run_dir: Path, *, purpose: str) -> RunReadiness:
    """Check only the artifacts needed by the requested command.

    ``web`` may display a tree while graph indexing is still in progress.
    ``ask`` must not start the heavyweight runtime until a full legacy
    BookIndex (or an explicit future readiness signal) is available.
    """

    run = run_dir.resolve()
    if purpose == "status":
        return RunReadiness(True, purpose, "directory")

    if purpose == "web":
        missing = () if _valid_tree_json(run) else ("valid non-empty tree.json",)
        return RunReadiness(not missing, purpose, "legacy_artifacts", missing)

    if purpose != "ask":
        raise ValueError(f"Unsupported run-readiness purpose: {purpose}")

    current_state = _current_index_state(run)
    published_roles: dict[str, str] | None = None
    artifact_manifest_path = run / "artifact-manifest.json"
    if current_state is None and (
        artifact_manifest_path.exists() or artifact_manifest_path.is_symlink()
    ):
        artifact_verification = verify_query_artifacts(run)
        detail = (
            artifact_verification.errors[0]
            if artifact_verification.errors
            else "terminal index state is missing"
        )
        return RunReadiness(
            False,
            purpose,
            "published_artifacts",
            (f"published artifact manifest is invalid: {detail}",),
        )
    if current_state:
        state_name, state_status, stages_succeeded, state_payload = current_state
        if state_status in {"running", "failed", "pending"}:
            return RunReadiness(
                False,
                purpose,
                f"state:{state_name}",
                (f"index state is {state_status}",),
            )
        if state_status == "succeeded" and not stages_succeeded:
            return RunReadiness(
                False,
                purpose,
                f"state:{state_name}",
                ("index state is inconsistent with its stage results",),
            )
        manifest_reference = state_payload.get("artifact_manifest")
        has_published_identity = bool(
            state_payload.get("attempt_id")
            or (run / "artifact-manifest.json").exists()
        )
        if (
            state_status == "succeeded"
            and has_published_identity
            and manifest_reference is None
        ):
            return RunReadiness(
                False,
                purpose,
                f"state:{state_name}",
                ("successful index state has no artifact manifest binding",),
            )
        if state_status == "succeeded" and manifest_reference is not None:
            expected_manifest = run / "artifact-manifest.json"
            referenced_manifest = Path(str(manifest_reference)).expanduser()
            if not referenced_manifest.is_absolute():
                referenced_manifest = run / referenced_manifest
            try:
                reference_matches = referenced_manifest.resolve() == expected_manifest.resolve()
            except (OSError, RuntimeError):
                reference_matches = False
            if not reference_matches:
                return RunReadiness(
                    False,
                    purpose,
                    f"state:{state_name}",
                    ("index state references an unsafe artifact manifest",),
                )
            artifact_verification = verify_query_artifacts(run)
            if not artifact_verification.valid:
                detail = (
                    artifact_verification.errors[0]
                    if artifact_verification.errors
                    else "invalid"
                )
                return RunReadiness(
                    False,
                    purpose,
                    f"state:{state_name}",
                    (f"published artifact manifest is invalid: {detail}",),
                )
            published_roles = _published_roles(run)

    signal = _state_readiness_signal(run)

    missing: list[str] = []
    for name in ("tree.pkl", "tree.json"):
        valid = (
            _valid_tree_json(run)
            if name == "tree.json"
            else _safe_nonempty_file(run, run / name)
        )
        if not valid:
            missing.append(name)

    if published_roles:
        graph_path = _role_path(run, published_roles, "graph")
        graph_ready = bool(graph_path and _valid_graph_json(run, graph_path))
    else:
        try:
            graph_ready = any(
                _valid_graph_json(run, path)
                for path in run.glob("graph_data*.json")
            )
        except OSError:
            graph_ready = False
    if not graph_ready:
        missing.append("graph_data*.json")
    if published_roles:
        retrieval_ready, entity_ready, vector_errors = _published_store_readiness(
            run, published_roles
        )
    else:
        retrieval_ready, entity_ready, vector_errors = _unpublished_store_readiness(run)
    if not retrieval_ready:
        missing.append("ready scoped retrieval index")
    if not entity_ready:
        missing.append("ready scoped entity index")
    missing.extend(vector_errors)

    source = (
        f"state:{signal[0]}+verified_artifacts"
        if signal
        else "legacy_artifacts"
    )
    return RunReadiness(
        not missing,
        purpose,
        source,
        tuple(missing),
    )


def find_runs(runs_dir: Path, *, purpose: str = "ask") -> list[Path]:
    root = runs_dir.expanduser().resolve()
    if not root.is_dir():
        return []

    candidates: set[Path] = set()
    discovery_names = {"tree.json", *_READY_MARKERS, *_STATE_FILES}
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.name not in discovery_names:
                continue
            run = path.parent.resolve()
            if _is_descendant(run, root) and run.is_dir():
                candidates.add(run)
    except OSError:
        pass

    ready = [
        run
        for run in candidates
        if inspect_run_readiness(run, purpose=purpose).ready
    ]

    def publication_time(path: Path) -> float:
        """Use immutable/publication files, never mutable directory mtime."""

        for state_name in ("index-state.json", "index_state.json"):
            state_path = path / state_name
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(state, dict):
                ended_at = state.get("ended_at")
                if isinstance(ended_at, str):
                    try:
                        return datetime.fromisoformat(
                            ended_at.replace("Z", "+00:00")
                        ).timestamp()
                    except ValueError:
                        pass
        for artifact_name in (
            "artifact-manifest.json",
            "index-state.json",
            "index_state.json",
            "tree.pkl",
            "tree.json",
        ):
            try:
                return (path / artifact_name).stat().st_mtime
            except OSError:
                continue
        return 0.0

    return sorted(
        ready,
        key=lambda path: (publication_time(path), str(path)),
        reverse=True,
    )


def resolve_run(
    runs_dir: Path,
    requested: str | None,
    *,
    purpose: str = "ask",
) -> Path:
    """Resolve a run strictly below ``runs_dir`` and validate its readiness."""

    root = runs_dir.expanduser().resolve()
    if requested:
        requested_path = Path(requested).expanduser()
        run = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (root / requested_path).resolve()
        )
        if run.is_file():
            run = run.parent.resolve()
    else:
        candidates = find_runs(root, purpose=purpose)
        if not candidates:
            raise FileNotFoundError(
                f"No {purpose}-ready runs were found under configured runs directory {root}"
            )
        run = candidates[0]

    if not _is_descendant(run, root):
        raise ValueError(
            f"Run must be a descendant of configured runs directory {root}: {run}"
        )
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run}")

    readiness = inspect_run_readiness(run, purpose=purpose)
    if not readiness.ready:
        missing = ", ".join(readiness.missing)
        raise FileNotFoundError(
            f"Run is not ready for {purpose}: {run}. Missing or empty: {missing}"
        )
    return run


def resolve_run_config(
    run_dir: Path,
    fallback: Path,
    *,
    allow_legacy_fallback: bool,
) -> Path:
    """Use the immutable run snapshot, with opt-in support for old runs."""

    run = run_dir.resolve()
    snapshot = run / "run_config.yaml"
    if snapshot.exists() and not _safe_nonempty_file(run, snapshot):
        raise ValueError(f"Run configuration snapshot is unsafe or empty: {snapshot}")
    if _safe_nonempty_file(run, snapshot):
        return snapshot.resolve()

    if not allow_legacy_fallback:
        raise FileNotFoundError(
            f"Legacy run has no configuration snapshot: {snapshot}. "
            "Re-index it, or explicitly pass --allow-legacy-config to use the "
            "current system configuration (results may not be reproducible)."
        )

    selected = fallback.expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Fallback system configuration not found: {selected}")
    return selected


def _write_execution_trace(path: Path, payload: dict[str, object]) -> str | None:
    try:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
        return None
    except (OSError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def run_provenance(run_dir: Path) -> dict[str, object]:
    """Load the non-secret identity fields that bind a query to its source."""

    manifest_path = run_dir / "run-manifest.json"
    if not _safe_nonempty_file(run_dir, manifest_path):
        return {"manifest": None, "run_dir": str(run_dir)}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"manifest": str(manifest_path), "run_dir": str(run_dir)}
    if not isinstance(payload, dict):
        return {"manifest": str(manifest_path), "run_dir": str(run_dir)}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    controller = (
        payload.get("controller")
        if isinstance(payload.get("controller"), dict)
        else {}
    )
    fingerprint = (
        controller.get("code_fingerprint")
        if isinstance(controller.get("code_fingerprint"), dict)
        else {}
    )
    artifact_manifest_path = run_dir / "artifact-manifest.json"
    artifact_payload: dict[str, object] = {}
    if _safe_nonempty_file(run_dir, artifact_manifest_path):
        try:
            loaded_artifacts = json.loads(
                artifact_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        else:
            if isinstance(loaded_artifacts, dict):
                artifact_payload = loaded_artifacts
    runtime_payload: dict[str, object] = {}
    runtime_path: Path | None = None
    runtime_reference = artifact_payload.get("runtime_manifest")
    if isinstance(runtime_reference, dict) and isinstance(
        runtime_reference.get("path"), str
    ):
        candidate = run_dir / str(runtime_reference["path"])
        if _safe_nonempty_file(run_dir, candidate):
            runtime_path = candidate.resolve()
            try:
                loaded_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            else:
                if isinstance(loaded_runtime, dict):
                    runtime_payload = loaded_runtime
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "run_dir": str(run_dir),
        "revision_id": payload.get("revision_id"),
        "document_id": payload.get("document_id"),
        "source_path": source.get("path"),
        "source_sha256": source.get("sha256"),
        "source_snapshot": source.get("snapshot_path"),
        "controller_code_sha256": fingerprint.get("sha256"),
        "artifact_manifest": (
            str(artifact_manifest_path) if artifact_payload else None
        ),
        "artifact_manifest_sha256": (
            _sha256_file(artifact_manifest_path) if artifact_payload else None
        ),
        "artifact_set_sha256": artifact_payload.get("artifact_set_sha256"),
        "artifact_file_count": artifact_payload.get("file_count"),
        "index_attempt_id": artifact_payload.get("attempt_id"),
        "runtime_manifest": str(runtime_path) if runtime_payload else None,
        "runtime_manifest_sha256": (
            _sha256_file(runtime_path) if runtime_payload and runtime_path else None
        ),
        "environment_fingerprint_sha256": runtime_payload.get(
            "environment_fingerprint_sha256"
        ),
    }


def ask(*, run_dir: Path, system_config: Path, question: str) -> dict[str, object]:
    """Run one compatibility BookRAG question and return trace references."""

    if not question.strip():
        raise ValueError("Question must not be empty")

    # End-to-end latency starts before importing or loading any runtime/model/index.
    started = time.monotonic()
    query_id = f"q_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    query_root = run_dir / "cli_queries"
    query_dir = query_root / query_id
    execution_path = query_dir / "execution.json"
    result_path = query_dir / "result.json"
    provenance = run_provenance(run_dir)
    config_sha256: str | None = None
    stage = "create_query_trace"
    try:
        query_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        query_root.chmod(0o700)
        query_dir.mkdir(exist_ok=False, mode=0o700)
    except OSError as exc:
        details: dict[str, object] = {
            "schema_version": 1,
            "status": "failed",
            "query_id": query_id,
            "stage": stage,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "run_dir": str(run_dir),
            "provenance": provenance,
            "failure_trace": None,
        }
        raise CompatibilityAskError(
            f"Could not create query trace directory under {run_dir}: {exc}",
            details,
        ) from exc

    agent = None
    answer: object = None
    node_ids: object = None
    failure: BaseException | None = None
    cleanup: dict[str, object] = {"attempted": False, "succeeded": True}
    token_scope = None
    token_usage: dict[str, object] | None = None
    run_log_info: RunLogInfo | None = None

    try:
        stage = "validate_configuration"
        if not system_config.is_file():
            raise FileNotFoundError(
                f"System configuration not found: {system_config}"
            )
        config_sha256 = _sha256_file(system_config)

        stage = "import_runtime"
        from Core.configs.system_config import load_system_config
        from Core.provider.TokenTracker import TokenTracker
        from Core.rag import create_rag_agent
        from Core.utils.resource_loader import prepare_rag_dependencies

        token_scope = TokenTracker.get_instance().scope(name=query_id)

        # Existing provider callbacks continue using the compatibility facades;
        # these two context-local scopes bind model usage and log records to the
        # same query even when several requests overlap in one service process.
        with RunLogScope(
            query_dir,
            logger_names=("Core", "docarag", __name__),
            run_id=query_id,
        ) as run_log_info:
            with token_scope:
                stage = "load_configuration"
                cfg = load_system_config(str(system_config))
                cfg.save_path = str(run_dir)

                stage = "load_index_and_models"
                dependencies = prepare_rag_dependencies(cfg=cfg)
                agent = create_rag_agent(
                    strategy_config=cfg.rag.strategy_config,
                    llm_config=cfg.llm,
                    vlm_config=cfg.vlm,
                    **dependencies,
                )

                stage = "generate_answer"
                answer, node_ids = agent.generation(question, query_dir)
    except (Exception, KeyboardInterrupt) as exc:
        failure = exc
    finally:
        if agent is not None:
            cleanup["attempted"] = True
            try:
                agent.close()
            except Exception as exc:  # cleanup must not hide the primary failure
                cleanup["succeeded"] = False
                cleanup["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        if token_scope is not None:
            token_scope.record_stage("query")
            token_usage = {"scope": query_id, **token_scope.snapshot()}

    elapsed = round(time.monotonic() - started, 3)
    retrieval_path = query_dir / "retrieval_res.json"
    if failure is not None:
        details = {
            "schema_version": 1,
            "status": "failed",
            "query_id": query_id,
            "stage": stage,
            "error": {
                "type": type(failure).__name__,
                "message": str(failure) or type(failure).__name__,
            },
            "elapsed_seconds": elapsed,
            "run_dir": str(run_dir),
            "system_config": str(system_config),
            "system_config_sha256": config_sha256,
            "question": question,
            "provenance": provenance,
            "token_usage": token_usage,
            "run_log": str(run_log_info.log_file) if run_log_info else None,
            "retrieval_trace": (
                str(retrieval_path) if retrieval_path.is_file() else None
            ),
            "cleanup": cleanup,
            "failure_trace": str(execution_path),
        }
        trace_error = _write_execution_trace(execution_path, details)
        if trace_error:
            details["failure_trace"] = None
            details["trace_write_error"] = trace_error
        raise CompatibilityAskError(
            f"Query failed during {stage}: {details['error']['message']}",
            details,
            exit_code=130 if isinstance(failure, KeyboardInterrupt) else 2,
        ) from failure

    if node_ids is None:
        normalized_node_ids: list[object] = []
    elif isinstance(node_ids, (str, bytes)):
        normalized_node_ids = [node_ids]
    else:
        try:
            normalized_node_ids = list(node_ids)
        except TypeError:
            normalized_node_ids = [node_ids]

    result_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "succeeded",
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "node_ids": normalized_node_ids,
        "elapsed_seconds": elapsed,
        "system_config": str(system_config),
        "system_config_sha256": config_sha256,
        "provenance": provenance,
        "token_usage": token_usage,
        "run_log": str(run_log_info.log_file) if run_log_info else None,
    }
    result_error = _write_execution_trace(result_path, result_payload)

    details = {
        "schema_version": 1,
        "status": "succeeded",
        "query_id": query_id,
        "stage": "complete",
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "system_config": str(system_config),
        "system_config_sha256": config_sha256,
        "question": question,
        "provenance": provenance,
        "token_usage": token_usage,
        "run_log": str(run_log_info.log_file) if run_log_info else None,
        "retrieval_trace": (
            str(retrieval_path) if retrieval_path.is_file() else None
        ),
        "cleanup": cleanup,
        "result": str(result_path) if result_error is None else None,
    }
    trace_error = _write_execution_trace(execution_path, details)

    result: dict[str, object] = {
        "answer": answer,
        "node_ids": normalized_node_ids,
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "query_dir": str(query_dir),
        "retrieval_trace": details["retrieval_trace"],
        "execution_trace": str(execution_path) if trace_error is None else None,
        "result_trace": str(result_path) if result_error is None else None,
        "provenance": provenance,
        "token_usage": token_usage,
        "run_log": str(run_log_info.log_file) if run_log_info else None,
    }
    if cleanup.get("succeeded") is False:
        result["cleanup_warning"] = cleanup.get("error")
    if trace_error:
        result["trace_warning"] = trace_error
    if result_error:
        result["result_warning"] = result_error
    return result


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal DocARAG compatibility bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("--run")
    ask_parser.add_argument("--runs-dir", required=True)
    ask_parser.add_argument(
        "--fallback-system-config",
        required=True,
        help="Configured fallback; run snapshot remains authoritative",
    )
    ask_parser.add_argument("--allow-legacy-config", action="store_true")
    ask_parser.add_argument("--json", action="store_true")
    ask_parser.add_argument("question")
    args = parser.parse_args(argv)

    try:
        run = resolve_run(Path(args.runs_dir), args.run, purpose="ask")
        system_config = resolve_run_config(
            run,
            Path(args.fallback_system_config),
            allow_legacy_fallback=args.allow_legacy_config,
        )
        result = ask(
            run_dir=run,
            system_config=system_config,
            question=args.question,
        )
    except CompatibilityAskError as exc:
        if args.json:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
            trace = exc.details.get("failure_trace")
            if trace:
                print(f"Failure trace: {trace}", file=sys.stderr)
        return exc.exit_code
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "stage": "validate_run",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "failure_trace": None,
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["answer"])
        if result["retrieval_trace"]:
            print(f"\nEvidence trace: {result['retrieval_trace']}")
        print(f"Execution trace: {result['execution_trace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
