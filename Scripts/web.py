#!/usr/bin/env python3
"""
Local BookRAG web UI.

Usage:
    conda run -n gbc-rag python Scripts/web.py [RUN_DIR] [--host 127.0.0.1] [--port 8765]
or:
    bash Scripts/web.sh [RUN_DIR]
"""
import argparse
import errno
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml


REPO = Path(__file__).resolve().parent.parent
WEB_ROOT = REPO / "Scripts" / "webui"
DEFAULT_CONFIG = os.environ.get("BOOKRAG_CONFIG", "config/ollama-gbc.yaml")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from docarag.cli import _load_project_environment
from docarag.compat import inspect_run_readiness, run_provenance
from docarag.provenance import query_readiness_stat_key
from docarag.provenance import sha256_file
from docarag.run_logging import RunLogInfo, RunLogScope
from Core.provider.TokenTracker import TokenTracker

MAX_REQUEST_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
STAGE_ORDER = ("tree", "graph", "vdb")
_STAGE_TOKEN_KEYS = {
    "tree": ("tree_index_construction", "tree_node_summary"),
    "graph": ("kg_extraction", "kg_refinement"),
    "vdb": (),
}
_GRAPH_VALIDATION_CACHE = {}
_ROLE_CACHE = {}


def uploads_dir():
    return runs_root() / "_uploads"


def safe_pdf_name(raw):
    """Return a bounded ``.pdf`` filename that cannot escape the upload dir."""

    stem = Path(unquote(str(raw or ""))).name
    cleaned = re.sub(r"[^\w.\- ]+", "_", stem).strip(" .") or "document"
    if cleaned.lower().endswith(".pdf"):
        cleaned = cleaned[:-4].strip(" .") or "document"
    return f"{cleaned[:80]}.pdf"


def find_run_dirs():
    """Return every run directory, including ones still being indexed.

    ``find_runs`` only sees admitted runs, so a run whose admission is still
    being written would be missing from the library exactly while its progress
    matters most.
    """

    root = runs_root()
    if not root.is_dir():
        return []
    runs = set()
    for marker in ("**/run-manifest.json", "**/index-state.json"):
        for path in root.glob(marker):
            run_dir = path.parent.resolve()
            try:
                relative = run_dir.relative_to(root)
            except ValueError:
                continue
            if relative != Path(".") and run_dir.is_dir():
                runs.add(run_dir)
    return sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)


def _iso_seconds(start, end):
    if not start or not end:
        return None
    try:
        started = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        ended = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((ended - started).total_seconds(), 0.0)


def _read_json_if_present(path):
    try:
        return read_json(path) or {}
    except (OSError, ValueError):
        return {}


_NODE_COUNT_CACHE = {}


def _tree_node_count(run_dir):
    """Count a run's tree nodes, cached against the published file's identity."""

    path = run_dir / "tree.json"
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _NODE_COUNT_CACHE:
        nodes = _read_json_if_present(path).get("nodes")
        if not isinstance(nodes, list):
            return None
        _NODE_COUNT_CACHE.clear()
        _NODE_COUNT_CACHE[key] = len(nodes)
    return _NODE_COUNT_CACHE[key]


def _stage_progress(run_dir, name):
    """Return measured progress for a stage that publishes per-item results."""

    if name != "graph":
        return None
    directory = run_dir / "kg_extractor_res"
    if not directory.is_dir():
        return None
    try:
        done = sum(1 for item in directory.iterdir() if item.suffix == ".json")
    except OSError:
        return None
    total = _tree_node_count(run_dir)
    if not total:
        return None
    return {"done": min(done, total), "total": total}


def _stage_history(run_dirs):
    """Median duration of each stage over runs that finished it."""

    samples = {name: [] for name in STAGE_ORDER}
    for run in run_dirs:
        stages = (
            _read_json_if_present(run / "index-state.json").get("stages") or {}
        )
        for name in STAGE_ORDER:
            entry = stages.get(name) or {}
            if entry.get("status") != "succeeded":
                continue
            seconds = _iso_seconds(entry.get("started_at"), entry.get("ended_at"))
            if seconds:
                samples[name].append(seconds)
    history = {}
    for name, values in samples.items():
        if values:
            values.sort()
            history[name] = values[len(values) // 2]
    return history


def _elapsed_seconds(started):
    if not started:
        return None
    try:
        start = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(start.tzinfo) if start.tzinfo else datetime.now()
    return max((now - start).total_seconds(), 0.0)


def run_progress(run_dir, history=None):
    """Report one run's stages and metrics from its own published state."""

    state = _read_json_if_present(run_dir / "index-state.json")
    stats = _read_json_if_present(run_dir / "indexing_stats.json")
    history = history or {}
    raw_stages = state.get("stages") or {}
    stages = []
    for name in STAGE_ORDER:
        entry = raw_stages.get(name) or {}
        status = entry.get("status") or "pending"
        elapsed = _elapsed_seconds(entry.get("started_at")) if status == "running" else None
        progress = _stage_progress(run_dir, name) if status == "running" else None
        eta = eta_basis = None
        if status == "running":
            if progress and progress["done"] and elapsed:
                rate = progress["done"] / elapsed
                if rate > 0:
                    eta = (progress["total"] - progress["done"]) / rate
                    eta_basis = "measured"
            if eta is None and history.get(name) and elapsed is not None:
                remaining = history[name] - elapsed
                if remaining > 0:
                    eta, eta_basis = remaining, "past runs"
        stages.append(
            {
                "name": name,
                "status": status,
                "startedAt": entry.get("started_at"),
                "endedAt": entry.get("ended_at"),
                "seconds": _iso_seconds(entry.get("started_at"), entry.get("ended_at")),
                "elapsed": elapsed,
                "progress": progress,
                "etaSeconds": eta,
                "etaBasis": eta_basis,
                "typicalSeconds": history.get(name),
                "tokens": {
                    key: value.get("total_tokens")
                    for key, value in (stats.get("token_stage_history") or {}).items()
                    if isinstance(value, dict) and key in _STAGE_TOKEN_KEYS[name]
                },
                "exitCode": entry.get("exit_code"),
                "error": entry.get("error"),
            }
        )
    tokens = {}
    for stage_name, usage in (stats.get("token_stage_history") or {}).items():
        if isinstance(usage, dict):
            tokens[stage_name] = usage.get("total_tokens")
    return {
        "runDir": str(run_dir),
        "phase": state.get("phase"),
        "overallStatus": state.get("overall_status"),
        "currentStage": state.get("current_stage"),
        "startedAt": state.get("started_at"),
        "updatedAt": state.get("updated_at"),
        "endedAt": state.get("ended_at"),
        "error": state.get("error"),
        "stages": stages,
        "viewable": run_completed(run_dir),
        "metrics": {
            "buildTreeSeconds": stats.get("build_tree_time"),
            "tokensByStage": tokens,
            "totalTokens": sum(v for v in tokens.values() if isinstance(v, int)),
        },
    }


def run_completed(run_dir):
    """True when the driver finished every stage and sealed the run."""

    state = _read_json_if_present(run_dir / "index-state.json")
    if not state:
        return False
    stages = state.get("stages") or {}
    return state.get("overall_status") == "succeeded" and all(
        (stages.get(name) or {}).get("status") == "succeeded" for name in STAGE_ORDER
    )


def run_summary(run_dir, history=None):
    progress = run_progress(run_dir, history)
    source = _read_json_if_present(run_dir / "run-manifest.json").get("source") or {}
    source_name = source.get("display_name")
    if not source_name and source.get("path"):
        source_name = Path(str(source["path"])).name
    name = source_name or run_dir.name
    progress["name"] = name
    progress["mtime"] = run_dir.stat().st_mtime
    return progress


def _qdrant_delete_scope(qdrant_url, timeout_seconds, collection, scope):
    """Delete exactly one run's points, leaving every other run untouched."""

    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=qdrant_url, timeout=timeout_seconds)
    try:
        names = {item.name for item in client.get_collections().collections}
        if collection not in names:
            return 0
        scope_filter = models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in sorted(scope.items())
            ]
        )
        before = client.count(collection, count_filter=scope_filter, exact=True).count
        if before:
            client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=scope_filter),
                wait=True,
            )
        return before
    finally:
        client.close()


def delete_run_everywhere(run_dir):
    """Remove one run's vectors, ledger rows, and files."""

    import shutil

    manifest = _read_json_if_present(run_dir / "run-manifest.json")
    config = read_run_config(run_dir) or {}
    removed = {"vectors": {}, "rows": {}, "runDir": str(run_dir)}

    tenant = manifest.get("tenant_id")
    corpus = manifest.get("corpus_id")
    document_id = manifest.get("document_id")
    document_version_id = manifest.get("document_version_id")
    revision_id = manifest.get("revision_id")
    identified = all(
        isinstance(value, str) and value
        for value in (tenant, corpus, document_id, document_version_id, revision_id)
    )

    if identified:
        vdb_cfg = config.get("vdb") or {}
        graph_cfg = config.get("graph") or {}
        qdrant_url = vdb_cfg.get("qdrant_url")
        scope = {
            "tenant_id": tenant,
            "corpus_id": corpus,
            "document_version_id": document_version_id,
            "index_revision_id": revision_id,
        }
        collections = [
            name
            for name in (
                vdb_cfg.get("collection_name"),
                graph_cfg.get("entity_collection_name"),
            )
            if isinstance(name, str) and name
        ]
        if qdrant_url:
            for collection in collections:
                try:
                    removed["vectors"][collection] = _qdrant_delete_scope(
                        qdrant_url,
                        float(vdb_cfg.get("timeout_seconds") or 10.0),
                        collection,
                        scope,
                    )
                except Exception as exc:
                    removed["vectors"][collection] = f"failed: {type(exc).__name__}"

        try:
            from docarag.persistence import DocumentScope, PostgresLedger
            from docarag.settings import load_settings

            ledger = PostgresLedger(load_settings().resolve_database_url())
            removed["rows"] = ledger.delete_revision_evidence(
                DocumentScope(tenant, corpus, document_id, document_version_id),
                revision_id,
            )
        except Exception as exc:
            removed["rows"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        removed["rows"] = {"error": "run has no published identity to purge"}

    shutil.rmtree(run_dir)
    removed["filesRemoved"] = True
    return removed


def start_index(pdf_path):
    """Launch the CLI indexer, which owns admission and the stage ledger."""

    completed = subprocess.run(
        [sys.executable, "-m", "docarag.cli", "index", str(pdf_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(output.strip() or "docarag index failed")
    match = re.search(r"run dir\s*:\s*(\S+)", output)
    return match.group(1) if match else None


def token_usage_payload(scope, query_id):
    """Freeze one query's usage and final stage delta for trace persistence."""

    scope.record_stage("query")
    return {"scope": query_id, **scope.snapshot()}


class RequestBodyError(ValueError):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class RunNotReadyError(RuntimeError):
    """A tree can be viewed, but the full BookRAG index is not queryable."""


def is_loopback_host(host):
    if str(host).lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def runs_root():
    configured = Path(os.environ.get("DOCARAG_RUNS_DIR", str(REPO / "runs"))).expanduser()
    if not configured.is_absolute():
        configured = REPO / configured
    return configured.resolve()


def read_run_config(run_dir):
    path = Path(run_dir) / "run_config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def configured_graph_variant(run_dir):
    graph = read_run_config(run_dir).get("graph") or {}
    if not isinstance(graph, dict):
        return None
    return "basic" if graph.get("refine_type") == "basic" else None


def published_role(run_dir, role):
    """Return ``(has_manifest, safe_role_path)`` using a small stat-keyed cache."""

    run_dir = Path(run_dir).resolve()
    manifest = run_dir / "artifact-manifest.json"
    try:
        stat_result = manifest.stat()
    except OSError:
        return False, None
    cache_key = (str(manifest), stat_result.st_mtime_ns, stat_result.st_size)
    roles = _ROLE_CACHE.get(cache_key)
    if roles is None:
        try:
            payload = read_json(manifest)
        except (OSError, ValueError, TypeError):
            return True, None
        loaded_roles = payload.get("roles") if isinstance(payload, dict) else None
        roles = loaded_roles if isinstance(loaded_roles, dict) else {}
        if len(_ROLE_CACHE) >= 64:
            _ROLE_CACHE.clear()
        _ROLE_CACHE[cache_key] = roles

    value = roles.get(role)
    if not isinstance(value, str) or not value:
        return True, None
    path = run_dir / value
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(run_dir)
    except (OSError, RuntimeError, ValueError):
        return True, None
    if path.is_symlink() or resolved != path.absolute():
        return True, None
    return True, resolved


def _valid_graph_artifact(path):
    try:
        stat_result = path.stat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    cache_key = (str(resolved), stat_result.st_mtime_ns, stat_result.st_size)
    cached = _GRAPH_VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = read_json(path)
    except (OSError, ValueError, TypeError):
        valid = False
    else:
        graph = data.get("graph", data) if isinstance(data, dict) else None
        valid = isinstance(graph, dict) and isinstance(graph.get("nodes"), list)
    if len(_GRAPH_VALIDATION_CACHE) >= 128:
        _GRAPH_VALIDATION_CACHE.clear()
    _GRAPH_VALIDATION_CACHE[cache_key] = valid
    return valid


def find_graph_artifact(run_dir):
    """Return the active graph variant, falling back to the freshest valid one."""
    run_dir = Path(run_dir)
    has_manifest, published = published_role(run_dir, "graph")
    if has_manifest:
        return published if published and _valid_graph_artifact(published) else None

    candidates = [
        path
        for path in run_dir.glob("graph_data*.json")
        if path.is_file() and _valid_graph_artifact(path)
    ]
    if not candidates:
        return None

    variant = configured_graph_variant(run_dir)
    preferred = run_dir / (f"graph_data_{variant}.json" if variant else "graph_data.json")
    if preferred in candidates:
        return preferred
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def configured_attestation_dir(run_dir):
    run_dir = Path(run_dir)
    vdb = read_run_config(run_dir).get("vdb") or {}
    if not isinstance(vdb, dict):
        return None
    value = vdb.get("manifest_dir_name")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return path


def _is_retrieval_artifact_dir(path):
    return (
        (path / "docarag-qdrant-index.ready.json").is_file()
        or (path / "bm25_index.pkl").is_file()
    )


def find_retrieval_artifact_dir(run_dir):
    """Find the run-local Qdrant attestation or BM25 artifact directory."""
    run_dir = Path(run_dir)
    has_manifest, published = published_role(run_dir, "retrieval_attestation")
    if has_manifest and published:
        return published.parent if published.is_file() else None
    has_manifest, published = published_role(run_dir, "lexical_store")
    if has_manifest:
        return published.parent if published and published.is_file() else None

    configured = configured_attestation_dir(run_dir)
    if configured is not None:
        return configured

    try:
        candidates = [
            path
            for path in run_dir.iterdir()
            if path.is_dir() and _is_retrieval_artifact_dir(path)
        ]
    except OSError:
        candidates = []
    if not candidates:
        return None

    non_entity = [
        path for path in candidates if not path.name.lower().startswith("entity_attestation")
    ]
    pool = non_entity or candidates
    return max(pool, key=lambda path: (path.stat().st_mtime_ns, path.name)).resolve()


def find_runs():
    runs = set()
    root = runs_root()
    if not root.is_dir():
        return []
    for manifest_path in root.glob("**/run-manifest.json"):
        run_dir = manifest_path.parent.resolve()
        try:
            relative = run_dir.relative_to(root)
        except ValueError:
            continue
        if relative != Path(".") and run_dir.is_dir():
            runs.add(run_dir)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def pick_run(path_arg):
    root = runs_root()
    if path_arg:
        path = Path(path_arg).expanduser()
        if path.is_file():
            path = path.parent
        path = path.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"Run must stay below configured runs directory {root}: {path}"
            ) from exc
        if relative == Path(".") or not path.is_dir():
            raise SystemExit(f"Invalid run directory: {path}")
        return path
    runs = find_runs()
    if not runs:
        raise SystemExit(f"No indexed runs found under {runs_root()}")
    return runs[0].resolve()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_private_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def node_content(node):
    """Return a node's own text.

    A table's markup is published separately as ``tableBody``; falling back to
    it here would render the grid a second time and title the node with raw
    HTML.
    """

    meta = node.get("meta_info") or {}
    return (
        meta.get("content")
        or meta.get("text")
        or meta.get("caption")
        or meta.get("title")
        or ""
    )


def normalize_type(value):
    return str(value or "").replace("NodeType.", "")


def citations_from_retrieval(retrieval):
    """Flatten each iteration's ordered evidence labels into one reference list.

    Labels are already unique and ordered by the answer generator (plain
    E-numbers for a simple question, Q1-E1/Q2-E1 style for a complex one),
    so this only concatenates -- it must not re-derive or re-sort them.
    """

    citations = []
    seen = set()
    for iteration in (retrieval or {}).get("iterations") or []:
        for citation in iteration.get("citations") or []:
            label = citation.get("label")
            if not label or label in seen:
                continue
            seen.add(label)
            citations.append(
                {
                    "label": label,
                    "nodeId": citation.get("index_id"),
                    "type": normalize_type(citation.get("type")),
                    "page": citation.get("page"),
                    "snippet": citation.get("snippet") or "",
                    "hasImage": bool(citation.get("has_image")),
                }
            )
    return citations


def bind_server(host, port, handler, attempts=20):
    last_error = None
    last_candidate = min(65535, port + attempts - 1)
    for candidate in range(port, last_candidate + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler), candidate
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
    raise SystemExit(
        f"Could not start web UI: ports {port}-{last_candidate} are in use"
    ) from last_error


class AppState:
    def __init__(self, run_dir=None):
        self.run_dir = run_dir
        self.chat_lock = threading.Lock()
        self.rag_agent = None
        self._rag_identity_key = None
        self._store = None
        self._tree_cache = None
        self._tree_cache_key = None
        self._graph_cache = None
        self._graph_cache_key = None
        self._readiness_cache_key = None
        self._readiness_cache = None

    def store(self):
        """The document store for this run; evidence is immutable per revision."""

        self.require_run()
        if self._store is None:
            from docarag.document_store import DocumentStore

            self._store = DocumentStore.open(self.run_dir)
        return self._store

    def require_run(self):
        """Return the selected run or reject document-scoped operations."""

        if self.run_dir is None:
            raise RunNotReadyError("Select a document from the Library first")
        return self.run_dir

    def switch_run(self, run_dir):
        """Rebind every cache and release the agent held for the old run."""

        with self.chat_lock:
            if run_dir == self.run_dir:
                return
            if self.rag_agent is not None:
                close = getattr(self.rag_agent, "close", None)
                if callable(close):
                    close()
            self.rag_agent = None
            self._rag_identity_key = None
            if self._store is not None:
                self._store.close()
            self._store = None
            self._tree_cache = None
            self._tree_cache_key = None
            self._graph_cache = None
            self._graph_cache_key = None
            self._readiness_cache_key = None
            self._readiness_cache = None
            self.run_dir = run_dir

    def chat_readiness(self):
        """Cache readiness until a query-loaded artifact/store identity changes."""

        self.require_run()
        key = query_readiness_stat_key(self.run_dir)
        if self._readiness_cache_key != key or self._readiness_cache is None:
            self._readiness_cache = inspect_run_readiness(
                self.run_dir, purpose="ask"
            )
            self._readiness_cache_key = key
        return self._readiness_cache

    @property
    def doc_name(self):
        self.require_run()
        return Handler._safe_doc_name(self.run_dir)

    def tree_payload(self):
        # Cached only until the run's own state changes -- a query made
        # while a stage was still running (before its evidence was fully
        # persisted) must not be served forever afterward as if it were
        # complete. See the matching note on graph_payload for the incident
        # this fixes.
        self.require_run()
        key = query_readiness_stat_key(self.run_dir)
        if self._tree_cache is not None and self._tree_cache_key == key:
            return self._tree_cache

        store = self.store()
        views = store.tree.nodes()
        structures = store.tree.structures_for_nodes(
            [view.index_id for view in views]
        )
        # The normalized SQL view may omit the legacy physical img_path while
        # the published BookRAG tree still owns a valid MinerU crop. Advertise
        # that crop explicitly so the frontend renders its Image section and
        # lets /api/tree-image perform the safe physical-path resolution.
        raw_image_node_ids = {
            item.get("index_id")
            for item in (
                _read_json_if_present(self.run_dir / "tree.json").get("nodes")
                or []
            )
            if isinstance((item.get("meta_info") or {}).get("img_path"), str)
            and (item.get("meta_info") or {}).get("img_path").strip()
        }
        children = {}
        for view in views:
            if view.parent_id is not None:
                children.setdefault(view.parent_id, []).append(view.index_id)

        flat = []
        root_id = None
        for view in views:
            evidence = dict(view.evidence or {})
            meta = view.meta_info
            if view.parent_id is None:
                root_id = view.index_id
            page = evidence.get("page")
            if meta.img_path:
                evidence.setdefault("crop_uri", meta.img_path)
            flat.append(
                {
                    "id": view.index_id,
                    "parentId": view.parent_id,
                    "type": view.type if view.type == "root" else view.type.upper(),
                    "page": page + 1 if isinstance(page, int) else None,
                    "titleLevel": evidence.get("title_level"),
                    # A node's display text is its evidence content -- caption
                    # and footnotes for a table, the description for an image.
                    # The retrieval feed prefers html/ocr channels and is for
                    # the reranker, not the reader.
                    "content": (
                        (evidence.get("content") or "")
                        if view.evidence
                        else meta.content or ""
                    ),
                    "tableBody": meta.table_body or "",
                    "caption": meta.caption or "",
                    "footnote": meta.footnote or "",
                    "summary": view.summary or "",
                    "meta": evidence,
                    "hasImage": bool(meta.img_path)
                    or view.index_id in raw_image_node_ids,
                    "structures": structures.get(view.index_id, []),
                    "children": children.get(view.index_id, []),
                }
            )

        payload = {
            "docName": self.doc_name,
            "runDir": str(self.run_dir),
            "rootId": root_id,
            "nodes": flat,
        }
        self._tree_cache = payload
        self._tree_cache_key = key
        return payload

    def graph_payload(self):
        # Cached only until the run's own state changes. Without this, a
        # request made while the graph stage was still running -- persist
        # had not written any entities/statements yet -- would cache an
        # empty payload and keep serving it forever, even long after the
        # real data existed in Postgres: the graph panel would show
        # "not loading" permanently for a run that actually finished fine.
        self.require_run()
        key = query_readiness_stat_key(self.run_dir)
        if self._graph_cache is not None and self._graph_cache_key == key:
            return self._graph_cache

        from Core.rag.stored_index import entity_node_name

        store = self.store()
        entities = store.tree.graph_entities()
        statements = store.tree.graph_statements()
        tree_nodes = {
            node["id"]: node for node in self.tree_payload().get("nodes", [])
        }

        degree = {}
        for statement in statements:
            for name_key, kind_key in (("source", "source_kind"), ("target", "target_kind")):
                node_id = entity_node_name(statement[name_key], statement[kind_key])
                degree[node_id] = degree.get(node_id, 0) + 1

        nodes = []
        for entity in entities:
            source_ids = entity["node_ids"]
            node_id = entity_node_name(entity["canonical"], entity["entity_kind"])
            nodes.append(
                {
                    "id": node_id,
                    "label": entity["canonical"],
                    "type": entity["entity_kind"] or "UNKNOWN",
                    "description": entity["description"],
                    "sourceIds": source_ids,
                    "sourceSnippets": [
                        tree_nodes[source_id].get("content")
                        or tree_nodes[source_id].get("summary")
                        or ""
                        for source_id in source_ids
                        if source_id in tree_nodes
                    ],
                    "degree": degree.get(node_id, 0),
                }
            )

        links = []
        for idx, statement in enumerate(statements):
            links.append(
                {
                    "id": f"rel-{idx}",
                    "source": entity_node_name(
                        statement["source"], statement["source_kind"]
                    ),
                    "target": entity_node_name(
                        statement["target"], statement["target_kind"]
                    ),
                    "sourceName": statement["source"],
                    "targetName": statement["target"],
                    "name": statement["predicate"],
                    "description": statement["description"],
                    "weight": statement["weight"],
                    "sourceIds": statement["node_ids"],
                }
            )

        payload = {
            "docName": self.doc_name,
            "runDir": str(self.run_dir),
            "nodes": nodes,
            "links": links,
        }
        self._graph_cache = payload
        self._graph_cache_key = key
        return payload

    def ensure_rag(self):
        readiness = self.chat_readiness()
        if not readiness.ready:
            raise RunNotReadyError(
                "Run is viewable but not ready for chat: " + ", ".join(readiness.missing)
            )
        current_identity = self._readiness_cache_key
        if self.rag_agent is not None and (
            self._rag_identity_key is None
            or self._rag_identity_key == current_identity
        ):
            self._rag_identity_key = current_identity
            return self.rag_agent
        if self.rag_agent is not None:
            close = getattr(self.rag_agent, "close", None)
            if callable(close):
                close()
            self.rag_agent = None

        from Core.configs.system_config import load_system_config
        from Core.rag import create_rag_agent
        from Core.utils.resource_loader import prepare_rag_dependencies

        run_config = self.run_dir / "run_config.yaml"
        cfg = load_system_config(str(run_config if run_config.is_file() else DEFAULT_CONFIG))
        cfg.save_path = str(self.run_dir)
        # cfg.graph.refine_type, loaded above from this run's own frozen
        # run_config.yaml, is already exactly what was used to build this
        # run -- graph_artifact_variant(cfg.graph) (Core/configs/graph_config.py)
        # is a pure function of cfg.graph alone, so recomputing it here would
        # always reproduce the same variant the build already used. There is
        # no build-time fallback/downgrade path that could make the actual
        # artifact disagree with a fresh config read. Inspecting a JSON
        # artifact's filename to "correct" refine_type was therefore
        # correcting a case that cannot happen, while depending on a JSON
        # file for something the store-backed config already answers.
        attestation_dir = find_retrieval_artifact_dir(self.run_dir)
        if attestation_dir is not None:
            cfg.vdb.manifest_dir_name = str(attestation_dir)
        dependencies = prepare_rag_dependencies(cfg=cfg)
        self.rag_agent = create_rag_agent(
            strategy_config=cfg.rag.strategy_config,
            llm_config=cfg.llm,
            vlm_config=cfg.vlm,
            **dependencies,
        )
        self._rag_identity_key = current_identity
        return self.rag_agent

    def ask(self, question):
        with self.chat_lock:
            started = time.monotonic()
            query_id = f"q_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            query_root = self.run_dir / "web_chat"
            query_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            query_root.chmod(0o700)
            q_dir = query_root / query_id
            q_dir.mkdir(mode=0o700)
            execution_path = q_dir / "execution.json"
            retrieval_path = q_dir / "retrieval_res.json"
            provenance = run_provenance(self.run_dir)
            config_path = self.run_dir / "run_config.yaml"
            if not config_path.is_file():
                config_path = Path(DEFAULT_CONFIG).expanduser().resolve()
            try:
                config_sha256 = sha256_file(config_path)
            except OSError:
                config_sha256 = None
            stage = "validate_readiness"
            agent_reused = self.rag_agent is not None
            usage_scope = TokenTracker.get_instance().scope(name=query_id)
            run_log_info: RunLogInfo | None = None
            try:
                with RunLogScope(
                    q_dir,
                    logger_names=("Core", "docarag", __name__),
                    run_id=query_id,
                ) as run_log_info:
                    readiness = self.chat_readiness()
                    if not readiness.ready:
                        raise RunNotReadyError(
                            "Run is viewable but not ready for chat: "
                            + ", ".join(readiness.missing)
                        )
                    with usage_scope:
                        stage = "load_index_and_models"
                        rag_agent = self.ensure_rag()
                        stage = "generate_answer"
                        answer, node_ids = rag_agent.generation(question, q_dir)
                    retrieval = (
                        read_json(retrieval_path) if retrieval_path.exists() else {}
                    )
            except Exception as exc:
                token_usage = token_usage_payload(usage_scope, query_id)
                write_private_json(
                    execution_path,
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "query_id": query_id,
                        "question": question,
                        "run_dir": str(self.run_dir),
                        "stage": stage,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "system_config": str(config_path),
                        "system_config_sha256": config_sha256,
                        "provenance": provenance,
                        "token_usage": token_usage,
                        "run_log": (
                            str(run_log_info.log_file) if run_log_info else None
                        ),
                        "agent_lifecycle": {
                            "mode": "persistent_web_process",
                            "reused": agent_reused,
                        },
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "retrieval_trace": (
                            str(retrieval_path) if retrieval_path.exists() else None
                        ),
                    },
                )
                raise

            elapsed = round(time.monotonic() - started, 3)
            token_usage = token_usage_payload(usage_scope, query_id)
            result_path = q_dir / "result.json"
            if node_ids is None:
                normalized_node_ids = []
            elif isinstance(node_ids, (str, bytes)):
                normalized_node_ids = [node_ids]
            else:
                try:
                    normalized_node_ids = list(node_ids)
                except TypeError:
                    normalized_node_ids = [node_ids]
            result = {
                "schema_version": 1,
                "status": "succeeded",
                "query_id": query_id,
                "question": question,
                "answer": answer,
                "node_ids": normalized_node_ids,
                "citations": citations_from_retrieval(retrieval),
                "run_dir": str(self.run_dir),
                "elapsed_seconds": elapsed,
                "system_config": str(config_path),
                "system_config_sha256": config_sha256,
                "provenance": provenance,
                "token_usage": token_usage,
                "run_log": str(run_log_info.log_file) if run_log_info else None,
                "agent_lifecycle": {
                    "mode": "persistent_web_process",
                    "reused": agent_reused,
                },
                "retrieval_trace": (
                    str(retrieval_path) if retrieval_path.exists() else None
                ),
            }
            write_private_json(result_path, result)
            write_private_json(
                execution_path,
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"answer", "node_ids"}
                }
                | {"result": str(result_path)},
            )
            return {
                "answer": answer,
                "nodeIds": result["node_ids"],
                "citations": result["citations"],
                "elapsedSec": elapsed,
                "retrieval": retrieval,
                "tokenUsage": token_usage,
                "runLog": str(run_log_info.log_file) if run_log_info else None,
                "queryId": query_id,
                "executionTrace": str(execution_path),
                "resultTrace": str(result_path),
            }


class Handler(SimpleHTTPRequestHandler):
    state: AppState = None

    def translate_path(self, path):
        parsed = urlparse(path)
        rel = parsed.path.lstrip("/") or "index.html"
        return str((WEB_ROOT / rel).resolve())

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message, detail=None):
        payload = {"error": message}
        if detail:
            payload["detail"] = detail
        self.send_json(payload, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            # Only completed indexes belong in the picker. A run whose vector
            # stage failed still publishes a tree, so it looks selectable while
            # its chat cannot answer anything.
            runs = []
            current = (
                self.state.run_dir.resolve()
                if self.state.run_dir is not None
                else None
            )
            for run in find_runs():
                if not run_completed(run) and run.resolve() != current:
                    continue
                runs.append(
                    {
                        "path": str(run),
                        "name": self._safe_doc_name(run),
                        "hasGraph": True,
                        "mtime": run.stat().st_mtime,
                    }
                )
            self.send_json(
                {
                    "current": None if current is None else str(current),
                    "runs": runs,
                }
            )
            return
        if parsed.path == "/api/tree":
            self.send_view_payload(self.state.tree_payload)
            return
        if parsed.path == "/api/graph":
            self.send_view_payload(self.state.graph_payload)
            return
        if parsed.path == "/api/tree-image":
            self.send_tree_image(parsed)
            return
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "runDir": (
                        None
                        if self.state.run_dir is None
                        else str(self.state.run_dir)
                    ),
                }
            )
            return
        if parsed.path == "/api/library":
            runs = find_run_dirs()
            history = _stage_history(runs)
            self.send_json(
                {
                    "current": (
                        None
                        if self.state.run_dir is None
                        else str(self.state.run_dir)
                    ),
                    "stageHistory": history,
                    "runs": [run_summary(item, history) for item in runs],
                }
            )
            return
        if parsed.path == "/api/progress":
            self.send_progress(parsed)
            return

        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        if not path.exists() or WEB_ROOT not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def resolve_run_argument(self, raw):
        """Resolve a caller-supplied run strictly below the runs directory."""

        root = runs_root().resolve()
        run_dir = Path(str(raw or "")).expanduser().resolve()
        try:
            relative = run_dir.relative_to(root)
        except ValueError:
            return None
        if relative == Path(".") or not run_dir.is_dir():
            return None
        return run_dir

    def send_view_payload(self, build):
        try:
            self.send_json(build())
        except RunNotReadyError as exc:
            self.send_error_json(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:
            self.send_error_json(
                HTTPStatus.CONFLICT,
                "Run has no persisted evidence to view",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def send_progress(self, parsed):
        requested = (parse_qs(parsed.query).get("run") or [""])[0]
        run_dir = (
            self.state.run_dir
            if not requested
            else self.resolve_run_argument(requested)
        )
        if run_dir is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown run")
            return
        self.send_json(run_summary(run_dir, _stage_history(find_run_dirs())))

    def send_tree_image(self, parsed):
        if self.state.run_dir is None:
            self.send_error_json(
                HTTPStatus.CONFLICT, "Select a document from the Library first"
            )
            return
        query = parse_qs(parsed.query)
        requested_run = (query.get("run") or [""])[0]
        if requested_run:
            resolved_run = self.resolve_run_argument(requested_run)
            if resolved_run is None or resolved_run.resolve() != self.state.run_dir.resolve():
                self.send_error_json(
                    HTTPStatus.CONFLICT,
                    "The selected document changed; reload the crop",
                )
                return
        try:
            node_id = int((query.get("id") or [""])[0])
        except ValueError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid node id")
            return

        node = next(
            (item for item in self.state.tree_payload()["nodes"] if item["id"] == node_id),
            None,
        )
        node_meta = (node or {}).get("meta") or {}
        run_dir = self.state.run_dir.resolve()
        candidates = [
            node_meta.get("crop_uri"),
            node_meta.get("img_path"),
        ]

        # The SQL projection can retain a stale or logical crop URI even when
        # the run's published tree.json contains the usable physical MinerU
        # crop. Fall back to that source-of-record rather than removing the
        # image from an otherwise valid IMAGE node.
        raw_tree = _read_json_if_present(run_dir / "tree.json")
        raw_node = next(
            (
                item
                for item in (raw_tree.get("nodes") or [])
                if item.get("index_id") == node_id
            ),
            None,
        )
        raw_meta = (raw_node or {}).get("meta_info") or {}
        candidates.extend(
            [raw_meta.get("img_path"), raw_meta.get("crop_uri")]
        )

        path = None
        for raw_path in candidates:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            candidate = candidate.resolve()
            if candidate.is_file() and run_dir in candidate.parents:
                path = candidate
                break
        if path is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Image is not available")
            return

        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        # `/api/tree-image?id=3` is document-scoped even though its old URL was
        # not. Do not let a browser retain one PDF's crop for another PDF that
        # happens to contain the same tree node id.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        origin = self.headers.get("Origin")
        if origin and not is_loopback_host(urlparse(origin).hostname or ""):
            self.send_error_json(HTTPStatus.FORBIDDEN, "Cross-origin request denied")
            return
        if parsed.path == "/api/run":
            self.set_run()
            return
        if parsed.path == "/api/upload":
            self.upload_document()
            return
        if parsed.path == "/api/edit-summary":
            self.edit_summary()
            return
        if parsed.path != "/api/chat":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            payload = self.read_json_body()
            question = str(payload.get("question") or "").strip()
            if not question:
                self.send_error_json(HTTPStatus.BAD_REQUEST, "Question is required")
                return
            self.send_json(self.state.ask(question))
        except RequestBodyError as exc:
            self.send_error_json(exc.status, str(exc))
        except RunNotReadyError as exc:
            self.send_error_json(HTTPStatus.CONFLICT, "Run is not ready for chat", str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Chat request failed",
                str(exc),
            )

    def do_DELETE(self):
        parsed = urlparse(self.path)
        origin = self.headers.get("Origin")
        if origin and not is_loopback_host(urlparse(origin).hostname or ""):
            self.send_error_json(HTTPStatus.FORBIDDEN, "Cross-origin request denied")
            return
        if parsed.path != "/api/run":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        run_dir = self.resolve_run_argument((parse_qs(parsed.query).get("run") or [""])[0])
        if run_dir is None:
            self.send_error_json(
                HTTPStatus.NOT_FOUND,
                "Run must be a directory below the configured runs directory",
            )
            return
        if (
            self.state.run_dir is not None
            and run_dir == self.state.run_dir.resolve()
        ):
            self.send_error_json(
                HTTPStatus.CONFLICT,
                "This document is open; open another one before deleting it",
            )
            return
        try:
            self.send_json(delete_run_everywhere(run_dir))
        except OSError as exc:
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Could not delete the run", str(exc)
            )
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Delete failed", str(exc)
            )

    def upload_document(self):
        name = safe_pdf_name(self.headers.get("X-Filename"))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Empty upload")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
            )
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Upload was truncated")
            return
        if not body.startswith(PDF_MAGIC):
            self.send_error_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Not a PDF file")
            return

        target = uploads_dir() / name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and hashlib.sha256(body).hexdigest() != sha256_file(
                target
            ):
                target = target.with_name(
                    f"{target.stem}-{uuid.uuid4().hex[:8]}{target.suffix}"
                )
            target.write_bytes(body)
        except OSError as exc:
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Could not store the upload", str(exc)
            )
            return

        try:
            run_dir = start_index(target)
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Indexing did not start", str(exc)
            )
            return
        self.send_json(
            {"ok": True, "name": name, "path": str(target), "runDir": run_dir}
        )

    def edit_summary(self):
        try:
            self.state.require_run()
            payload = self.read_json_body()
            node_id = payload.get("nodeId")
            text = payload.get("text")
            if not isinstance(node_id, int) or not isinstance(text, str) or not text.strip():
                self.send_error_json(
                    HTTPStatus.BAD_REQUEST, "nodeId and non-empty text are required"
                )
                return
            from docarag.edit import edit_node_summary

            edit_node_summary(self.state.run_dir, node_id, text.strip())
            # The tree changed under the viewer; its cache must not outlive it.
            self.state._tree_cache = None
            self.send_json({"ok": True, "nodeId": node_id, "vdb": "stale"})
        except SystemExit as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Edit failed",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def set_run(self):
        try:
            payload = self.read_json_body()
            run_dir = self.resolve_run_argument(payload.get("runDir"))
            if run_dir is None:
                self.send_error_json(
                    HTTPStatus.NOT_FOUND,
                    "Run must be an indexed directory below the configured runs "
                    "directory",
                )
                return
            if not (run_dir / "run-manifest.json").is_file():
                self.send_error_json(
                    HTTPStatus.CONFLICT, "Run has not been admitted yet"
                )
                return
            self.state.switch_run(run_dir)
            self.send_json({"ok": True, "runDir": str(run_dir)})
        except RequestBodyError as exc:
            self.send_error_json(exc.status, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not switch run",
                str(exc),
            )

    def read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestBodyError("Invalid Content-Length") from exc
        if length <= 0:
            raise RequestBodyError("Request body is empty")
        if length > MAX_REQUEST_BYTES:
            raise RequestBodyError(
                f"Request body exceeds {MAX_REQUEST_BYTES} bytes",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestBodyError("Request body is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RequestBodyError("JSON body must be an object")
        return payload

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")

    @staticmethod
    def _safe_doc_name(run):
        try:
            manifest = read_json(run / "run-manifest.json")
            source = manifest.get("source") or {}
            return Path(source.get("path") or "").name or run.name
        except Exception:
            return run.name


def main():
    # The store, Qdrant client and BookRAG subprocess must see the same local
    # storage endpoints the CLI uses, so the project .env is loaded the same
    # way -- here and not at import, which would leak it into any importer.
    _load_project_environment(REPO)
    # This executable is the composition root and may configure its own
    # console. Importing it for tests or embedding must not mutate the host
    # process's root logging configuration.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run the local BookRAG web UI.")
    parser.add_argument("run_dir", nargs="?", help="BookRAG run directory")
    parser.add_argument("--host", default=os.environ.get("BOOKRAG_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BOOKRAG_WEB_PORT", "8765")))
    args = parser.parse_args()

    if not is_loopback_host(args.host):
        raise SystemExit(
            "The compatibility Web UI is intentionally loopback-only because it "
            "has no authentication. Use 127.0.0.1/localhost and place a production "
            "API behind authenticated infrastructure."
        )
    if not 1 <= args.port <= 65535:
        raise SystemExit("Web port must be between 1 and 65535")

    run_dir = pick_run(args.run_dir) if args.run_dir else None
    if run_dir is not None and not (run_dir / "run-manifest.json").exists():
        raise SystemExit(f"Missing run-manifest.json in run directory: {run_dir}")

    Handler.state = AppState(run_dir)
    server, actual_port = bind_server(args.host, args.port, Handler)
    if actual_port != args.port:
        print(f"Port {args.port} is busy; using {actual_port} instead.")
    print(f"BookRAG web UI: http://{args.host}:{actual_port}")
    print(f"Runs directory: {runs_root()}")
    if run_dir is not None:
        print(f"Initially selected run: {run_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
    finally:
        try:
            if Handler.state.rag_agent:
                Handler.state.rag_agent.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
