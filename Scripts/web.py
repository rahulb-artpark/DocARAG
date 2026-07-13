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
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO = Path(__file__).resolve().parent.parent
WEB_ROOT = REPO / "Scripts" / "webui"
DEFAULT_CONFIG = os.environ.get("BOOKRAG_CONFIG", "config/ollama-gbc.yaml")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def find_runs():
    runs = []
    for tree_path in (REPO / "runs").glob("**/tree.json"):
        run_dir = tree_path.parent
        runs.append(run_dir)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def pick_run(path_arg):
    if path_arg:
        path = Path(path_arg).expanduser()
        if path.is_file():
            path = path.parent
        return path.resolve()
    runs = find_runs()
    if not runs:
        raise SystemExit("No indexed runs found under runs/ with tree.json")
    return runs[0].resolve()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def node_content(node):
    meta = node.get("meta_info") or {}
    return (
        meta.get("content")
        or meta.get("text")
        or meta.get("caption")
        or meta.get("table_body")
        or meta.get("title")
        or ""
    )


def normalize_type(value):
    return str(value or "").replace("NodeType.", "")


def bind_server(host, port, handler, attempts=20):
    last_error = None
    for candidate in range(port, port + attempts):
        try:
            return ThreadingHTTPServer((host, candidate), handler), candidate
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
    raise SystemExit(
        f"Could not start web UI: ports {port}-{port + attempts - 1} are in use"
    ) from last_error


class AppState:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.chat_lock = threading.Lock()
        self.rag_agent = None
        self.chat_counter = 0

    def set_run_dir(self, run_dir):
        with self.chat_lock:
            if self.rag_agent:
                try:
                    self.rag_agent.close()
                except Exception:
                    pass
            self.run_dir = run_dir
            self.rag_agent = None
            self.chat_counter = 0

    @property
    def doc_name(self):
        try:
            meta = read_json(self.run_dir / "tree.json").get("meta_info") or {}
            return meta.get("file_name") or self.run_dir.name
        except Exception:
            return self.run_dir.name

    def tree_payload(self):
        data = read_json(self.run_dir / "tree.json")
        nodes = data.get("nodes") or []
        children = {}
        by_id = {}
        for node in nodes:
            nid = node.get("index_id")
            by_id[nid] = node
            children.setdefault(nid, [])
        for node in nodes:
            parent = node.get("parent_id")
            if parent is not None:
                children.setdefault(parent, []).append(node.get("index_id"))

        flat = []
        for node in nodes:
            meta = node.get("meta_info") or {}
            nid = node.get("index_id")
            flat.append(
                {
                    "id": nid,
                    "parentId": node.get("parent_id"),
                    "type": normalize_type(node.get("type")),
                    "page": (
                        meta.get("page_idx") + 1
                        if isinstance(meta.get("page_idx"), int)
                        else None
                    ),
                    "titleLevel": meta.get("title_level"),
                    "content": node_content(node),
                    "summary": node.get("summary") or "",
                    "meta": meta,
                    "children": children.get(nid, []),
                }
            )

        root_id = next((n["id"] for n in flat if n["parentId"] is None), None)
        return {
            "docName": self.doc_name,
            "runDir": str(self.run_dir),
            "rootId": root_id,
            "nodes": flat,
        }

    def graph_payload(self):
        path = self.run_dir / "graph_data.json"
        if not path.exists():
            return {"docName": self.doc_name, "nodes": [], "links": []}
        data = read_json(path)
        graph = data.get("graph") or data
        raw_nodes = graph.get("nodes") or []
        raw_links = graph.get("links") or graph.get("edges") or []
        tree_nodes = {
            node["id"]: node
            for node in self.tree_payload().get("nodes", [])
        }

        degree = {}
        for link in raw_links:
            for key in ("source", "target"):
                val = link.get(key)
                if val is not None:
                    degree[val] = degree.get(val, 0) + 1

        nodes = []
        for idx, node in enumerate(raw_nodes):
            node_id = node.get("id") or f"entity-{idx}"
            label = node.get("entity_name") or node_id
            source_ids = node.get("source_ids") or []
            source_snippets = [
                tree_nodes[source_id].get("content")
                or tree_nodes[source_id].get("summary")
                or ""
                for source_id in source_ids
                if source_id in tree_nodes
            ]
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "type": node.get("entity_type") or "UNKNOWN",
                    "description": node.get("description") or "",
                    "sourceIds": source_ids,
                    "sourceSnippets": source_snippets,
                    "degree": degree.get(node_id, 0),
                    "raw": node,
                }
            )

        links = []
        for idx, link in enumerate(raw_links):
            links.append(
                {
                    "id": f"rel-{idx}",
                    "source": link.get("source"),
                    "target": link.get("target"),
                    "sourceName": link.get("src_entity_name") or "",
                    "targetName": link.get("tgt_entity_name") or "",
                    "name": link.get("relation_name") or "",
                    "description": link.get("description") or "",
                    "weight": link.get("weight"),
                    "sourceIds": link.get("source_ids") or [],
                    "raw": link,
                }
            )

        return {
            "docName": self.doc_name,
            "runDir": str(self.run_dir),
            "variant": data.get("variant"),
            "nodes": nodes,
            "links": links,
        }

    def ensure_rag(self):
        if self.rag_agent is not None:
            return self.rag_agent

        from Core.configs.system_config import load_system_config
        from Core.rag import create_rag_agent
        from Core.utils.resource_loader import prepare_rag_dependencies

        cfg = load_system_config(DEFAULT_CONFIG)
        cfg.save_path = str(self.run_dir)
        dependencies = prepare_rag_dependencies(cfg=cfg)
        self.rag_agent = create_rag_agent(
            strategy_config=cfg.rag.strategy_config,
            llm_config=cfg.llm,
            vlm_config=cfg.vlm,
            **dependencies,
        )
        return self.rag_agent

    def ask(self, question):
        with self.chat_lock:
            rag_agent = self.ensure_rag()
            self.chat_counter += 1
            q_dir = self.run_dir / "web_chat" / f"q_{self.chat_counter:03d}"
            q_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            answer, node_ids = rag_agent.generation(question, q_dir)
            retrieval = {}
            retrieval_path = q_dir / "retrieval_res.json"
            if retrieval_path.exists():
                retrieval = read_json(retrieval_path)
            return {
                "answer": answer,
                "nodeIds": node_ids,
                "elapsedSec": round(time.time() - started, 2),
                "retrieval": retrieval,
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
            runs = [
                {
                    "path": str(run),
                    "name": self._safe_doc_name(run),
                    "hasGraph": (run / "graph_data.json").exists(),
                    "mtime": run.stat().st_mtime,
                }
                for run in find_runs()
            ]
            self.send_json({"current": str(self.state.run_dir), "runs": runs})
            return
        if parsed.path == "/api/tree":
            self.send_json(self.state.tree_payload())
            return
        if parsed.path == "/api/graph":
            self.send_json(self.state.graph_payload())
            return
        if parsed.path == "/api/tree-image":
            self.send_tree_image(parsed)
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "runDir": str(self.state.run_dir)})
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

    def send_tree_image(self, parsed):
        query = parse_qs(parsed.query)
        try:
            node_id = int((query.get("id") or [""])[0])
        except ValueError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid node id")
            return

        node = next(
            (item for item in self.state.tree_payload()["nodes"] if item["id"] == node_id),
            None,
        )
        img_path = ((node or {}).get("meta") or {}).get("img_path")
        if not img_path:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Node has no image")
            return

        path = Path(img_path).resolve()
        run_dir = self.state.run_dir.resolve()
        if not path.exists() or run_dir not in path.parents:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Image is not available")
            return

        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            self.set_run()
            return
        if parsed.path != "/api/chat":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question") or "").strip()
            if not question:
                self.send_error_json(HTTPStatus.BAD_REQUEST, "Question is required")
                return
            self.send_json(self.state.ask(question))
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Chat request failed",
                str(exc),
            )

    def set_run(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            run_dir = Path(str(payload.get("runDir") or "")).expanduser().resolve()
            allowed = {run.resolve() for run in find_runs()}
            if run_dir not in allowed:
                self.send_error_json(HTTPStatus.BAD_REQUEST, "Unknown run directory")
                return
            self.state.set_run_dir(run_dir)
            self.send_json({"ok": True, "runDir": str(run_dir)})
        except Exception as exc:
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not switch run",
                str(exc),
            )

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")

    @staticmethod
    def _safe_doc_name(run):
        try:
            meta = read_json(run / "tree.json").get("meta_info") or {}
            return meta.get("file_name") or run.name
        except Exception:
            return run.name


def main():
    parser = argparse.ArgumentParser(description="Run the local BookRAG web UI.")
    parser.add_argument("run_dir", nargs="?", help="BookRAG run directory")
    parser.add_argument("--host", default=os.environ.get("BOOKRAG_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BOOKRAG_WEB_PORT", "8765")))
    args = parser.parse_args()

    run_dir = pick_run(args.run_dir)
    if not (run_dir / "tree.json").exists():
        raise SystemExit(f"Missing tree.json in run directory: {run_dir}")

    Handler.state = AppState(run_dir)
    server, actual_port = bind_server(args.host, args.port, Handler)
    if actual_port != args.port:
        print(f"Port {args.port} is busy; using {actual_port} instead.")
    print(f"BookRAG web UI: http://{args.host}:{actual_port}")
    print(f"Run directory: {run_dir}")
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
