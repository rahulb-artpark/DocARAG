#!/usr/bin/env python3
"""
Interactive BookRAG chat over an already-built index.

Loads the GBC index + reranker + multimodal model ONCE, then answers questions
in a loop until you type a terminate command. This is far faster than re-running
main.py per question (which reloads all models each time).

Usage (from repo root):
    BOOKRAG_SAVE_PATH=/absolute/path/to/run \
      conda run -n gbc-rag python Scripts/chat.py
or via the wrapper:
    bash Scripts/chat.sh

Terminate commands: exit, quit, q, :q  (or Ctrl-D / Ctrl-C)
"""
import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path

# Keep the interactive compatibility shell free of library deprecation noise.
warnings.filterwarnings("ignore")

# Repo root on the path, and run from it so relative config paths resolve.
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --- knobs (override via env) ---
CONFIG = os.environ.get("BOOKRAG_CONFIG", "config/ollama-gbc.yaml")
# The clean workspace has no implicit historical run. Select one explicitly so
# this helper cannot accidentally reopen stale data after a storage reset.
SAVE_PATH = os.environ.get("BOOKRAG_SAVE_PATH")
if not SAVE_PATH:
    raise SystemExit(
        "BOOKRAG_SAVE_PATH must name an indexed DocARAG run; "
        "for example: BOOKRAG_SAVE_PATH=/absolute/path/to/run "
        "python Scripts/chat.py"
    )
VERBOSE = os.environ.get("BOOKRAG_VERBOSE", "0") == "1"
# How to show retrieved nodes: "short" (snippets), "full" (whole text), "off".
NODE_MODE = os.environ.get("BOOKRAG_SHOW_NODES", "short").lower()
SNIPPET = int(os.environ.get("BOOKRAG_SNIPPET", "220"))

# Keep the screen clean unless the user wants the pipeline logs.
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(message)s",
)
from Core.configs.system_config import load_system_config
from Core.rag import create_rag_agent
from Core.utils.resource_loader import prepare_rag_dependencies

EXIT_WORDS = {"exit", "quit", "q", ":q", "bye"}

HELP = """commands:
  :nodes short   show retrieved-node snippets before each answer (default)
  :nodes full    show the full retrieved node text (the LLM's actual input)
  :nodes off     hide retrieved nodes, show only the answer
  :help          show this help
  exit / quit    leave
"""


def _clip(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _plain(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {}


def _as_list(value):
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _show_retrieval_iterations(iterations, qtype, mode):
    if mode == "off" or not iterations:
        return False

    print(
        f"\n\033[1;33m┌─ Retrieved context\033[0m "
        f"\033[2m(query_type: {qtype})\033[0m"
    )

    seen = set()
    printed = False
    for raw_it in iterations:
        it = _plain(raw_it)
        if not it:
            continue
        sub = it.get("sub_query")
        if sub:
            printed = True
            print(f"\033[1;33m│ sub-question:\033[0m {sub}")

        text_nodes = _as_list(
            it.get("iteration_text_nodes") or it.get("retrieval_nodes")
        )
        for raw_nd in text_nodes:
            nd = _plain(raw_nd)
            if not nd:
                continue
            nid = nd.get("index_id")
            key = (it.get("sub_number"), nid, nd.get("content"))
            if key in seen:
                continue
            seen.add(key)
            printed = True
            page = nd.get("page")
            tag = f"§{nid}" + (f" p{page}" if page not in (None, "") else "")
            body = _clip(nd.get("content"), 100000 if mode == "full" else SNIPPET)
            if mode == "full":
                print(f"\033[33m│  {tag}\033[0m\n{body}\n")
            else:
                print(f"\033[33m│  {tag}\033[0m  {body}")

        imgs = _as_list(it.get("iteration_image_nodes"))
        for raw_im in imgs:
            im = _plain(raw_im)
            if not im:
                continue
            printed = True
            print(
                f"\033[33m│  [image §{im.get('index_id')}]\033[0m  "
                f"{_clip(im.get('content'), SNIPPET)}"
            )

        gnodes = _as_list(it.get("iteration_graph_nodes"))
        ents = []
        for raw_g in gnodes[:12]:
            g = _plain(raw_g)
            if g:
                name = g.get("entity_name")
                etype = g.get("entity_type")
                if name or etype:
                    ents.append(f"{name} ({etype})")
            else:
                ents.append(str(raw_g))
        if ents:
            printed = True
            print(
                f"\033[33m│  graph entities:\033[0m "
                f"\033[2m{', '.join(ents)}\033[0m"
            )

    print("\033[1;33m└─\033[0m", flush=True)
    return printed


def show_retrieval_step(context, step, mode):
    """Print one live retrieval step before the answer agent uses it."""
    context_data = _plain(context)
    plan = _plain(context_data.get("plan") or {})
    return _show_retrieval_iterations(
        [step],
        plan.get("query_type", ""),
        mode,
    )


def show_retrieval(q_dir, mode):
    """Print the nodes that were retrieved & fed to the LLM for this query."""
    if mode == "off":
        return False
    f = Path(q_dir) / "retrieval_res.json"
    if not f.exists():
        return False
    try:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
    except Exception:
        return False

    plan = _plain(r.get("plan") or {})
    return _show_retrieval_iterations(
        r.get("iterations") or [],
        plan.get("query_type", ""),
        mode,
    )


def main():
    if not Path(SAVE_PATH).exists():
        print(f"[error] Index dir not found: {SAVE_PATH}", file=sys.stderr)
        print("        Build the index first (index stages tree -> graph -> vdb).",
              file=sys.stderr)
        sys.exit(1)

    print("Loading BookRAG index and models (one-time, ~30-60s)...", flush=True)
    t0 = time.time()
    cfg = load_system_config(CONFIG)
    cfg.save_path = SAVE_PATH  # GBC.load_gbc_index reads from here

    dependencies = prepare_rag_dependencies(cfg=cfg)
    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    print(f"Ready in {time.time() - t0:.0f}s. Strategy: {rag_agent.name}", flush=True)
    print("Ask a question. Type ':help' for commands, 'exit' (or Ctrl-D) to quit.\n",
          flush=True)

    node_mode = NODE_MODE if NODE_MODE in ("short", "full", "off") else "short"
    printed_retrieval = False

    def print_retrieval_before_answer(context, step):
        nonlocal printed_retrieval
        if show_retrieval_step(context, step, node_mode):
            printed_retrieval = True

    if hasattr(rag_agent, "retrieval_callback"):
        rag_agent.retrieval_callback = print_retrieval_before_answer

    out_dir = Path(SAVE_PATH) / "interactive"
    out_dir.mkdir(exist_ok=True)
    n = 0
    try:
        while True:
            try:
                query = input("\033[1;36m❓ > \033[0m").strip()
            except EOFError:
                print()
                break
            if not query:
                continue
            if query.lower() in EXIT_WORDS:
                break

            # in-REPL commands
            if query.lower() in (":help", "help", "?"):
                print(HELP, flush=True)
                continue
            if query.lower().startswith(":nodes"):
                arg = query.split(maxsplit=1)
                val = arg[1].strip().lower() if len(arg) > 1 else ""
                if val in ("short", "full", "off"):
                    node_mode = val
                    print(f"[retrieved-nodes display: {node_mode}]\n", flush=True)
                else:
                    print("usage: :nodes short|full|off\n", flush=True)
                continue

            n += 1
            q_dir = out_dir / f"q_{n:03d}"
            q_dir.mkdir(exist_ok=True)
            print("\033[2m…thinking…\033[0m", flush=True)
            t = time.time()
            printed_retrieval = False
            try:
                answer, node_ids = rag_agent.generation(query, q_dir)
            except Exception as e:
                print(f"\033[31m[error answering]\033[0m {e}\n", flush=True)
                continue
            # Fallback for agents that do not expose the live retrieval hook.
            if not printed_retrieval:
                show_retrieval(q_dir, node_mode)
            print("\n" + "=" * 70)
            print(answer)
            print("=" * 70)
            print(
                f"\033[2m({time.time() - t:.0f}s · {len(node_ids)} nodes retrieved)\033[0m\n",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        try:
            rag_agent.close()
        except Exception:
            pass
        print("Bye.")


if __name__ == "__main__":
    main()
