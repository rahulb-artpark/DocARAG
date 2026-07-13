#!/usr/bin/env python3
"""
Terminal viewer for a BookRAG index: browse the document TREE and knowledge GRAPH
that were built for a PDF.

Usage (from repo root):
    conda run -n gbc-rag python Scripts/view.py [RUN_DIR]
or:
    bash Scripts/view.sh [RUN_DIR]

RUN_DIR defaults to the most recently modified run under runs/ that has a tree.json.
"""
import os
import sys
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
REPO = Path(__file__).resolve().parent.parent


# ----------------------------- run discovery -----------------------------
def find_runs():
    runs = []
    for tj in glob.glob(str(REPO / "runs" / "**" / "tree.json"), recursive=True):
        runs.append(Path(tj).parent)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def pick_run(arg):
    if arg:
        p = Path(arg)
        if p.is_file():
            p = p.parent
        return p
    runs = find_runs()
    if not runs:
        console.print("[red]No indexed PDFs found under runs/ (no tree.json).[/red]")
        sys.exit(1)
    if len(runs) == 1:
        return runs[0]
    console.print("[bold]Indexed PDFs:[/bold]")
    for i, r in enumerate(runs):
        name = doc_name(r)
        console.print(f"  [cyan]{i+1}[/cyan]. {name}  [dim]{r}[/dim]")
    while True:
        sel = input("Pick a run number: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(runs):
            return runs[int(sel) - 1]


def doc_name(run_dir):
    try:
        t = json.load(open(run_dir / "tree.json"))
        return (t.get("meta_info") or {}).get("file_name", run_dir.name)
    except Exception:
        return run_dir.name


# ------------------------------- tree view -------------------------------
def is_title(n):
    return "TITLE" in str(n.get("type", "")) or n.get("type") == "root"


def short(s, n=90):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def node_text(n):
    mi = n.get("meta_info") or {}
    return (mi.get("content") or mi.get("text") or mi.get("caption")
            or mi.get("title") or "")


def show_tree(run_dir, with_summary=False):
    data = json.load(open(run_dir / "tree.json"))
    nodes = data["nodes"]
    by_id = {n["index_id"]: n for n in nodes}
    children = defaultdict(list)
    for n in nodes:
        if n["parent_id"] is not None:
            children[n["parent_id"]].append(n["index_id"])

    counts = Counter(str(n["type"]).replace("NodeType.", "") for n in nodes)
    header = "  ".join(f"[cyan]{k}[/cyan]:{v}" for k, v in counts.items())
    console.print(Panel(f"{doc_name(run_dir)}\n{header}",
                        title="Document Tree", border_style="cyan"))

    root_id = next((n["index_id"] for n in nodes if n["type"] == "root"), 0)
    rtree = Tree(f"[bold]{doc_name(run_dir)}[/bold]")

    def walk(nid, branch):
        kids = children.get(nid, [])
        title_kids = [k for k in kids if is_title(by_id[k])]
        other = [k for k in kids if not is_title(by_id[k])]
        if other:
            c = Counter(str(by_id[k]["type"]).replace("NodeType.", "") for k in other)
            branch.add("[dim]" + ", ".join(f"{v} {k.lower()}" for k, v in c.items()) + "[/dim]")
        for tk in title_kids:
            n = by_id[tk]
            label = f"[yellow]§[/yellow] {short(node_text(n), 80)} [dim](#{tk})[/dim]"
            sub = branch.add(label)
            if with_summary and n.get("summary"):
                sub.add(f"[green italic]{short(n['summary'], 120)}[/green italic]")
            walk(tk, sub)

    walk(root_id, rtree)
    console.print(rtree)


# ------------------------------ graph view -------------------------------
def load_graph(run_dir):
    g = json.load(open(run_dir / "graph_data.json"))
    G = g["graph"]
    ents = G.get("nodes", [])
    rels = G.get("links", G.get("edges", []))
    return ents, rels


def show_graph_overview(run_dir):
    ents, rels = load_graph(run_dir)
    types = Counter(e.get("entity_type", "?") for e in ents)

    console.print(Panel(f"[bold]{len(ents)}[/bold] entities   "
                        f"[bold]{len(rels)}[/bold] relations",
                        title="Knowledge Graph", border_style="magenta"))

    tbl = Table(title="Entity types", box=box.SIMPLE)
    tbl.add_column("type", style="cyan")
    tbl.add_column("count", justify="right")
    for t, c in types.most_common():
        tbl.add_row(str(t), str(c))
    console.print(tbl)


def show_top_entities(run_dir, topn=20):
    ents, rels = load_graph(run_dir)
    deg = Counter()
    for r in rels:
        deg[r.get("src_entity_name")] += 1
        deg[r.get("tgt_entity_name")] += 1
    etype = {e.get("entity_name"): e.get("entity_type", "?") for e in ents}

    tbl = Table(title=f"Top {topn} entities by connections", box=box.SIMPLE)
    tbl.add_column("entity", style="yellow")
    tbl.add_column("type", style="cyan")
    tbl.add_column("degree", justify="right")
    for name, d in deg.most_common(topn):
        tbl.add_row(short(name, 45), str(etype.get(name, "?")), str(d))
    console.print(tbl)


def show_relations(run_dir, topn=25):
    ents, rels = load_graph(run_dir)
    rels_sorted = sorted(rels, key=lambda r: float(r.get("weight") or 0), reverse=True)
    tbl = Table(title=f"Top {topn} relations by weight", box=box.SIMPLE)
    tbl.add_column("source", style="yellow")
    tbl.add_column("→", justify="center")
    tbl.add_column("target", style="yellow")
    tbl.add_column("w", justify="right")
    tbl.add_column("description", style="dim")
    for r in rels_sorted[:topn]:
        tbl.add_row(short(r.get("src_entity_name"), 24), "→",
                    short(r.get("tgt_entity_name"), 24),
                    str(r.get("weight", "")), short(r.get("description"), 50))
    console.print(tbl)


# --------------------------------- menu ----------------------------------
MENU = """
[bold]BookRAG index viewer[/bold] — [dim]{name}[/dim]
  [cyan]1[/cyan]  Document tree (outline)
  [cyan]2[/cyan]  Document tree + node summaries
  [cyan]3[/cyan]  Knowledge graph overview (entity types)
  [cyan]4[/cyan]  Top entities (most connected)
  [cyan]5[/cyan]  Top relations
  [cyan]q[/cyan]  quit
"""


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = pick_run(arg)
    has_graph = (run_dir / "graph_data.json").exists()
    console.print(f"[green]Viewing:[/green] {run_dir}")
    if not has_graph:
        console.print("[yellow](graph not built yet — only tree available)[/yellow]")

    while True:
        console.print(MENU.format(name=doc_name(run_dir)))
        choice = input("select> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        try:
            if choice == "1":
                show_tree(run_dir, with_summary=False)
            elif choice == "2":
                show_tree(run_dir, with_summary=True)
            elif choice == "3":
                show_graph_overview(run_dir) if has_graph else console.print("[yellow]no graph[/yellow]")
            elif choice == "4":
                show_top_entities(run_dir) if has_graph else console.print("[yellow]no graph[/yellow]")
            elif choice == "5":
                show_relations(run_dir) if has_graph else console.print("[yellow]no graph[/yellow]")
            else:
                console.print("[dim]unknown option[/dim]")
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")
    console.print("Bye.")


if __name__ == "__main__":
    main()
