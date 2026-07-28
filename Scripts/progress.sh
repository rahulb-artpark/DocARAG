#!/usr/bin/env bash
# Show BookRAG indexing progress for a run directory (clean summary + timing).
#
#   bash Scripts/progress.sh                 # latest run under runs/
#   bash Scripts/progress.sh <run_dir>       # a specific run
#   bash Scripts/progress.sh -w [run_dir]    # live watch (refresh every 8s)
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RUNS_DIR="${DOCARAG_RUNS_DIR:-$REPO/runs}"
if [[ "$RUNS_DIR" != /* ]]; then
  RUNS_DIR="$REPO/$RUNS_DIR"
fi

WATCH=0
RUN_DIR=""
for a in "$@"; do
  case "$a" in
    -w|--watch) WATCH=1 ;;
    *) RUN_DIR="$a" ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(python3 - "$RUNS_DIR" <<'PY' 2>/dev/null
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
candidates = set()
if root.is_dir():
    for name in ("index-state.json", "tree.json", "tree.pkl"):
        for artifact in root.glob(f"**/{name}"):
            run = artifact.parent.resolve()
            try:
                run.relative_to(root)
            except ValueError:
                continue
            candidates.add(run)
if candidates:
    print(max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))))
PY
)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "[error] no run directory found. Pass one explicitly." >&2
  exit 1
fi
RUN_DIR="${RUN_DIR%/}"

fmt() {  # seconds -> human readable
  local s=$1; (( s < 0 )) && s=0
  if   (( s >= 3600 )); then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
  elif (( s >= 60 ));   then printf '%dm%02ds' $((s/60)) $((s%60))
  else printf '%ds' "$s"; fi
}
mtime() { stat -c %Y "$1" 2>/dev/null; }

select_graph_artifact() {
  python3 - "$1" <<'PY' 2>/dev/null
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
refine_type = None
config_path = run_dir / "run_config.yaml"
try:
    section = None
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw[0].isspace():
            section = stripped.rstrip(":") if stripped.endswith(":") else None
        elif section == "graph" and stripped.startswith("refine_type:"):
            refine_type = stripped.split(":", 1)[1].split("#", 1)[0].strip(" '\"")
            break
except OSError:
    pass

valid = []
for path in run_dir.glob("graph_data*.json"):
    try:
        if path.stat().st_size <= 0:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = payload.get("graph", payload) if isinstance(payload, dict) else None
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            valid.append(path)
    except (OSError, ValueError, TypeError):
        pass

preferred = run_dir / ("graph_data_basic.json" if refine_type == "basic" else "graph_data.json")
if preferred in valid:
    print(preferred)
elif valid:
    print(max(valid, key=lambda item: (item.stat().st_mtime_ns, item.name)))
PY
}

discover_retrieval_artifact() {
  python3 - "$1" <<'PY' 2>/dev/null
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
published = run_dir / "artifact-manifest.json"
try:
    payload = json.loads(published.read_text(encoding="utf-8"))
    roles = payload.get("roles") if isinstance(payload, dict) else None
    if isinstance(roles, dict):
        value = roles.get("retrieval_attestation") or roles.get("lexical_store")
        if isinstance(value, str) and value:
            print((run_dir / value).resolve())
            raise SystemExit
except (OSError, ValueError, TypeError):
    pass

configured = None
index_type = None
config_path = run_dir / "run_config.yaml"
try:
    section = None
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw[0].isspace():
            section = stripped.rstrip(":") if stripped.endswith(":") else None
            if stripped.startswith("index_type:"):
                index_type = stripped.split(":", 1)[1].split("#", 1)[0].strip(" '\"")
        elif section == "vdb" and stripped.startswith("manifest_dir_name:"):
            value = stripped.split(":", 1)[1].split("#", 1)[0].strip(" '\"")
            if value:
                configured = Path(os.path.expanduser(value))
                if not configured.is_absolute():
                    configured = run_dir / configured
                configured = Path(os.path.abspath(configured))
            break
except OSError:
    pass

if configured is not None:
    name = "bm25_index.pkl" if index_type == "bm25" else "docarag-qdrant-index.ready.json"
    print(configured / name)
    raise SystemExit

candidates = []
try:
    children = list(run_dir.rglob("docarag-qdrant-index.ready.json"))
    children.extend(run_dir.rglob("bm25_index.pkl"))
except OSError:
    children = []
for path in children:
    if not path.is_file() or path.stat().st_size <= 0:
        continue
    if path.name == "docarag-qdrant-index.ready.json" and path.parent.name.startswith("entity_attestation"):
        continue
    candidates.append(path)
if candidates:
    print(max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name)).resolve())
PY
}

retrieval_is_ready() {
  local run="$1" path="$2"
  local readiness_python="${DOCARAG_PYTHON:-python3}"
  [[ -n "$path" && -s "$path" ]] || return 1
  [[ "$(basename "$path")" == "bm25_index.pkl" ]] && return 0
  "$readiness_python" -m docarag.qdrant_readiness \
    --config "$run/run_config.yaml" \
    --attestation "$path" \
    --revision-id "$(basename "$run")" >/dev/null 2>&1
}

render() {
  local d="$RUN_DIR" now; now=$(date +%s)
  local dlog="$d/logs/index-driver.log"
  local name; name="$(python3 - "$d/tree.json" <<'PY' 2>/dev/null
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as source:
        print((json.load(source).get("meta_info") or {}).get("file_name", ""))
except (OSError, ValueError, TypeError):
    pass
PY
)"
  [[ -z "$name" ]] && name="$(basename "$d")"

  local total=0
  if [[ -f "$d/tree.json" ]]; then
    total="$(python3 - "$d/tree.json" <<'PY' 2>/dev/null
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as source:
        nodes = json.load(source).get("nodes") or []
    print(len(nodes))
except (OSError, ValueError, TypeError):
    print(0)
PY
)"
  fi

  # --- stage timestamps from driver log (index_pdf.sh writes these) ---
  ep_start() { local ln; ln=$(grep -a "starting $1$" "$dlog" 2>/dev/null | tail -1); [[ -z "$ln" ]] && return 1
               ln=${ln#\[driver\] }; ln=${ln% starting $1}; date -d "$ln" +%s 2>/dev/null; }
  ep_done()  { local ln; ln=$(grep -a "$1 done" "$dlog" 2>/dev/null | tail -1); [[ -z "$ln" ]] && return 1
               ln=${ln#\[driver\] }; ln=${ln% $1 done*}; date -d "$ln" +%s 2>/dev/null; }

  # New drivers publish run-local state. Never infer one run's state from a
  # machine-global process search; legacy runs without a terminal log marker
  # remain explicitly unknown.
  local state_present=0 run_status="unknown" current_stage="" state_finished=0 run_phase=""
  local -A stage_status=() stage_started=() stage_finished=() stage_error=()
  while IFS=$'\t' read -r kind key status started ended error; do
    if [[ "$kind" == "RUN" ]]; then
      state_present=1
      run_status="$key"
      current_stage="$status"
      state_finished="$ended"
      run_phase="$error"
    elif [[ "$kind" == "STAGE" ]]; then
      stage_status["$key"]="$status"
      stage_started["$key"]="$started"
      stage_finished["$key"]="$ended"
      stage_error["$key"]="$error"
    fi
  done < <(python3 - "$d/index-state.json" <<'PY' 2>/dev/null
import json
import sys
from datetime import datetime

def epoch(value):
    if not value:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return 0

try:
    with open(sys.argv[1], encoding="utf-8") as source:
        state = json.load(source)
except (OSError, ValueError, TypeError):
    raise SystemExit
if not isinstance(state, dict):
    raise SystemExit

status = str(state.get("overall_status") or state.get("status") or "")
if status not in {"running", "succeeded", "failed"}:
    raise SystemExit
current = str(state.get("current_stage") or "")
phase = str(state.get("phase") or "")
print("RUN", status, current, epoch(state.get("started_at")), epoch(state.get("ended_at") or state.get("finished_at")), phase, sep="\t")
stages = state.get("stages") or {}
for name in ("tree", "graph", "vdb"):
    item = stages.get(name) or {}
    error = str(item.get("error") or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")[:160]
    print("STAGE", name, str(item.get("status") or ""), epoch(item.get("started_at")), epoch(item.get("ended_at") or item.get("finished_at")), error, sep="\t")
PY
)

  case "$current_stage" in
    tree|graph|vdb)
      if [[ "$run_status" == "running" && -z "${stage_status[$current_stage]:-}" ]]; then
        stage_status["$current_stage"]="running"
      fi
      ;;
  esac

  local finished=0
  if (( state_present )); then
    [[ "$run_status" == "succeeded" || "$run_status" == "failed" ]] && finished=1
  elif grep -q "ALL DONE" "$dlog" 2>/dev/null; then
    run_status="succeeded"; finished=1
  elif grep -qE "\[driver\].* (tree|graph|vdb) FAILED" "$dlog" 2>/dev/null; then
    run_status="failed"; finished=1
  fi

  # freeze the clock at the last driver event when finished
  local asof=$now
  if (( finished )); then
    if (( state_finished > 0 )); then
      asof=$state_finished
    else
      local ll; ll=$(tail -1 "$dlog" 2>/dev/null); ll=${ll#\[driver\] }
      ll=$(printf '%s' "$ll" | sed -E 's/ (ALL DONE|starting [a-z_]+|[a-z_]+ done.*|[a-z_]+ FAILED.*)$//')
      local e; e=$(date -d "$ll" +%s 2>/dev/null) && asof=$e
    fi
  fi

  # latest failure reason — from the error file (clean), for this run's uuid
  local tree_err
  tree_err="$(python3 - "$REPO" "$(basename "$d")" <<'PY' 2>/dev/null
import sys, glob, os, ast
repo, uuid = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(repo, "*-index_error_*.txt")), key=os.path.getmtime, reverse=True)
for f in files:
    for line in open(f, encoding="utf-8", errors="ignore"):
        if uuid in line:
            try:
                _, reason = ast.literal_eval(line.strip())
                print(str(reason)[:120]); sys.exit()
            except Exception:
                pass
PY
)"

  local t_start="${stage_started[tree]:-}"
  (( t_start > 0 )) 2>/dev/null || t_start=$(ep_start tree || echo "")

  echo "──────────────────────────────────────────────────────────────"

  if [[ "$run_phase" == "waiting_for_capacity" ]]; then
    echo " [•] queued — waiting for the DocARAG indexing/GPU admission lock"
  fi
  echo " BookRAG indexing progress"
  echo "   doc : $name"
  echo "   dir : $d"
  if [[ -n "$t_start" ]]; then
    if (( finished )); then echo "   started : $(date -d @"$t_start" '+%H:%M:%S')   ran for : $(fmt $((asof - t_start)))"
    else echo "   started : $(date -d @"$t_start" '+%H:%M:%S')   elapsed : $(fmt $((now - t_start)))"; fi
  fi
  echo "──────────────────────────────────────────────────────────────"

  # ---- TREE ----
  if [[ -f "$d/tree.pkl" ]] && { (( ! state_present )) || [[ -z "${stage_status[tree]:-}" || "${stage_status[tree]}" == "succeeded" ]]; }; then
    local te="${stage_finished[tree]:-}" ts dur=""
    (( te > 0 )) 2>/dev/null || te=$(ep_done tree || mtime "$d/tree.pkl")
    ts=${t_start:-$(ep_start tree)}
    [[ -n "$ts" && -n "$te" ]] && dur="   took $(fmt $((te - ts)))"
    echo " [1] tree   ✅ done   ($total nodes)$dur"
  elif [[ "${stage_status[tree]:-}" == "failed" ]] || { [[ "$run_status" == "failed" ]] && [[ "$current_stage" == "tree" ]]; }; then
    local terr="${stage_error[tree]:-$tree_err}"
    echo " [1] tree   ❌ FAILED${terr:+   $terr}"
  elif [[ "${stage_status[tree]:-}" == "running" ]]; then
    local slog; slog="$(ls -t "$d"/logs/run_*.log 2>/dev/null | head -1)"
    local el=""; [[ -n "$t_start" ]] && el="   elapsed $(fmt $((now - t_start)))"
    echo " [1] tree   ⏳ running$el"
    [[ -n "$slog" ]] && python3 - "$slog" <<'PY'
import sys, re
try:
    lines = open(sys.argv[1], errors="ignore").read().splitlines()
except Exception:
    sys.exit(0)
def ts(l):
    m = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', l)
    return None if not m else int(m[1])*3600+int(m[2])*60+int(m[3])
def span(a,b):
    if a is None or b is None: return 0
    d=b-a; return d+86400 if d<0 else d
def human(s):
    s=int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s>=3600 else (f"{s//60}m{s%60:02d}s" if s>=60 else f"{s}s")
def last_int(pat):
    v=[int(m[1]) for l in lines for m in [re.search(pat,l)] if m]
    return v[-1] if v else None

mods=[m[1] for l in lines for m in [re.search(r'Core\.(?:pipelines|provider)\.([a-z_]+)', l)] if m]
lastmod = mods[-1] if mods else ""
out=[]
if lastmod=="extract_pdf_info":
    out.append("   ↳ parsing PDF with the VLM (MinerU gives no per-page counter)")
elif lastmod=="pdf_refiner":
    total=last_int(r'Found (\d+) candidates for merging') or 0
    calls=[l for l in lines if 'pdf_refiner' in l and 'number of tokens in prompt' in l]
    done=len(calls); line=f"   ↳ refining paragraphs: {done}"
    if total: line+=f" / {total} candidates (~{done*100//total}%)"
    if len(calls)>=2:
        dur=span(ts(calls[0]),ts(calls[-1])) or 1; rate=done/dur*60; line+=f" · {rate:.1f}/min"
        if total and done: line+=f" · ETA {human((total-done)/(done/dur))}"
    out.append(line)
    out.append("   ↳ next: build tree + summarize every node (the long phase)")
elif lastmod in ("tree_node_summary","doc_tree_builder","outline_extractor"):
    total=last_int(r'Total (\d+) nodes added to the tree')
    lm=[m for l in lines for m in [re.search(r'Processing level (\d+) with (\d+) nodes',l)] if m]
    lvl=lm[-1][1] if lm else None
    gen=[l for l in lines if 'summary generated' in l]; done=len(gen)
    if done==0:
        oc=[l for l in lines if 'outline_extractor' in l and 'tokens in prompt' in l.lower()]
        line=f"   ↳ building tree structure (LLM outline): {len(oc)} chunks"
        if len(oc)>=2:
            dur=span(ts(oc[0]),ts(oc[-1])) or 1; line+=f" · {len(oc)/dur*60:.1f}/min"
        line+=f"{f' · {total} nodes' if total else ''}"
        out.append(line)
        out.append("   ↳ (node total & summary ETA appear once the structure is built)")
    else:
        line=f"   ↳ summarizing nodes: {done}"
        if total: line+=f" / {total} (~{done*100//total}%)"
        if lvl: line+=f" · level {lvl}"
        if len(gen)>=2:
            dur=span(ts(gen[0]),ts(gen[-1])) or 1; rate=done/dur*60; line+=f" · {rate:.1f}/min"
            if total and done: line+=f" · ETA {human((total-done)/(done/dur))}"
        out.append(line)
else:
    out.append(f"   ↳ {lastmod or 'working'}…")
print("\n".join(out))
PY
  elif (( ! state_present )) && [[ -n "$t_start" ]]; then
    echo " [1] tree   ⚠️  incomplete — legacy run activity is unknown"
  elif (( finished )); then
    echo " [1] tree   ❌ FAILED${tree_err:+   $tree_err}"
  else
    echo " [1] tree   ⬜ pending"
  fi

  # ---- GRAPH ----
  local gstart="${stage_started[graph]:-}"
  (( gstart > 0 )) 2>/dev/null || gstart=$(ep_start graph || echo "")
  local graph_artifact; graph_artifact="$(select_graph_artifact "$d")"
  if [[ -n "$graph_artifact" && -f "$graph_artifact" ]] && { (( ! state_present )) || [[ -z "${stage_status[graph]:-}" || "${stage_status[graph]}" == "succeeded" ]]; }; then
    local ents; ents="$(python3 - "$graph_artifact" <<'PY' 2>/dev/null
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as source:
        payload = json.load(source)
        graph = payload.get("graph", payload) if isinstance(payload, dict) else {}
        print(len(graph.get("nodes") or []))
except (OSError, ValueError, TypeError):
    print("?")
PY
)"
    local ge="${stage_finished[graph]:-}" dur=""
    (( ge > 0 )) 2>/dev/null || ge=$(ep_done graph || mtime "$graph_artifact")
    [[ -n "$gstart" && -n "$ge" ]] && dur="   took $(fmt $((ge - gstart)))"
    echo " [2] graph  ✅ done   ($ents entities, $(basename "$graph_artifact"))$dur"
  elif [[ -f "$d/tree.pkl" ]]; then
    local gdir="$d/kg_extractor_res" done_n=0
    # Extraction results survive across attempts, so counting everything in the
    # directory reports a previous run's work as this one's progress -- and on a
    # resumed stage that reads as nearly complete from the first second. Count
    # only what this attempt produced, and only the result files themselves.
    if [[ -n "${gstart:-}" ]] && (( gstart > 0 )); then
      done_n="$(find "$gdir" -maxdepth 1 -type f -name '*.json' \
        -newermt "@$gstart" 2>/dev/null | wc -l | tr -d ' ')"
    else
      done_n="$(find "$gdir" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    local pct=""; (( total > 0 && done_n > 0 )) && pct="  (~$(( done_n * 100 / total ))%)"
    if [[ "${stage_status[graph]:-}" == "failed" ]] || { [[ "$run_status" == "failed" ]] && [[ "$current_stage" == "graph" ]]; }; then
      local gerr="${stage_error[graph]:-}"
      echo " [2] graph  ❌ FAILED / incomplete   ${done_n}/${total} nodes${pct}${gerr:+   $gerr}"
    elif [[ "${stage_status[graph]:-}" == "running" ]]; then
      local tstr=""
      if (( done_n > 0 )); then
        local oldest newest
        oldest="$(find "$gdir" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f1 | cut -d. -f1)"
        newest="$(find "$gdir" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f1 | cut -d. -f1)"
        local base=${gstart:-$oldest}; local elapsed=$(( now - base )); (( elapsed < 1 )) && elapsed=1
        local per=$(( elapsed / (done_n>0?done_n:1) ))
        local remain=$(( total - done_n )); (( remain < 0 )) && remain=0
        local idle=$(( now - newest ))
        tstr="   elapsed $(fmt $elapsed) · ${per}s/node · ETA $(fmt $((remain * per)))"
        (( idle > 90 )) && tstr="$tstr · last node ${idle}s ago"
      fi
      echo " [2] graph  ⏳ extracting   ${done_n}/${total} nodes${pct}"
      [[ -n "$tstr" ]] && echo "            $tstr"
    elif (( ! state_present )) && [[ -n "$gstart" || "$done_n" -gt 0 ]]; then
      echo " [2] graph  ⚠️  incomplete   ${done_n}/${total} nodes${pct} — legacy run activity is unknown"
    elif (( finished )); then
      echo " [2] graph  ❌ FAILED / incomplete   ${done_n}/${total} nodes${pct}"
    else
      echo " [2] graph  ⬜ pending"
    fi
  elif (( finished )); then
    echo " [2] graph  ⬜ not built (tree failed)"
  else
    echo " [2] graph  ⬜ waiting for tree"
  fi

  # ---- VDB ----
  local vstart="${stage_started[vdb]:-}"
  (( vstart > 0 )) 2>/dev/null || vstart=$(ep_start vdb || echo "")
  local retrieval_artifact; retrieval_artifact="$(discover_retrieval_artifact "$d")"
  if retrieval_is_ready "$d" "$retrieval_artifact" && { (( ! state_present )) || [[ -z "${stage_status[vdb]:-}" || "${stage_status[vdb]}" == "succeeded" ]]; }; then
    local ve="${stage_finished[vdb]:-}" dur=""
    (( ve > 0 )) 2>/dev/null || ve=$(ep_done vdb || mtime "$retrieval_artifact")
    [[ -n "$vstart" && -n "$ve" ]] && dur="   took $(fmt $((ve - vstart)))"
    echo " [3] vdb    ✅ done   ($(basename "$retrieval_artifact"))$dur"
  elif [[ "${stage_status[vdb]:-}" == "failed" ]] || { [[ "$run_status" == "failed" ]] && [[ "$current_stage" == "vdb" ]]; }; then
    local verr="${stage_error[vdb]:-}"
    echo " [3] vdb    ❌ FAILED${verr:+   $verr}"
  elif [[ "${stage_status[vdb]:-}" == "running" ]]; then
    local el=""; [[ -n "$vstart" ]] && el="   elapsed $(fmt $((now - vstart)))"
    echo " [3] vdb    ⏳ embedding nodes…$el${retrieval_artifact:+   → $retrieval_artifact}"
  elif (( ! state_present )) && [[ -n "$vstart" ]]; then
    echo " [3] vdb    ⚠️  incomplete — legacy run activity is unknown${retrieval_artifact:+   ($retrieval_artifact)}"
  elif (( finished )); then
    echo " [3] vdb    ⬜ not built"
  else
    echo " [3] vdb    ⬜ pending"
  fi

  if [[ "$run_status" == "succeeded" ]] && [[ -f "$d/tree.pkl" ]] && [[ -n "$graph_artifact" ]] && retrieval_is_ready "$d" "$retrieval_artifact"; then
    echo " [✓] index complete — ask with: docarag ask --run '$d' 'your question'"
  elif [[ "$run_status" == "succeeded" ]]; then
    echo " ⚠ driver reported success, but one or more verified index artifacts are missing"
  elif [[ "$run_status" == "failed" ]]; then
    echo " ✗ RUN FINISHED WITH ERRORS."
    echo "   see: $d/logs/index-${current_stage:-tree}.log"
  elif [[ "$run_status" == "unknown" ]]; then
    echo " ⚠ run activity unknown (legacy run has no index-state.json or terminal driver event)"
  fi
  local drv; drv="$(tail -1 "$dlog" 2>/dev/null)"
  [[ -n "$drv" ]] && echo " driver: $drv"
  echo "──────────────────────────────────────────────────────────────"
}

if [[ "$WATCH" == "1" ]]; then
  trap 'echo; exit 0' INT
  while true; do clear; render; echo " (live — Ctrl-C to stop)"; sleep 8; done
else
  render
fi
