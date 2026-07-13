#!/usr/bin/env bash
# Show BookRAG indexing progress for a run directory (clean summary + timing).
#
#   bash Scripts/progress.sh                 # latest run under runs/
#   bash Scripts/progress.sh <run_dir>       # a specific run
#   bash Scripts/progress.sh -w [run_dir]    # live watch (refresh every 8s)
cd "$(dirname "$0")/.."
REPO="$(pwd)"

WATCH=0
RUN_DIR=""
for a in "$@"; do
  case "$a" in
    -w|--watch) WATCH=1 ;;
    *) RUN_DIR="$a" ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -dt "$REPO"/runs/*/ "$REPO"/runs/*/*/ 2>/dev/null \
            | while read -r d; do [[ -d "$d/logs" || -f "$d/tree.json" || -f "$d/tree.pkl" ]] && echo "$d"; done \
            | head -1)"
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

render() {
  local d="$RUN_DIR" now; now=$(date +%s)
  local dlog="$d/logs/index-driver.log"
  local name; name="$(python3 -c "import json;print(json.load(open('$d/tree.json')).get('meta_info',{}).get('file_name',''))" 2>/dev/null)"
  [[ -z "$name" ]] && name="$(basename "$d")"

  local total=0
  [[ -f "$d/tree.json" ]] && total="$(python3 -c "import json;print(len(json.load(open('$d/tree.json'))['nodes']))" 2>/dev/null || echo 0)"

  # --- stage timestamps from driver log (index_pdf.sh writes these) ---
  ep_start() { local ln; ln=$(grep -a "starting $1$" "$dlog" 2>/dev/null | tail -1); [[ -z "$ln" ]] && return 1
               ln=${ln#\[driver\] }; ln=${ln% starting $1}; date -d "$ln" +%s 2>/dev/null; }
  ep_done()  { local ln; ln=$(grep -a "$1 done" "$dlog" 2>/dev/null | tail -1); [[ -z "$ln" ]] && return 1
               ln=${ln#\[driver\] }; ln=${ln% $1 done*}; date -d "$ln" +%s 2>/dev/null; }

  # --- is the run still active? (driver finished, or no index process alive) ---
  local proc_alive=0
  pgrep -f "main.py .*index --stage" >/dev/null 2>&1 && proc_alive=1
  local finished=0
  if grep -q "ALL DONE" "$dlog" 2>/dev/null; then finished=1
  elif [[ -f "$dlog" ]] && (( ! proc_alive )); then finished=1; fi

  # freeze the clock at the last driver event when finished
  local asof=$now
  if (( finished )); then
    local ll; ll=$(tail -1 "$dlog" 2>/dev/null); ll=${ll#\[driver\] }
    ll=$(printf '%s' "$ll" | sed -E 's/ (ALL DONE|starting [a-z_]+|[a-z_]+ done.*)$//')
    local e; e=$(date -d "$ll" +%s 2>/dev/null) && asof=$e
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

  local t_start; t_start=$(ep_start tree || echo "")

  echo "──────────────────────────────────────────────────────────────"
  echo " BookRAG indexing progress"
  echo "   doc : $name"
  echo "   dir : $d"
  if [[ -n "$t_start" ]]; then
    if (( finished )); then echo "   started : $(date -d @"$t_start" '+%H:%M:%S')   ran for : $(fmt $((asof - t_start)))"
    else echo "   started : $(date -d @"$t_start" '+%H:%M:%S')   elapsed : $(fmt $((now - t_start)))"; fi
  fi
  echo "──────────────────────────────────────────────────────────────"

  # ---- TREE ----
  if [[ -f "$d/tree.pkl" ]]; then
    local te ts dur=""; te=$(ep_done tree || mtime "$d/tree.pkl"); ts=${t_start:-$(ep_start tree)}
    [[ -n "$ts" && -n "$te" ]] && dur="   took $(fmt $((te - ts)))"
    echo " [1] tree   ✅ done   ($total nodes)$dur"
  elif (( finished )); then
    echo " [1] tree   ❌ FAILED${tree_err:+   $tree_err}"
  else
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
  fi

  # ---- GRAPH ----
  local gstart; gstart=$(ep_start graph || echo "")
  if [[ -f "$d/graph_data.json" ]]; then
    local ents; ents="$(python3 -c "import json;print(len(json.load(open('$d/graph_data.json'))['graph'].get('nodes',[])))" 2>/dev/null || echo '?')"
    local ge dur=""; ge=$(ep_done graph || mtime "$d/graph_data.json")
    [[ -n "$gstart" && -n "$ge" ]] && dur="   took $(fmt $((ge - gstart)))"
    echo " [2] graph  ✅ done   ($ents entities)$dur"
  elif [[ -f "$d/tree.pkl" ]]; then
    local gdir="$d/kg_extractor_res" done_n=0
    done_n="$(ls "$gdir" 2>/dev/null | wc -l | tr -d ' ')"
    local pct=""; (( total > 0 && done_n > 0 )) && pct="  (~$(( done_n * 100 / total ))%)"
    if (( finished )); then
      echo " [2] graph  ❌ FAILED / incomplete   ${done_n}/${total} nodes${pct}"
    else
      local tstr=""
      if (( done_n > 0 )); then
        local oldest newest; oldest=$(mtime "$(ls -tr "$gdir"/*.json 2>/dev/null | head -1)")
        newest=$(mtime "$(ls -t "$gdir"/*.json 2>/dev/null | head -1)")
        local base=${gstart:-$oldest}; local elapsed=$(( now - base )); (( elapsed < 1 )) && elapsed=1
        local per=$(( elapsed / (done_n>0?done_n:1) ))
        local remain=$(( total - done_n )); (( remain < 0 )) && remain=0
        local idle=$(( now - newest ))
        tstr="   elapsed $(fmt $elapsed) · ${per}s/node · ETA $(fmt $((remain * per)))"
        (( idle > 90 )) && tstr="$tstr · last node ${idle}s ago"
      fi
      echo " [2] graph  ⏳ extracting   ${done_n}/${total} nodes${pct}"
      [[ -n "$tstr" ]] && echo "            $tstr"
    fi
  elif (( finished )); then
    echo " [2] graph  ⬜ not built (tree failed)"
  else
    echo " [2] graph  ⬜ waiting for tree"
  fi

  # ---- VDB ----
  local vstart; vstart=$(ep_start vdb || echo "")
  if [[ -d "$d/Tree_vdb" ]] && find "$d/Tree_vdb" \( -name '*.bin' -o -name 'data_level0.bin' \) 2>/dev/null | grep -q .; then
    local ve dur=""; ve=$(ep_done vdb || mtime "$d/Tree_vdb")
    [[ -n "$vstart" && -n "$ve" ]] && dur="   took $(fmt $((ve - vstart)))"
    echo " [3] vdb    ✅ done$dur"
  elif [[ -d "$d/Tree_vdb" ]] && (( ! finished )); then
    local el=""; [[ -n "$vstart" ]] && el="   elapsed $(fmt $((now - vstart)))"
    echo " [3] vdb    ⏳ embedding nodes…$el"
  elif (( finished )); then
    echo " [3] vdb    ⬜ not built"
  else
    echo " [3] vdb    ⬜ pending"
  fi

  if ls "$d"/eval_*/final_results.json >/dev/null 2>&1; then
    echo " [✓] index complete — ask via Scripts/chat.sh · view via Scripts/view.sh"
  elif (( finished )) && [[ ! -f "$d/tree.pkl" ]]; then
    echo " ✗ RUN FINISHED WITH ERRORS — nothing was built."
    echo "   see: $d/logs/index-tree.log"
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
