#!/usr/bin/env bash
# "Upload" a PDF: build a BookRAG index (tree -> graph -> vdb) for any PDF,
# then view it with Scripts/view.sh.
#
#   bash Scripts/index_pdf.sh /abs/path/to/document.pdf
#   bash Scripts/index_pdf.sh                # prompts for the path
#
# The index runs DETACHED (survives closing the terminal). Tree is usually
# ready in a few minutes; the knowledge graph can take 1-2h on a local LLM.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

PDF="${1:-}"
if [[ -z "$PDF" ]]; then
  read -rp "Path to PDF: " PDF
fi
PDF="$(readlink -f "$PDF")" || true
if [[ ! -f "$PDF" ]]; then
  echo "[error] PDF not found: $PDF" >&2
  exit 1
fi

# Stable per-document id + names
UUID="$(python3 -c "import uuid,sys; print(uuid.uuid5(uuid.NAMESPACE_URL, sys.argv[1]))" "$PDF")"
NAME="$(basename "$PDF" .pdf | tr ' /' '__' | tr -cd 'A-Za-z0-9_-')"
WORKDIR="$REPO/runs"
DSDIR="$REPO/runs/_datasets"
mkdir -p "$DSDIR" "$WORKDIR/$UUID/logs"
JSON="$DSDIR/${NAME}.json"
CFG="$DSDIR/${NAME}.cfg.yaml"

# dataset JSON (question is only needed for rag; empty is fine for indexing)
UUID="$UUID" PDF="$PDF" python3 - "$JSON" <<'PY'
import json, os, sys
json.dump([{ "question": "", "answer": "",
             "doc_uuid": os.environ["UUID"], "doc_path": os.environ["PDF"] }],
          open(sys.argv[1], "w"), indent=2, ensure_ascii=False)
PY

# dataset config -> working_dir/<uuid> becomes the run dir
cat > "$CFG" <<EOF
dataset_path: $JSON
working_dir: $WORKDIR
dataset_name: $NAME
EOF

RUN_DIR="$WORKDIR/$UUID"
DRIVER_LOG="$RUN_DIR/logs/index-driver.log"

# Detached driver: tree -> graph -> vdb
setsid nohup bash -c '
  cd "'"$REPO"'"
  C="-c config/ollama-gbc.yaml -d '"$CFG"' --num 1 --nsplit 1"
  for stage in tree graph vdb; do
    echo "[driver] $(date) starting $stage"
    conda run -n gbc-rag python main.py $C index --stage $stage \
      > "'"$RUN_DIR"'/logs/index-$stage.log" 2>&1
    echo "[driver] $(date) $stage done (exit $?)"
  done
  echo "[driver] $(date) ALL DONE"
' > "$DRIVER_LOG" 2>&1 < /dev/null &
disown

echo "Indexing started (detached) for: $PDF"
echo "  run dir : $RUN_DIR"
echo "  uuid    : $UUID"
echo
echo "Watch progress : tail -f $DRIVER_LOG"
echo "                 tail -f $RUN_DIR/logs/index-tree.log   (per-stage detail)"
echo "View when ready: bash Scripts/view.sh $RUN_DIR"
echo "  (tree is viewable in a few minutes; full graph takes longer)"
