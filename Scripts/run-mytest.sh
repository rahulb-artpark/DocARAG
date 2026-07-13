#!/usr/bin/env bash
# Single-PDF end-to-end driver for the Ollama setup.
# Runs with --num 1 --nsplit 1 (odd num, no --debug) so main.py's process_resource()
# does NOT override devices to cuda:3/cuda:4 or MinerU to port 30001.
#
# Usage:
#   bash Scripts/run-mytest.sh index   # build BookIndex (tree -> graph -> vdb)
#   bash Scripts/run-mytest.sh rag     # run retrieval+generation over the QA file
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root

SYS_CONFIG="config/ollama-gbc.yaml"
DATASET_CONFIG="Scripts/cfg/mytest.yaml"
MODE="${1:-index}"

mkdir -p Scripts/logs/mytest

if [[ "$MODE" == "index" ]]; then
  for stage in tree graph vdb; do
    echo "=== INDEX stage: $stage ==="
    python main.py -c "$SYS_CONFIG" -d "$DATASET_CONFIG" \
      --num 1 --nsplit 1 index --stage "$stage" \
      2>&1 | tee "Scripts/logs/mytest/index-$stage.log"
  done
elif [[ "$MODE" == "rag" ]]; then
  echo "=== RAG ==="
  python main.py -c "$SYS_CONFIG" -d "$DATASET_CONFIG" \
    --num 1 --nsplit 1 rag \
    2>&1 | tee "Scripts/logs/mytest/rag.log"
else
  echo "Unknown mode: $MODE (use 'index' or 'rag')"; exit 1
fi
