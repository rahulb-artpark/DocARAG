#!/usr/bin/env bash
# Terminal viewer for a built BookRAG index (document tree + knowledge graph).
#   bash Scripts/view.sh              # pick from indexed PDFs (or auto if only one)
#   bash Scripts/view.sh <run_dir>    # view a specific run directory
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n gbc-rag python Scripts/view.py "$@"
