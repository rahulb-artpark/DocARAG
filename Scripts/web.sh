#!/usr/bin/env bash
# Launch the local BookRAG web UI.
#   bash Scripts/web.sh
#   bash Scripts/web.sh <run_dir>
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n gbc-rag python Scripts/web.py "$@"
