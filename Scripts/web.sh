#!/usr/bin/env bash
# Manage the DocARAG web UI service.
#   bash Scripts/web.sh              start (or report already running)
#   bash Scripts/web.sh stop         stop it
#   bash Scripts/web.sh status       is it up, and what is it serving
#   bash Scripts/web.sh restart      stop then start
#   bash Scripts/web.sh start --port 8888 --run <run_dir>
cd "$(dirname "$0")/.."
action="${1:-start}"
shift 2>/dev/null || true
exec conda run --no-capture-output -n gbc-rag docarag web "$action" "$@"
