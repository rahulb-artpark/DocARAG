#!/usr/bin/env bash
# Ask a question against a published DocARAG run.
# Usage:  bash Scripts/ask.sh "your question here" [run-name-or-path]
set -euo pipefail
cd "$(dirname "$0")/.."

QUESTION="${1:-}"
RUN="${2:-}"
if [[ -z "$QUESTION" ]]; then
  echo "Usage: bash Scripts/ask.sh \"your question\" [run-name-or-path]" >&2
  exit 1
fi

COMMAND=(python -m docarag ask)
if [[ -n "$RUN" ]]; then
  COMMAND+=(--run "$RUN")
fi
COMMAND+=("$QUESTION")
exec "${COMMAND[@]}"
