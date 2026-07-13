#!/usr/bin/env bash
# Launch the interactive BookRAG chat REPL.
# Models load once, then ask questions in a loop until you type exit/quit/Ctrl-D.
#
#   bash Scripts/chat.sh            # clean UI
#   BOOKRAG_VERBOSE=1 bash Scripts/chat.sh   # show pipeline logs (retrieval/entity steps)
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n gbc-rag python Scripts/chat.py
