#!/usr/bin/env bash
# Detached driver: finish graph (resumes from temp files), then vdb, then rag.
# Launched via setsid+nohup so it survives terminal/session teardown.
cd "$(dirname "$0")/.."
SYS="config/ollama-gbc.yaml"
DS="Scripts/cfg/mytest.yaml"
LOGDIR="Scripts/logs/mytest"
mkdir -p "$LOGDIR"
COMMON="-c $SYS -d $DS --num 1 --nsplit 1"

echo "[driver] $(date) starting graph (resume)"
conda run -n gbc-rag python main.py $COMMON index --stage graph > "$LOGDIR/index-graph.log" 2>&1
echo "[driver] $(date) graph done (exit $?)"

echo "[driver] $(date) starting vdb"
conda run -n gbc-rag python main.py $COMMON index --stage vdb > "$LOGDIR/index-vdb.log" 2>&1
echo "[driver] $(date) vdb done (exit $?)"

echo "[driver] $(date) starting rag"
conda run -n gbc-rag python main.py $COMMON rag > "$LOGDIR/rag.log" 2>&1
echo "[driver] $(date) rag done (exit $?)"
echo "[driver] $(date) ALL DONE"
