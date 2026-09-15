#!/usr/bin/env bash
# Start / stop the BookRAG web UI as a shareable background service.
#
#   bash Scripts/webservice.sh start      # start (detached) and print the URL
#   bash Scripts/webservice.sh stop       # stop it
#   bash Scripts/webservice.sh status     # is it running?
#   bash Scripts/webservice.sh restart
#
# 'run' is an alias for start; 'close'/'end' are aliases for stop.
#
# Override defaults with env vars, e.g.:
#   HOST=127.0.0.1 PORT=9000 RUN=runs/<uuid> bash Scripts/webservice.sh start
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

HOST="${HOST:-0.0.0.0}"                                       # all interfaces (LAN + NetBird)
PORT="${PORT:-8765}"
RUN="${RUN:-runs/666bc8af-5c07-5487-8511-68a6029324e2}"       # document to open (dropdown can switch)
ENVN="${ENVN:-gbc-rag}"
LOG="$REPO/Scripts/logs/web.log"
mkdir -p "$REPO/Scripts/logs"

port_pids() { ss -tlnp 2>/dev/null | grep -E "[.:]$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u; }
show_url() { [ "$HOST" != 0.0.0.0 ] && echo "  Share:  http://$HOST:$PORT"; echo "  Local:  http://127.0.0.1:$PORT"; }

start() {
  if [ -n "$(port_pids)" ]; then
    echo "Already running on port $PORT (pid $(port_pids | tr '\n' ' '))."
    show_url; return 0
  fi
  echo "Starting BookRAG web UI on $HOST:$PORT ..."
  setsid nohup env BOOKRAG_WEB_HOST="$HOST" BOOKRAG_WEB_PORT="$PORT" \
    conda run --no-capture-output -n "$ENVN" python Scripts/web.py "$RUN" \
    > "$LOG" 2>&1 < /dev/null & disown
  for _ in $(seq 1 25); do [ -n "$(port_pids)" ] && break; sleep 1; done
  if [ -n "$(port_pids)" ]; then
    echo "Started (pid $(port_pids | tr '\n' ' ')).  Log: $LOG"
    show_url
  else
    echo "Failed to start — last log lines:"; tail -8 "$LOG" 2>/dev/null; return 1
  fi
}

stop() {
  local pids; pids="$(port_pids)"
  if [ -z "$pids" ]; then echo "Not running (nothing on port $PORT)."; return 0; fi
  echo "Stopping (pid $(echo "$pids" | tr '\n' ' ')) ..."
  kill $pids 2>/dev/null; sleep 2
  [ -n "$(port_pids)" ] && { kill -9 $(port_pids) 2>/dev/null; sleep 1; }
  if [ -z "$(port_pids)" ]; then echo "Stopped. Port $PORT is free."
  else echo "Could not stop; try: fuser -k $PORT/tcp"; return 1; fi
}

status() {
  if [ -n "$(port_pids)" ]; then echo "RUNNING on $HOST:$PORT (pid $(port_pids | tr '\n' ' '))"; show_url
  else echo "STOPPED (nothing on port $PORT)"; fi
}

case "${1:-}" in
  start|run)        start ;;
  stop|close|end)   stop ;;
  restart)          stop; sleep 1; start ;;
  status)           status ;;
  *) echo "usage: bash Scripts/webservice.sh {start|stop|status|restart}"; exit 1 ;;
esac
