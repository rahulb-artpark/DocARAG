#!/usr/bin/env bash
# Let Ollama run more than one generation at a time.
#
# Measured before writing this: two identical generations took 2.50s against
# 2.60s serial and ~1.30s parallel; four took 4.89s against 5.20s. The server
# was running them one at a time, which is why refine_entities costs its full
# ~126 minutes while already threading 8-way through iter_bounded_ordered, and
# why both forms of request batching measured ~1.00x.
#
# OLLAMA_CONTEXT_LENGTH matters as much as the slot count. The resident model
# reported ctx=262144 and held 32.2 GiB against a 22.3 GiB model -- ~10 GiB of
# KV cache for a context config/ollama-gbc.yaml never asked for. Four slots do
# not fit until that is corrected; at 16K they do.
#
# Run: sudo bash Scripts/enable_ollama_parallelism.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "needs root: sudo bash $0" >&2
    exit 1
fi

OVERRIDE=/etc/systemd/system/ollama.service.d/override.conf

if pgrep -af "index_pdf|docarag" | grep -v grep >/dev/null; then
    echo "refusing to restart: indexing is running" >&2
    exit 1
fi

mkdir -p "$(dirname "$OVERRIDE")"
if [[ -f "$OVERRIDE" ]]; then
    cp "$OVERRIDE" "$OVERRIDE.bak.$(date +%Y%m%d%H%M%S)"
    echo "backed up the existing override"
fi

cat > "$OVERRIDE" <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
EOF

systemctl daemon-reload
systemctl restart ollama
sleep 8

systemctl is-active ollama
systemctl show ollama --property=Environment | tr ' ' '\n' | grep -iE "PARALLEL|CONTEXT" || true

echo
echo "now measure it:"
echo "  python Scripts/probes/ollama_parallelism_probe.py"
