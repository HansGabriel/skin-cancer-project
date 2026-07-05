#!/bin/bash
# One-time Ollama + Gemma 3 270M setup for the DermaScan Pi 4 kiosk.
#
# ⚠ This is the ONLY step that needs internet. After `ollama pull` completes,
#   the assistant runs fully offline.
#
# Run on the Pi:  bash scripts/setup_ollama_pi.sh

set -euo pipefail

MODEL="${SKIN_OLLAMA_MODEL:-gemma3:270m}"

echo "== 1/4 Installing Ollama (needs internet, once) =="
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "ollama already installed: $(ollama --version)"
fi

echo "== 2/4 Low-RAM systemd config for the Pi 4 (2-4GB) =="
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/dermascan.conf >/dev/null <<'EOF'
[Service]
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_KEEP_ALIVE=30m
Environment=OLLAMA_CONTEXT_LENGTH=1024
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now ollama

echo "== 3/4 Pulling $MODEL (needs internet, once; ~300MB) =="
ollama pull "$MODEL"

echo "== 4/4 Smoke test =="
curl -s http://127.0.0.1:11434/api/tags | grep -q "$(echo "$MODEL" | cut -d: -f1)" \
    && echo "OK: $MODEL is available at http://127.0.0.1:11434" \
    || { echo "FAILED: model not visible via /api/tags"; exit 1; }

echo
echo "Done. The DermaScan assistant will auto-detect Ollama; toggle 'AI wording'"
echo "in Settings. If Ollama is stopped, the assistant falls back to verbatim"
echo "doctor-reviewed answers automatically."
