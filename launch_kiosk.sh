#!/bin/bash
# DermaScan kiosk launcher — starts Streamlit, waits for it, opens surf fullscreen.
# Closing surf (or the Stop launcher) tears everything down. Intended for the Pi
# desktop double-click icon so students can open/close the demo easily.

set -u

PROJECT_DIR="$HOME/Documents/skin-cancer-project"
VENV="$PROJECT_DIR/venv"
URL="http://127.0.0.1:8501"
LOG="/tmp/dermascan_kiosk.log"

cd "$PROJECT_DIR" || exit 1

# If a previous session is still running, clean it up first.
pkill -f "streamlit run" 2>/dev/null
pkill surf 2>/dev/null
sleep 1

# Quality gates relaxed (default brightness threshold rejects real photos).
export SKIN_QUALITY_BLUR_MIN=0
export SKIN_QUALITY_V_MIN=0
export SKIN_QUALITY_V_MAX=254
export SKIN_QUALITY_SKIN_MIN=0

# Start Streamlit in the background, logging to a file.
source "$VENV/bin/activate"
nohup streamlit run frontend/app.py >"$LOG" 2>&1 &
STREAMLIT_PID=$!

# Wait (up to ~40s) for Streamlit to answer before opening the browser.
for _ in $(seq 1 40); do
    if curl -s -o /dev/null "$URL"; then
        break
    fi
    sleep 1
done

# Open surf fullscreen on the LCD. surf blocks until the window is closed.
DISPLAY=:0 surf -F "$URL"

# When surf closes, shut Streamlit down so nothing lingers.
kill "$STREAMLIT_PID" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null
