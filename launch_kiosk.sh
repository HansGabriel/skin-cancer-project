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

# Tell the app it's in kiosk mode so it shows the on-screen Exit button.
export SKIN_KIOSK=1

# Clear any stale quit flag from a previous session.
QUIT_FLAG="/tmp/dermascan_quit"
rm -f "$QUIT_FLAG"

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

# Open surf fullscreen on the LCD, zoomed out to fit the 480x320 screen.
# Adjust SURF_ZOOM (e.g. 0.5, 0.6, 0.7) if it's too big or too small.
SURF_ZOOM="${SURF_ZOOM:-0.5}"
DISPLAY=:0 surf -F -z "$SURF_ZOOM" "$URL" &
SURF_PID=$!

# Watch for the on-screen Exit button's quit flag; close surf when it appears.
# Also exit the loop if surf is closed manually (Ctrl+W / Alt+F4).
while kill -0 "$SURF_PID" 2>/dev/null; do
    if [ -f "$QUIT_FLAG" ]; then
        kill "$SURF_PID" 2>/dev/null
        break
    fi
    sleep 0.5
done

# Tear everything down so nothing lingers.
rm -f "$QUIT_FLAG"
pkill surf 2>/dev/null
kill "$STREAMLIT_PID" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null
