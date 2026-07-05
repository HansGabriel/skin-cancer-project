#!/bin/bash
# DermaScan kiosk launcher — starts Streamlit, waits for it, opens surf fullscreen.
# Closing surf (or the Stop launcher) tears everything down. Intended for the Pi
# desktop double-click icon so students can open/close the demo easily.

set -u

# Resolve the project from wherever this script lives (no hardcoded path).
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv"
URL="http://127.0.0.1:8501"
SPLASH="file://$PROJECT_DIR/frontend/static/splash.html"
LOG="/tmp/dermascan_kiosk.log"

cd "$PROJECT_DIR" || exit 1

# If a previous session is still running, clean it up first.
pkill -f "streamlit run" 2>/dev/null
pkill surf 2>/dev/null
sleep 1

# Quality gates: relaxed enough for real close-up camera shots, but still strict
# enough to reject an all-black/blank/blown-out frame. Bump toward the defaults
# (35 / 35 / 220 / 0.15) if too many bad photos slip through during testing.
export SKIN_QUALITY_BLUR_MIN=8
export SKIN_QUALITY_V_MIN=15
export SKIN_QUALITY_V_MAX=245
export SKIN_QUALITY_SKIN_MIN=0.03

# Tell the app it's in kiosk mode so it shows the on-screen Exit button.
export SKIN_KIOSK=1

# Use all 4 Pi cores for TFLite inference.
export SKIN_NUM_THREADS=4

# 7" touchscreen profile (1024x600 HDMI IPS): wider frame, larger type/touch targets.
export SKIN_DISPLAY=7in

# Clear any stale quit flag from a previous session.
QUIT_FLAG="/tmp/dermascan_quit"
rm -f "$QUIT_FLAG"

# Start Streamlit in the background, logging to a file.
source "$VENV/bin/activate"
nohup streamlit run frontend/app.py >"$LOG" 2>&1 &
STREAMLIT_PID=$!

# Open surf immediately on the local splash page — it polls Streamlit and
# redirects itself once the app answers (no blank screen during boot).
# SURF_ZOOM 1.0 suits the 7" 1024x600 panel; tweak (0.9 / 1.1) if needed.
SURF_ZOOM="${SURF_ZOOM:-1.0}"
DISPLAY=:0 surf -F -z "$SURF_ZOOM" "$SPLASH" &
SURF_PID=$!

# Watchdog only: give Streamlit up to ~40s; log if it never comes up.
for _ in $(seq 1 40); do
    if curl -s -o /dev/null "$URL"; then
        break
    fi
    sleep 1
done

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
