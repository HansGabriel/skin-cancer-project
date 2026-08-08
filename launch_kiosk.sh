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
STREAMLIT_PIDFILE="/tmp/dermascan_streamlit.pid"
SURF_PIDFILE="/tmp/dermascan_surf.pid"

cd "$PROJECT_DIR" || exit 1

# Kill exactly the PID recorded in a pidfile — never pkill by name, which would
# take down unrelated streamlit/surf processes running on the same box.
kill_pidfile() {
    local pidfile="$1" pid
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile" 2>/dev/null)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
        rm -f "$pidfile"
    fi
}

# If a previous session is still running, clean it up first (exact PIDs only).
kill_pidfile "$STREAMLIT_PIDFILE"
kill_pidfile "$SURF_PIDFILE"
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

# Staff passcode (locks History/Cases/Settings behind the on-screen keypad).
# Set it in .streamlit/secrets.toml ([dermascan] passcode = "...") or export
# DERMASCAN_PASSCODE here from a staff-held file — never commit either. See docs/PRIVACY.md.

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
echo "$STREAMLIT_PID" >"$STREAMLIT_PIDFILE"

# Open surf immediately on the local splash page — it polls Streamlit and
# redirects itself once the app answers (no blank screen during boot).
# SURF_ZOOM 1.0 suits the 7" 1024x600 panel; tweak (0.9 / 1.1) if needed.
SURF_ZOOM="${SURF_ZOOM:-1.0}"
DISPLAY=:0 surf -F -z "$SURF_ZOOM" "$SPLASH" &
SURF_PID=$!
echo "$SURF_PID" >"$SURF_PIDFILE"

# Give Streamlit up to ~40s. If it never answers, log it, tear down, and exit
# nonzero so a supervisor (systemd Restart=on-failure) can retry the launch.
STREAMLIT_UP=0
for _ in $(seq 1 40); do
    if curl -s -o /dev/null "$URL"; then
        STREAMLIT_UP=1
        break
    fi
    sleep 1
done
if [ "$STREAMLIT_UP" -ne 1 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Streamlit did not answer on $URL within 40s; aborting kiosk (streamlit output above)." >>"$LOG"
    kill "$SURF_PID" 2>/dev/null
    kill "$STREAMLIT_PID" 2>/dev/null
    rm -f "$STREAMLIT_PIDFILE" "$SURF_PIDFILE" "$QUIT_FLAG"
    exit 1
fi

# Watch for the on-screen Exit button's quit flag; close surf when it appears.
# Also exit the loop if surf is closed manually (Ctrl+W / Alt+F4).
while kill -0 "$SURF_PID" 2>/dev/null; do
    if [ -f "$QUIT_FLAG" ]; then
        kill "$SURF_PID" 2>/dev/null
        break
    fi
    sleep 0.5
done

# Tear everything down so nothing lingers (exact PIDs only — no pkill).
rm -f "$QUIT_FLAG"
kill "$SURF_PID" 2>/dev/null
kill "$STREAMLIT_PID" 2>/dev/null
rm -f "$STREAMLIT_PIDFILE" "$SURF_PIDFILE"
