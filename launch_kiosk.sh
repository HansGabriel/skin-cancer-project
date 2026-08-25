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

# Capture quality. These only decide whether a photo earns an advisory note —
# a genuinely unreadable frame is caught by the separate hard thresholds, and
# "this is not a skin spot" is services/lesion_gate.py, not these.
# Focus is left at the services/quality.py default, which is now calibrated
# against real HAM10000 dermoscopy (median 79) rather than the synthetic
# patches in samples/. The old export here was 120 — identical to the old
# default, so the kiosk relaxed nothing while refusing 72% of real lesions.
export SKIN_QUALITY_V_MIN=25
export SKIN_QUALITY_V_MAX=235

# No file watcher. `fileWatcherType` is unset by default, which resolves to
# "auto" — and because watchdog is not in requirements-pi.txt, Streamlit falls
# back to a polling watcher that stats every source file on a 0.2s cycle, off
# the SD card, for the life of the process. A device that is not being edited
# gains nothing from it. Set here rather than in .streamlit/config.toml so the
# Mac keeps hot reload and Streamlit Cloud is untouched.
# Editing code on the Pi now needs a restart, which it effectively did anyway.
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Kiosk mode: staff passcode defaults, no free-text inputs, offline pill, and
# the Exit control in Settings. (There is no longer an Exit key in the nav —
# see components/bottom_nav.py for why.)
export SKIN_KIOSK=1

# Show the assistant's answers before a clinician has signed them off. All 18
# entries in data/assistant_kb.json lack reviewed_by/reviewed_date, so without
# this the "Questions" tab and both "Ask about this" buttons hide themselves —
# working as designed, not a bug. Demo only: fill in the review fields (or drop
# this line) before the kiosk is used with real participants.
export SKIN_KB_DEV=1

# Staff passcode (locks History/Cases/Settings behind the on-screen keypad).
# Read from a staff-held file that is NOT in git. Without it the staff area
# stays closed rather than open — see enforce_staff_gate() in services/auth.py.
PASSCODE_FILE="${DERMASCAN_PASSCODE_FILE:-$HOME/.dermascan_passcode}"
if [ -r "$PASSCODE_FILE" ]; then
  DERMASCAN_PASSCODE="$(tr -d "[:space:]" < "$PASSCODE_FILE")"
  export DERMASCAN_PASSCODE
  echo "Staff passcode loaded from $PASSCODE_FILE"
else
  echo "WARNING: no staff passcode at $PASSCODE_FILE — History, saved scans and"
  echo "         Settings will be CLOSED on this kiosk. Create the file with:"
  echo "           printf %s '<your-code>' > $PASSCODE_FILE && chmod 600 $PASSCODE_FILE"
fi

# How long a staff unlock lasts before it re-locks itself (seconds).
export DERMASCAN_STAFF_TIMEOUT=300

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
# SURF_ZOOM scales CSS pixels on the 7" 1024x600 panel. At 1.0 the type is
# legible on a monitor but too small at arm's length on a 7" screen, so the
# The layout is designed for exactly 1024x600, so the panel wants no zoom.
# Only raise this if the compositor is NOT running the panel at its native
# mode — check with `wlr-randr` first, since these boards advertise 1920x1080
# over EDID and a zoomed 1080p desktop is the wrong fix for small text.
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
