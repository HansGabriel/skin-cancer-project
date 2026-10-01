#!/bin/bash
# E.P.I.V.U.E. kiosk launcher for the Raspberry Pi.
# Starts the server, waits for it, opens the browser full screen, and tears
# everything down when the on-screen "Exit kiosk" button is pressed or the
# browser window is closed. Double-click EPIVUE.desktop to run this.
set -u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
URL="http://127.0.0.1:8080"
LOG="/tmp/dermascan_kiosk.log"
QUIT_FLAG="/tmp/dermascan_quit"

cd "$PROJECT_DIR" || exit 1
rm -f "$QUIT_FLAG"

# A previous run still alive? Stop exactly that one (never pkill by name).
if [ -f /tmp/dermascan_server.pid ] && kill -0 "$(cat /tmp/dermascan_server.pid)" 2>/dev/null; then
  kill "$(cat /tmp/dermascan_server.pid)"; sleep 1
fi

# shellcheck disable=SC1091
source "$PROJECT_DIR/venv/bin/activate"
python -m kiosk.server >"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >/tmp/dermascan_server.pid

# Wait up to 30 s for the model to load and the server to answer.
for _ in $(seq 1 30); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 1
done
if ! curl -sf "$URL/health" >/dev/null; then
  echo "server did not start; see $LOG" >&2
  kill "$SERVER_PID" 2>/dev/null; exit 1
fi

# surf is the small WebKitGTK browser installed on the Pi; -F is full screen.
# Fall back to chromium if surf is missing.
if command -v surf >/dev/null; then
  DISPLAY=:0 surf -F "$URL" &
else
  DISPLAY=:0 chromium --kiosk --noerrdialogs --disable-infobars "$URL" &
fi
BROWSER_PID=$!

# Stay here until the browser closes or the Exit button writes the quit flag.
while kill -0 "$BROWSER_PID" 2>/dev/null; do
  [ -f "$QUIT_FLAG" ] && break
  sleep 0.5
done

kill "$BROWSER_PID" 2>/dev/null
kill "$SERVER_PID" 2>/dev/null
rm -f "$QUIT_FLAG" /tmp/dermascan_server.pid
