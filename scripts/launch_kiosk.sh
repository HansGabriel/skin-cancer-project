#!/bin/bash
# E.P.I.V.U.E. kiosk launcher for the Raspberry Pi.
# Starts the server, waits for it, opens the browser full screen, and tears
# everything down when the on-screen "Exit kiosk" button is pressed or the
# browser window is closed. Double-click EPIVUE.desktop to run this.
#
# PORT and QUIT_FLAG must match KIOSK_PORT and QUIT_FLAG in dermascan/config.py.
set -u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8080
QUIT_FLAG="/tmp/dermascan_quit"
URL="http://127.0.0.1:$PORT"
LOG="/tmp/dermascan_kiosk.log"
PIDFILE="/tmp/dermascan_server.pid"

cd "$PROJECT_DIR" || exit 1
rm -f "$QUIT_FLAG"

if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
  echo "No venv in $PROJECT_DIR. Set it up first (docs/DEPLOYMENT.md):" >&2
  echo "  python3 -m venv --system-site-packages venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# A previous run still alive? Stop exactly that process, and only if it really is
# our server (a stale pidfile can point at a PID some other program now has).
if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE")"
  if grep -qs "kiosk.server" "/proc/$OLD_PID/cmdline"; then
    kill "$OLD_PID"; sleep 1
  fi
  rm -f "$PIDFILE"
fi

# shellcheck disable=SC1091
source "$PROJECT_DIR/venv/bin/activate"
python -m kiosk.server --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PIDFILE"

# Wait up to 30 s for the model to load and the server to answer.
for _ in $(seq 1 30); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 1
done
if ! curl -sf "$URL/health" >/dev/null; then
  echo "The kiosk server did not start; the reason is at the end of $LOG" >&2
  kill "$SERVER_PID" 2>/dev/null; rm -f "$PIDFILE"; exit 1
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
rm -f "$QUIT_FLAG" "$PIDFILE"
