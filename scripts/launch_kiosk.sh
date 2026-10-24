#!/bin/bash
# E.P.I.V.U.E. kiosk launcher for the Raspberry Pi.
# Starts the server, waits for it, opens the browser full screen, and tears
# everything down when the on-screen "Exit kiosk" button is pressed or the
# browser window is closed. Double-click EPIVUE.desktop to run this.
#
# "Update from GitHub" on the Settings screen writes UPDATE_FLAG: the kiosk is
# stopped, scripts/update.sh runs, and the kiosk starts again on the new code.
#
# PORT, QUIT_FLAG, UPDATE_FLAG and the version file must match dermascan/config.py.
set -u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8080
QUIT_FLAG="/tmp/dermascan_quit"
UPDATE_FLAG="/tmp/dermascan_update"
VERSION_FILE="$HOME/.dermascan/version.txt"
URL="http://127.0.0.1:$PORT"
LOG="/tmp/dermascan_kiosk.log"
PIDFILE="/tmp/dermascan_server.pid"

cd "$PROJECT_DIR" || exit 1
rm -f "$QUIT_FLAG" "$UPDATE_FLAG"

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

while :; do
  # The Settings screen shows this: short commit hash and its date.
  mkdir -p "$(dirname "$VERSION_FILE")"
  git log -1 --format='%h · %cs' >"$VERSION_FILE" 2>/dev/null || echo "unknown" >"$VERSION_FILE"

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
    # After an update this is the line staff will see at the next start.
    [ -n "${JUST_UPDATED:-}" ] && echo "$(date '+%Y-%m-%d %H:%M') · failed: the kiosk did not start after the update — ask Hans, send /tmp/dermascan_kiosk.log" >"$HOME/.dermascan/last_update.txt"
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

  # Stay here until the browser closes or a button writes the quit or update flag.
  while kill -0 "$BROWSER_PID" 2>/dev/null; do
    [ -f "$QUIT_FLAG" ] || [ -f "$UPDATE_FLAG" ] && break
    sleep 0.5
  done

  kill "$BROWSER_PID" 2>/dev/null
  kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  rm -f "$QUIT_FLAG" "$PIDFILE"

  [ -f "$UPDATE_FLAG" ] || break
  rm -f "$UPDATE_FLAG"
  bash "$PROJECT_DIR/scripts/update.sh" "$PROJECT_DIR"   # the outcome goes to ~/.dermascan/last_update.txt
  JUST_UPDATED=1
  # This launcher itself stays the old copy until the next double-click; the server and page are new.
done
