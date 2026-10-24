#!/bin/bash
# Update the kiosk from GitHub. Pressed as "Update from GitHub" on the Settings
# screen (kiosk/server.py writes /tmp/dermascan_update, scripts/launch_kiosk.sh
# stops the kiosk and runs this), or by hand: bash scripts/update.sh
#
# Steps, each one stopping with a plain message if it cannot go on:
#   1. the Pi must be on branch main            -> "refused: ... ask Hans"
#   2. no local edits                           -> "refused: ... ask Hans"
#   3. fetch from GitHub (needs wifi)           -> "failed: GitHub not reachable"
#   4. nothing new?                             -> "already up to date"
#   5. fast-forward only, never a merge commit  -> "failed: history diverged"
#   6. pip install only if requirements.txt changed; a failure rolls back
#   7. pytest -q -x; a failure rolls back to the old commit
#   8. "updated OLD -> NEW"
#
# The outcome is one line in ~/.dermascan/last_update.txt, which the Settings
# screen shows. This script always exits 0 so the launcher relaunches the kiosk
# whatever happened. The paths must match dermascan/config.py (LAST_UPDATE_FILE).
#
#   bash scripts/update.sh [PROJECT_DIR] [STATE_DIR]     (both default to the usual places)
set -u

PROJECT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
STATE_DIR="${2:-$HOME/.dermascan}"
RESULT="$STATE_DIR/last_update.txt"
LOG="$STATE_DIR/update.log"

mkdir -p "$STATE_DIR"
cd "$PROJECT_DIR" || { echo "$(date '+%Y-%m-%d %H:%M') · failed: $PROJECT_DIR is missing" >"$RESULT"; exit 0; }

finish() {  # finish "<message>": record it, print it, leave
  local msg="$1"
  echo "$(date '+%Y-%m-%d %H:%M') · $msg" | tee "$RESULT"
  echo "--- $(date) · $msg" >>"$LOG"
  exit 0
}

BRANCH="$(git branch --show-current 2>/dev/null)"
[ "$BRANCH" = "main" ] || finish "refused: the Pi is on branch '${BRANCH:-?}', not main — ask Hans"

DIRTY="$(git status --porcelain --untracked-files=no | awk '{print $2}' | head -3 | paste -sd' ')"
[ -z "$DIRTY" ] || finish "refused: local edits in $DIRTY — ask Hans"

git fetch origin main >>"$LOG" 2>&1 || finish "failed: GitHub not reachable — is the Pi on wifi?"

OLD="$(git rev-parse --short HEAD)"
NEW="$(git rev-parse --short origin/main)"
[ "$OLD" != "$NEW" ] || finish "already up to date ($OLD)"

git merge --ff-only origin/main >>"$LOG" 2>&1 || finish "failed: history diverged from GitHub — ask Hans"

rollback() { git reset --hard "$OLD" >>"$LOG" 2>&1; }

if [ -x venv/bin/pip ] && git diff --name-only "$OLD" "$NEW" | grep -qx "requirements.txt"; then
  venv/bin/pip install -r requirements.txt >>"$LOG" 2>&1 || { rollback; finish "failed: pip install — rolled back to $OLD, ask Hans"; }
fi

if [ -x venv/bin/python ] && [ -d tests ]; then
  venv/bin/python -m pytest -q -x tests >>"$LOG" 2>&1 || { rollback; finish "failed: tests — rolled back to $OLD, ask Hans"; }
fi

finish "updated $OLD → $NEW"
