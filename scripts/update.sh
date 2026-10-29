#!/bin/bash
# Update the kiosk from GitHub. Pressed as "Update from GitHub" on the Settings
# screen (kiosk/server.py writes /tmp/dermascan_update, scripts/launch_kiosk.sh
# stops the kiosk and runs this), or by hand: bash scripts/update.sh
#
# Steps, each one stopping with a plain message if it cannot go on:
#   1. the Pi must be on branch main            -> "refused: ... ask Hans"
#   2. no local edits                           -> "refused: ... ask Hans"
#   3. is there internet?                       -> "failed: no internet"
#   4. fetch from GitHub                        -> "failed: could not fetch"
#   5. nothing new?                             -> "already up to date"
#   6. fast-forward only, never a merge commit  -> "failed: history diverged"
#   7. pip install only if requirements.txt changed; a failure puts the old code back
#      (packages already upgraded stay upgraded; the log says which)
#   8. pytest -q -x; a failure puts the old code back
#   9. "updated OLD -> NEW"
#
# The fetch goes to the public HTTPS address by name, not to whatever `origin` is
# set to: on the Pi `origin` is git@github.com (SSH), which wants a key and a
# terminal to ask on, and the launcher has neither. That fetch failed and the old
# message blamed the wifi (29 Oct 2026). Every git prompt is switched off, so a
# bad fetch fails in words instead of hanging the launcher.
#
# The outcome is one line in ~/.dermascan/last_update.txt, which the Settings
# screen shows. This script always exits 0 so the launcher relaunches the kiosk
# whatever happened. The paths must match dermascan/config.py (LAST_UPDATE_FILE).
#
#   bash scripts/update.sh [PROJECT_DIR] [STATE_DIR] [REPO_URL]   (all default to the usual places)
set -u

PROJECT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
STATE_DIR="${2:-$HOME/.dermascan}"
REPO_URL="${3:-https://github.com/HansGabriel/skin-cancer-project.git}"
RESULT="$STATE_DIR/last_update.txt"
LOG="$STATE_DIR/update.log"
export GIT_TERMINAL_PROMPT=0   # never wait for a username or password nobody can type
TIMEOUT=""; command -v timeout >/dev/null && TIMEOUT="timeout 90"

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

DIRTY="$(git diff --name-only HEAD | head -3 | paste -sd' ' -)"   # the "-" keeps BSD paste (a Mac) happy too
[ -z "$DIRTY" ] || finish "refused: local edits in $DIRTY — ask Hans"

# Internet first, so "no wifi" and "GitHub said no" are two different messages.
case "$REPO_URL" in
  http*) curl -sSfI --max-time 8 -o /dev/null "$REPO_URL" >>"$LOG" 2>&1 || finish "failed: no internet — is the Pi on wifi?" ;;
esac
echo "--- $(date) · fetch $REPO_URL" >>"$LOG"
$TIMEOUT git fetch "$REPO_URL" main >>"$LOG" 2>&1 || finish "failed: could not fetch from GitHub — see update.log, ask Hans"

OLD="$(git rev-parse --short HEAD)"
NEW="$(git rev-parse --short FETCH_HEAD)"
[ "$OLD" != "$NEW" ] || finish "already up to date ($OLD)"

git merge --ff-only FETCH_HEAD >>"$LOG" 2>&1 || finish "failed: history diverged from GitHub — ask Hans"

rollback() { git reset --hard "$OLD" >>"$LOG" 2>&1; }

if [ -x venv/bin/pip ] && git diff --name-only "$OLD" "$NEW" | grep -qx "requirements.txt"; then
  venv/bin/pip install -r requirements.txt >>"$LOG" 2>&1 || { rollback; finish "failed: pip install — rolled back to $OLD, ask Hans"; }
fi

if [ -x venv/bin/python ] && [ -d tests ]; then
  venv/bin/python -m pytest -q -x tests >>"$LOG" 2>&1 || { rollback; finish "failed: tests — rolled back to $OLD, ask Hans"; }
fi

finish "updated $OLD → $NEW"
