#!/bin/bash
# Quality gates. These must match launch_kiosk.sh, or testing here proves
# nothing about the device people actually use.
#
# SKIN_QUALITY_BLUR_MIN used to be exported as 8 here — five times below the
# calibrated default, and never set in launch_kiosk.sh at all, so the kiosk
# silently ran 40 while this script ran 8. It only moves the *advisory*
# threshold (the hard floor is 20 and is what actually blocks), so all it did
# was hide "slightly blurry" during dev and show it on the demo. It is dropped
# rather than copied across: soft close-ups are a fixed-focus IMX219 focused for
# ~1m, and the fix is refocusing the lens for ~6cm, not lowering the bar.
export SKIN_QUALITY_V_MIN=25
export SKIN_QUALITY_V_MAX=235
# No skin-gate override. This file used to export SKIN_QUALITY_SKIN_MIN=0.03,
# which nothing reads — the real name is SKIN_GATE_SKIN_MIN — so the Pi silently
# ran the strict default for its whole life. Rather than resurrect it, the
# default in services/lesion_gate.py is now calibrated against real dermoscopy.
# Do not set it to 0.03 here: that is below the grey-surface score and would
# make a photo of a desk read as skin.

# Show the assistant's knowledge base before a clinician has signed it off.
#
# data/assistant_kb.json holds 18 entries and NONE of them carry reviewed_by /
# reviewed_date, so backend/assistant.py hides all 18 and kb_is_live() returns
# False — which is why the "Questions" tab and both "Ask about this" buttons
# were invisible. Nothing was missing; the doctor-review gate was doing its job.
#
# This flag is for the demo. The gate exists so unreviewed medical wording
# cannot reach a patient, so before this device is used with real participants,
# either fill in reviewed_by/reviewed_date in the KB or drop this line.
export SKIN_KB_DEV=1

# Pi 4 tuning: 4 inference threads; 7" 1024x600 display profile (larger type +
# touch targets). Tokens are read at import — export BEFORE streamlit run.
export SKIN_NUM_THREADS=4
export SKIN_DISPLAY=7in

# No file watcher. `fileWatcherType` is unset by default, which resolves to
# "auto" — and because watchdog is not in requirements-pi.txt, Streamlit falls
# back to a polling watcher that stats every source file on a 0.2s cycle, off
# the SD card, for the life of the process. A device that is not being edited
# gains nothing from it. Set here rather than in .streamlit/config.toml so the
# Mac keeps hot reload and Streamlit Cloud is untouched.
# Editing code on the Pi now needs a restart, which it effectively did anyway.
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Use the project venv, not whatever streamlit happens to be on PATH. Without
# this the script launched the system interpreter, which has no ai-edge-litert.
VENV="$(cd "$(dirname "$0")" && pwd)/venv"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

streamlit run frontend/app.py
