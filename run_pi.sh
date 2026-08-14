#!/bin/bash
# Quality gates: relaxed enough for real close-up camera shots, but still strict
# enough to reject an all-black/blank/blown-out frame.
export SKIN_QUALITY_BLUR_MIN=8
export SKIN_QUALITY_V_MIN=15
export SKIN_QUALITY_V_MAX=245
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

streamlit run frontend/app.py
