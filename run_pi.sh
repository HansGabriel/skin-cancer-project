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

# Pi 4 tuning: 4 inference threads; 7" 1024x600 display profile (larger type +
# touch targets). Tokens are read at import — export BEFORE streamlit run.
export SKIN_NUM_THREADS=4
export SKIN_DISPLAY=7in

streamlit run frontend/app.py
