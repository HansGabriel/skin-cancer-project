#!/bin/bash
# Quality gates: relaxed enough for real close-up camera shots, but still strict
# enough to reject an all-black/blank/blown-out frame. Bump these toward the
# defaults (35 / 35 / 220 / 0.15) if too many bad photos get through.
export SKIN_QUALITY_BLUR_MIN=8
export SKIN_QUALITY_V_MIN=15
export SKIN_QUALITY_V_MAX=245
export SKIN_QUALITY_SKIN_MIN=0.03

# Pi 4 tuning: 4 inference threads; 7" 1024x600 display profile (larger type +
# touch targets). Tokens are read at import — export BEFORE streamlit run.
export SKIN_NUM_THREADS=4
export SKIN_DISPLAY=7in

streamlit run frontend/app.py
