#!/bin/bash
# Quality gates: relaxed enough for real close-up camera shots, but still strict
# enough to reject an all-black/blank/blown-out frame. Bump these toward the
# defaults (35 / 35 / 220 / 0.15) if too many bad photos get through.
export SKIN_QUALITY_BLUR_MIN=8
export SKIN_QUALITY_V_MIN=15
export SKIN_QUALITY_V_MAX=245
export SKIN_QUALITY_SKIN_MIN=0.03

streamlit run frontend/app.py
