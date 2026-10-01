"""The web demo on Streamlit Community Cloud: upload a photo, read the verdict.

This is the whole of it. Everything that decides anything lives in dermascan/,
the same code the Pi kiosk runs, so the two can never disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dermascan.scan import run_scan  # noqa: E402

st.set_page_config(page_title="E.P.I.V.U.E. skin check", page_icon="🔬", layout="centered")
st.title("E.P.I.V.U.E. — skin check")
st.caption("An educational screening aid, not a diagnosis. Only a health worker can tell you what a spot is.")

photo = st.file_uploader("A close-up photo of one spot on skin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
force = st.session_state.get("force", False)

if photo is not None:
    data = photo.getvalue()
    left, right = st.columns([2, 3])
    left.image(data, use_container_width=True)
    with st.spinner("Reading the spot…"):
        out = run_scan(data, force=force)
    v = out.verdict
    with right:
        st.subheader(v.headline)
        st.write(v.body)
        st.markdown(f"**{v.advice}**")
        if v.note:
            st.info(f"**{v.note_label}** — {v.note}")
        if out.forced:
            st.warning("This photo did not pass the usual checks and was read anyway. Treat the result with extra caution.")
        if out.status == "refused" and out.refusal and out.refusal.can_override and not force:
            if st.button("Check it anyway"):
                st.session_state["force"] = True
                st.rerun()
    with st.expander("Details for staff"):
        if out.prediction:
            st.write({k: f"{v:.0f}%" for k, v in out.prediction.probs_pct.items()})
            st.write("flagged for a doctor" if out.prediction.flagged else "not flagged")
        st.write({k: round(v, 3) for k, v in out.measured.items()})
        st.write({f"{k} ms": v for k, v in out.stage_ms.items()})
else:
    st.session_state["force"] = False
    st.write("Upload a photo to try it. Nothing is stored.")
