"""3-class and optional 7-class probability bars."""

from __future__ import annotations

import streamlit as st

from services.gradcam import HAM7_LABELS


PRETTY7 = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}


def render_three_class_probs(probs: dict[str, float]) -> None:
    st.subheader("CNN (3-class)")
    for key in ("benign", "pre_cancerous", "malignant"):
        p = float(probs.get(key, 0.0))
        st.progress(p / 100.0, text=f"{key.replace('_', ' ')}: {p:.1f}%")


def render_seven_class_expander(seven: dict[str, float] | None, keras_path: str) -> None:
    with st.expander("Differential (7-class HAM10000)"):
        if not seven:
            st.info(
                "No 7-class head available. Export a Keras model whose softmax has 7 outputs "
                f"in order {', '.join(HAM7_LABELS)} and point **SKIN_KERAS_PATH** to it "
                f"(currently: `{keras_path}`)."
            )
            return
        for dx in HAM7_LABELS:
            p = float(seven.get(dx, 0.0))
            label = PRETTY7.get(dx, dx)
            st.progress(p / 100.0, text=f"{label}: {p:.1f}%")
