"""Streamlit frontend for Pi-based skin lesion screening."""

from __future__ import annotations

import base64
import os

import requests
import streamlit as st


PI_SCAN_URL = os.getenv("PI_SCAN_URL", "http://raspberrypi.local:5000/scan")


def render_result(result: dict) -> None:
    urgency_color = {"LOW CONCERN": "green", "IMPORTANT": "orange", "URGENT": "red"}
    color = urgency_color.get(result.get("urgency", ""), "blue")

    st.markdown(f"### :{color}[{result['icon']} {result['urgency']}]")
    st.metric("Detected", result["label"].upper(), f"{result['confidence']:.1f}% confidence")

    st.write("**All probabilities**")
    for label, prob in result["probs"].items():
        st.progress(float(prob) / 100.0, text=f"{label}: {float(prob):.1f}%")

    if result.get("image"):
        st.image(base64.b64decode(result["image"]), caption="Captured image", use_container_width=True)

    st.info(f"Recommendation: {result['action']}")
    st.warning("For screening only - not a medical diagnosis.")


def main() -> None:
    st.set_page_config(page_title="DermaScan AI", page_icon="🧬", layout="wide")
    st.title("🧬 DermaScan AI - Skin Lesion Risk Screening")
    st.caption("AI-assisted dermatoscope prototype. For screening only.")

    with st.sidebar:
        st.subheader("Connection")
        st.code(PI_SCAN_URL, language="text")
        st.caption("Set `PI_SCAN_URL` environment variable to change endpoint.")

    left, right = st.columns(2)
    with left:
        st.subheader("Capture")
        run_scan = st.button("Capture from Pi Camera", type="primary", use_container_width=True)
        if run_scan:
            with st.spinner("Capturing and analyzing..."):
                try:
                    response = requests.post(PI_SCAN_URL, timeout=30)
                    response.raise_for_status()
                    st.session_state["result"] = response.json()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Scan failed: {exc}")

    with right:
        st.subheader("Results")
        if "result" in st.session_state:
            render_result(st.session_state["result"])
        else:
            st.info("No scan yet. Click 'Capture from Pi Camera'.")


if __name__ == "__main__":
    main()
