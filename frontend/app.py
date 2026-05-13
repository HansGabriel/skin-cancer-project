"""Streamlit skin lesion screening demo (swappable inference backends)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.contracts import BackendKind, ScanResult
from backend.streamlit_resources import get_cached_backend


def _model_path() -> str:
    return os.environ.get("SKIN_MODEL_PATH", str(ROOT / "models" / "skin_classifier.tflite"))


def _labels_path() -> str:
    return os.environ.get("SKIN_LABELS_PATH", str(ROOT / "models" / "labels.txt"))


def _default_pi_url() -> str:
    return os.environ.get("PI_BASE_URL", "http://raspberrypi.local:5000")


def render_scan_result(result: ScanResult) -> None:
    urgency_color = {"LOW CONCERN": "green", "IMPORTANT": "orange", "URGENT": "red"}
    color = urgency_color.get(result.urgency, "blue")
    st.markdown(f"### :{color}[{result.icon} {result.urgency}]")
    st.metric(
        "Detected",
        result.label.upper().replace("_", " "),
        f"{result.confidence:.1f}% confidence",
    )
    st.write("**All probabilities**")
    for lab, prob in result.probs.items():
        st.progress(float(prob) / 100.0, text=f"{lab}: {float(prob):.1f}%")
    if result.image_jpg_bytes:
        st.image(
            result.image_jpg_bytes,
            caption=f"{result.timestamp} — {result.inference_ms} ms — backend: {result.backend_id}",
            use_container_width=True,
        )
    st.info(f"Recommendation: {result.action}")
    st.warning("For screening only — not a medical diagnosis.")


def list_sample_paths() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    samples_dir = ROOT / "samples"
    if samples_dir.is_dir():
        for p in sorted(samples_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((p.name, p))
    test_samples = ROOT / "datasets" / "ham10000" / "test_samples"
    if test_samples.is_dir():
        for p in sorted(test_samples.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((f"ham10000/{p.name}", p))
    return items


def main() -> None:
    st.set_page_config(page_title="DermaScan AI", page_icon="🧬", layout="wide")
    st.title("🧬 DermaScan AI — Skin lesion risk screening")
    st.caption("AI-assisted dermatoscope prototype. For screening only.")

    with st.sidebar:
        st.subheader("Inference backend")
        kind = cast(
            BackendKind,
            st.radio(
                "Source",
                ("mock", "local", "pi"),
                format_func=lambda k: {
                    "mock": "Upload / sample images (no camera)",
                    "local": "Browser camera (this machine)",
                    "pi": "Raspberry Pi (HTTP)",
                }[k],
                key="inference_backend_kind",
            ),
        )
        pi_base = st.text_input("Pi base URL", value=_default_pi_url(), key="pi_base_url_input")
        st.caption("Env: `SKIN_MODEL_PATH`, `SKIN_LABELS_PATH`, `PI_BASE_URL`.")

        backend = get_cached_backend(
            kind,
            _model_path(),
            _labels_path(),
            pi_base.rstrip("/"),
        )

        if st.button("Check backend health", key="health_btn"):
            status = backend.health()
            if status.get("status") == "ok":
                st.success(status)
            else:
                st.error(status)

    left, right = st.columns(2)

    with left:
        st.subheader("Capture / input")

        if kind == "mock":
            samples = list_sample_paths()
            chosen_path: Path | None = None
            if samples:
                labels_sel = [label for label, _ in samples]
                pick = st.selectbox("Sample image", labels_sel, key="mock_sample_pick")
                path_by_label = dict(samples)
                chosen_path = path_by_label[pick]
                st.caption(str(chosen_path.relative_to(ROOT)))
            else:
                st.warning("No sample images under `samples/` or `datasets/ham10000/test_samples/`.")

            up = st.file_uploader("Or upload JPG/PNG", type=["jpg", "jpeg", "png"], key="mock_upload")

            if st.button("Run classification", type="primary", key="mock_run"):
                image_bytes: bytes | None = None
                if up is not None:
                    image_bytes = up.getvalue()
                elif chosen_path is not None:
                    image_bytes = chosen_path.read_bytes()
                if not image_bytes:
                    st.error("Choose a sample or upload an image.")
                else:
                    with st.spinner("Running TFLite…"):
                        try:
                            st.session_state["result"] = backend.scan(image_bytes)
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))

        elif kind == "local":
            shot = st.camera_input("Camera", key="local_camera")
            if st.button("Classify snapshot", type="primary", key="local_run"):
                if shot is None:
                    st.error("Take a photo first.")
                else:
                    with st.spinner("Running TFLite…"):
                        try:
                            st.session_state["result"] = backend.scan(shot.getvalue())
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))

        else:
            pi_upload = st.file_uploader(
                "Optional: upload JPG/PNG to run on Pi (skips camera)",
                type=["jpg", "jpeg", "png"],
                key="pi_upload",
            )
            st.caption("Leave empty to capture from the Pi Camera Module.")
            if st.button("Run scan on Pi", type="primary", key="pi_run"):
                with st.spinner("Waiting for Pi…"):
                    try:
                        if pi_upload is not None:
                            st.session_state["result"] = backend.scan(pi_upload.getvalue())
                        else:
                            st.session_state["result"] = backend.scan()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

    with right:
        st.subheader("Latest result")
        if "result" in st.session_state:
            render_scan_result(st.session_state["result"])
        else:
            st.info("Run a classification to see results here.")

    st.divider()
    st.subheader("Scan history")
    if st.button("Refresh history", key="log_refresh"):
        rows, warn = backend.fetch_log()
        if warn:
            st.warning(warn)
        if rows:
            try:
                st.dataframe(pd.json_normalize(rows), use_container_width=True)
            except Exception:  # noqa: BLE001
                st.dataframe(rows, use_container_width=True)
        elif not warn:
            st.info("No rows logged yet.")


if __name__ == "__main__":
    main()
