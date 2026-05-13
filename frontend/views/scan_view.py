"""Capture / upload / Pi controls and ``run_pipeline``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.gradcam import enrich_result_with_vis
from services.pipeline import run_pipeline


@st.cache_resource
def load_keras_vis_cached(path: str):
    from services.gradcam import load_keras_model

    return load_keras_model(path)


def list_sample_paths(root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    samples_dir = root / "samples"
    if samples_dir.is_dir():
        for p in sorted(samples_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((p.name, p))
    test_samples = root / "datasets" / "ham10000" / "test_samples"
    if test_samples.is_dir():
        for p in sorted(test_samples.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((f"ham10000/{p.name}", p))
    return items


def _finalize_pipeline_result(pl: dict, keras_path: str) -> None:
    rgb = pl.get("rgb")
    if rgb is None or not keras_path or not Path(keras_path).is_file():
        return
    try:
        model = load_keras_vis_cached(keras_path)
        enrich_result_with_vis(pl, rgb, model)
    except Exception as exc:  # noqa: BLE001
        pl["vis_error"] = str(exc)


def render_scan_tab(
    *,
    root: Path,
    backend,
    kind: str,
) -> None:
    st.subheader("Capture / input")
    pixels = float(st.session_state.get("pixels_per_mm_ui", 10.0))
    keras_path = str(st.session_state.get("SKIN_KERAS_PATH_UI", ""))
    strict_q = bool(st.session_state.get("strict_quality_gate", True))

    if kind == "mock":
        samples = list_sample_paths(root)
        chosen_path: Path | None = None
        if samples:
            labels_sel = [label for label, _ in samples]
            pick = st.selectbox("Sample image", labels_sel, key="mock_sample_pick")
            path_by_label = dict(samples)
            chosen_path = path_by_label[pick]
            st.caption(str(chosen_path.relative_to(root)))
        else:
            st.warning("No sample images under `samples/` or `datasets/ham10000/test_samples/`.")

        up = st.file_uploader("Or upload JPG/PNG", type=["jpg", "jpeg", "png"], key="mock_upload")

        if st.button("Run pipeline", type="primary", key="mock_run"):
            image_bytes: bytes | None = None
            if up is not None:
                image_bytes = up.getvalue()
            elif chosen_path is not None:
                image_bytes = chosen_path.read_bytes()
            if not image_bytes:
                st.error("Choose a sample or upload an image.")
            else:
                with st.spinner("Running quality checks, segmentation, TFLite…"):
                    try:
                        pl = run_pipeline(
                            backend,
                            image_bytes,
                            pixels_per_mm=pixels,
                            strict_quality=strict_q,
                        )
                        _finalize_pipeline_result(pl, keras_path)
                        st.session_state["last_result"] = pl
                        if pl.get("vis_error"):
                            st.caption(f"Grad-CAM / Keras extras: {pl['vis_error']}")
                        st.success("Done — open the **Results** tab.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

    elif kind == "local":
        shot = st.camera_input("Camera", key="local_camera")
        if st.button("Run pipeline", type="primary", key="local_run"):
            if shot is None:
                st.error("Take a photo first.")
            else:
                with st.spinner("Running pipeline…"):
                    try:
                        pl = run_pipeline(
                            backend,
                            shot.getvalue(),
                            pixels_per_mm=pixels,
                            strict_quality=strict_q,
                        )
                        _finalize_pipeline_result(pl, keras_path)
                        st.session_state["last_result"] = pl
                        if pl.get("vis_error"):
                            st.caption(f"Grad-CAM / Keras extras: {pl['vis_error']}")
                        st.success("Done — open the **Results** tab.")
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
                        pl = run_pipeline(
                            backend,
                            pi_upload.getvalue(),
                            pixels_per_mm=pixels,
                            strict_quality=strict_q,
                        )
                    else:
                        pl = run_pipeline(
                            backend,
                            None,
                            pixels_per_mm=pixels,
                            strict_quality=strict_q,
                        )
                    _finalize_pipeline_result(pl, keras_path)
                    st.session_state["last_result"] = pl
                    if pl.get("vis_error"):
                        st.caption(f"Grad-CAM / Keras extras: {pl['vis_error']}")
                    st.success("Done — open the **Results** tab.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
