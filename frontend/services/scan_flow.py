from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from services.gradcam import enrich_result_with_vis
from services.pipeline import run_pipeline

logger = logging.getLogger("dermascan.scan")


@st.cache_resource
def load_keras_vis_cached(path: str):
    from services.gradcam import load_keras_model

    return load_keras_model(path)


def finalize_pipeline_result(pl: dict, keras_path: str) -> None:
    rgb = pl.get("rgb")
    if rgb is None:
        return
    if keras_path and Path(keras_path).is_file():
        try:
            enrich_result_with_vis(pl, rgb, load_keras_vis_cached(keras_path))
        except Exception as exc:  # noqa: BLE001
            pl["vis_error"] = str(exc)
    # No Keras on this device (the kiosk): gradient-free Eigen-CAM from the
    # two-output TFLite export, if present. Silent no-op otherwise.
    if not pl.get("gradcam_overlay_jpg"):
        from services.eigencam import enrich_with_eigencam

        enrich_with_eigencam(pl)


def run_scan_and_store(backend, image_bytes: bytes | None, *, pixels_per_mm: float, strict_quality: bool, keras_path: str, case_id: str | None = None) -> dict:
    backend_id = getattr(backend, "backend_id", "?")
    logger.info(
        "starting scan backend=%s upload=%s",
        backend_id,
        image_bytes is not None,
    )
    pl = run_pipeline(
        backend,
        image_bytes,
        pixels_per_mm=pixels_per_mm,
        strict_quality=strict_quality,
        case_id=case_id,
    )
    finalize_pipeline_result(pl, keras_path)
    pl["pixels_per_mm"] = pixels_per_mm
    return pl
