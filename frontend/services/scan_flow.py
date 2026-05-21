from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.gradcam import enrich_result_with_vis
from services.pipeline import run_pipeline


@st.cache_resource
def load_keras_vis_cached(path: str):
    from services.gradcam import load_keras_model

    return load_keras_model(path)


def finalize_pipeline_result(pl: dict, keras_path: str) -> None:
    rgb = pl.get("rgb")
    if rgb is None or not keras_path or not Path(keras_path).is_file():
        return
    try:
        enrich_result_with_vis(pl, rgb, load_keras_vis_cached(keras_path))
    except Exception as exc:  # noqa: BLE001
        pl["vis_error"] = str(exc)


def run_scan_and_store(backend, image_bytes: bytes | None, *, pixels_per_mm: float, strict_quality: bool, keras_path: str, case_id: str | None = None) -> dict:
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
