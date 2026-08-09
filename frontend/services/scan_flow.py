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
    if not pl.get("attention_overlay_jpg"):
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
    _apply_out_of_distribution(pl)
    pl["pixels_per_mm"] = pixels_per_mm
    return pl


def _apply_out_of_distribution(pl: dict) -> None:
    """Withdraw the result when the model's own features say it saw something
    unlike anything it was trained on.

    The colour and shape stages in ``services.lesion_gate`` stop a wall, but not
    something merely skin-coloured. This is the last stage, and it can only run
    after inference because it reads the model's features. ``None`` means the
    check did not run (no models/feature_stats.json) — treated as "not checked",
    never as a pass.
    """
    if pl.get("out_of_distribution") is not True or pl.get("scan_result") is None:
        return
    from services.verdict import no_lesion_verdict

    logger.info("scan withdrawn: features are unlike the training data")
    pl["blocked"] = True
    pl["scan_result"] = None
    pl["verdict"] = no_lesion_verdict(
        ("This does not look like the skin spots the scanner was taught to read.",)
    )
