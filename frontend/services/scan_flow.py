from __future__ import annotations

import logging
import time
from pathlib import Path

import streamlit as st

from services.gradcam import enrich_result_with_vis
from services.pipeline import run_pipeline

logger = logging.getLogger("dermascan.scan")


@st.cache_resource
def load_keras_vis_cached(path: str):
    from services.gradcam import load_keras_model

    return load_keras_model(path)


def finalize_pipeline_result(
    pl: dict, keras_path: str, *, want_overlay: bool = False, on_stage=None
) -> None:
    rgb = pl.get("rgb")
    if rgb is None:
        return
    # A refused scan has no result to explain, so there is nothing for an
    # attention map to attend to. It used to run anyway — every blocked result
    # carries `rgb` — and because @st.cache_resource does not cache raises, a
    # machine with TensorFlow installed retried a full Keras load on every
    # rejected photo. The one case that has to stay fast paid the most.
    if pl.get("blocked") and not want_overlay:
        return
    t0 = time.perf_counter()
    if keras_path and Path(keras_path).is_file():
        try:
            enrich_result_with_vis(pl, rgb, load_keras_vis_cached(keras_path))
        except Exception as exc:  # noqa: BLE001
            pl["vis_error"] = str(exc)
    # No Keras on this device (the kiosk): gradient-free Eigen-CAM from the
    # two-output TFLite export, if present. Silent no-op otherwise.
    #
    # want_overlay is False on the scan path — the heatmap is built on demand
    # when staff ask for it (see views/results_view.py). The call is still made
    # so stage 3 of the content gate runs when it is available.
    if not pl.get("attention_overlay_jpg"):
        from services.eigencam import enrich_with_eigencam

        # The reading screen's last checklist step. Announced before the work
        # rather than after, so the step is lit while it is happening.
        if on_stage is not None:
            on_stage("ood")
        t_ood = time.perf_counter()
        enrich_with_eigencam(pl, want_overlay=want_overlay)
        ood_ms = int((time.perf_counter() - t_ood) * 1000)
        if ood_ms >= 1:
            pl.setdefault("stage_ms", {})["ood"] = ood_ms
    ms = int((time.perf_counter() - t0) * 1000)
    if ms >= 1:
        pl.setdefault("stage_ms", {})["attention"] = ms


def build_attention_overlay(pl: dict, keras_path: str) -> None:
    """Produce the attention heatmap now, on staff request.

    Kept out of the scan path deliberately — see ``finalize_pipeline_result``.
    """
    finalize_pipeline_result(pl, keras_path, want_overlay=True)


def run_scan_and_store(backend, image_bytes: bytes | None, *, pixels_per_mm: float, strict_quality: bool, keras_path: str, case_id: str | None = None, force: bool = False, on_stage=None, trusted_pixels_per_mm: float | None = None) -> dict:
    backend_id = getattr(backend, "backend_id", "?")
    logger.info(
        "starting scan backend=%s upload=%s force=%s",
        backend_id,
        image_bytes is not None,
        force,
    )
    pl = run_pipeline(
        backend,
        image_bytes,
        pixels_per_mm=pixels_per_mm,
        strict_quality=strict_quality,
        case_id=case_id,
        force=force,
        on_stage=on_stage,
        trusted_pixels_per_mm=trusted_pixels_per_mm,
    )
    finalize_pipeline_result(pl, keras_path, on_stage=on_stage)
    _apply_out_of_distribution(pl)
    # run_pipeline already records the scale it measured at, which differs
    # from the requested one when the frame was resolution-capped. Only fill
    # it in for paths that returned before setting it (blocked scans).
    pl.setdefault("pixels_per_mm", pixels_per_mm)
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
    distance = pl.get("ood_distance")
    threshold = pl.get("ood_threshold")
    # Logged on every scan, refused or not. Tuning SKIN_GATE_MAX_OOD for this
    # camera is guesswork without a record of where real captures actually land,
    # and the statistics behind the threshold come from dermoscopy rather than
    # from this lens.
    if distance is not None:
        logger.info(
            "feature distance %.1f (threshold %s) blocked=%s",
            distance,
            f"{threshold:.1f}" if threshold is not None else "not set",
            pl.get("out_of_distribution") is True,
        )

    if pl.get("out_of_distribution") is not True or pl.get("scan_result") is None:
        return
    from services.lesion_gate import FrameCheck
    from services.verdict import no_lesion_verdict

    logger.info("scan withdrawn: features are unlike the training data")
    pl["blocked"] = True
    pl["scan_result"] = None
    reason = "This does not look like the skin spots the scanner was taught to read."
    pl["verdict"] = no_lesion_verdict((reason,), code="ood")
    # A FrameCheck so the result screen applies the same rule it applies to
    # every other refusal — and this one is hard. There is no cautious answer
    # available by forcing it: the classifier's three labels are benign,
    # pre-cancerous and malignant, so whatever this is would be called one of
    # them.
    pl["frame_check"] = FrameCheck(
        False,
        float(pl.get("frame_check").skin_fraction) if pl.get("frame_check") else 1.0,
        0.0,
        0.0,
        (reason,),
        code="ood",
        severity="hard",
    )
