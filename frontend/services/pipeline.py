"""Orchestrate preprocess → quality → segmentation → TFLite → ABCDE → composite risk."""

from __future__ import annotations

import os
import time
from typing import Any, NotRequired, TypedDict

import numpy as np

from backend.contracts import ScanResult
import cv2

from backend.tflite_shared import decode_image_bytes_to_rgb
from services.abcde import LetterResult, compute_abcde
from services.evolving import apply_to_abcde
from services.preprocess import enhance_lesion_image
from services.quality import check_quality
from services.risk import composite_risk_score, risk_band
from services.segmentation import segment_safe

APP_VERSION = "0.4.0"


def _rgb_to_jpeg_bytes(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


class PipelineResult(TypedDict, total=False):
    blocked: bool
    quality: dict[str, Any]
    rgb: np.ndarray
    mask: np.ndarray | None
    scan_result: ScanResult | None
    abcde: dict[str, LetterResult] | None
    composite: float
    risk_band: str
    seven_class_probs: dict[str, float] | None
    gradcam_overlay_jpg: bytes | None
    gradcam_disclaimer: str | None
    rgb_before: NotRequired[np.ndarray]
    error: str
    quality_warnings: NotRequired[list[str]]
    vis_error: NotRequired[str]
    trust_line: NotRequired[str]
    inference_ms: NotRequired[int]
    model_path: NotRequired[str]
    tta_enabled: NotRequired[bool]


def _preprocess_enabled() -> bool:
    try:
        import streamlit as st

        if st.session_state.get("preprocess_enabled") is False:
            return False
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("SKIN_PREPROCESS", "1") == "1"


def _preprocess_debug() -> bool:
    try:
        import streamlit as st

        return bool(st.session_state.get("preprocess_debug"))
    except Exception:  # noqa: BLE001
        return False


def _trust_line(sr: ScanResult | None, *, model_path: str, tta: bool, quality_ok: bool) -> str:
    ms = sr.inference_ms if sr else 0
    model_name = os.path.basename(model_path)
    q = "OK" if quality_ok else "WARN"
    tta_s = "on" if tta else "off"
    return f"Inference: {ms/1000:.1f}s · Model: {model_name} · v{APP_VERSION} · Quality: {q} · TTA: {tta_s}"


def run_pipeline(
    backend,
    image_bytes: bytes | None,
    *,
    pixels_per_mm: float,
    strict_quality: bool = True,
    case_id: str | None = None,
    preprocess: bool = True,
) -> PipelineResult:
    tta = os.environ.get("SKIN_TTA", "1") == "1"
    model_path = os.environ.get("SKIN_MODEL_PATH", "skin_classifier.tflite")
    out: PipelineResult = {}

    def _finish(rgb, mask, scan_result, abcde, q, *, rgb_before: np.ndarray | None = None) -> PipelineResult:
        abcde = apply_to_abcde(case_id, abcde, rgb=rgb, mask=mask)
        p_mal = float(scan_result.probs.get("malignant", 0.0))
        comp = composite_risk_score(p_mal, abcde)
        result: PipelineResult = {
            "blocked": False,
            "quality": q,
            "rgb": rgb,
            "mask": mask,
            "scan_result": scan_result,
            "abcde": abcde,
            "composite": comp,
            "risk_band": risk_band(comp),
            "seven_class_probs": None,
            "gradcam_overlay_jpg": None,
            "gradcam_disclaimer": None,
            "inference_ms": scan_result.inference_ms,
            "model_path": model_path,
            "tta_enabled": tta,
            "trust_line": _trust_line(scan_result, model_path=model_path, tta=tta, quality_ok=q.get("ok", True)),
        }
        if not q.get("ok"):
            result["quality_warnings"] = list(q.get("reasons", []))
        if rgb_before is not None:
            result["rgb_before"] = rgb_before
        return result

    if image_bytes is not None:
        try:
            rgb = decode_image_bytes_to_rgb(image_bytes)
        except ValueError as exc:
            return {"blocked": True, "error": str(exc), "scan_result": None}
        rgb_before: np.ndarray | None = None
        if _preprocess_enabled():
            if _preprocess_debug():
                rgb_before = rgb.copy()
            rgb = enhance_lesion_image(rgb)
        q = check_quality(rgb)
        out["quality"] = q
        if not q["ok"] and strict_quality:
            return {"blocked": True, "quality": q, "rgb": rgb, "scan_result": None}
        if not q["ok"] and not strict_quality:
            out["quality_warnings"] = list(q["reasons"])
        mask = segment_safe(rgb)
        scan_bytes = _rgb_to_jpeg_bytes(rgb) if _preprocess_enabled() else image_bytes
        try:
            scan_result = backend.scan(scan_bytes)
        except Exception as exc:  # noqa: BLE001
            return {"blocked": False, "error": str(exc), "rgb": rgb, "mask": mask, "quality": q}
        abcde = compute_abcde(rgb, mask, pixels_per_mm=pixels_per_mm) if mask is not None else None
        return _finish(rgb, mask, scan_result, abcde, q, rgb_before=rgb_before)

    try:
        scan_result = backend.scan(None)
    except Exception as exc:  # noqa: BLE001
        return {"blocked": False, "error": str(exc), "scan_result": None}
    if not scan_result.image_jpg_bytes:
        return {"blocked": False, "error": "Pi returned no image bytes.", "scan_result": scan_result}
    try:
        rgb = decode_image_bytes_to_rgb(scan_result.image_jpg_bytes)
    except ValueError as exc:
        return {"blocked": False, "error": str(exc), "scan_result": scan_result}
    rgb_before_pi: np.ndarray | None = None
    if _preprocess_enabled():
        if _preprocess_debug():
            rgb_before_pi = rgb.copy()
        rgb = enhance_lesion_image(rgb)
    q = check_quality(rgb)
    mask = segment_safe(rgb)
    abcde = compute_abcde(rgb, mask, pixels_per_mm=pixels_per_mm) if mask is not None else None
    result = _finish(rgb, mask, scan_result, abcde, q, rgb_before=rgb_before_pi)
    if not q["ok"]:
        result["quality_warnings"] = list(q["reasons"])
    return result
