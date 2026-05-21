"""Orchestrate quality → TFLite (original image) → segmentation/ABCDE (optional enhance)."""

from __future__ import annotations

import os
from typing import Any, NotRequired, TypedDict

import cv2
import numpy as np

from backend.contracts import ScanResult
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
    rgb_analysis: NotRequired[np.ndarray]
    error: str
    quality_warnings: NotRequired[list[str]]
    vis_error: NotRequired[str]
    trust_line: NotRequired[str]
    inference_ms: NotRequired[int]
    model_path: NotRequired[str]
    tta_enabled: NotRequired[bool]
    borderline_note: NotRequired[str]


def _preprocess_for_abcde() -> bool:
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
        pass
    return False


def _analysis_rgb(rgb: np.ndarray) -> np.ndarray:
    """Enhancement for segmentation/ABCDE only — never fed to the TFLite classifier."""
    if not _preprocess_for_abcde():
        return rgb
    return enhance_lesion_image(rgb)


def _borderline_note(probs: dict[str, float], label: str) -> str | None:
    """Flag near-tie predictions the UI should not over-trust."""
    malignant = float(probs.get("malignant", 0.0))
    benign = float(probs.get("benign", 0.0))
    if label == "benign" and malignant >= 25.0:
        return (
            f"Borderline: CNN says {label} but malignant probability is still {malignant:.1f}%. "
            "Treat as screening only — clinical ABCDE flags may warrant follow-up."
        )
    if label == "malignant" and benign >= 25.0:
        return (
            f"Borderline: CNN says {label} but benign probability is still {benign:.1f}%. "
            "Treat as screening only."
        )
    ordered = sorted(probs.values(), reverse=True)
    if len(ordered) >= 2 and (ordered[0] - ordered[1]) < 15.0:
        return "Borderline: top two class probabilities are very close — interpret with caution."
    return None


def _trust_line(sr: ScanResult | None, *, model_path: str, tta: bool, quality_ok: bool) -> str:
    ms = sr.inference_ms if sr else 0
    model_name = os.path.basename(model_path)
    q = "OK" if quality_ok else "WARN"
    tta_s = "on" if tta else "off"
    pp = "on" if _preprocess_for_abcde() else "off"
    return (
        f"Inference: {ms/1000:.1f}s · Model: {model_name} · v{APP_VERSION} · "
        f"Quality: {q} · TTA: {tta_s} · ABCDE enhance: {pp}"
    )


def run_pipeline(
    backend,
    image_bytes: bytes | None,
    *,
    pixels_per_mm: float,
    strict_quality: bool = True,
    case_id: str | None = None,
    preprocess: bool = True,  # noqa: ARG001 — kept for API compat; session/env controls ABCDE enhance
) -> PipelineResult:
    tta = os.environ.get("SKIN_TTA", "1") == "1"
    model_path = os.environ.get("SKIN_MODEL_PATH", "skin_classifier.tflite")

    def _finish(
        rgb_display: np.ndarray,
        rgb_for_abcde: np.ndarray,
        mask,
        scan_result: ScanResult,
        abcde,
        q,
        *,
        rgb_before: np.ndarray | None = None,
        model_jpg: bytes,
    ) -> PipelineResult:
        abcde = apply_to_abcde(case_id, abcde, rgb=rgb_for_abcde, mask=mask)
        p_mal = float(scan_result.probs.get("malignant", 0.0))
        comp = composite_risk_score(p_mal, abcde)
        note = _borderline_note(scan_result.probs, scan_result.label)
        result: PipelineResult = {
            "blocked": False,
            "quality": q,
            "rgb": rgb_display,
            "rgb_analysis": rgb_for_abcde,
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
        if note:
            result["borderline_note"] = note
        if not q.get("ok"):
            result["quality_warnings"] = list(q.get("reasons", []))
        if rgb_before is not None:
            result["rgb_before"] = rgb_before
        return result

    if image_bytes is not None:
        try:
            rgb_display = decode_image_bytes_to_rgb(image_bytes)
        except ValueError as exc:
            return {"blocked": True, "error": str(exc), "scan_result": None}

        rgb_for_abcde = _analysis_rgb(rgb_display)
        rgb_before = rgb_display.copy() if _preprocess_debug() and _preprocess_for_abcde() else None

        q = check_quality(rgb_display)
        if not q["ok"] and strict_quality:
            return {"blocked": True, "quality": q, "rgb": rgb_display, "scan_result": None}
        out: PipelineResult = {"quality": q}
        if not q["ok"]:
            out["quality_warnings"] = list(q["reasons"])

        mask = segment_safe(rgb_for_abcde)
        # Classifier always sees the original capture (matches HAM10000 / training preprocessing).
        model_jpg = image_bytes
        try:
            scan_result = backend.scan(model_jpg)
        except Exception as exc:  # noqa: BLE001
            return {"blocked": False, "error": str(exc), "rgb": rgb_display, "mask": mask, "quality": q}

        abcde = compute_abcde(rgb_for_abcde, mask, pixels_per_mm=pixels_per_mm) if mask is not None else None
        return _finish(
            rgb_display,
            rgb_for_abcde,
            mask,
            scan_result,
            abcde,
            q,
            rgb_before=rgb_before,
            model_jpg=model_jpg,
        )

    try:
        scan_result = backend.scan(None)
    except Exception as exc:  # noqa: BLE001
        return {"blocked": False, "error": str(exc), "scan_result": None}
    if not scan_result.image_jpg_bytes:
        return {"blocked": False, "error": "Pi returned no image bytes.", "scan_result": scan_result}
    try:
        rgb_display = decode_image_bytes_to_rgb(scan_result.image_jpg_bytes)
    except ValueError as exc:
        return {"blocked": False, "error": str(exc), "scan_result": scan_result}

    rgb_for_abcde = _analysis_rgb(rgb_display)
    rgb_before = rgb_display.copy() if _preprocess_debug() and _preprocess_for_abcde() else None
    q = check_quality(rgb_display)
    mask = segment_safe(rgb_for_abcde)
    abcde = compute_abcde(rgb_for_abcde, mask, pixels_per_mm=pixels_per_mm) if mask is not None else None
    result = _finish(
        rgb_display,
        rgb_for_abcde,
        mask,
        scan_result,
        abcde,
        q,
        rgb_before=rgb_before,
        model_jpg=scan_result.image_jpg_bytes,
    )
    if not q.get("ok"):
        result["quality_warnings"] = list(q["reasons"])
    return result
