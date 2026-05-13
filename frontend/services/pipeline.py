"""Orchestrate quality → segmentation → TFLite → ABCDE → composite risk."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import numpy as np

from backend.contracts import ScanResult
from backend.tflite_shared import decode_image_bytes_to_rgb
from services.abcde import LetterResult, compute_abcde
from services.quality import check_quality
from services.risk import composite_risk_score, risk_band
from services.segmentation import segment_safe


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
    error: str
    quality_warnings: NotRequired[list[str]]
    vis_error: NotRequired[str]


def run_pipeline(
    backend,
    image_bytes: bytes | None,
    *,
    pixels_per_mm: float,
    strict_quality: bool = True,
) -> PipelineResult:
    """Run full analysis. ``image_bytes`` may be ``None`` only for Pi camera capture.

    If ``strict_quality`` is False, failed quality checks still produce warnings but the
    pipeline continues (segmentation, TFLite, ABCDE).
    """
    out: PipelineResult = {}

    if image_bytes is not None:
        try:
            rgb = decode_image_bytes_to_rgb(image_bytes)
        except ValueError as exc:
            return {"blocked": True, "error": str(exc), "scan_result": None}

        q = check_quality(rgb)
        out["quality"] = q
        if not q["ok"] and strict_quality:
            return {
                "blocked": True,
                "quality": q,
                "rgb": rgb,
                "scan_result": None,
            }
        if not q["ok"] and not strict_quality:
            out["quality_warnings"] = list(q["reasons"])

        mask = segment_safe(rgb)
        out["rgb"] = rgb
        out["mask"] = mask

        try:
            scan_result = backend.scan(image_bytes)
        except Exception as exc:  # noqa: BLE001
            return {
                "blocked": False,
                "error": str(exc),
                "rgb": rgb,
                "mask": mask,
                "quality": q,
            }

        abcde: dict[str, LetterResult] | None
        if mask is not None:
            abcde = compute_abcde(rgb, mask, pixels_per_mm=pixels_per_mm)
        else:
            abcde = None

        p_mal = float(scan_result.probs.get("malignant", 0.0))
        comp = composite_risk_score(p_mal, abcde)
        out.update(
            {
                "blocked": False,
                "scan_result": scan_result,
                "abcde": abcde,
                "composite": comp,
                "risk_band": risk_band(comp),
                "seven_class_probs": None,
                "gradcam_overlay_jpg": None,
                "gradcam_disclaimer": None,
            }
        )
        return out

    # Pi: capture on device first (no bytes from PC)
    try:
        scan_result = backend.scan(None)
    except Exception as exc:  # noqa: BLE001
        return {"blocked": False, "error": str(exc), "scan_result": None}

    if not scan_result.image_jpg_bytes:
        return {
            "blocked": False,
            "error": "Pi returned no image bytes.",
            "scan_result": scan_result,
        }

    try:
        rgb = decode_image_bytes_to_rgb(scan_result.image_jpg_bytes)
    except ValueError as exc:
        return {"blocked": False, "error": str(exc), "scan_result": scan_result}

    q = check_quality(rgb)
    out["quality"] = q
    mask = segment_safe(rgb)
    out["rgb"] = rgb
    out["mask"] = mask

    abcde: dict[str, LetterResult] | None
    if mask is not None:
        abcde = compute_abcde(rgb, mask, pixels_per_mm=pixels_per_mm)
    else:
        abcde = None

    p_mal = float(scan_result.probs.get("malignant", 0.0))
    comp = composite_risk_score(p_mal, abcde)
    out.update(
        {
            "blocked": False,
            "scan_result": scan_result,
            "abcde": abcde,
            "composite": comp,
            "risk_band": risk_band(comp),
            "seven_class_probs": None,
            "gradcam_overlay_jpg": None,
            "gradcam_disclaimer": None,
        }
    )
    if not q["ok"]:
        out["quality_warnings"] = list(q["reasons"])
    return out
