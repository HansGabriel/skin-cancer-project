"""Orchestrate quality → TFLite (original image) → segmentation/ABCDE (optional enhance)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, TypedDict

# NotRequired moved into typing only in 3.11. Streamlit Community Cloud picks
# its own interpreter unless runtime.txt pins one, so importing it from typing
# made the whole app fail to start there with a redacted ImportError that
# pointed at an unrelated module. typing_extensions ships with Streamlit.
try:  # pragma: no cover - trivial version shim
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired

import cv2
import numpy as np

from backend.contracts import ScanResult
from backend.tflite_shared import decode_image_bytes_to_rgb
from services.abcde import LetterResult, compute_abcde
from services.evolving import apply_to_abcde
from services.lesion_gate import FrameCheck, check_skin, check_spot, quick_reject
from services.preprocess import enhance_lesion_image
from services.quality import check_quality
from services.risk import composite_risk_score, risk_band
from services.segmentation import segment_safe
from services.verdict import UIVerdict, no_lesion_verdict, resolve_verdict, retake_verdict

APP_VERSION = "0.4.0"

# Longest edge the pipeline will work at. Mirrors the Pi camera's fixed 1024
# centre crop (services/pi_camera.py) so an upload and a device capture cost the
# same and are measured the same way. See the resize in run_pipeline.
MAX_WORK_PX = int(os.environ.get("SKIN_MAX_WORK_PX", "1024"))

logger = logging.getLogger("dermascan.pipeline")


class Stages(dict):
    """Stage timings, plus an optional "I am starting X now" callback.

    A plain dict works everywhere this is used; the callback exists so the
    reading screen can tick its checklist as the work happens instead of
    animating a guess. It is carried on the dict rather than threaded through
    ``_stage``'s signature so the twenty-odd call sites stay untouched, and it
    is per-call state rather than a module global — Streamlit runs each session
    on its own thread, and a global would let two visitors' scans drive each
    other's progress.
    """

    def __init__(self, on_enter=None) -> None:
        super().__init__()
        self.on_enter = on_enter


@contextmanager
def _stage(stages: dict[str, int], name: str) -> Iterator[None]:
    """Record how long one pipeline stage took, in ms.

    The only timer this pipeline used to have lived inside the TTA loop
    (``backend.tflite_shared``), so the figure shown to staff described the
    model and nothing else: a 30-second scan reported "Inference: 0.8s". That
    number sent people looking at the classifier when ~85% of the time was in
    segmentation. Measure the whole thing or the measurement misleads.
    """
    on_enter = getattr(stages, "on_enter", None)
    if on_enter is not None:
        on_enter(name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stages[name] = int((time.perf_counter() - t0) * 1000)


def _log_stages(stages: dict[str, int]) -> None:
    """One grep-able line per scan: `grep 'scan stages' /tmp/dermascan_kiosk.log`."""
    if not stages:
        return
    total = sum(stages.values())
    detail = " ".join(f"{k}={v}ms" for k, v in stages.items())
    logger.info("scan stages %s total=%dms", detail, total)


def _log_pi_error(backend, message: str) -> None:
    if getattr(backend, "backend_id", None) != "pi":
        return
    logger.error("Pi scan failed: %s", message)


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
    attention_overlay_jpg: bytes | None
    attention_note: str | None
    rgb_before: NotRequired[np.ndarray]
    rgb_analysis: NotRequired[np.ndarray]
    error: str
    frame_check: NotRequired[FrameCheck]
    vis_error: NotRequired[str]
    trust_line: NotRequired[str]
    inference_ms: NotRequired[int]
    model_path: NotRequired[str]
    tta_enabled: NotRequired[bool]
    verdict: NotRequired[UIVerdict]
    stage_ms: NotRequired[dict[str, int]]
    forced: NotRequired[bool]


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


def trust_line(pl: dict) -> str:
    """The staff timing line, built at RENDER time.

    It has to be built here rather than inside ``run_pipeline`` because the
    attention overlay is produced after the pipeline returns — so a line built
    in the pipeline structurally cannot report it. Leading with the scan total
    and its breakdown is the point: the old line reported only the model's own
    milliseconds, which read as "0.8s" while the user watched a 30-second
    spinner.
    """
    base = pl.get("trust_line", "")
    stages = pl.get("stage_ms") or {}
    if not stages:
        return base
    total = sum(stages.values())
    # Only the stages worth a staff member's attention; the sub-10ms ones are
    # noise on a line that has to fit a 1024px panel.
    parts = " · ".join(f"{k} {v/1000:.1f}s" for k, v in stages.items() if v >= 100)
    head = f"Scan: {total/1000:.1f}s"
    if parts:
        head += f" ({parts})"
    return f"{head} · {base}"


def _gate(
    rgb_display: np.ndarray,
    q: dict,
    *,
    strict_quality: bool,
    force: bool = False,
    stages: dict[str, int] | None = None,
) -> tuple[PipelineResult | None, Any, np.ndarray]:
    """Run every stop-check in order.

    Returns ``(blocking_result_or_None, mask, rgb_for_abcde)``.

    Both backends call this so they cannot drift apart: the upload path once
    honoured ``strict_quality`` while the Pi path ignored it, which meant the
    same photo could pass on one device and be rejected on the other.

    Order is deliberate. "There is no skin here" is answered first because a
    photo of a wall also fails the focus check, and "hold the camera still" is
    useless advice for it.

    ``force`` is the user saying "I have looked at this photo and I want it
    read anyway" after a refusal. Every check still RUNS — the mask and the
    frame check are still computed and returned, so the result screen can carry
    the warning — but none of them stops the scan. Without this the gate is a
    dead end, and a health worker holding a lesion the scanner will not look at
    has no way forward. The caller is responsible for making the caveat visible.

    The ABCDE enhancement is built HERE, after the two cheap stop-checks, and
    handed back — it costs ~90 ms (≈1 s on a Pi 4) and used to be paid before
    any check ran, so a photo of a wall was colour-corrected and de-haired
    purely to be thrown away.
    """
    st_ms = stages if stages is not None else {}
    with _stage(st_ms, "skin"):
        no_skin, skin = check_skin(rgb_display)
    if no_skin is not None and not force:
        return {
            "blocked": True,
            "quality": q,
            "frame_check": no_skin,
            "rgb": rgb_display,
            "scan_result": None,
            "verdict": no_lesion_verdict(no_skin.reasons),
        }, None, rgb_display

    if (not q["ok"] or (q["reasons"] and strict_quality)) and not force:
        return {
            "blocked": True,
            "quality": q,
            "rgb": rgb_display,
            "scan_result": None,
            "verdict": retake_verdict(q),
        }, None, rgb_display

    # Stop-check C, cheap half. A frame with no lesion in it is the one that
    # used to pay the most: it makes the cheap segmentation candidates score out
    # of band, which is the single condition that forces GrabCut to run — at
    # native resolution, after a full colour-constancy and hair-removal pass.
    # Measured at 12 MP that was 107 s of work to reach a refusal. This answers
    # the obvious cases from a 384px copy in a few milliseconds, and returns
    # None whenever it is not sure (see lesion_gate.quick_reject).
    with _stage(st_ms, "prespot"):
        early = quick_reject(rgb_display, skin=skin)
    if early is not None and not force:
        return {
            "blocked": True,
            "quality": q,
            "frame_check": early,
            "rgb": rgb_display,
            "scan_result": None,
            "verdict": no_lesion_verdict(early.reasons),
        }, None, rgb_display

    with _stage(st_ms, "enhance"):
        rgb_for_abcde = _analysis_rgb(rgb_display)
    with _stage(st_ms, "segment"):
        mask = segment_safe(rgb_for_abcde)
    # A 3-class softmax always sums to 1, so without this a photo of a bare
    # forearm comes back as a confident "benign".
    with _stage(st_ms, "spot"):
        frame = check_spot(rgb_display, mask, skin=skin)
    if not frame.is_lesion_photo and not force:
        return {
            "blocked": True,
            "quality": q,
            "frame_check": frame,
            "rgb": rgb_display,
            "scan_result": None,
            "verdict": no_lesion_verdict(frame.reasons),
        }, mask, rgb_for_abcde
    return None, mask, rgb_for_abcde


def run_pipeline(
    backend,
    image_bytes: bytes | None,
    *,
    pixels_per_mm: float,
    strict_quality: bool = True,
    case_id: str | None = None,
    force: bool = False,
    preprocess: bool = True,  # noqa: ARG001 — kept for API compat; session/env controls ABCDE enhance
    on_stage=None,
) -> PipelineResult:
    tta = os.environ.get("SKIN_TTA", "1") == "1"
    model_path = os.environ.get("SKIN_MODEL_PATH", "skin_classifier.tflite")
    stages: dict[str, int] = Stages(on_enter=on_stage)
    # ``strict_quality`` now only decides whether *advisory* quality notes (a
    # slightly soft or dim photo) also block. It is no longer forced on for the
    # kiosk: services.lesion_gate is what keeps junk input from reaching the
    # classifier, and it runs unconditionally. Blocking every imperfect photo
    # was rejecting real lesions shot in ordinary indoor light.

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
        result: PipelineResult = {
            "blocked": False,
            "quality": q,
            "verdict": resolve_verdict(scan_result, q),
            "rgb": rgb_display,
            "rgb_analysis": rgb_for_abcde,
            "mask": mask,
            "scan_result": scan_result,
            "abcde": abcde,
            "composite": comp,
            "risk_band": risk_band(comp),
            "seven_class_probs": None,
            "attention_overlay_jpg": None,
            "attention_note": None,
            "inference_ms": scan_result.inference_ms,
            "model_path": model_path,
            "tta_enabled": tta,
            "trust_line": _trust_line(scan_result, model_path=model_path, tta=tta, quality_ok=q.get("ok", True)),
        }
        if rgb_before is not None:
            result["rgb_before"] = rgb_before
        # The scale the ABCDE numbers were actually measured at, which is not
        # the caller's value once the frame has been capped (see the resize in
        # run_pipeline). Storage persists this, so a saved scan records how it
        # was measured rather than what was requested.
        result["pixels_per_mm"] = pixels_per_mm
        result["stage_ms"] = stages
        _log_stages(stages)
        return result

    if image_bytes is not None:
        try:
            with _stage(stages, "decode"):
                rgb_display = decode_image_bytes_to_rgb(image_bytes)
        except ValueError as exc:
            return {"blocked": True, "error": str(exc), "scan_result": None, "verdict": retake_verdict(None)}

        # Cap the working resolution. Every stage after this scales with pixel
        # count — enhance measured 3.6 s and segmentation 13.6 s on a 12 MP
        # frame — while the Pi's own captures are a fixed 1024 centre crop
        # (services/pi_camera.py), so the device was always cheap and only
        # uploads were not. Capping HERE rather than only in the upload widget
        # means no caller can route around it.
        #
        # The classifier sees the capped frame too: it is re-encoded below so
        # the pixels the model reads are the pixels that were measured. It then
        # resizes to 224 regardless, and INTER_AREA matches the reduction in
        # backend/preprocessing.py, so this is one extra area-average step and
        # not a different image.
        if max(rgb_display.shape[:2]) > MAX_WORK_PX:
            with _stage(stages, "resize"):
                h, w = rgb_display.shape[:2]
                scale = MAX_WORK_PX / float(max(h, w))
                rgb_display = cv2.resize(
                    rgb_display,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                image_bytes = _rgb_to_jpeg_bytes(rgb_display)
                # The scale MUST travel with the image. `pixels_per_mm` is a
                # property of the capture, so shrinking the frame without
                # shrinking it makes every millimetre measurement wrong by the
                # resize factor — measured, the same lesion read 80.28 mm at
                # full size and 25.58 mm capped, which moves the D tier, the
                # composite risk score, and the "About N mm across" line a
                # visitor reads.
                pixels_per_mm = pixels_per_mm * scale

        rgb_before = rgb_display.copy() if _preprocess_debug() and _preprocess_for_abcde() else None

        with _stage(stages, "quality"):
            q = check_quality(rgb_display)
        blocking, mask, rgb_for_abcde = _gate(
            rgb_display, q, strict_quality=strict_quality, force=force, stages=stages
        )
        if blocking is not None:
            blocking["stage_ms"] = stages
            _log_stages(stages)
            return blocking

        # Classifier sees the capture as-is (matches HAM10000 / training
        # preprocessing) — resolution-capped above, never enhanced. The ABCDE
        # path gets the colour-corrected, de-haired copy; the model does not.
        model_jpg = image_bytes
        try:
            with _stage(stages, "model"):
                scan_result = backend.scan(model_jpg)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            _log_pi_error(backend, msg)
            return {"blocked": False, "error": msg, "rgb": rgb_display, "mask": mask, "quality": q}

        with _stage(stages, "abcde"):
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

    # NOTE: on this path the stop-checks below run *after* inference, unlike the
    # upload path where they run before it. That is not an oversight that can be
    # fixed here: `POST /scan` on the Pi captures and classifies in one call
    # (scripts/pi_server.py), so there is no way to get the frame without also
    # paying for the model. Splitting it needs a capture-only endpoint on the
    # device. Timed so the cost is at least visible in the staff readout, which
    # it was not before — neither this call nor the decode below was wrapped, so
    # the Pi path's dominant stage was missing from every `scan stages` line.
    try:
        with _stage(stages, "capture+model"):
            scan_result = backend.scan(None)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        _log_pi_error(backend, msg)
        return {"blocked": False, "error": msg, "scan_result": None}
    if not scan_result.image_jpg_bytes:
        msg = "Pi returned no image bytes."
        _log_pi_error(backend, msg)
        return {"blocked": False, "error": msg, "scan_result": scan_result}
    try:
        with _stage(stages, "decode"):
            rgb_display = decode_image_bytes_to_rgb(scan_result.image_jpg_bytes)
    except ValueError as exc:
        msg = str(exc)
        _log_pi_error(backend, msg)
        return {"blocked": False, "error": msg, "scan_result": scan_result}

    rgb_before = rgb_display.copy() if _preprocess_debug() and _preprocess_for_abcde() else None
    with _stage(stages, "quality"):
        q = check_quality(rgb_display)
    blocking, mask, rgb_for_abcde = _gate(
        rgb_display, q, strict_quality=strict_quality, force=force, stages=stages
    )
    if blocking is not None:
        blocking["stage_ms"] = stages
        _log_stages(stages)
        return blocking
    with _stage(stages, "abcde"):
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
    return result
