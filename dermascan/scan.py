"""run_scan: the one function the kiosk and the web page call.

    JPEG bytes
      -> decode, reduce to 1024 px
      -> gate.check          (is this a readable photo of one spot on skin?)
      -> classifier.predict  (only if the gate passed, or the visitor insisted)
      -> signs.measure       (A B C on the gate's outline; shown, never decides)
      -> verdict             (the words on screen)

check_photo is the first half alone (decode + gate, no model), for the "check the
photo" screen: it answers in milliseconds, before the visitor commits to a scan.
The kiosk keeps that CheckedPhoto and hands it to run_scan, so the photo is not
decoded and gated a second time (the Streamlit page and the kiosk's web mode keep
nothing between requests, so their scans run both).

Every stage is timed in milliseconds and logged on one line per scan, so a slow
device can be diagnosed from the log alone:

    check status=pass focus=41.2 brightness=128 ms decode=18 gate=55 total=73
    scan status=ok label=malignant conf=61 ms decode=0 gate=0 model=640 signs=9 total=649 check=reused
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from dermascan import config, gate, signs, verdict
from dermascan.classifier import Classifier, Prediction, get_classifier

log = logging.getLogger("dermascan.scan")


@dataclass
class ScanOutcome:
    status: str  # ok | refused | error
    verdict: verdict.Verdict
    prediction: Prediction | None = None
    refusal: gate.Refusal | None = None
    measured: dict[str, float] = field(default_factory=dict)
    stage_ms: dict[str, int] = field(default_factory=dict)
    forced: bool = False
    model_name: str = ""
    abc_signs: list[signs.Sign] | None = None

    def to_dict(self) -> dict:
        """What the web page receives. No image bytes: the page already has the photo."""
        return {
            "status": self.status,
            "verdict": self.verdict.to_dict(),
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "refusal": (
                {"code": self.refusal.code, "can_override": self.refusal.can_override}
                if self.refusal
                else None
            ),
            "measured": {k: round(v, 3) for k, v in self.measured.items()},
            "stage_ms": self.stage_ms,
            "total_ms": sum(self.stage_ms.values()),
            "forced": self.forced,
            "caveat": verdict.FORCED_CAVEAT if self.forced else "",
            "model": self.model_name,
            "signs": [s.to_dict() for s in self.abc_signs] if self.abc_signs else None,
            "sign_lines": verdict.sign_lines(self.abc_signs),
        }


_decode_slots = threading.BoundedSemaphore(config.MAX_PARALLEL_DECODES)


def decode_jpeg(data: bytes, max_px: int) -> np.ndarray:
    """Open a photo and shrink it to at most `max_px` on its longest edge, as RGB.

    The full-size picture exists only inside a decode slot, so at most
    MAX_PARALLEL_DECODES of them are in memory at once.
    """
    with _decode_slots:
        try:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:  # larger than MAX_DECODE_PIXELS, or a broken file
            bgr = None
        if bgr is None:
            raise ValueError("not an image")
        return cv2.cvtColor(gate.shrink(bgr, max_px), cv2.COLOR_BGR2RGB)


@dataclass
class CheckedPhoto:
    """A photo after decode + gate: everything run_scan needs before the model.

    `rgb` is None when the bytes were not a picture at all.
    """

    rgb: np.ndarray | None
    report: gate.GateReport | None
    stage_ms: dict[str, int]  # decode, gate


def check_photo(jpeg_bytes: bytes) -> CheckedPhoto:
    """Decode to at most MAX_WORK_PX and run the gate. Logged as one `check` line."""
    ms: dict[str, int] = {}
    t = time.perf_counter()
    try:
        rgb = decode_jpeg(jpeg_bytes, config.MAX_WORK_PX)
    except ValueError:
        log.info("check status=error reason=not_a_picture")
        return CheckedPhoto(None, None, ms)
    ms["decode"] = int((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    report = gate.check(rgb)
    ms["gate"] = int((time.perf_counter() - t) * 1000)
    _maybe_save_capture(jpeg_bytes, report)
    r = report.refusal
    log.info(
        "check status=%s focus=%.1f brightness=%.0f ms decode=%d gate=%d total=%d",
        r.code if r else "pass", report.measured.get("focus", -1), report.measured.get("brightness", -1),
        ms["decode"], ms["gate"], ms["decode"] + ms["gate"],
    )
    return CheckedPhoto(rgb, report, ms)


def run_scan(
    jpeg_bytes: bytes,
    *,
    force: bool = False,
    classifier: Classifier | None = None,
    checked: CheckedPhoto | None = None,
) -> ScanOutcome:
    """`force=True` reads the photo even when the gate's spot check refused it.

    The gate still runs and its numbers are still reported, but a refusal the
    visitor is allowed to override no longer stops the scan. Refusals that
    cannot be overridden (no skin, unreadable) always stop it.

    `checked` is the CheckedPhoto of exactly these bytes from the "check the
    photo" step; given one, decode and gate are not run again (their stage
    times are then 0 and the log line says `check=reused`).
    """
    ms: dict[str, int] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        ms[name] = int((now - t) * 1000)
        t = now

    reused = checked is not None
    if checked is None:
        checked = check_photo(jpeg_bytes)
    if checked.rgb is None or checked.report is None:
        log.info("scan status=error reason=not_a_picture")
        return ScanOutcome("error", verdict.error_verdict("not_a_picture"), stage_ms=dict(checked.stage_ms))
    rgb, report = checked.rgb, checked.report
    ms["decode"], ms["gate"] = (0, 0) if reused else (checked.stage_ms["decode"], checked.stage_ms["gate"])
    t = time.perf_counter()

    refusal = report.refusal
    if refusal is not None and not (force and refusal.can_override):
        out = ScanOutcome("refused", verdict.for_refusal(refusal), refusal=refusal, measured=report.measured, stage_ms=ms)
        _log(out, reused)
        return out

    clf = classifier or get_classifier()
    try:
        pred = clf.predict(rgb)
    except Exception:  # noqa: BLE001 - a model failure must reach the screen as words
        log.exception("scan status=error reason=model")
        return ScanOutcome("error", verdict.error_verdict("scanner"), stage_ms=ms)
    lap("model")
    # Signs only on an outline the gate accepted: a "read anyway" photo's outline is
    # exactly what the gate did not trust.
    small = report.small if report.small is not None else gate.shrink(rgb)
    measured_signs = signs.measure(small, report.outline) if refusal is None else None
    lap("signs")

    out = ScanOutcome(
        "ok",
        verdict.for_prediction(pred),
        prediction=pred,
        refusal=refusal,  # kept when the visitor overrode it, so the screen can say so
        measured=report.measured,
        stage_ms=ms,
        forced=bool(force and refusal is not None),
        model_name=clf.model_path.name,
        abc_signs=measured_signs,
    )
    _log(out, reused)
    return out


def check_result(checked: CheckedPhoto) -> dict:
    """What the "check the photo" screen shows, from a CheckedPhoto.

    {"status": "pass" | "refused" | "error", "headline", "lede", "readings": [...],
     "verdict": refusal/error words or None, "refusal": {...} or None, "measured": {...}}
    """
    if checked.report is None:
        v = verdict.error_verdict("not_a_picture")
        return _check_result("error", v.headline, v.body, [], v, None, {})
    report = checked.report
    r = report.refusal
    readings = verdict.photo_readings(report.measured, r.code if r else None)
    if r is None:
        return _check_result("pass", verdict.PHOTO_OK_HEADLINE, verdict.photo_lede(readings), readings, None, None, report.measured)
    v = verdict.for_refusal(r)
    return _check_result("refused", v.headline, v.body, readings, v, r, report.measured)


def run_check(jpeg_bytes: bytes) -> dict:
    """Decode + gate only, as one call: check_photo then check_result."""
    return check_result(check_photo(jpeg_bytes))


def _check_result(status, headline, lede, readings, v, refusal, measured) -> dict:
    return {
        "status": status, "headline": headline, "lede": lede, "readings": readings,
        "verdict": v.to_dict() if v else None,
        "refusal": {"code": refusal.code, "can_override": refusal.can_override} if refusal else None,
        "measured": {k: round(x, 3) for k, x in measured.items()},
    }


def _log(out: ScanOutcome, reused_check: bool) -> None:
    stages = " ".join(f"{k}={v}" for k, v in out.stage_ms.items())
    if out.prediction:
        head = f"label={out.prediction.label} conf={out.prediction.confidence_pct:.0f}"
    else:
        head = f"refused={out.refusal.code if out.refusal else '?'}"
    tail = " check=reused" if reused_check else ""
    log.info("scan status=%s %s ms %s total=%d%s", out.status, head, stages, sum(out.stage_ms.values()), tail)


def _maybe_save_capture(jpeg_bytes: bytes, report: gate.GateReport) -> None:
    """Optional, off by default: keep raw captures to calibrate the gate later.

    Runs at the check step (since 2026-10-11), so a photo the gate refused and the
    visitor never scanned is kept too: those are the ones a gate calibration needs.
    A scan that reuses its check does not save the photo a second time.
    """
    folder = config.SAVE_CAPTURES_DIR
    if folder is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        (folder / f"{stamp}.jpg").write_bytes(jpeg_bytes)
        numbers = " ".join(f"{k}={v:.3f}" for k, v in report.measured.items())
        code = report.refusal.code if report.refusal else "pass"
        (folder / f"{stamp}.txt").write_text(f"{code} {numbers}\n")
    except OSError:
        log.warning("could not save capture", exc_info=True)
