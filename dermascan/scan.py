"""run_scan: the one function the kiosk and the web page call.

    JPEG bytes
      -> decode, reduce to 1024 px
      -> gate.check          (is this a readable photo of one spot on skin?)
      -> classifier.predict  (only if the gate passed, or the visitor insisted)
      -> signs.measure       (A B C on the gate's outline; shown, never decides)
      -> verdict             (the words on screen)

run_check is the first half alone (decode + gate, no model), for the "check the
photo" screen: it answers in milliseconds, before the visitor commits to a scan.

Every stage is timed in milliseconds and logged on one line per scan, so a slow
device can be diagnosed from the log alone:

    scan status=ok label=malignant conf=61 ms decode=18 gate=55 model=640 signs=9 total=722
"""

from __future__ import annotations

import logging
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
    signs: list[signs.Sign] | None = None

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
            "signs": [s.to_dict() for s in self.signs] if self.signs else None,
            "sign_lines": verdict.sign_lines(self.signs),
        }


def decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_scan(
    jpeg_bytes: bytes,
    *,
    force: bool = False,
    classifier: Classifier | None = None,
) -> ScanOutcome:
    """`force=True` reads the photo even when the gate's spot check refused it.

    The gate still runs and its numbers are still reported, but a refusal the
    visitor is allowed to override no longer stops the scan. Refusals that
    cannot be overridden (no skin, unreadable) always stop it.
    """
    ms: dict[str, int] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        ms[name] = int((now - t) * 1000)
        t = now

    try:
        rgb = decode_jpeg(jpeg_bytes)
        rgb = gate.shrink(rgb, config.MAX_WORK_PX)
        lap("decode")
    except ValueError:
        log.info("scan status=error reason=not_a_picture")
        return ScanOutcome("error", verdict.error_verdict("not_a_picture"), stage_ms=ms)

    report = gate.check(rgb)
    lap("gate")
    _maybe_save_capture(jpeg_bytes, report)

    refusal = report.refusal
    if refusal is not None and not (force and refusal.can_override):
        out = ScanOutcome("refused", verdict.for_refusal(refusal), refusal=refusal, measured=report.measured, stage_ms=ms)
        _log(out)
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
    measured_signs = signs.measure(gate.shrink(rgb), report.outline) if refusal is None else None
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
        signs=measured_signs,
    )
    _log(out)
    return out


def run_check(jpeg_bytes: bytes) -> dict:
    """Decode + gate only. What the "check the photo" screen shows.

    {"status": "pass" | "refused" | "error", "headline", "lede", "readings": [...],
     "verdict": refusal/error words or None, "refusal": {...} or None, "measured": {...}}
    Logged like a scan:  check status=pass ms decode=18 gate=52
    """
    t = time.perf_counter()
    try:
        rgb = gate.shrink(decode_jpeg(jpeg_bytes), config.MAX_WORK_PX)
    except ValueError:
        log.info("check status=error reason=not_a_picture")
        v = verdict.error_verdict("not_a_picture")
        return _check_result("error", v.headline, v.body, [], v, None, {})
    decode_ms = int((time.perf_counter() - t) * 1000)
    report = gate.check(rgb)
    gate_ms = int((time.perf_counter() - t) * 1000) - decode_ms
    r = report.refusal
    readings = verdict.photo_readings(report.measured, r.code if r else None)
    log.info(
        "check status=%s focus=%.1f brightness=%.0f ms decode=%d gate=%d",
        r.code if r else "pass", report.measured.get("focus", -1), report.measured.get("brightness", -1), decode_ms, gate_ms,
    )
    if r is None:
        return _check_result("pass", verdict.PHOTO_OK_HEADLINE, verdict.photo_lede(readings), readings, None, None, report.measured)
    v = verdict.for_refusal(r)
    return _check_result("refused", v.headline, v.body, readings, v, r, report.measured)


def _check_result(status, headline, lede, readings, v, refusal, measured) -> dict:
    return {
        "status": status, "headline": headline, "lede": lede, "readings": readings,
        "verdict": v.to_dict() if v else None,
        "refusal": {"code": refusal.code, "can_override": refusal.can_override} if refusal else None,
        "measured": {k: round(x, 3) for k, x in measured.items()},
    }


def _log(out: ScanOutcome) -> None:
    stages = " ".join(f"{k}={v}" for k, v in out.stage_ms.items())
    if out.prediction:
        head = f"label={out.prediction.label} conf={out.prediction.confidence_pct:.0f}"
    else:
        head = f"refused={out.refusal.code if out.refusal else '?'}"
    log.info("scan status=%s %s ms %s total=%d", out.status, head, stages, sum(out.stage_ms.values()))


def _maybe_save_capture(jpeg_bytes: bytes, report: gate.GateReport) -> None:
    """Optional, off by default: keep raw captures to calibrate the gate later."""
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
