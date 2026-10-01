"""run_scan: the one function the kiosk and the web page call.

    JPEG bytes
      -> decode, reduce to 1024 px
      -> gate.check          (is this a readable photo of one spot on skin?)
      -> classifier.predict  (only if the gate passed, or the visitor insisted)
      -> verdict             (the words on screen)

Every stage is timed in milliseconds and logged on one line per scan, so a slow
device can be diagnosed from the log alone:

    scan status=ok label=malignant conf=61 ms decode=18 gate=55 model=640 total=713
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from dermascan import config, gate, verdict
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
            "model": self.model_name,
        }


def decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("That file is not a picture the scanner can open.")
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
    except ValueError as exc:
        return ScanOutcome("error", verdict.error_verdict(str(exc)), stage_ms=ms)

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
    except Exception as exc:  # noqa: BLE001 - a model failure must reach the screen as words
        log.exception("model failed")
        return ScanOutcome("error", verdict.error_verdict(f"The model did not answer ({exc})."), stage_ms=ms)
    lap("model")

    out = ScanOutcome(
        "ok",
        verdict.for_prediction(pred),
        prediction=pred,
        refusal=refusal,  # kept when the visitor overrode it, so the screen can say so
        measured=report.measured,
        stage_ms=ms,
        forced=bool(force and refusal is not None),
        model_name=clf.model_path.name,
    )
    _log(out)
    return out


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
