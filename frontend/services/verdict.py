"""Single source of truth for the one verdict the results screen shows.

Exactly one ``UIVerdict`` is produced per scan. This is the only place allowed
to combine the screening decision (``ScanResult.label`` — already thresholded,
sensitivity-first, by ``backend.tflite_shared.decide_index``), the calibrated
confidence, and capture quality into user-facing risk language. Views render
from it and never add their own risk copy — separate rules in separate places
is what previously produced "URGENT" and "Inconclusive" on the same screen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from backend.contracts import ScanResult
from backend.recommendations import RECOMMENDATIONS

VerdictState = Literal[
    "retake",
    "low_concern",
    "uncertain",
    "needs_attention",
    "urgent",
    "uncertain_caution",
]
Tone = Literal["info", "success", "warning", "urgent"]

# Shown with every low-concern verdict: false reassurance is the failure mode
# that hurts people, so a clean screen still points at the warning signs.
LOW_RISK_SAFETY_LINE = (
    "Low risk is not no risk — if this spot changes, grows, or bleeds, "
    "have it checked by a health professional."
)

_CANCER_LABELS = ("pre_cancerous", "malignant")


def _confidence_floor() -> float:
    """Calibrated top-class confidence (%) below which no single label is trusted."""
    try:
        return float(os.environ.get("SKIN_INCONCLUSIVE_BELOW", "45"))
    except ValueError:
        return 45.0


@dataclass(frozen=True)
class UIVerdict:
    state: VerdictState
    headline: str
    body: str
    advice: str
    tone: Tone
    reasons: tuple[str, ...] = ()

    def text(self) -> str:
        """Every user-visible string in one place (used by copy-invariant tests)."""
        return " ".join((self.headline, self.body, self.advice, *self.reasons))


def retake_verdict(quality: dict | None) -> UIVerdict:
    reasons = tuple(quality.get("reasons", [])) if quality else ()
    return UIVerdict(
        state="retake",
        headline="RETAKE PHOTO",
        body="The photo quality was too low to analyze.",
        advice="Adjust and take another photo.",
        tone="info",
        reasons=reasons,
    )


def resolve_verdict(
    scan_result: ScanResult | None,
    quality: dict | None,
    *,
    confidence_floor: float | None = None,
) -> UIVerdict:
    """Map one scan to the single verdict the user sees.

    No scan (quality gate blocked it) → retake. Screened negative → low concern,
    or uncertain when the calibrated confidence is under the floor. Screened
    positive → needs-attention/urgent, or a no-urgent-styling "needs a check"
    that keeps the referral advice when confidence is under the floor.
    """
    if scan_result is None:
        return retake_verdict(quality)
    floor = _confidence_floor() if confidence_floor is None else float(confidence_floor)
    confident = float(scan_result.confidence) >= floor
    flagged = scan_result.label in _CANCER_LABELS
    if not flagged:
        if confident:
            return UIVerdict(
                state="low_concern",
                headline="LOW CONCERN",
                body="The AI did not flag this spot.",
                advice=f"{RECOMMENDATIONS['benign']['action']} {LOW_RISK_SAFETY_LINE}",
                tone="success",
            )
        return UIVerdict(
            state="uncertain",
            headline="UNCERTAIN",
            body="The AI could not assess this spot confidently.",
            advice=(
                "Retake the photo with better lighting and focus, or consult a "
                "health professional if the spot worries you."
            ),
            tone="info",
        )
    if confident:
        if scan_result.label == "malignant":
            return UIVerdict(
                state="urgent",
                headline="URGENT",
                body="The AI flagged this spot as high concern.",
                advice=RECOMMENDATIONS["malignant"]["action"],
                tone="urgent",
            )
        return UIVerdict(
            state="needs_attention",
            headline="NEEDS ATTENTION",
            body="The AI flagged this spot for follow-up.",
            advice=RECOMMENDATIONS["pre_cancerous"]["action"],
            tone="warning",
        )
    return UIVerdict(
        state="uncertain_caution",
        headline="NEEDS A CHECK",
        body="The AI flagged this spot but is not confident.",
        advice=(
            "It could not rule out risk — please have this spot checked by a "
            "health professional."
        ),
        tone="warning",
    )
