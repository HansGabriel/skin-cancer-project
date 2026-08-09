"""Single source of truth for the one verdict any screen shows.

Exactly one ``UIVerdict`` is produced per scan. This is the only place allowed
to turn a screening decision (``ScanResult.label`` — already thresholded,
sensitivity-first, by ``backend.tflite_shared.decide_index``), the calibrated
confidence, and capture quality into user-facing risk language. Views render
from it and never add their own risk copy — separate rules in separate places
is what previously produced "URGENT" and "Inconclusive" on the same screen.

Saved scans go through :func:`verdict_for_saved_scan` so history, case and
assistant screens speak with this same voice instead of re-deriving wording
from the stored composite risk band.

Copy rules: short everyday words, no clinical jargon, no percentages. Every
headline names what the person should DO, not what the model computed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from backend.contracts import ScanResult
from backend.recommendations import CANCER_LABELS, RECOMMENDATIONS

VerdictState = Literal[
    "retake",
    "no_lesion",
    "low_concern",
    "uncertain",
    "needs_attention",
    "urgent",
    "uncertain_caution",
]
# "neutral" deliberately has no colour: a clean screening result must not be
# dressed up as an all-clear (see LOW_RISK_SAFETY_LINE).
Tone = Literal["neutral", "info", "warning", "urgent"]

# Shown with every low-concern verdict: false reassurance is the failure mode
# that hurts people, so a clean screen still points at the warning signs.
LOW_RISK_SAFETY_LINE = (
    "This does not promise the spot is safe. If it changes, grows, itches, "
    "or bleeds, have it checked."
)


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
    """The photo could not be read — a camera problem, not a skin finding."""
    reasons = tuple(quality.get("reasons", [])) if quality else ()
    return UIVerdict(
        state="retake",
        headline="TAKE ANOTHER PHOTO",
        body="This photo was too hard to read.",
        advice="Move into better light, hold still, and take a new photo.",
        tone="info",
        reasons=reasons,
    )


def no_lesion_verdict(reasons: tuple[str, ...] | list[str] | None = None) -> UIVerdict:
    """Nothing on screen to screen — the input is not a photo of a skin spot.

    Kept apart from :func:`retake_verdict` on purpose: "I cannot read this
    photo" and "this is not a skin spot" are different problems, and telling
    someone to fix the lighting when they photographed a wall is useless.
    """
    return UIVerdict(
        state="no_lesion",
        headline="NO SKIN SPOT FOUND",
        body="This photo does not show a spot on skin that the scanner can read.",
        advice="Point the camera at a mole or mark on skin, fill the ring with it, "
        "and take a new photo.",
        tone="info",
        reasons=tuple(reasons or ()),
    )


def resolve_verdict(
    scan_result: ScanResult | None,
    quality: dict | None,
    *,
    confidence_floor: float | None = None,
) -> UIVerdict:
    """Map one scan to the single verdict the user sees.

    No scan (the gate blocked it) → retake. Screened negative → low concern, or
    "not sure" when the calibrated confidence is under the floor. Screened
    positive → get-checked/see-a-doctor, or a no-urgent-styling "better to get
    checked" that keeps the referral advice when confidence is under the floor.
    """
    if scan_result is None:
        return retake_verdict(quality)
    return verdict_for_label(
        scan_result.label,
        float(scan_result.confidence),
        confidence_floor=confidence_floor,
    )


def verdict_for_label(
    label: str,
    confidence: float,
    *,
    confidence_floor: float | None = None,
) -> UIVerdict:
    """Core mapping shared by live scans and saved scans — the one rule set."""
    floor = _confidence_floor() if confidence_floor is None else float(confidence_floor)
    confident = float(confidence) >= floor
    flagged = label in CANCER_LABELS
    if not flagged:
        if confident:
            return UIVerdict(
                state="low_concern",
                headline="NOTHING STOOD OUT",
                body="This spot looks like a common, ordinary mark.",
                advice=f"{RECOMMENDATIONS['benign']['action']} {LOW_RISK_SAFETY_LINE}",
                tone="neutral",
            )
        return UIVerdict(
            state="uncertain",
            headline="NOT SURE",
            body="The scan could not read this spot well enough to say.",
            advice=(
                "Try another photo in better light. If the spot worries you, "
                "show it to a health worker."
            ),
            tone="info",
        )
    if confident:
        if label == "malignant":
            return UIVerdict(
                state="urgent",
                headline="SEE A DOCTOR SOON",
                body="The scan found signs that need a doctor's opinion.",
                advice=RECOMMENDATIONS["malignant"]["action"],
                tone="urgent",
            )
        return UIVerdict(
            state="needs_attention",
            headline="GET THIS CHECKED",
            body="The scan found something worth a closer look.",
            advice=RECOMMENDATIONS["pre_cancerous"]["action"],
            tone="warning",
        )
    return UIVerdict(
        state="uncertain_caution",
        headline="BETTER TO GET CHECKED",
        body="The scan saw something here, but is not sure about it.",
        advice=(
            "It could not rule out a problem, so please show this spot to a "
            "health worker."
        ),
        tone="warning",
    )


def verdict_for_saved_scan(scan, *, confidence_floor: float | None = None) -> UIVerdict:
    """Rebuild the verdict for a stored scan row (``services.storage.Scan``).

    History/case screens used to render risk language from ``risk_band``, a
    separate composite score — which is how a scan could show "URGENT" in a
    list and "NOT SURE" on its own result screen. Recomputing here from the
    same label and confidence keeps every screen on one story.
    """
    return verdict_for_label(
        str(scan.label),
        float(scan.confidence),
        confidence_floor=confidence_floor,
    )
