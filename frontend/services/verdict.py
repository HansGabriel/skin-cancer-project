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
    # The tinted panel under the result: a label in small caps and one short
    # paragraph. It carries the thing the headline cannot — what to do today,
    # what to bring, or (for a clean result) that "clean" is not a promise.
    # Kept on the verdict rather than in the view so there is still exactly one
    # place allowed to write risk language.
    note_label: str = ""
    note: str = ""

    def text(self) -> str:
        """Every user-visible string in one place (used by copy-invariant tests)."""
        return " ".join(
            (self.headline, self.body, self.advice, self.note, *self.reasons)
        )


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
        note_label="WHAT HELPS",
        note=(
            "Even light with no shadow and no glare, the spot filling the ring, "
            "and two seconds held still before you tap."
        ),
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
        note_label="WHAT TO POINT AT",
        note=(
            "A mole or mark on skin, filling the ring, with a little plain skin "
            "around its edge so the scanner can see where the spot ends."
        ),
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
                # The safety line used to be glued onto the end of `advice`,
                # where it read as an afterthought to "no action needed". It is
                # the more important half of a clean result, so it now gets the
                # panel to itself.
                advice=RECOMMENDATIONS["benign"]["action"],
                tone="neutral",
                note_label="KEEP AN EYE ON IT",
                note=LOW_RISK_SAFETY_LINE,
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
            note_label="WHY THIS HAPPENS",
            note=(
                "Some spots sit right between the patterns this scanner knows. "
                "A clearer photo often settles it; a health worker always can."
            ),
        )
    if confident:
        if label == "malignant":
            return UIVerdict(
                state="urgent",
                headline="SEE A DOCTOR SOON",
                body="The scan found signs that need a doctor's opinion.",
                advice=RECOMMENDATIONS["malignant"]["action"],
                tone="urgent",
                note_label="WHAT TO DO TODAY",
                note=(
                    "Book the appointment now — most spots the scanner picks out "
                    "turn out to be harmless, and the ones that are not are far "
                    "easier to treat early."
                ),
            )
        return UIVerdict(
            state="needs_attention",
            headline="GET THIS CHECKED",
            body="The scan found something worth a closer look.",
            advice=RECOMMENDATIONS["pre_cancerous"]["action"],
            tone="warning",
            note_label="WHAT TO BRING",
            note=(
                "Tell them when you first noticed the spot and whether it has "
                "changed. Saving this scan keeps a photo to compare against."
            ),
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
        note_label="WHAT TO BRING",
        note=(
            "Tell them when you first noticed the spot and whether it has "
            "changed. Saving this scan keeps a photo to compare against."
        ),
    )


# --- What the scan actually saw ---------------------------------------------
#
# The ABCDE numbers (A 0.37, B 4.58, C 2, D 17.41) are meaningless to a visitor,
# and until now they only ever appeared in the staff panel. These turn the same
# measurements into the sentences a person can act on — "Edges are uneven and
# blurred" — without inventing a second risk vocabulary: the copy lives here,
# beside the verdicts, for the reason the module docstring gives.
#
# Tier is carried rather than a colour so nothing in this module has to know
# about the theme; ``services.format.sign_ink`` maps it.


@dataclass(frozen=True)
class ScanSign:
    text: str
    tier: int  # 0 normal, 1 borderline, 2 stands out


_BORDER_TEXT = {
    0: "Edges are smooth and even",
    1: "Edges are slightly uneven",
    2: "Edges are uneven and blurred",
}
_ASYMMETRY_TEXT = {
    0: "Both halves look alike",
    1: "The two halves differ a little",
    2: "The two halves do not match",
}
# The classic ABCDE "D" cue is 6 mm — about a pencil eraser — which is exactly
# where services.abcde.TIER_D_MM puts tier 2, so the comparison is not a
# separate rule, just the same threshold said in an object people can picture.
_ERASER_CLAUSE = " — wider than a pencil eraser"


def _colour_text(count: int) -> str:
    if count <= 1:
        return "One even colour throughout"
    if count == 2:
        return "Two different colours inside"
    return "Three or more colours inside"


def _tier_of(letter: dict) -> int:
    try:
        return int(letter.get("tier", 0) or 0)
    except (TypeError, ValueError):
        return 0


def scan_signs(abcde: dict | None) -> tuple[ScanSign, ...]:
    """Plain-language lines for what the five checks measured.

    Border, colour and size are always reported — they are the three a person
    can see for themselves in the photo. Asymmetry joins them only when it
    stands out, and Evolving only when there is an earlier scan to compare
    against; otherwise they add two lines that say "nothing to report" and push
    the real findings off a 600px panel.
    """
    if not abcde:
        return ()
    signs: list[ScanSign] = []

    a = abcde.get("A") or {}
    if _tier_of(a) >= 2:
        signs.append(ScanSign(_ASYMMETRY_TEXT[2], 2))

    b = abcde.get("B") or {}
    if b:
        signs.append(ScanSign(_BORDER_TEXT.get(_tier_of(b), _BORDER_TEXT[0]), _tier_of(b)))

    c = abcde.get("C") or {}
    if c and c.get("value") is not None:
        signs.append(ScanSign(_colour_text(int(c["value"])), _tier_of(c)))

    d = abcde.get("D") or {}
    if d and d.get("value") is not None:
        mm = float(d["value"])
        tier = _tier_of(d)
        text = f"About {mm:.0f} mm across" + (_ERASER_CLAUSE if tier >= 2 else "")
        signs.append(ScanSign(text, tier))

    e = abcde.get("E") or {}
    if e and e.get("verdict") not in (None, "", "needs history") and e.get("value") is not None:
        grown = float(e["value"])
        if grown > 0:
            signs.append(ScanSign(f"About {grown:.1f} mm bigger than last time", _tier_of(e)))
        else:
            signs.append(ScanSign("About the same size as last time", _tier_of(e)))

    return tuple(signs)


def signs_sentence(abcde: dict | None) -> str:
    """The same findings as one sentence, for the assistant to lead with.

    Composed here in Python from the measured numbers rather than generated, so
    it cannot state a size the scan did not measure. See docs/ASSISTANT.md —
    the on-device model may reword approved text, never author it.
    """
    signs = scan_signs(abcde)
    if not signs:
        return ""
    parts = [s.text[0].lower() + s.text[1:] for s in signs]
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"Looking at your spot, the scan measured: {body}."


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
