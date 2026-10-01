"""The words a visitor reads. One verdict per scan, and this is the only file allowed to write risk language.

Rules for every string here: short everyday words, no clinical or model jargon
(never "malignant", "softmax", "inconclusive"), no percentages, and every
headline names what the person should DO, not what the model computed.
tests/test_verdict.py enforces the jargon ban.

Two kinds of verdict:

* `for_prediction`  - the model answered. Flagged + confident -> see a doctor;
                      flagged + unsure -> better to get checked; not flagged +
                      confident -> nothing stood out (with the safety line);
                      not flagged + unsure -> not sure, try again.
* `for_refusal`     - the gate stopped the scan. One headline per refusal code,
                      each saying what to point the camera at instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dermascan import config
from dermascan.classifier import Prediction
from dermascan.gate import Refusal

Tone = Literal["neutral", "info", "warning", "urgent"]
State = Literal["low_concern", "uncertain", "needs_attention", "urgent", "uncertain_caution", "refused", "error"]
# "neutral" has no colour on purpose: a clean result must never look like an
# all-clear. Green is banned for the same reason.

LOW_RISK_SAFETY_LINE = (
    "This does not promise the spot is safe. If it changes, grows, itches, "
    "or bleeds, have it checked."
)

ACTION = {
    "benign": "No action needed now. Keep checking your skin from time to time.",
    "pre_cancerous": "See a skin doctor within a month.",
    "malignant": "See a skin doctor within one to two weeks.",
}


@dataclass(frozen=True)
class Verdict:
    state: State
    tone: Tone
    headline: str
    body: str
    advice: str
    note_label: str = ""
    note: str = ""

    def text(self) -> str:
        """Every visible string, for the copy tests."""
        return " ".join((self.headline, self.body, self.advice, self.note_label, self.note))

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def for_prediction(p: Prediction, *, confidence_floor: float = config.CONFIDENCE_FLOOR_PCT) -> Verdict:
    confident = p.confidence_pct >= confidence_floor
    if not p.flagged:
        if confident:
            return Verdict(
                "low_concern", "neutral",
                "NOTHING STOOD OUT",
                "This spot looks like a common, ordinary mark.",
                ACTION["benign"],
                "KEEP AN EYE ON IT", LOW_RISK_SAFETY_LINE,
            )
        return Verdict(
            "uncertain", "info",
            "NOT SURE",
            "The scan could not read this spot well enough to say.",
            "Try another photo in better light. If the spot worries you, show it to a health worker.",
            "WHY THIS HAPPENS",
            "Some spots sit right between the patterns this scanner knows. "
            "A clearer photo often settles it; a health worker always can.",
        )
    if confident and p.label == "malignant":
        return Verdict(
            "urgent", "urgent",
            "SEE A DOCTOR SOON",
            "The scan found signs that need a doctor's opinion.",
            ACTION["malignant"],
            "WHAT TO DO TODAY",
            "Book the appointment now. Most spots the scanner picks out turn out to be "
            "harmless, and the ones that are not are far easier to treat early.",
        )
    if confident:
        return Verdict(
            "needs_attention", "warning",
            "GET THIS CHECKED",
            "The scan found something worth a closer look.",
            ACTION["pre_cancerous"],
            "WHAT TO BRING",
            "Tell them when you first noticed the spot and whether it has changed.",
        )
    return Verdict(
        "uncertain_caution", "warning",
        "BETTER TO GET CHECKED",
        "The scan saw something here, but is not sure about it.",
        "It could not rule out a problem, so please show this spot to a health worker.",
        "WHAT TO BRING",
        "Tell them when you first noticed the spot and whether it has changed.",
    )


# headline, body, advice - one per gate refusal code.
_REFUSAL_COPY: dict[str, tuple[str, str, str]] = {
    "no_skin": (
        "NO SKIN IN THIS PHOTO",
        "The scanner could not find skin in this photo.",
        "Point the camera at a mole or mark on skin, fill the ring with it, and take a new photo.",
    ),
    "too_blurry": (
        "TOO BLURRY TO READ",
        "The photo is out of focus.",
        "Hold the camera still for two seconds, then take a new photo.",
    ),
    "too_dark": (
        "TOO DARK TO READ",
        "There is not enough light on the skin.",
        "Move into brighter light and take a new photo.",
    ),
    "too_bright": (
        "TOO BRIGHT TO READ",
        "The photo is washed out by glare.",
        "Move out of the direct light and take a new photo.",
    ),
    "no_spot": (
        "NO SPOT COULD BE PICKED OUT",
        "The scanner found skin, but could not pick out a mark on it to read.",
        "Put the ring over a mole or mark, move a little closer, and take a new photo.",
    ),
    "fills_frame": (
        "NO EDGE TO READ",
        "The whole photo is one even tone, so the scanner cannot see where a spot ends.",
        "Move back so some plain skin shows around the spot, or put the ring over a mole or mark.",
    ),
    "soft_edge": (
        "THAT LOOKS LIKE A SHADOW",
        "The edge of what the scanner outlined fades away slowly, the way a shadow does, "
        "rather than stopping the way a mole does.",
        "Move so the light falls evenly on the skin, then put the ring over the mole or mark itself.",
    ),
    "plain_skin": (
        "NO SPOT ON THIS SKIN",
        "The scanner found skin, but no clear mark on it to read.",
        "Put the ring over a mole or mark, or move closer if the spot is small.",
    ),
}
_REFUSAL_DEFAULT = (
    "NOT A SPOT IT CAN READ",
    "This photo does not show a spot on skin that the scanner can read.",
    "Point the camera at a mole or mark on skin, fill the ring with it, and take a new photo.",
)


def for_refusal(r: Refusal) -> Verdict:
    headline, body, advice = _REFUSAL_COPY.get(r.code, _REFUSAL_DEFAULT)
    return Verdict(
        "refused", "info", headline, body, advice,
        "WHAT TO POINT AT",
        "A mole or mark on skin, filling the ring, with a little plain skin around "
        "its edge so the scanner can see where the spot ends.",
    )


# Shown under a result the visitor asked to be read after the photo check refused it.
FORCED_CAVEAT = (
    "This photo did not pass the usual checks and was read anyway. "
    "Treat the result with extra caution."
)

# What went wrong, in words, for the few ways a scan can fail outright. The
# technical detail goes to the log, never to the visitor.
_ERROR_BODY = {
    "not_a_picture": "That file is not a picture the scanner can open.",
    "no_photo": "There is no photo to check yet.",
    "camera": "The camera did not take the photo.",
    "scanner": "The scanner could not finish reading this photo.",
}


def error_verdict(kind: str) -> Verdict:
    return Verdict(
        "error", "info",
        "THAT SCAN DID NOT FINISH",
        _ERROR_BODY.get(kind, _ERROR_BODY["scanner"]),
        "Take another photo. If this keeps happening, ask a staff member.",
    )
