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
* `cannot_judge`    - the visitor pressed "read anyway" on a photo the gate found no
                      spot in (gate.NOT_A_SPOT_CODES). The model ran for staff, but
                      on plain skin or hair its numbers mean nothing, so no verdict.
* `for_refusal`     - the gate stopped the scan. One headline per refusal code,
                      each saying what to point the camera at instead.

Also here: the chip above each verdict (`CHIP`), the "what the scan saw" lines
for the A B C signs (`sign_lines`), and the "check the photo" screen's chip
(`check_label`: SPOT FOUND / NOT A SPOT · PLAIN SKIN / SPOT · NOT IN THE RING ...)
and three readings (`photo_readings`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dermascan import config
from dermascan.classifier import Prediction
from dermascan.gate import Refusal
from dermascan.signs import Sign

Tone = Literal["neutral", "info", "warning", "urgent"]
State = Literal["low_concern", "uncertain", "needs_attention", "urgent", "uncertain_caution", "cannot_judge", "refused", "error"]
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
        return {**self.__dict__, "chip": CHIP[self.state]}


# The small label above the headline. Never "safe", never "clear".
CHIP: dict[str, str] = {
    "low_concern": "LOW CONCERN",
    "uncertain": "NOT SURE",
    "needs_attention": "IMPORTANT",
    "uncertain_caution": "IMPORTANT",
    "urgent": "URGENT",
    "cannot_judge": "NOT READ",
    "refused": "NOT READ",
    "error": "NOT READ",
}


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


def cannot_judge() -> Verdict:
    """A forced read of a photo with no spot in it: the model's answer is withheld."""
    return Verdict(
        "cannot_judge", "info",
        "WE CAN'T JUDGE THIS PHOTO",
        "The photo check did not find one spot on skin, so the scan had nothing to read.",
        "Put the ring over one mole or mark and take a new photo. If a spot worries you, show it to a health worker.",
        "WHY NO ANSWER",
        "The scanner only knows spots. On plain skin, hair or several marks its numbers "
        "mean nothing, so it does not give one.",
    )


# Beside the "Where it looked" circle (check screen and result). The heatmap is class-agnostic:
# it shows where the scanner looked, never why. "off" is its one nudge - retake, not a verdict.
HEAT_WORDS: dict[str, str] = {
    "loading": "Looking…",
    "on": "Bright area is on your spot.",
    "off": "Bright area off your spot. For a more accurate reading, retake with the lesion highlighted in the ring.",
    "unknown": "Bright areas are where the scanner looked. Not on your mark? Take another.",
}


# headline, body, advice - one per gate refusal code.
_REFUSAL_COPY: dict[str, tuple[str, str, str]] = {
    "no_skin": (
        "NO SKIN RECOGNISED",
        "The scanner did not recognise skin here. Glare, shadow or an odd tint can hide it.",
        "Fill the ring with the skin around one mole or mark, in even light, and take a new photo.",
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
        "The scanner found skin, but no mark on it to read.",
        "Put the ring over a mole or mark, move a little closer, and take a new photo.",
    ),
    "fills_frame": (
        "TOO CLOSE TO SEE THE EDGE",
        "What the scanner outlined fills the photo, so it cannot see where the spot ends.",
        "Move back until plain skin shows all round the spot, then take a new photo.",
    ),
    "soft_edge": (
        "THAT LOOKS LIKE A SHADOW",
        "The outline fades out slowly, the way a shadow does, not the way a mole stops.",
        "Move so the light falls evenly on the skin, then take a new photo.",
    ),
    "plain_skin": (
        "NO SPOT ON THIS SKIN",
        "The scanner found skin, but no clear mark on it to read.",
        "Put the ring over a mole or mark, or move closer if the spot is small.",
    ),
    "ragged_spot": (
        "THE EDGE IS HARD TO FOLLOW",
        "The scanner found a mark, but its outline breaks up into pieces.",
        "Move a little closer so the mark fills the ring, then take a new photo.",
    ),
    "not_in_ring": (
        "PUT THE SPOT IN THE RING",
        "The mark the scanner found sits away from the middle or runs off the edge.",
        "Move so the spot sits in the middle of the ring, then take a new photo.",
    ),
    "several_marks": (
        "MORE THAN ONE MARK",
        "This photo has several marks in it, not one spot on skin.",
        "Bring the camera close to one mole or mark so it fills the ring, and take a new photo.",
    ),
}
_REFUSAL_DEFAULT = (
    "NOT A SPOT IT CAN READ",
    "This photo does not show a spot on skin that the scanner can read.",
    "Point the camera at a mole or mark on skin, fill the ring with it, and take a new photo.",
)

# The chip on the "check the photo" screen: what the gate decided, in two or three words.
# This is the label the kiosk puts on a photo that is not a lesion. Three families:
# "NOT A SPOT" (nothing to read, so no verdict even when forced), "SPOT" (a real mark,
# badly framed: a forced read answers, with the caveat), "CAN'T READ" (the photo itself).
CHECK_LABEL: dict[str, str] = {
    "pass": "SPOT FOUND",
    "no_skin": "SKIN NOT RECOGNISED",
    "too_blurry": "CAN'T READ · BLURRY",
    "too_dark": "CAN'T READ · DARK",
    "too_bright": "CAN'T READ · GLARE",
    "no_spot": "NOT A SPOT · NONE FOUND",
    "plain_skin": "NOT A SPOT · PLAIN SKIN",
    "several_marks": "NOT A SPOT · SEVERAL MARKS",
    "not_in_ring": "SPOT · NOT IN THE RING",
    "fills_frame": "SPOT · TOO CLOSE",
    "soft_edge": "SPOT · LOOKS LIKE A SHADOW",
    "ragged_spot": "SPOT · EDGE HARD TO FOLLOW",
    "error": "NOT READ",
}
_LABEL_DEFAULT = "NOT A SPOT"


def check_label(status: str, refusal_code: str | None) -> str:
    """The chip for a pass, a refusal or an error. A code this file does not know is "NOT A SPOT"."""
    if status == "pass":
        return CHECK_LABEL["pass"]
    if status == "error" or not refusal_code:
        return CHECK_LABEL["error"]
    return CHECK_LABEL.get(refusal_code, _LABEL_DEFAULT)


# The line above the headline on the "check the photo" screen. One short line for every
# outcome: the chip (check_label) and the headline carry the reason, so this does not.
PHOTO_CHECK_EYEBROW = "PHOTO CHECK"


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
    "too_big": "That picture is too large to send. Choose a smaller one.",
    "camera": "The camera did not take the photo.",
    "scanner": "The scanner could not finish reading this photo.",
}


# The web demo has no staff member to send a visitor to (kiosk/server.py swaps this in).
ERROR_ADVICE_WEB = "Take another photo, or try again in a minute."


def error_verdict(kind: str) -> Verdict:
    return Verdict(
        "error", "info",
        "THAT SCAN DID NOT FINISH",
        _ERROR_BODY.get(kind, _ERROR_BODY["scanner"]),
        "Take another photo. If this keeps happening, ask a staff member.",
    )


# --- E (evolving): what a visitor reads when comparing a spot's first and latest photo.
# The kiosk does not judge change itself (docs/METRICS.md: from photos it misses most
# real change), so these words hand the judgement to a person.
EVOLVING_COMPARE = (
    "Look for any change in size, shape or colour between the two photos. "
    "A spot that is growing or changing should be shown to a health worker, "
    "whatever any single scan says."
)


# --- What the scan saw: one line per measured sign, by tier (normal, borderline, stands out)
_SIGN_WORDS: dict[str, tuple[str, str, str]] = {
    "A": ("The two halves match", "The two halves differ a little", "The two halves look different"),
    "B": ("Edges are smooth and even", "Edges are slightly uneven", "Edges are uneven and ragged"),
    "C": ("One even colour throughout", "Two different colours inside", "Several different colours inside"),
}


SIGN_LINE_TEXTS = frozenset(t for words in _SIGN_WORDS.values() for t in words)


def sign_lines(signs: list[Sign] | None) -> list[dict]:
    """[{letter, tier, text}] for the signs that were measured; [] when none were."""
    if not signs:
        return []
    return [
        {"letter": s.letter, "tier": s.tier, "text": _SIGN_WORDS[s.letter][s.tier]}
        for s in signs
        if s.letter in _SIGN_WORDS and s.tier is not None
    ]


# --- The "check the photo" screen: light, focus, spot, each as 0-3 bars and a word
PHOTO_OK_HEADLINE = "This photo can be read"
PHOTO_OK_LEDE = "All three readings are good enough for a reliable answer."
PHOTO_SOFT_LEDE = "It can be read. A sharper, evenly lit photo gives a steadier answer."


def photo_lede(readings: list[dict]) -> str:
    """The sentence under "This photo can be read": a nudge when any reading is below full."""
    soft = any(r["level"] is not None and r["level"] < 3 for r in readings)
    return PHOTO_SOFT_LEDE if soft else PHOTO_OK_LEDE


def photo_readings(measured: dict[str, float], refusal_code: str | None) -> list[dict]:
    """[{name, level, word}]. level is 0-3 bars, or None when the gate stopped before measuring it."""
    code = refusal_code or ""

    def light() -> tuple[int | None, str]:
        if code == "too_dark":
            return 0, "too dark"
        if code == "too_bright":
            return 0, "too bright"
        b = measured.get("brightness")
        if b is None:
            return None, "not checked"
        lo, hi = config.BRIGHTNESS_GOOD
        return (3, "good") if lo <= b <= hi else (2, "dim" if b < lo else "glary")

    def focus() -> tuple[int | None, str]:
        if code == "too_blurry":
            return 1, "blurry"
        f = measured.get("focus")
        if f is None:
            return None, "not checked"
        return (3, "sharp") if f >= config.FOCUS_SHARP else (2, "soft")

    def spot() -> tuple[int | None, str]:
        if code == "no_skin":
            return 0, "not seen"
        if code in ("no_spot", "plain_skin"):
            return 1, "not found"
        if code == "fills_frame":
            return 1, "too close"
        if code == "soft_edge":
            return 2, "faint edge"
        if code == "ragged_spot":
            return 2, "ragged edge"
        if code == "not_in_ring":
            return 1, "not in ring"
        if code == "several_marks":
            return 1, "more than one"
        if "spot_fraction" not in measured:
            return None, "not checked"
        return 3, "yes"

    rows = (("Light", light()), ("Focus", focus()), ("Spot in frame", spot()))
    return [{"name": name, "level": level, "word": word} for name, (level, word) in rows]
