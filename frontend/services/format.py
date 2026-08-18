"""Display formatting helpers."""

from __future__ import annotations

from theme.tokens import TOKENS as T

# Verdict tone → (ink, fill). "neutral" is intentionally colourless: a clean
# screening result must never be dressed up as an all-clear badge.
_TONE_COLORS = {
    "neutral": (T.text, T.surface),
    "info": (T.info, T.info_tint),
    "warning": (T.warning, T.warning_tint),
    "urgent": (T.urgent, T.urgent_tint),
}


def fmt_pct(x: float, decimals: int = 1) -> str:
    if x <= 1.0:
        x *= 100.0
    return f"{x:.{decimals}f}%"


def fmt_mm(x: float | None, decimals: int = 2) -> str:
    if x is None or x <= 0:
        return "—"
    return f"{x:.{decimals}f} mm"


def fmt_score(x: float | None, decimals: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f}"


def tier_label(tier: int, *, evolving: bool = False) -> tuple[str, str]:
    if evolving:
        return ("NEEDS HISTORY", T.info)
    labels = {0: ("NORMAL ↓", T.success), 1: ("BORDERLINE", T.warning), 2: ("SUSPICIOUS ↑", T.urgent)}
    return labels.get(tier, ("—", T.outline))


# Tier → ink for the plain-language sign lines on white paper. Tier 0 is grey,
# not green, for the reason in theme.tokens: five green ticks read as a clean
# bill of health, which is the one thing a screening result must not promise.
_SIGN_INK = {0: T.sign_neutral, 1: T.melanin, 2: T.erythema}

# The same three tiers as they appear on the instrument's dark field. Kept
# apart from _SIGN_INK because the paper inks sit at about 2:1 against #0B1220
# and the rim marks turn to mud; see the on_field_* tokens.
_FIELD_INK = {0: T.on_field_neutral, 1: T.on_field_melanin, 2: T.on_field_erythema}


def sign_ink(tier: int) -> str:
    """Ink for a ``services.verdict.ScanSign`` rendered on the page band."""
    return _SIGN_INK.get(int(tier), _SIGN_INK[0])


def field_ink(tier: int) -> str:
    """Ink for the same tier rendered on the dark instrument band."""
    return _FIELD_INK.get(int(tier), _FIELD_INK[0])


# The short badge beside the "RESULT" eyebrow. Keyed off the tone rather than
# the label so it can never disagree with the colour it is printed in.
_TONE_CHIP = {
    "neutral": "LOW CONCERN",
    "info": "CHECK THE PHOTO",
    "warning": "IMPORTANT",
    "urgent": "URGENT",
}


def tone_chip(tone: str) -> str:
    return _TONE_CHIP.get(tone, _TONE_CHIP["info"])


def tone_colors(tone: str) -> tuple[str, str]:
    """Resolve a ``UIVerdict.tone`` to its ``(ink, fill)`` pair.

    Every screen that shows verdict wording colours it through here, so the
    palette stays consistent and no view invents its own risk colour.
    """
    return _TONE_COLORS.get(tone, _TONE_COLORS["info"])
