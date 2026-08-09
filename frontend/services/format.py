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


def tone_colors(tone: str) -> tuple[str, str]:
    """Resolve a ``UIVerdict.tone`` to its ``(ink, fill)`` pair.

    Every screen that shows verdict wording colours it through here, so the
    palette stays consistent and no view invents its own risk colour.
    """
    return _TONE_COLORS.get(tone, _TONE_COLORS["info"])
