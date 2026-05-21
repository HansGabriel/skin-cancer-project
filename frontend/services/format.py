"""Display formatting helpers."""

from __future__ import annotations

from theme.tokens import TOKENS as T

_BAND_URGENCY = {
    "low": ("LOW CONCERN", T.success),
    "moderate": ("IMPORTANT", T.warning),
    "high": ("URGENT", T.urgent),
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


def urgency_from_band(band: str) -> tuple[str, str]:
    return _BAND_URGENCY.get(band, ("SCREENING", T.info))
