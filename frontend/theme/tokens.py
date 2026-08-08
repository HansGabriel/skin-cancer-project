"""Single source of truth for DermaScan v2 design tokens.

Display profiles (env ``SKIN_DISPLAY``, read once at import):
- ``desktop`` (default) — PC / Streamlit Cloud browser; values unchanged from MVP 1.
- ``7in`` — the 7" 1024x600 (or 800x480) touchscreen kiosk at surf zoom 1.0:
  wider frame, larger type, and >=56px touch targets.

Set the env var BEFORE ``streamlit run`` (launch_kiosk.sh exports SKIN_DISPLAY=7in).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Tokens:
    # Clean clinical light palette — high contrast on a small panel.
    bg: str = "#FFFFFF"
    bg_elev: str = "#F4F6FB"
    surface: str = "#EDEFF6"
    outline: str = "#D4D9E6"
    text: str = "#14181F"
    text_muted: str = "#5A6273"
    violet: str = "#6C4AB6"  # darkened so it reads on white
    violet_strong: str = "#4A2F86"
    teal: str = "#0EA5A4"  # medical secondary accent
    teal_strong: str = "#0B7E7D"
    success: str = "#15803D"
    warning: str = "#B45309"
    urgent: str = "#DC2626"
    info: str = "#2563EB"
    # Soft tints for card fills behind risk content (not the saturated badge colors).
    success_tint: str = "#E7F6EC"
    warning_tint: str = "#FDF3E4"
    urgent_tint: str = "#FCEBEA"
    info_tint: str = "#E8F0FE"
    violet_tint: str = "#F0EBFA"
    # Elevation — consistent depth instead of flat 1px borders.
    shadow_sm: str = "0 1px 3px rgba(20,24,31,.06), 0 1px 2px rgba(20,24,31,.04)"
    shadow_md: str = "0 6px 18px rgba(20,24,31,.10), 0 2px 6px rgba(20,24,31,.06)"
    radius_xs: int = 8
    radius_sm: int = 12
    radius_md: int = 16
    radius_pill: int = 999
    mobile_width: int = 460
    space_2: int = 2
    space_4: int = 4
    space_8: int = 8
    space_12: int = 12
    space_16: int = 16
    space_20: int = 20
    space_24: int = 24
    space_32: int = 32
    # Type scale (px). Desktop values match MVP 1 exactly.
    font_xs: int = 13
    font_sm: int = 15
    font_base: int = 17
    font_md: int = 19
    font_lg: int = 23
    font_xl: int = 27
    font_2xl: int = 30
    # Small-print scale — every size a component may render must come from here
    # (no inline px literals in components; enforced by tests/components/test_tokens.py).
    font_2xs: int = 9  # ABCDE chip captions / tier pills
    chip_font: int = 10  # case chips, scan-row labels, ABCDE letter labels
    pill_font: int = 12  # urgency pill, row dates
    stat_font: int = 18  # ABCDE values, empty-state titles
    # Touch
    touch_min: int = 48  # min button height (px)
    font_family: str = "'Plus Jakarta Sans', Inter, system-ui, sans-serif"
    # Brand (display only — internal identifiers, paths and env vars stay "dermascan").
    brand_name: str = "E.P.I.V.U.E."
    brand_tagline: str = "Screening aid — not a diagnosis"
    # Aliases used by older components (plan naming: type_*)
    type_xs: int = 13
    type_sm: int = 15
    type_base: int = 17
    type_md: int = 19
    type_lg: int = 23
    type_xl: int = 27
    type_2xl: int = 30


def _seven_inch(base: Tokens) -> Tokens:
    """1024x600 (or 800x480) panel at surf zoom 1.0: wider frame, ~x1.15 type, 56px targets."""
    return replace(
        base,
        mobile_width=744,  # fits 800-wide minus padding; centers on 1024
        font_xs=15,
        font_sm=17,
        font_base=19,
        font_md=22,
        font_lg=26,
        font_xl=30,
        font_2xl=34,
        font_2xs=12,
        chip_font=13,
        pill_font=15,
        stat_font=21,
        touch_min=56,
        type_xs=15,
        type_sm=17,
        type_base=19,
        type_md=22,
        type_lg=26,
        type_xl=30,
        type_2xl=34,
    )


def get_tokens(profile: str | None = None) -> Tokens:
    """Resolve tokens for a display profile (default: env SKIN_DISPLAY or desktop)."""
    p = (profile or os.environ.get("SKIN_DISPLAY", "desktop")).strip().lower()
    base = Tokens()
    if p == "7in":
        return _seven_inch(base)
    return base


@dataclass(frozen=True)
class EvolutionThresholds:
    diam_stable_mm: float = 0.5
    diam_watch_mm: float = 2.0
    de_stable: float = 8.0
    de_watch: float = 18.0
    border_change_suspicious: float = 0.5


TOKENS = get_tokens()
EVOLUTION = EvolutionThresholds()
