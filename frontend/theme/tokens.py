"""Single source of truth for DermaScan v2 design tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokens:
    # Clean clinical light palette — high contrast for the cheap 480x320 LCD.
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
    # Type scale bumped up — surf runs at 0.5 zoom on the LCD, halving on-screen px.
    font_xs: int = 13
    font_sm: int = 15
    font_base: int = 17
    font_md: int = 19
    font_lg: int = 23
    font_xl: int = 27
    font_2xl: int = 30
    font_family: str = "'Plus Jakarta Sans', Inter, system-ui, sans-serif"
    # Aliases used by older components (plan naming: type_*)
    type_xs: int = 13
    type_sm: int = 15
    type_base: int = 17
    type_md: int = 19
    type_lg: int = 23
    type_xl: int = 27
    type_2xl: int = 30


@dataclass(frozen=True)
class EvolutionThresholds:
    diam_stable_mm: float = 0.5
    diam_watch_mm: float = 2.0
    de_stable: float = 8.0
    de_watch: float = 18.0
    border_change_suspicious: float = 0.5


TOKENS = Tokens()
EVOLUTION = EvolutionThresholds()
