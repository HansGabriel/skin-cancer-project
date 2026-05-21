"""Single source of truth for DermaScan v2 design tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokens:
    bg: str = "#0B0B14"
    bg_elev: str = "#15131F"
    surface: str = "#1B1830"
    outline: str = "#2A2540"
    text: str = "#F5F2FF"
    text_muted: str = "#B6B0CC"
    violet: str = "#B58CF0"
    violet_strong: str = "#6C4AB6"
    success: str = "#34D399"
    warning: str = "#F59E0B"
    urgent: str = "#EF4444"
    info: str = "#60A5FA"
    radius_xs: int = 8
    radius_sm: int = 12
    radius_md: int = 16
    radius_pill: int = 999
    mobile_width: int = 440
    space_2: int = 2
    space_4: int = 4
    space_8: int = 8
    space_12: int = 12
    space_16: int = 16
    space_20: int = 20
    space_24: int = 24
    space_32: int = 32
    font_xs: int = 11
    font_sm: int = 13
    font_base: int = 14
    font_md: int = 16
    font_lg: int = 20
    font_xl: int = 24
    font_2xl: int = 32
    font_family: str = "'Plus Jakarta Sans', Inter, system-ui, sans-serif"
    # Aliases used by older components (plan naming: type_*)
    type_xs: int = 11
    type_sm: int = 13
    type_base: int = 14
    type_md: int = 16
    type_lg: int = 20
    type_xl: int = 24
    type_2xl: int = 32


@dataclass(frozen=True)
class EvolutionThresholds:
    diam_stable_mm: float = 0.5
    diam_watch_mm: float = 2.0
    de_stable: float = 8.0
    de_watch: float = 18.0
    border_change_suspicious: float = 0.5


TOKENS = Tokens()
EVOLUTION = EvolutionThresholds()
