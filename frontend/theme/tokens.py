"""Single source of truth for the design tokens.

The look is built around one idea: **the instrument is the interface**. A single
dark circular field with a cyan reticle — a dermatoscope's view — carries every
state of a scan (waiting, captured, judged, empty). Everything around it stays
quiet: white paper, hairline rules, near-black text.

Palette notes, because two choices here are deliberate and easy to "fix" by
accident:

- Colours are named after what a dermatoscope actually shows — ``melanin``
  brown, ``erythema`` red, ``veil`` blue — not after traffic lights.
- A clean result has **no colour at all**. Green reads as "you are fine", and
  false reassurance is the one failure mode that hurts people here (see
  ``services.verdict.LOW_RISK_SAFETY_LINE``). Clean results get plain ink and
  the safety line instead of a reassuring badge.

Display profiles (env ``SKIN_DISPLAY``, read once at import):
- ``desktop`` (default) — PC / Streamlit browser.
- ``7in`` — the 1024x600 landscape touchscreen kiosk at zoom 1.0: wider frame,
  larger type, >=56px touch targets.

Set the env var BEFORE ``streamlit run`` (launch_kiosk.sh exports SKIN_DISPLAY=7in).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Tokens:
    # --- Paper and ink ---------------------------------------------------
    bg: str = "#FFFFFF"
    bg_elev: str = "#F5F7FA"
    surface: str = "#EDF1F7"
    outline: str = "#E2E6EE"  # hairline rule
    text: str = "#131A22"
    text_muted: str = "#5B6675"

    # --- The instrument --------------------------------------------------
    field: str = "#0B1220"  # the aperture's dark field — the ONE dark element
    field_ink: str = "#E8EEF6"  # text/marks on the dark field
    reticle: str = "#3DDBD9"  # graticule + focus marks. Never a button.

    # --- Verdict accents, after dermoscopic signs -------------------------
    erythema: str = "#C0362C"  # urgent
    melanin: str = "#7A4B2A"  # needs a check
    veil: str = "#4A6FB0"  # unsure (the blue-white veil)

    # --- Legacy aliases --------------------------------------------------
    # Existing components refer to these names; they now resolve to the new
    # palette so one edit here restyles the whole app.
    violet: str = "#16233A"  # primary action — instrument navy, not purple
    violet_strong: str = "#0B1220"
    violet_tint: str = "#EDF1F7"
    teal: str = "#3DDBD9"
    teal_strong: str = "#0F7A6C"
    success: str = "#0F7A6C"  # non-verdict "good" only (ABCDE tiers)
    warning: str = "#7A4B2A"
    urgent: str = "#C0362C"
    info: str = "#4A6FB0"
    success_tint: str = "#E6F2F0"
    warning_tint: str = "#F6EEE7"
    urgent_tint: str = "#FAECEA"
    info_tint: str = "#ECF1F9"

    # --- Depth -----------------------------------------------------------
    # Restrained on purpose: shadows are a compositing cost on the Pi 4 GPU, so
    # only the aperture and raised buttons get one.
    shadow_sm: str = "0 1px 2px rgba(19,26,34,.06)"
    shadow_md: str = "0 8px 24px rgba(11,18,32,.14)"

    radius_xs: int = 6
    radius_sm: int = 10
    radius_md: int = 14
    radius_pill: int = 999

    # --- Layout ----------------------------------------------------------
    # Landscape two-pane: the 1024x600 panel is only 600px tall, so stacking
    # the photo above the verdict forces scrolling on a kiosk nobody scrolls.
    mobile_width: int = 1040  # frame width (name kept for existing components)
    pane_gap: int = 28
    # Sized so a whole results screen — bar, aperture, verdict, actions, staff
    # details, disclaimer and nav — fits 600px without scrolling.
    aperture_px: int = 250  # diameter of the instrument circle

    space_2: int = 2
    space_4: int = 4
    space_8: int = 8
    space_12: int = 12
    space_16: int = 16
    space_20: int = 20
    space_24: int = 24
    space_32: int = 32

    # --- Type ------------------------------------------------------------
    font_2xs: int = 10
    chip_font: int = 11
    pill_font: int = 12
    font_xs: int = 13
    font_sm: int = 15
    font_base: int = 16
    font_md: int = 19
    font_lg: int = 24
    font_xl: int = 30
    font_2xl: int = 38
    stat_font: int = 18
    verdict_font: int = 34  # the one headline
    advice_font: int = 20  # the one serif sentence

    touch_min: int = 48

    # Archivo carries the UI and the data (tabular figures). Source Serif sets
    # exactly one thing: the sentence of advice addressed to a person — the
    # machine speaks in grotesque, the advice to a human is set in serif.
    font_family: str = "'Archivo', 'Helvetica Neue', Arial, sans-serif"
    serif_family: str = "'Source Serif 4', Georgia, 'Times New Roman', serif"

    # Brand (display only — internal identifiers, paths and env vars stay "dermascan").
    brand_name: str = "E.P.I.V.U.E."
    brand_tagline: str = "Skin check — not a diagnosis"

    # Aliases used by older components (plan naming: type_*)
    type_xs: int = 13
    type_sm: int = 15
    type_base: int = 16
    type_md: int = 19
    type_lg: int = 24
    type_xl: int = 30
    type_2xl: int = 38


def _seven_inch(base: Tokens) -> Tokens:
    """1024x600 landscape panel at zoom 1.0: wider frame, larger type, 56px targets."""
    return replace(
        base,
        mobile_width=1000,
        aperture_px=225,
        pane_gap=24,
        font_2xs=12,
        chip_font=13,
        pill_font=14,
        font_xs=15,
        font_sm=17,
        font_base=18,
        font_md=21,
        font_lg=26,
        font_xl=32,
        font_2xl=40,
        stat_font=21,
        verdict_font=36,
        advice_font=22,
        touch_min=56,
        type_xs=15,
        type_sm=17,
        type_base=18,
        type_md=21,
        type_lg=26,
        type_xl=32,
        type_2xl=40,
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
