import re
from pathlib import Path

from theme.tokens import EVOLUTION, TOKENS, Tokens, get_tokens

COMPONENTS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "components"


def test_desktop_tokens_defaults():
    t = get_tokens("desktop")
    # Light clinical theme: white background, near-black text.
    assert t.bg == "#FFFFFF"
    assert t.text == "#131A22"
    # Landscape two-pane frame — the kiosk panel is 1024x600, not a tall phone.
    assert t.mobile_width == 1040
    assert t.font_base == 16
    assert t.touch_min == 48
    assert t.space_24 == 24
    assert EVOLUTION.diam_watch_mm == 2.0


def test_design_tokens_present():
    # Refresh adds a medical accent, elevation shadows, and risk tints.
    assert TOKENS.teal.startswith("#")
    assert "rgba" in TOKENS.shadow_sm
    assert TOKENS.success_tint.startswith("#")


def test_brand_tokens():
    # EPIVUE display branding; internals stay "dermascan".
    assert TOKENS.brand_name == "E.P.I.V.U.E."
    assert "not a diagnosis" in TOKENS.brand_tagline.lower()
    assert TOKENS.violet == "#16233A"  # instrument navy, the primary action colour


def test_instrument_palette_present():
    """The dark field and its reticle are the design's signature elements."""
    assert TOKENS.field == "#0B1220"
    assert TOKENS.reticle == "#3DDBD9"
    assert TOKENS.field_ink.startswith("#")


def test_no_green_reassurance_colour_for_verdicts():
    """A clean screening result must not be styled as an all-clear.

    services.verdict maps low_concern to the "neutral" tone precisely so it
    renders in plain ink; if a green ever becomes a verdict tone, the UI starts
    promising safety it cannot promise.
    """
    from services.format import tone_colors

    ink, fill = tone_colors("neutral")
    assert ink == TOKENS.text
    assert fill == TOKENS.surface


def test_7in_profile_scales_up():
    t = get_tokens("7in")
    d = get_tokens("desktop")
    assert t.mobile_width == 1000
    assert t.touch_min >= 56
    # Every font token must be at least as large as desktop.
    for name in ("font_xs", "font_sm", "font_base", "font_md", "font_lg",
                 "font_xl", "font_2xl", "font_2xs", "chip_font", "pill_font", "stat_font"):
        assert getattr(t, name) >= getattr(d, name), name
    # Nothing on the 7" panel may render below 12px physical.
    assert t.font_2xs >= 12
    assert t.chip_font >= 12
    # type_* aliases stay in sync with font_*.
    assert t.type_base == t.font_base
    assert t.type_2xl == t.font_2xl


def test_unknown_profile_falls_back_to_desktop():
    assert get_tokens("weird") == Tokens()
    assert TOKENS in (get_tokens("desktop"), get_tokens("7in"))


def test_no_hardcoded_font_px_in_components():
    """All component font sizes must come from tokens (display-profile scaling)."""
    offenders = []
    for py in COMPONENTS_DIR.glob("*.py"):
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"font-size:\s*\d+px", line):
                offenders.append(f"{py.name}:{n}: {line.strip()}")
    assert not offenders, "hardcoded font-size px found:\n" + "\n".join(offenders)
