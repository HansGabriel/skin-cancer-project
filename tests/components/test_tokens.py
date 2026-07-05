import re
from pathlib import Path

from theme.tokens import EVOLUTION, TOKENS, Tokens, get_tokens

COMPONENTS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "components"


def test_desktop_tokens_defaults():
    t = get_tokens("desktop")
    # Light clinical theme: white background, near-black text.
    assert t.bg == "#FFFFFF"
    assert t.text == "#14181F"
    assert t.mobile_width == 460
    assert t.font_base == 17
    assert t.touch_min == 48
    assert t.space_24 == 24
    assert EVOLUTION.diam_watch_mm == 2.0


def test_design_tokens_present():
    # Refresh adds a medical accent, elevation shadows, and risk tints.
    assert TOKENS.teal.startswith("#")
    assert "rgba" in TOKENS.shadow_sm
    assert TOKENS.success_tint.startswith("#")


def test_7in_profile_scales_up():
    t = get_tokens("7in")
    d = get_tokens("desktop")
    assert t.mobile_width == 744
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
