from theme.tokens import EVOLUTION, TOKENS, Tokens


def test_tokens():
    # Light clinical theme: white background, near-black text.
    assert TOKENS.bg == "#FFFFFF"
    assert TOKENS.text == "#14181F"
    assert Tokens().mobile_width == 460
    assert TOKENS.space_24 == 24
    assert EVOLUTION.diam_watch_mm == 2.0


def test_design_tokens_present():
    # Refresh adds a medical accent, elevation shadows, and risk tints.
    assert TOKENS.teal.startswith("#")
    assert "rgba" in TOKENS.shadow_sm
    assert TOKENS.success_tint.startswith("#")
