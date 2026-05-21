from theme.tokens import EVOLUTION, TOKENS, Tokens


def test_tokens():
    assert TOKENS.bg == "#0B0B14"
    assert Tokens().mobile_width == 440
    assert TOKENS.space_24 == 24
    assert EVOLUTION.diam_watch_mm == 2.0
