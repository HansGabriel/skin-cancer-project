"""The ABCDE numbers, said in words a visitor can check against their own skin.

The measurements used to appear only in the staff panel, so the result screen
told people *what to do* without ever telling them *what was seen*. These lines
close that gap, and they are generated from the real measurements rather than
written per tone — a spot measured at 17 mm has to say 17 mm.
"""

from __future__ import annotations

import pytest

from services.verdict import ScanSign, scan_signs, signs_sentence


def _abcde(a_tier=0, b_tier=0, colours=1, mm=4.0, d_tier=0, e=None):
    return {
        "A": {"value": 0.1, "tier": a_tier, "verdict": "normal"},
        "B": {"value": 1.5, "tier": b_tier, "verdict": "normal"},
        "C": {"value": colours, "tier": 0 if colours <= 1 else 1, "verdict": "normal"},
        "D": {"value": mm, "tier": d_tier, "verdict": "normal"},
        "E": e or {"value": None, "tier": 0, "verdict": "needs history"},
    }


def test_no_abcde_gives_no_signs() -> None:
    assert scan_signs(None) == ()
    assert scan_signs({}) == ()
    assert signs_sentence(None) == ""


def test_border_colour_and_size_are_always_reported() -> None:
    """The three a person can check against the photo in front of them."""
    signs = scan_signs(_abcde())
    assert len(signs) == 3
    assert all(isinstance(s, ScanSign) for s in signs)
    text = " | ".join(s.text for s in signs)
    assert "Edges" in text
    assert "colour" in text
    assert "mm across" in text


@pytest.mark.parametrize(
    "tier,fragment",
    [(0, "smooth and even"), (1, "slightly uneven"), (2, "uneven and blurred")],
)
def test_border_wording_tracks_its_tier(tier: int, fragment: str) -> None:
    signs = scan_signs(_abcde(b_tier=tier))
    border = next(s for s in signs if "Edges" in s.text)
    assert fragment in border.text
    assert border.tier == tier


@pytest.mark.parametrize(
    "colours,fragment", [(1, "One even colour"), (2, "Two different colours"), (4, "Three or more")]
)
def test_colour_wording_counts_what_was_measured(colours: int, fragment: str) -> None:
    signs = scan_signs(_abcde(colours=colours))
    assert any(fragment in s.text for s in signs)


def test_asymmetry_appears_only_when_it_stands_out() -> None:
    """Otherwise it adds a "nothing to report" line to a 600px panel."""
    for tier in (0, 1):
        assert not any("halves" in s.text for s in scan_signs(_abcde(a_tier=tier)))
    assert any("halves" in s.text for s in scan_signs(_abcde(a_tier=2)))


def test_eraser_comparison_only_above_the_six_millimetre_cue() -> None:
    """6 mm is the classic ABCDE "D" cue and where services.abcde puts tier 2,
    so the pencil-eraser line is the same threshold said in a picturable object,
    not a second rule."""
    small = next(s for s in scan_signs(_abcde(mm=4.0, d_tier=0)) if "mm across" in s.text)
    assert "eraser" not in small.text
    big = next(s for s in scan_signs(_abcde(mm=17.4, d_tier=2)) if "mm across" in s.text)
    assert "eraser" in big.text
    assert "17 mm" in big.text


def test_evolving_is_silent_without_history_and_speaks_with_it() -> None:
    """A first scan cannot know whether the spot changed — it must not imply it did."""
    assert not any("last time" in s.text for s in scan_signs(_abcde()))
    grown = _abcde(e={"value": 2.4, "tier": 2, "verdict": "changing"})
    assert any("bigger than last time" in s.text for s in scan_signs(grown))
    same = _abcde(e={"value": 0.0, "tier": 0, "verdict": "stable"})
    assert any("same size as last time" in s.text for s in scan_signs(same))


def test_sentence_reads_as_prose_and_keeps_every_measurement() -> None:
    """The assistant leads with this, so it has to be a sentence, not a list."""
    sentence = signs_sentence(_abcde(a_tier=2, b_tier=2, colours=2, mm=17.4, d_tier=2))
    assert sentence.endswith(".")
    assert ", and " in sentence
    assert "17 mm" in sentence
    # Composed from the numbers in Python — never generated — so it cannot
    # report a size the scan did not measure (docs/ASSISTANT.md).
    for sign in scan_signs(_abcde(a_tier=2, b_tier=2, colours=2, mm=17.4, d_tier=2)):
        assert sign.text.lower()[:12] in sentence.lower()


def test_signs_survive_a_partial_abcde_dict() -> None:
    """Segmentation can fail per-letter; a missing value must not raise."""
    assert scan_signs({"B": {"tier": 1}}) is not None
    assert scan_signs({"D": {"value": None, "tier": 0}}) == ()
    assert scan_signs({"C": {"value": 2, "tier": 1}})[0].text == "Two different colours inside"
