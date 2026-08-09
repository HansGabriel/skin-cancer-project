"""The content gate: does this photo show a skin spot at all?

The two failures this guards against are opposite in direction:
  1. junk input (a wall, a screen) sailing through to a confident "benign";
  2. a real lesion on deeply pigmented skin being rejected as "not skin",
     which is what the old light-skin-tuned HSV bounds did.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.lesion_gate import check_frame, lesion_contrast, skin_fraction

SIZE = 256

# Sampled across the Fitzpatrick range. The gate must accept all of them.
SKIN_TONES = {
    "very_light": (255, 224, 196),
    "light": (241, 194, 165),
    "medium": (198, 134, 66),
    "olive": (161, 102, 94),
    "brown": (120, 77, 48),
    "dark_brown": (89, 61, 47),
    "deep": (61, 42, 33),
}

NOT_SKIN = {
    "white_wall": (245, 245, 245),
    "grey_desk": (128, 128, 128),
    "blue_screen": (30, 80, 200),
    "green_grass": (60, 150, 60),
    "black": (10, 10, 10),
}


def _field(rgb: tuple[int, int, int], size: int = SIZE) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def _with_spot(base: np.ndarray, spot_rgb: tuple[int, int, int], radius: int = 40) -> np.ndarray:
    """Paint a circular lesion in the centre."""
    img = base.copy()
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    m = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    img[m] = spot_rgb
    return img


def _mask(size: int = SIZE, radius: int = 40) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    m[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= radius**2] = 255
    return m


@pytest.mark.parametrize("name,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_every_skin_tone_is_recognised_as_skin(name: str, rgb: tuple[int, int, int]) -> None:
    """The regression that mattered: dark skin used to fail the skin check."""
    assert skin_fraction(_field(rgb)) > 0.5, f"{name} not detected as skin"


@pytest.mark.parametrize("name,rgb", NOT_SKIN.items(), ids=list(NOT_SKIN))
def test_non_skin_surfaces_are_rejected(name: str, rgb: tuple[int, int, int]) -> None:
    assert skin_fraction(_field(rgb)) < 0.12, f"{name} wrongly detected as skin"


@pytest.mark.parametrize("name,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_lesion_on_any_skin_tone_passes_the_gate(name: str, rgb: tuple[int, int, int]) -> None:
    img = _with_spot(_field(rgb), (40, 26, 20))
    check = check_frame(img, _mask())
    assert check.is_lesion_photo, f"{name}: {check.reasons}"


def test_wall_is_stopped_before_the_classifier() -> None:
    check = check_frame(_field(NOT_SKIN["white_wall"]), _mask())
    assert not check.is_lesion_photo
    assert "No skin" in check.reasons[0]


def test_bare_skin_with_no_spot_is_stopped() -> None:
    """Flat skin must not be screened — there is nothing on it to screen."""
    check = check_frame(_field(SKIN_TONES["light"]), _mask())
    assert not check.is_lesion_photo
    assert "plain skin" in check.reasons[0]


def test_missing_mask_reports_no_spot_not_no_skin() -> None:
    check = check_frame(_field(SKIN_TONES["medium"]), None)
    assert not check.is_lesion_photo
    assert "No spot" in check.reasons[0]


def test_spot_filling_the_frame_is_stopped() -> None:
    """Camera pressed right against the skin: the spot has no border left.

    The radius leaves the corners as visible skin on purpose — a frame with
    *no* skin in it at all is a different failure ("no skin found"), and this
    test is about the framing message.
    """
    radius = SIZE // 2
    img = _with_spot(_field(SKIN_TONES["light"]), (40, 26, 20), radius=radius)
    check = check_frame(img, _mask(radius=radius))
    assert not check.is_lesion_photo
    assert "whole photo" in check.reasons[0]


def test_speck_of_noise_is_stopped() -> None:
    img = _with_spot(_field(SKIN_TONES["light"]), (40, 26, 20), radius=3)
    check = check_frame(img, _mask(radius=3))
    assert not check.is_lesion_photo
    assert "too small" in check.reasons[0]


def test_contrast_separates_a_real_spot_from_flat_skin() -> None:
    skin = _field(SKIN_TONES["medium"])
    assert lesion_contrast(_with_spot(skin, (40, 26, 20)), _mask()) > 5.0
    assert lesion_contrast(skin, _mask()) < 1.0


def test_reasons_are_plain_language() -> None:
    """No jargon in anything a visitor can read."""
    jargon = ("mask", "segmentation", "HSV", "YCrCb", "Lab", "threshold", "fraction", "pixel")
    checks = [
        check_frame(_field(NOT_SKIN["grey_desk"]), _mask()),
        check_frame(_field(SKIN_TONES["light"]), _mask()),
        check_frame(_field(SKIN_TONES["light"]), None),
    ]
    for c in checks:
        for reason in c.reasons:
            assert not any(j.lower() in reason.lower() for j in jargon), reason


def _through_jpeg(rgb: np.ndarray, quality: int = 92) -> np.ndarray:
    """Round-trip through the exact encode the upload path uses."""
    import io

    from PIL import Image

    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    with Image.open(io.BytesIO(png.tobytes())) as im:
        rgb_im = im.convert("RGB")
    buf = io.BytesIO()
    rgb_im.save(buf, format="JPEG", quality=quality)
    decoded = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _noisy(rgb: tuple[int, int, int], amp: int = 30, size: int = 320) -> np.ndarray:
    """A real surface has texture; a flat colour block is not a fair test."""
    rng = np.random.default_rng(3)
    img = _field(rgb, size).astype(np.int16)
    img += rng.integers(-amp, amp, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "name,rgb",
    [("grey desk", (120, 120, 125)), ("white wall", (240, 240, 240)), ("dark grey", (70, 70, 72))],
)
def test_neutral_surfaces_stay_rejected_after_jpeg_encoding(name, rgb) -> None:
    """JPEG chroma quantisation drags neutral pixels toward the skin box.

    Tested through the real encode because raw-array tests miss it entirely: a
    grey desk measured 5% skin as an array and 15% after JPEG — over the
    threshold — which is how a photo of a desk reached stage 2 claiming to be
    skin.
    """
    assert skin_fraction(_through_jpeg(_noisy(rgb))) < 0.12, name


@pytest.mark.parametrize("name,rgb", SKIN_TONES.items(), ids=list(SKIN_TONES))
def test_skin_survives_jpeg_encoding_on_every_tone(name, rgb) -> None:
    assert skin_fraction(_through_jpeg(_noisy(rgb))) > 0.5, name
