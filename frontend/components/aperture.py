"""The aperture — a dermatoscope's circular field, and this app's one bold element.

Every state of a scan is the same object seen differently:

===============  ==========================================================
waiting          dark field, cyan reticle ticks, a line telling you what to do
holding a photo  the photo clipped inside the field, reticle still showing
judged           a coloured ring around it, with the five ABCDE marks on the rim
nothing found    a dashed ring — the instrument found nothing to measure
===============  ==========================================================

Drawn as inline SVG rather than stacked divs: one element, no blurred shadows,
and it scales to any panel size — all of which matter on a Raspberry Pi 4.
"""

from __future__ import annotations

import base64
import html
import math
from typing import Any

import streamlit as st

from services.format import tone_colors
from services.verdict import UIVerdict
from theme.tokens import TOKENS as T

_LETTERS = ("A", "B", "C", "D", "E")
# ABCDE severity → rim colour. Tier 0 is deliberately colourless, for the same
# reason a clean verdict is (see theme.tokens): a green tick invites people to
# read "safe", and five green ticks read as a clean bill of health.
_TIER_INK = {0: T.text_muted, 1: T.melanin, 2: T.erythema}

_FIELD_R = 46.0  # the dark field
# The specimen inside it. The photo renders at (2 * _PHOTO_R / 104) *
# aperture_px, so on the 7" kiosk (aperture_px 225, device pixel ratio 1) the
# old 38.0 put a 1024px capture into ~164 real pixels — small enough that people
# asked whether that thumbnail was what the model saw. It is not: the classifier
# always receives the original capture bytes (services/pipeline.py).
#
# 41.0 gives ~177px and is the most that fits: the ABCDE marks are stroked 3
# units wide centred on _RIM_R, so anything past ~41.5 puts the photo edge under
# the marks and the rim stops reading as a rim. Making the circle itself bigger
# is a tokens change (aperture_px), and that trades against the 600px height
# budget the results screen is built to fit without scrolling — verify on the
# panel before raising it.
_PHOTO_R = 41.0
_RIM_R = 42.0  # graticule + ABCDE marks live on this radius


def _data_uri(image: Any) -> str | None:
    """Encode bytes or an RGB array as a data URI for embedding in the SVG."""
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(image)).decode()
    try:
        import cv2
        import numpy as np

        arr = np.asarray(image)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:  # noqa: BLE001 — the aperture must never break a result screen
        return None


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
    """SVG path for an arc, angles measured clockwise from 12 o'clock."""
    def point(deg: float) -> tuple[float, float]:
        rad = math.radians(deg - 90.0)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    x1, y1 = point(start_deg)
    x2, y2 = point(end_deg)
    large = 1 if (end_deg - start_deg) % 360 > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f}"


def _abcde_marks(abcde: dict[str, Any] | None) -> str:
    """Five arcs on the rim, one per warning sign, plus its letter."""
    if not abcde:
        return ""
    parts = []
    span = 360.0 / len(_LETTERS)
    for i, letter in enumerate(_LETTERS):
        d = abcde.get(letter) or {}
        if d.get("verdict") == "needs history":
            ink, opacity = T.info, 0.9
        else:
            ink, opacity = _TIER_INK.get(int(d.get("tier", 0) or 0), T.outline), 0.9
        start = i * span + 3.0
        end = (i + 1) * span - 3.0
        parts.append(
            f'<path d="{_arc(50, 50, _RIM_R, start, end)}" fill="none" '
            f'stroke="{ink}" stroke-width="3" stroke-linecap="round" opacity="{opacity}"/>'
        )
        mid = math.radians((start + end) / 2.0 - 90.0)
        lx = 50 + (_RIM_R - 6.5) * math.cos(mid)
        ly = 50 + (_RIM_R - 6.5) * math.sin(mid)
        parts.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" fill="{T.field_ink}" font-size="5.5" '
            f'font-weight="700" text-anchor="middle" dominant-baseline="central" '
            f'font-family="{T.font_family}">{letter}</text>'
        )
    return "".join(parts)


def _label(verdict: UIVerdict | None, hint: str | None) -> str:
    """Accessible name for the aperture, escaped for attribute context."""
    return html.escape((verdict.headline if verdict else hint) or "camera view", quote=True)


def _reticle() -> str:
    """Four cardinal ticks — the graticule that says 'instrument, at rest'."""
    ticks = []
    for deg in (0, 90, 180, 270):
        rad = math.radians(deg - 90.0)
        x1 = 50 + (_RIM_R - 3) * math.cos(rad)
        y1 = 50 + (_RIM_R - 3) * math.sin(rad)
        x2 = 50 + (_RIM_R + 2) * math.cos(rad)
        y2 = 50 + (_RIM_R + 2) * math.sin(rad)
        ticks.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{T.reticle}" stroke-width="1" opacity=".8"/>'
        )
    return "".join(ticks)


def aperture_svg(
    *,
    image: Any = None,
    hint: str | None = None,
    verdict: UIVerdict | None = None,
    abcde: dict[str, Any] | None = None,
    dashed: bool = False,
) -> str:
    """Build the aperture markup. See the module docstring for the four states."""
    uri = _data_uri(image)
    ring_ink = tone_colors(verdict.tone)[0] if verdict else T.reticle
    # A neutral verdict gets a quiet rim, not a bright one. Drawing the reticle
    # cyan at verdict weight made "nothing stood out" the loudest thing on the
    # screen — the opposite of what a colourless result is meant to convey.
    quiet = bool(verdict and verdict.tone == "neutral")
    if quiet:
        ring_ink = T.text_muted

    photo = ""
    if uri:
        photo = (
            f'<clipPath id="ds-clip"><circle cx="50" cy="50" r="{_PHOTO_R}"/></clipPath>'
            f'<image href="{uri}" x="{50 - _PHOTO_R}" y="{50 - _PHOTO_R}" '
            f'width="{_PHOTO_R * 2}" height="{_PHOTO_R * 2}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#ds-clip)"/>'
        )

    hint_text = ""
    if hint and not uri:
        words, lines, current = hint.split(), [], ""
        for w in words:
            if len(current) + len(w) + 1 > 22:
                lines.append(current)
                current = w
            else:
                current = f"{current} {w}".strip()
        lines.append(current)
        start_y = 50 - (len(lines) - 1) * 3.4
        hint_text = "".join(
            f'<text x="50" y="{start_y + i * 6.8:.2f}" fill="{T.field_ink}" font-size="5" '
            f'text-anchor="middle" dominant-baseline="central" opacity=".85" '
            f'font-family="{T.font_family}">{line}</text>'
            for i, line in enumerate(lines)
        )

    # The thick outer ring is the verdict. When the scan was stopped there is no
    # verdict to draw, so it is omitted entirely and the rim goes finely dashed:
    # "the instrument found nothing to measure", not "here is a blue finding".
    # (Dashing a thick ring turns it into a sunburst, which reads as decoration.)
    rim_dash = ' stroke-dasharray="1.5 2"' if dashed else ""
    outer = ""
    if verdict and not dashed:
        outer = (
            f'<circle cx="50" cy="50" r="{_FIELD_R + 1.5}" fill="none" stroke="{ring_ink}" '
            f'stroke-width="{1.5 if quiet else 3}" opacity="{.45 if quiet else .95}"/>'
        )
    return (
        f'<div class="ds-aperture-wrap" style="max-width:{T.aperture_px}px;margin:0 auto">'
        f'<svg viewBox="-2 -2 104 104" width="100%" role="img" '
        f'aria-label="{_label(verdict, hint)}">'
        f'<circle cx="50" cy="50" r="{_FIELD_R}" fill="{T.field}"/>'
        f"{photo}{hint_text}"
        f'<circle cx="50" cy="50" r="{_RIM_R}" fill="none" stroke="{ring_ink}" '
        f'stroke-width="{1 if dashed else 1.5}" opacity="{.55 if dashed else .7}"{rim_dash}/>'
        f"{_reticle() if not abcde else ''}"
        f"{_abcde_marks(abcde)}"
        f"{outer}"
        f"</svg></div>"
    )


def render_aperture(**kwargs: Any) -> None:
    """Render the aperture into the page."""
    st.markdown(aperture_svg(**kwargs), unsafe_allow_html=True)


def live_aperture_html(stream_url: str) -> str:
    """The same field around a live camera stream.

    A streaming ``<img>`` cannot go inside the SVG, so this variant is built
    from CSS instead. It lives here rather than in the camera view so the app's
    one signature element has a single home: two implementations in two files
    is how the still and live views drift apart.
    """
    ticks = "".join(
        f'<span class="ds-tick ds-tick-{side}"></span>' for side in ("t", "b", "l", "r")
    )
    return (
        f'<div class="ds-aperture"><img src="{html.escape(stream_url, quote=True)}" alt="live camera view"/>'
        f'<div class="ds-aperture-ring"></div>{ticks}</div>'
    )


def render_live_aperture(stream_url: str) -> None:
    st.markdown(live_aperture_html(stream_url), unsafe_allow_html=True)
