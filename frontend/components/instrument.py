"""The dark band: the instrument, its readout, and the nav keys at its foot.

This is the left 45% of the shell and it is present on every screen. It holds
three things, top to bottom:

1. **Who this is** — the brand mark, the time, and whether the device is offline.
2. **The instrument** — the aperture, showing whichever state the current route
   is in (at rest, live, holding a photo, working, judged), plus one caption
   line underneath that says what the marks on the rim mean.
3. **The nav keys** — flat, no fill, a cyan cap on the live one.

The band reads its state from the route and ``last_result`` rather than taking
them as arguments. That is deliberate: the alternative is every view passing the
photo and verdict back up to ``app.py`` so it can pass them down again, and the
two would drift the first time a screen forgot to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from components.aperture import render_aperture, render_live_aperture
from components.bottom_nav import render_nav
from services.format import field_ink
from services.kiosk import is_kiosk
from services.pipeline import APP_VERSION
from services.verdict import UIVerdict, resolve_verdict, retake_verdict
from theme.tokens import TOKENS as T

# Routes where the instrument is idle: nothing has been captured, so it shows
# the reticle at rest and an instruction rather than a stale photo.
_AT_REST = ("home", "settings")


def _clock() -> str:
    now = datetime.now()
    hour = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    return f"{hour}:{now.minute:02d}{ampm} · {now.day:02d} {now.strftime('%b')}"


def _render_header() -> None:
    offline = '<span class="ds-inst-pill">OFFLINE</span>' if is_kiosk() else ""
    st.markdown(
        '<div class="ds-inst-head">'
        f'<div class="ds-inst-brand"><span class="ds-inst-dot"></span>{T.brand_name}</div>'
        f'<div class="ds-inst-meta"><span>{_clock()}</span>{offline}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _abcde_legend() -> str:
    """What the five coloured arcs on the rim mean.

    Without this the rim is decoration. The wording stays plain — "the five
    things doctors look at" rather than asymmetry/border/colour/diameter/
    evolving, which is staff vocabulary and lives on the staff screen.
    """
    keys = (("Normal", 0), ("Borderline", 1), ("Stands out", 2))
    swatches = "".join(
        f'<span><span class="ds-inst-swatch" style="background:{field_ink(tier)}"></span>'
        f"{label}</span>"
        for label, tier in keys
    )
    return (
        '<div class="ds-inst-caption">A B C D E · the five things doctors look at</div>'
        f'<div class="ds-inst-legend">{swatches}</div>'
    )


def _timeline(count: int, title: str) -> str:
    """One node per saved scan, newest in reticle cyan."""
    nodes = []
    for i in range(min(count, 5)):
        if i:
            nodes.append('<span class="ds-inst-link"></span>')
        last = " is-last" if i == min(count, 5) - 1 else ""
        nodes.append(f'<span class="ds-inst-node{last}"></span>')
    return (
        f'<div class="ds-inst-caption">{title}</div>'
        f'<div class="ds-inst-track">{"".join(nodes)}</div>'
    )


def _standby() -> str:
    return f'<div class="ds-inst-caption">Ready · on-device · v{APP_VERSION}</div>'


def _result_verdict(pl: dict) -> UIVerdict | None:
    if pl.get("blocked") and not pl.get("scan_result"):
        return pl.get("verdict") or retake_verdict(pl.get("quality"))
    if pl.get("scan_result") is None:
        return None
    return pl.get("verdict") or resolve_verdict(pl.get("scan_result"), pl.get("quality"))


def _render_scan_state(pl: dict) -> str:
    """Aperture + caption for a finished scan. Returns the caption markup."""
    verdict = _result_verdict(pl)
    blocked = bool(pl.get("blocked")) and not pl.get("scan_result")
    abcde = None if blocked else pl.get("abcde")
    render_aperture(
        image=pl.get("rgb"),
        verdict=verdict,
        abcde=abcde,
        dashed=blocked,
    )
    return _abcde_legend() if abcde else ""


def _render_camera_state(kind: str) -> str:
    captured = st.session_state.get("capture_image_bytes")
    if captured:
        # Through the aperture component, which embeds the photo in its SVG. A
        # bare <div> wrapper would not work: Streamlit closes each st.markdown
        # block, so the image would land outside it.
        render_aperture(image=captured)
        return '<div class="ds-inst-caption">Photo taken · check the readings</div>'

    from services import pi_camera

    if pi_camera.AVAILABLE and kind in ("pi", "local"):
        cam = pi_camera.get_camera()
        if cam is not None:
            render_live_aperture(pi_camera.preview_url())
            return '<div class="ds-inst-caption">Hold still · fill the ring</div>'
    render_aperture(hint="Put the spot inside the ring")
    return '<div class="ds-inst-caption">Waiting for a photo</div>'


def _render_saved_state() -> str:
    from services.storage import get_storage

    store = get_storage()
    case_id = st.session_state.get("selected_case_id")
    if case_id:
        case = store.get_case(case_id)
        scans = store.list_scans(case_id) if case else []
        if case and scans:
            latest = scans[-1]
            try:
                photo = (store.root / latest.image_path).read_bytes()
            except OSError:
                # The instrument renders on every request, so a deleted or
                # unreadable image file here would take down every screen, not
                # just this one. history_view._thumb guards the same read.
                photo = None
            render_aperture(image=photo, hint=None if photo else "Photo unavailable")
            return _timeline(len(scans), f"{case.name} · {len(scans)} scans")
    render_aperture(hint="Saved spots stay on this device")
    return _standby()


def _render_body(route: str, kind: str) -> None:
    pl = st.session_state.get("last_result")
    caption = ""
    if route in _AT_REST:
        render_aperture(hint="Put the spot inside the ring")
        caption = _standby()
    elif route == "camera":
        caption = _render_camera_state(kind)
    elif route == "reading":
        render_aperture(image=st.session_state.get("capture_image_bytes"), working=True)
        caption = '<div class="ds-inst-caption">Reading · everything on this device</div>'
    elif route in ("results", "assistant", "staff") and isinstance(pl, dict):
        caption = _render_scan_state(pl)
    elif route in ("history", "folder", "case"):
        caption = _render_saved_state()
    else:
        render_aperture(hint="Put the spot inside the ring")
        caption = _standby()
    if caption:
        st.markdown(caption, unsafe_allow_html=True)


def render_instrument(route: str, *, kind: str = "local") -> None:
    """Draw the whole dark band for the current route."""
    _render_header()
    with st.container(key="epv-inst-body"):
        _render_body(route, kind)
    with st.container(key="epv-nav"):
        render_nav()


def render_disclaimer_strip() -> None:
    """The permanent safety line at the foot of the page band.

    Copy is duplicated from ``components.app_bar.render_disclaimer_footer`` on
    purpose — that function is what
    ``tests/components/test_no_dev_leaks.py::test_mandated_safety_copy_present``
    pins, and it is still the single source the wording is taken from.
    """
    st.markdown(
        '<div class="ds-foot"><span class="ds-foot-tag">Not a diagnosis</span>'
        '<span class="ds-foot-text">An educational screening aid. Always contact '
        "a qualified health professional.</span></div>",
        unsafe_allow_html=True,
    )


def render_head(eyebrow: str, title: str, lede: str = "", *, extra: Any = None) -> None:
    """The top of a page-band screen: small-caps eyebrow, headline, one line.

    An empty ``title`` emits no ``<h1>`` at all. The results screen passes one,
    because its headline is the verdict and that is rendered separately in the
    verdict's own colour — and an empty ``<h1>`` still carries its font-size and
    margins, which cost about 100px of a 600px panel and pushed the action
    buttons off the bottom of the screen.
    """
    chip = extra or ""
    head = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'gap:{T.space_12}px"><span class="ds-eyebrow">{eyebrow}</span>{chip}</div>'
        if chip
        else f'<p class="ds-eyebrow">{eyebrow}</p>'
    )
    title_html = f'<h1 class="ds-title">{title}</h1>' if title else ""
    lede_html = f'<p class="ds-lede">{lede}</p>' if lede else ""
    st.markdown(f"{head}{title_html}{lede_html}", unsafe_allow_html=True)
