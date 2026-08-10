"""Inject the global stylesheet into Streamlit.

Two constraints shape everything here:

**Fonts must load offline.** They are vendored in ``frontend/static/fonts`` and
served by Streamlit's static file server at ``app/static/...`` (enabled in
``.streamlit/config.toml``). The previous version pointed ``@font-face`` at a
``file://`` path, which browsers refuse to load from an ``http://`` page — so no
custom font ever loaded, and the kiosk silently fell back to whatever generic
sans the Pi had. Base64 data-URIs would work too but Streamlit re-sends this
stylesheet on every rerun; served as files, the browser caches them once.

**It has to stay fast on a Raspberry Pi 4.** No ``backdrop-filter`` (very slow
on the Pi GPU), no ``:has()`` (unreliable in the WebKitGTK build ``surf`` uses),
shadows only where they carry meaning, and one short transition.
"""

from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T

# Below this width the two panes stack. The 1024x600 kiosk never crosses it at
# zoom 1.0, and leaves headroom up to ~1.3x if a panel ever needs zooming.
_STACK_BP = 760

_FONT_DIR = "app/static/fonts"


def _font_faces() -> str:
    faces = [
        ("Archivo", 400, "archivo-400.woff2"),
        ("Archivo", 600, "archivo-600.woff2"),
        ("Archivo", 700, "archivo-700.woff2"),
        ("Source Serif 4", 400, "sourceserif-400.woff2"),
    ]
    return "".join(
        f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
        f"font-display:swap;src:url('{_FONT_DIR}/{file}') format('woff2');}}"
        for family, weight, file in faces
    )


def inject_global_css() -> None:
    st.markdown(
        f"""<style>
{_font_faces()}

/* ---------- Page ---------- */
[data-testid="stAppViewContainer"],.stApp{{background:{T.bg}!important;color:{T.text}!important;
  font-family:{T.font_family}!important;font-size:{T.font_base}px;
  font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased;}}
[data-testid="stSidebar"]{{background:{T.bg_elev}!important;border-right:1px solid {T.outline}!important;}}
[data-testid="stHeader"]{{background:transparent!important;}}
.stApp a{{color:{T.violet};}}
h1,h2,h3,h4,h5{{font-family:{T.font_family}!important;color:{T.text}!important;
  letter-spacing:-.015em;font-weight:700;}}
/* Streamlit sets its own font-family on markdown/caption/widget subtrees, which
   stops the .stApp family from being inherited. Without these selectors the
   headings pick up Archivo but every paragraph silently stays on Streamlit's
   default face. */
.stMarkdown,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,
.stCaption,label,input,select,textarea,[data-testid="stExpander"] summary,
[data-testid="stAlert"] *{{font-family:{T.font_family}!important;}}
/* The one exception, applied after and at higher specificity. */
.stMarkdown .ds-advice,[data-testid="stMarkdownContainer"] .ds-advice{{
  font-family:{T.serif_family}!important;}}

/* The kiosk panel is only 600px tall — every default gap has to earn its
   place or the answer ends up below the fold on a screen nobody scrolls.
   Width is set here rather than on .ds-mobile-frame: that wrapper emits its
   opening and closing div in separate st.markdown calls, so Streamlit closes
   each one and it never actually wraps anything. */
/* Fill the viewport height and centre the content in it. Screens here are
   short (600px) and the content is deliberately sparse, so left at auto
   height everything crowds into a band at the top with dead space below.
   min-height rather than height so a taller screen (Results) still grows and
   scrolls instead of being clipped. */
.block-container{{padding-top:{T.space_8}px!important;padding-bottom:{T.space_8}px!important;
  max-width:{T.mobile_width}px!important;
  min-height:100vh!important;display:flex!important;flex-direction:column!important;
  justify-content:center!important;}}
[data-testid="stVerticalBlock"]{{gap:{T.space_8}px!important;}}
[data-testid="stElementContainer"]:empty{{display:none!important;}}
hr{{margin:{T.space_8}px 0!important;}}

/* ---------- Buttons ---------- */
.stButton>button{{border-radius:{T.radius_sm}px!important;background:{T.bg}!important;
  color:{T.text}!important;border:1px solid {T.outline}!important;
  min-height:{T.touch_min}px!important;font-family:{T.font_family}!important;
  font-size:{T.font_base}px!important;font-weight:600!important;letter-spacing:.01em!important;
  padding:10px 18px!important;box-shadow:none!important;transition:border-color .15s ease;}}
.stButton>button:hover{{border-color:{T.violet}!important;color:{T.violet}!important;}}
.stButton>button:focus-visible{{outline:3px solid {T.reticle}!important;outline-offset:2px!important;}}
.stButton>button[kind="primary"]{{background:{T.violet}!important;color:#fff!important;
  border:1px solid {T.violet}!important;box-shadow:{T.shadow_sm}!important;}}
.stButton>button[kind="primary"]:hover{{background:{T.violet_strong}!important;color:#fff!important;}}
.stButton>button:disabled{{opacity:.45!important;}}

/* ---------- Structure ---------- */
.ds-card{{background:{T.bg};border:1px solid {T.outline};border-radius:{T.radius_md}px;
  padding:{T.space_16}px;margin:{T.space_12}px 0;}}
.ds-section-title{{font-size:{T.font_2xs}px;letter-spacing:.14em;text-transform:uppercase;
  color:{T.text_muted};font-weight:700;margin:0 0 {T.space_8}px;}}
.ds-rule{{border:0;border-top:1px solid {T.outline};margin:{T.space_16}px 0;}}
.ds-empty{{text-align:center;color:{T.text_muted};padding:{T.space_24}px;}}

/* ---------- The instrument: one dark circular field ---------- */
.ds-aperture{{position:relative;width:100%;max-width:{T.aperture_px}px;aspect-ratio:1/1;
  margin:0 auto;border-radius:50%;background:{T.field};overflow:hidden;
  box-shadow:{T.shadow_md};display:flex;align-items:center;justify-content:center;}}
.ds-aperture img{{width:100%;height:100%;object-fit:cover;display:block;}}
.ds-aperture-ring{{position:absolute;inset:0;border-radius:50%;pointer-events:none;
  border:2px solid {T.reticle}66;}}
.ds-aperture-ring.is-empty{{border-style:dashed;}}
.ds-aperture-verdict{{position:absolute;inset:-6px;border-radius:50%;pointer-events:none;
  border:4px solid currentColor;}}
.ds-aperture-hint{{color:{T.field_ink};font-size:{T.font_sm}px;text-align:center;
  padding:0 {T.space_24}px;line-height:1.5;opacity:.85;}}
/* Reticle ticks: four short marks at the cardinal points, like a graticule. */
.ds-tick{{position:absolute;background:{T.reticle};opacity:.75;}}
.ds-tick-t,.ds-tick-b{{left:50%;width:1px;height:10px;margin-left:-.5px;}}
.ds-tick-l,.ds-tick-r{{top:50%;height:1px;width:10px;margin-top:-.5px;}}
.ds-tick-t{{top:8px;}} .ds-tick-b{{bottom:8px;}}
.ds-tick-l{{left:8px;}} .ds-tick-r{{right:8px;}}

/* ---------- Verdict ---------- */
.ds-verdict-head{{font-size:{T.verdict_font}px;font-weight:700;letter-spacing:-.02em;
  line-height:1.05;margin:0;}}
.ds-verdict-mark{{width:44px;height:4px;border-radius:2px;margin:{T.space_12}px 0 {T.space_16}px;}}
.ds-verdict-body{{font-size:{T.font_base}px;color:{T.text_muted};margin:0 0 {T.space_12}px;line-height:1.5;}}
/* The one serif element: the sentence addressed to a person. Capped at a
   comfortable measure — the right pane is wide enough to run to 100 characters
   a line, which is past the point where the eye loses the next line. */
.ds-advice{{font-family:{T.serif_family};font-size:{T.advice_font}px;line-height:1.45;
  color:{T.text};margin:0;max-width:56ch;}}
.ds-gap{{height:{T.space_16}px;}}
.ds-reason{{font-size:{T.font_sm}px;color:{T.text_muted};margin-top:{T.space_8}px;}}

/* ---------- Live capture meters ---------- */
.ds-meter{{display:flex;align-items:center;gap:{T.space_8}px;padding:{T.space_4}px 0;
  font-size:{T.font_sm}px;color:{T.text_muted};}}
.ds-meter-name{{min-width:92px;}}
.ds-meter-bars{{display:flex;gap:3px;}}
.ds-meter-bar{{width:14px;height:6px;border-radius:2px;background:{T.outline};}}
.ds-meter-bar.on{{background:{T.violet};}}
.ds-meter-value{{color:{T.text};font-weight:600;}}

/* ---------- Lists ---------- */
.ds-case-row,.ds-scan-row{{padding:{T.space_12}px 0;border-bottom:1px solid {T.outline};
  font-size:{T.font_sm}px;}}
.ds-folder-card{{border:1px solid {T.outline};border-radius:{T.radius_sm}px;
  padding:{T.space_12}px {T.space_12}px;margin-bottom:{T.space_4}px;font-size:{T.font_sm}px;}}
.ds-history-scroll{{display:flex;flex-wrap:nowrap;gap:{T.space_12}px;overflow-x:auto;
  padding:{T.space_8}px 0 {T.space_16}px;-webkit-overflow-scrolling:touch;}}
.ds-history-scroll .ds-folder-card{{min-width:120px;flex:0 0 auto;}}
.ds-tier-dot{{width:8px;height:8px;border-radius:999px;display:inline-block;
  margin-right:{T.space_4}px;vertical-align:middle;}}
.ds-prob-track{{height:10px;background:{T.surface};border-radius:999px;flex:1;overflow:hidden;}}
.ds-prob-fill{{height:100%;background:{T.violet};transition:width .4s ease;}}

/* ---------- Brand bar ---------- */
.ds-brand{{display:flex;align-items:center;gap:{T.space_8}px;font-weight:700;
  font-size:{T.font_md}px;color:{T.text};letter-spacing:.02em;}}
.ds-brand-dot{{width:10px;height:10px;border-radius:999px;background:{T.reticle};
  box-shadow:0 0 0 3px {T.violet_tint};}}
.ds-brand-sub{{font-size:{T.font_xs}px;color:{T.text_muted};font-weight:400;}}
.ds-offline-pill{{margin-left:{T.space_8}px;padding:2px 10px;border-radius:999px;
  background:{T.surface};color:{T.text_muted};font-size:{T.pill_font}px;font-weight:600;
  letter-spacing:.06em;vertical-align:middle;}}
.ds-iconbtn{{display:flex;justify-content:center;color:{T.text_muted};}}
.ds-disclaimer{{font-size:{T.font_2xs}px;letter-spacing:.14em;text-transform:uppercase;
  color:{T.text_muted};text-align:center;}}
.ds-disclaimer-sub{{font-size:{T.font_sm}px;color:{T.text_muted};text-align:center;}}
.ds-app-bar-time{{font-size:{T.font_xs}px;color:{T.text_muted};text-align:right;}}
.ds-rec-card{{background:{T.surface};border:1px solid {T.outline};
  border-radius:{T.radius_md}px;padding:{T.space_16}px;margin:{T.space_16}px 0;}}

/* ---------- Streamlit widgets ---------- */
[data-testid="stExpander"]{{border:1px solid {T.outline}!important;
  border-radius:{T.radius_md}px!important;background:{T.bg}!important;}}
[data-testid="stExpander"] summary{{font-size:{T.font_sm}px!important;
  color:{T.text_muted}!important;font-weight:600!important;}}
.stTextInput input,.stSelectbox div[data-baseweb="select"]{{border-radius:{T.radius_sm}px!important;
  font-size:{T.font_base}px!important;}}
[data-testid="stCameraInput"]{{overflow:visible!important;}}
[data-testid="stCameraInput"] video{{width:100%;height:auto;object-fit:contain;border-radius:{T.radius_md}px;}}
[data-testid="stImage"] img{{border-radius:{T.radius_sm}px;}}
[data-testid="stCaptionContainer"],.stCaption{{color:{T.text_muted}!important;}}

/* ---------- Responsive ---------- */
@media (max-width:{_STACK_BP}px){{
  [data-testid="stHorizontalBlock"]{{flex-direction:column!important;}}
  [data-testid="stHorizontalBlock"]>div{{width:100%!important;}}
  .ds-verdict-head{{font-size:{T.font_xl}px;}}
  .ds-aperture{{max-width:240px;}}
}}
@media (prefers-reduced-motion:reduce){{
  *{{transition:none!important;animation:none!important;}}
}}
</style>""",
        unsafe_allow_html=True,
    )
