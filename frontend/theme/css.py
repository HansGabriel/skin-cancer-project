"""Inject the global stylesheet into Streamlit.

Three constraints shape everything here:

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

**The shell is two bands, and Streamlit has to be talked into it.** The layout
is a dark *instrument* band (45%, aperture + nav keys at its foot) beside a
white *page* band (one job per screen, actions pinned to the bottom). The only
DOM hook Streamlit gives us is ``st.container(key="x")`` → ``.st-key-x``, so
every shell rule targets one of those keyed blocks **by its own class** rather
than the shared ``[data-testid="stHorizontalBlock"]``. That matters: styling the
horizontal block directly would also hit every ``st.columns`` a screen nests
inside the page band. Each band carries its own ``height:100vh``, which makes
the row full-height without the row being styled at all.

The design expresses type as ``clamp(min, N·cqh, max)``, which needs
``container-type: size``. Container queries are a risk in the kiosk's WebKitGTK
build and are unnecessary here — as the design's own TYPE note says, that clamp
encodes "the same rule the desktop / 7in token profiles already encode". So the
profiles in ``theme.tokens`` do the job and no container query is used.
"""

from __future__ import annotations

import streamlit as st

from theme.tokens import TOKENS as T

# Below this width the two bands stack. The 1024x600 kiosk never crosses it at
# zoom 1.0, and it leaves headroom up to ~1.3x if a panel ever needs zooming.
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
/* Zero-height so the shell's 100vh really is the viewport. The toolbar inside
   it stays clickable — it is positioned, not in flow — which keeps the sidebar
   reachable on a PC without stealing a strip from the top of the instrument. */
[data-testid="stHeader"]{{background:transparent!important;height:0!important;}}
/* The Deploy button and the running-man overlay sit on top of the page band's
   eyebrow. Neither belongs on a device a member of the public walks up to. The
   sidebar's collapsed control is a separate element and is left alone, so the
   power-user sidebar is still reachable on a PC. */
[data-testid="stToolbar"],[data-testid="stStatusWidget"],
[data-testid="stDecoration"]{{display:none!important;}}
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
/* Establish a block formatting context in every markdown block. Without it a
   child's top margin collapses *through* the container, so the container
   measures shorter than what it paints and the next element starts too early —
   measured as a 7px overlap of the staff readout's timing line by the Close
   button, i.e. content sitting under a button and unreachable. Every .ds-* rule
   that opens with a top margin depends on this. */
.stMarkdown,[data-testid="stMarkdownContainer"]{{display:flow-root;}}

/* Full-bleed: the shell paints both bands edge to edge, so the container that
   used to centre a 1040px column now gets out of the way entirely.
   ``stMainBlockContainer`` is the 1.5x name for the same element; both are
   listed so a Streamlit rename does not silently restore the gutters. */
.block-container,.stMainBlockContainer{{padding:0!important;max-width:100%!important;
  overflow:hidden!important;}}
[data-testid="stVerticalBlock"]{{gap:{T.space_8}px!important;}}
/* ...except the root block, which holds the shell AND the element this very
   stylesheet is injected into. That <style> tag is an invisible flex item, so
   the gap after it pushed the whole shell 8px down the page and put 8px of the
   nav below the fold on a 600px panel. Direct child only — nested blocks keep
   their spacing. */
.stMainBlockContainer > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"]{{gap:0!important;}}
[data-testid="stElementContainer"]:empty{{display:none!important;}}
hr{{margin:{T.space_8}px 0!important;}}

/* ---------- The shell: instrument band | page band ---------- */
/* Two DOM shapes are covered because Streamlit has moved the st-key- class
   between the vertical block and its wrapper across releases. Neither selector
   can reach a nested st.columns: those sit three levels deeper, inside a
   stColumn within .st-key-epv-page. */
.st-key-epv-shell > div[data-testid="stHorizontalBlock"],
.st-key-epv-shell > div > div[data-testid="stHorizontalBlock"]{{
  gap:0!important;align-items:stretch!important;}}

/* Both bands are full-viewport-height, and getting there takes three declarations
   rather than one.

   `height` alone does nothing here. Streamlit makes every stVerticalBlock a flex
   item with `flex: 1 1 0%` inside a *column* wrapper, and on the main axis
   `flex-basis` beats `height` — measured, the bands sized to their content at
   385px with `height:100vh` set and winning the cascade. So `flex` is reset to
   `0 0 auto` to hand control back to `height`, and `min-height` backs it up
   because a flex item's basis can still shrink it.

   Sizing the bands themselves rather than the ancestor chain is deliberate:
   every element between them and .block-container grows to fit its content, so
   a 100vh band pushes the whole chain to full height without this file having
   to know the chain's shape. That shape already changed once — stLayoutWrapper
   is new in 1.57 — and a hard-coded ancestor path would have broken silently. */
.st-key-epv-band,.st-key-epv-page,.st-key-epv-page-dark{{
  flex:0 0 auto!important;height:100vh!important;min-height:100vh!important;
  box-sizing:border-box;
  display:flex!important;flex-direction:column!important;gap:0!important;}}

.st-key-epv-band{{background:{T.field};color:{T.field_ink};
  padding:{T.space_16}px 0 0;overflow:hidden;position:relative;}}
.st-key-epv-band .stMarkdown,.st-key-epv-band [data-testid="stMarkdownContainer"] *{{
  color:{T.field_ink};}}

/* The band itself never scrolls: the safety strip is the last thing in it and
   has to stay visible. Overflow is handled one level in, by .st-key-epv-body. */
.st-key-epv-page,.st-key-epv-page-dark{{
  padding:{T.space_24}px {T.page_pad_x}px {T.space_12}px;
  overflow:hidden;}}
.st-key-epv-page{{background:{T.bg};}}

/* The staff readout takes the page band whole and dark — the design's overlay,
   achieved by being a route rather than by stacking anything. */
.st-key-epv-page-dark{{background:{T.field};color:{T.field_ink};}}
.st-key-epv-page-dark .ds-title,.st-key-epv-page-dark .stMarkdown,
.st-key-epv-page-dark [data-testid="stMarkdownContainer"] *{{color:{T.field_ink};}}
.st-key-epv-page-dark .ds-eyebrow{{color:{T.reticle}!important;}}
.st-key-epv-page-dark .ds-foot{{border-top-color:{T.field_rule};}}
.st-key-epv-page-dark .ds-foot-tag{{color:{T.field_ink}!important;}}
.st-key-epv-page-dark .ds-foot-text,.st-key-epv-page-dark .stCaption,
.st-key-epv-page-dark [data-testid="stCaptionContainer"]{{color:{T.field_muted}!important;}}
.st-key-epv-page-dark .stButton>button{{background:none!important;color:{T.field_ink}!important;
  border-color:rgba(232,238,246,.22)!important;}}
.st-key-epv-page-dark .stButton>button:hover{{border-color:{T.reticle}!important;
  color:{T.field_ink}!important;}}
.st-key-epv-page-dark [data-testid="stExpander"]{{background:none!important;
  border-color:{T.field_rule}!important;}}
.st-key-epv-page-dark [data-testid="stExpander"] summary{{color:{T.field_muted}!important;}}

/* ---------- Pinning things to the foot of a band ----------
   Streamlit wraps every container in an anonymous stElementContainer /
   stLayoutWrapper, so `.st-key-epv-nav {{ margin-top:auto }}` styles an element
   that is NOT the flex item — the wrapper is, and it stays put. `:has()` would
   let us reach the wrapper but is banned here (WebKitGTK).

   So position carries the meaning instead, and app.py / instrument.py keep to
   the contract:

     .st-key-epv-band  -> [header] [instrument body] [nav]
     .st-key-epv-page  -> [screen] [safety strip]
     .st-key-epv-body  -> [...screen content..., actions last]

   Growing the middle child is what pins the ones after it. */
.st-key-epv-band > *:nth-child(2){{flex:1 1 auto!important;min-height:0!important;}}
.st-key-epv-page > *:first-child,.st-key-epv-page-dark > *:first-child{{
  flex:1 1 auto!important;min-height:0!important;}}

.st-key-epv-inst-body,.st-key-epv-body{{height:100%!important;
  display:flex!important;flex-direction:column!important;}}
/* min-height:0 is what actually lets a flex item shrink below its content;
   without it a tall screen spilled over the safety strip below, and the strip
   then sat on top of the action buttons and swallowed their taps. Scrolling
   here rather than on the band keeps the strip pinned and visible. */
.st-key-epv-body{{min-height:0!important;overflow-y:auto!important;overflow-x:hidden!important;}}
.st-key-epv-inst-body{{justify-content:center!important;gap:{T.space_12}px!important;}}
/* The actions row is always the last thing a screen renders. */
.st-key-epv-body > *:last-child{{margin-top:auto!important;}}

/* ---------- Instrument band internals ---------- */
.ds-inst-head{{display:flex;align-items:center;justify-content:space-between;
  gap:{T.space_12}px;padding:0 {T.band_pad_x}px;}}
.ds-inst-brand{{display:flex;align-items:center;gap:{T.space_8}px;font-weight:700;
  letter-spacing:.17em;color:{T.field_ink};font-size:{T.font_sm}px;}}
.ds-inst-dot{{width:9px;height:9px;border-radius:999px;background:{T.reticle};
  box-shadow:0 0 0 4px rgba(61,219,217,.16);}}
.ds-inst-meta{{display:flex;align-items:center;gap:{T.space_8}px;color:{T.field_muted};
  font-size:{T.font_xs}px;white-space:nowrap;}}
.ds-inst-pill{{border:1px solid rgba(232,238,246,.18);border-radius:999px;
  padding:3px 9px;font-weight:600;letter-spacing:.13em;font-size:{T.font_2xs}px;}}
.ds-inst-body{{flex:1;min-height:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:{T.space_12}px;
  padding:{T.space_8}px {T.band_pad_x}px;}}
.ds-inst-caption{{color:{T.field_muted};font-weight:600;letter-spacing:.16em;
  text-align:center;font-size:{T.font_2xs}px;text-transform:uppercase;}}
.ds-inst-legend{{display:flex;justify-content:center;gap:{T.space_12}px;
  color:{T.field_body};font-size:{T.font_2xs}px;margin-top:{T.space_4}px;}}
.ds-inst-legend span{{display:inline-flex;align-items:center;gap:4px;}}
.ds-inst-swatch{{width:14px;height:3px;border-radius:2px;display:inline-block;}}
/* The saved-spot timeline: one dot per scan, the newest in reticle cyan. */
.ds-inst-track{{display:flex;justify-content:center;align-items:center;gap:{T.space_8}px;
  margin-top:{T.space_8}px;}}
.ds-inst-node{{width:9px;height:9px;border-radius:999px;background:rgba(232,238,246,.28);}}
.ds-inst-node.is-last{{width:11px;height:11px;background:{T.reticle};}}
.ds-inst-link{{width:26px;height:1px;background:rgba(232,238,246,.2);}}

/* Nav keys live at the instrument's foot: flat, no fill, a cyan cap when live. */
.st-key-epv-nav{{border-top:1px solid {T.field_rule};}}
.st-key-epv-nav [data-testid="stHorizontalBlock"]{{gap:0!important;}}
.st-key-epv-nav .stButton>button{{
  height:{T.nav_h}px!important;min-height:{T.nav_h}px!important;width:100%!important;
  background:none!important;border:0!important;border-radius:0!important;
  border-left:1px solid {T.field_rule}!important;border-top:2px solid transparent!important;
  color:{T.field_muted}!important;font-weight:600!important;letter-spacing:.04em!important;
  font-size:{T.font_sm}px!important;box-shadow:none!important;padding:0 4px!important;
  white-space:nowrap!important;}}
.st-key-epv-nav [data-testid="stColumn"]:first-child .stButton>button{{border-left:0!important;}}
.st-key-epv-nav .stButton>button:hover{{color:{T.field_ink}!important;
  background:rgba(232,238,246,.05)!important;}}
.st-key-epv-nav .stButton>button[kind="primary"]{{
  color:#FFFFFF!important;border-top:2px solid {T.reticle}!important;background:none!important;}}
.st-key-epv-nav .stButton>button:focus-visible{{outline:3px solid {T.reticle}!important;
  outline-offset:-3px!important;}}

/* ---------- Page band internals ---------- */
.ds-eyebrow{{font-weight:600;letter-spacing:.18em;color:{T.text_muted};
  font-size:{T.font_xs}px;text-transform:uppercase;margin:0;}}
.ds-title{{margin:{T.space_12}px 0 0;font-weight:700;letter-spacing:-.022em;
  line-height:1.05;color:{T.text};font-size:{T.font_2xl}px;}}
.ds-lede{{margin:{T.space_12}px 0 0;color:{T.text_muted};line-height:1.5;
  max-width:42ch;font-size:{T.font_md}px;}}
/* The middle region of a screen: centred in whatever space the eyebrow/title
   above and the pinned actions below leave over. */
.ds-mid{{padding:{T.space_12}px 0 0;margin:0;width:100%;box-sizing:border-box;
  display:flex;flex-direction:column;}}
.ds-row{{width:100%;box-sizing:border-box;display:flex;align-items:center;
  gap:{T.space_16}px;padding:{T.space_12}px 0;border-top:1px solid {T.hairline};
  font-size:{T.font_sm}px;}}
.ds-row:last-child{{border-bottom:1px solid {T.hairline};}}
.ds-row-num{{flex:0 0 auto;width:32px;height:32px;border-radius:999px;background:{T.chip};
  color:{T.violet};display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:{T.font_xs}px;}}
.ds-row-grow{{flex:1;min-width:0;}}
.ds-dot{{width:7px;height:7px;border-radius:999px;background:{T.reticle};flex:0 0 auto;}}
.ds-note{{margin-top:{T.space_12}px;padding:{T.space_12}px {T.space_16}px;
  border-radius:{T.radius_md}px;}}
.ds-note-label{{font-weight:600;letter-spacing:.16em;font-size:{T.font_2xs}px;
  text-transform:uppercase;}}
.ds-note-body{{margin-top:{T.space_4}px;color:{T.text};line-height:1.45;font-size:{T.font_sm}px;}}
.ds-sign{{width:100%;box-sizing:border-box;display:flex;align-items:center;
  gap:{T.space_12}px;padding:{T.space_8}px 0;border-bottom:1px solid {T.hairline};
  font-size:{T.font_sm}px;}}
.ds-sign-mark{{flex:0 0 auto;width:14px;height:3px;border-radius:2px;}}
.ds-chiprow{{display:flex;gap:{T.space_8}px;padding-top:{T.space_12}px;flex-wrap:wrap;}}
.ds-chip{{padding:6px 13px;border-radius:999px;background:{T.chip};color:{T.text_muted};
  font-weight:600;font-size:{T.font_xs}px;}}
.ds-chip.is-on{{background:{T.violet};color:#FFFFFF;}}

/* The permanent safety strip at the foot of the page band. */
.ds-foot{{width:100%;box-sizing:border-box;display:flex;align-items:center;
  gap:{T.space_8}px;padding:{T.space_12}px 0;border-top:1px solid {T.hairline};}}
.ds-foot-tag{{flex:0 0 auto;font-weight:700;letter-spacing:.14em;white-space:nowrap;
  color:{T.text};font-size:{T.font_2xs}px;text-transform:uppercase;}}
.ds-foot-text{{color:{T.text_muted};line-height:1.4;font-size:{T.font_xs}px;}}

/* ---------- The staff readout ---------- */
.ds-staff-row{{width:100%;box-sizing:border-box;display:flex;align-items:center;
  gap:{T.space_12}px;padding:{T.space_8}px 0;border-top:1px solid {T.field_rule};
  font-size:{T.font_sm}px;}}
.ds-staff-letter{{flex:0 0 auto;width:26px;color:{T.reticle};font-weight:700;}}
.ds-staff-name{{flex:1;min-width:0;color:{T.field_body};}}
.ds-staff-value{{flex:0 0 auto;min-width:72px;text-align:right;color:{T.field_ink};
  font-weight:700;font-variant-numeric:tabular-nums;}}
.ds-staff-pill{{flex:0 0 auto;min-width:140px;text-align:right;font-weight:700;
  letter-spacing:.08em;font-size:{T.font_2xs}px;}}
.ds-staff-track{{flex:1;height:8px;border-radius:999px;background:rgba(232,238,246,.12);
  overflow:hidden;}}
.ds-staff-fill{{display:block;height:100%;border-radius:999px;}}
.ds-staff-meta{{color:{T.field_muted};line-height:1.5;font-variant-numeric:tabular-nums;
  font-size:{T.font_2xs}px;padding:{T.space_8}px 0 {T.space_12}px;}}

/* ---------- Buttons ---------- */
.stButton>button{{border-radius:{T.radius_sm}px!important;background:{T.bg}!important;
  color:{T.text}!important;border:1px solid {T.border}!important;
  min-height:{T.touch_min}px!important;font-family:{T.font_family}!important;
  font-size:{T.font_base}px!important;font-weight:600!important;letter-spacing:.01em!important;
  padding:10px 18px!important;box-shadow:none!important;transition:border-color .15s ease;}}
.stButton>button:hover{{border-color:{T.violet}!important;color:{T.violet}!important;}}
.stButton>button:focus-visible{{outline:3px solid {T.reticle}!important;outline-offset:2px!important;}}
.stButton>button[kind="primary"]{{background:{T.violet}!important;color:#fff!important;
  border:1px solid {T.violet}!important;box-shadow:{T.shadow_sm}!important;}}
.stButton>button[kind="primary"]:hover{{background:{T.violet_strong}!important;color:#fff!important;}}
.stButton>button:disabled{{opacity:.45!important;}}
/* The two pinned actions are the tallest targets on a screen. */
.st-key-epv-actions .stButton>button{{min-height:{T.touch_min + 8}px!important;
  border-radius:{T.radius_md}px!important;font-size:{T.font_md}px!important;}}

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
/* Badges that sit over the live/working aperture. */
.ds-aperture-badge{{position:absolute;left:50%;bottom:6%;transform:translateX(-50%);
  display:flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;
  background:rgba(2,6,13,.6);color:{T.field_ink};font-weight:600;letter-spacing:.14em;
  font-size:{T.font_2xs}px;}}
.ds-aperture-live{{width:6px;height:6px;border-radius:999px;background:{T.reticle};
  animation:dsPulse 1.4s ease-in-out infinite;}}
@keyframes dsPulse{{0%,100%{{opacity:1;}}50%{{opacity:.3;}}}}
@keyframes dsSpin{{to{{transform:rotate(360deg);}}}}

/* ---------- Verdict ---------- */
.ds-verdict-head{{font-size:{T.verdict_font}px;font-weight:700;letter-spacing:-.024em;
  line-height:1.02;margin:0;padding-top:{T.space_12}px;}}
.ds-verdict-mark{{width:44px;height:4px;border-radius:2px;margin:{T.space_8}px 0 {T.space_12}px;}}
.ds-verdict-body{{font-size:{T.font_base}px;color:{T.text_muted};margin:0 0 {T.space_8}px;line-height:1.45;}}
/* The one serif element: the sentence addressed to a person. Capped at a
   comfortable measure — the page band is wide enough to run to 100 characters
   a line, which is past the point where the eye loses the next line. */
.ds-advice{{font-family:{T.serif_family};font-size:{T.advice_font}px;line-height:1.45;
  color:{T.text};margin:0;max-width:44ch;}}
.ds-gap{{height:{T.space_16}px;}}
.ds-reason{{font-size:{T.font_sm}px;color:{T.text_muted};margin-top:{T.space_8}px;}}

/* ---------- Live capture meters ---------- */
.ds-meter{{display:flex;align-items:center;gap:{T.space_16}px;padding:{T.space_12}px 0;
  font-size:{T.font_sm}px;color:{T.text_muted};border-top:1px solid {T.hairline};}}
.ds-meter-name{{flex:0 0 auto;min-width:110px;}}
.ds-meter-bars{{flex:1;display:flex;gap:4px;}}
.ds-meter-bar{{flex:1;height:7px;border-radius:3px;background:{T.hairline};}}
.ds-meter-bar.on{{background:{T.violet};}}
.ds-meter-value{{flex:0 0 auto;min-width:72px;text-align:right;color:{T.text};font-weight:600;}}

/* ---------- Reading ---------- */
.ds-progress{{margin-top:{T.space_16}px;height:6px;border-radius:999px;
  background:{T.hairline};overflow:hidden;}}
.ds-progress-fill{{height:100%;background:{T.violet};}}
.ds-step{{display:flex;align-items:center;gap:{T.space_12}px;padding:{T.space_8}px 0;
  font-size:{T.font_sm}px;}}
.ds-step-dot{{width:9px;height:9px;border-radius:999px;background:{T.outline};flex:0 0 auto;}}
.ds-step.is-done .ds-step-dot{{background:{T.success};}}
.ds-step.is-now{{color:{T.text_muted};}}
.ds-step.is-now .ds-step-dot{{animation:dsPulse 1.2s ease-in-out infinite;}}

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
.ds-thumb{{flex:0 0 auto;width:46px;height:46px;border-radius:999px;background:{T.field};
  background-size:cover;background-position:center;}}
.ds-pill{{flex:0 0 auto;padding:5px 11px;border-radius:999px;font-weight:700;
  letter-spacing:.06em;font-size:{T.font_2xs}px;white-space:nowrap;}}

/* ---------- Brand bar (legacy screens) ---------- */
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
  /* Too narrow for two bands: stack them and let the page scroll. The kiosk
     never reaches here, but a phone-sized browser window otherwise gets a
     45%-wide instrument with nothing legible in it. */
  .st-key-epv-band,.st-key-epv-page{{height:auto!important;min-height:0!important;}}
  .st-key-epv-shell > div[data-testid="stHorizontalBlock"],
  .st-key-epv-shell > div > div[data-testid="stHorizontalBlock"]{{flex-direction:column!important;}}
  .st-key-epv-shell [data-testid="stColumn"]{{width:100%!important;flex:1 1 100%!important;}}
  .ds-verdict-head{{font-size:{T.font_xl}px;}}
  .ds-aperture{{max-width:240px;}}
}}
@media (prefers-reduced-motion:reduce){{
  *{{transition:none!important;animation:none!important;}}
}}
</style>""",
        unsafe_allow_html=True,
    )
