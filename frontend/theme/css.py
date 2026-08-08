"""Inject global DermaScan theme CSS into Streamlit.

Profile-aware via theme.tokens (SKIN_DISPLAY env): ``desktop`` keeps MVP 1
values; ``7in`` targets the 1024x600/800x480 touchscreen at surf zoom 1.0.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from theme.tokens import TOKENS as T

_IS_7IN = os.environ.get("SKIN_DISPLAY", "desktop").strip().lower() == "7in"
# Column-stacking breakpoint: desktop keeps 560 (phones stack); on the 7" panel
# (>=800px viewport) drop to 480 so nothing ever stacks on the kiosk itself.
_STACK_BP = 480 if _IS_7IN else 560

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_HAS_PJS = (_FONTS_DIR / "PlusJakartaSans-Regular.woff2").is_file()
_FONT_STACK = (
    "'Plus Jakarta Sans', Inter, system-ui, sans-serif"
    if _HAS_PJS
    else "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
)
_FONT_FACE = (
    f"@font-face{{font-family:'Plus Jakarta Sans';src:url('file://{_FONTS_DIR}/PlusJakartaSans-Regular.woff2') format('woff2');}}"
    if _HAS_PJS
    else ""
)


def inject_global_css() -> None:
    st.markdown(
        f"""<style>
{_FONT_FACE}
[data-testid="stAppViewContainer"],.stApp{{background:{T.bg}!important;color:{T.text}!important;font-family:{_FONT_STACK}!important;font-size:{T.font_base}px;}}
[data-testid="stSidebar"]{{background:{T.bg_elev}!important;border-right:1px solid {T.outline}!important;}}
/* Bigger, readable tap targets on the LCD. Secondary buttons are white-on-outline. */
.stButton>button{{border-radius:999px!important;background:#fff!important;color:{T.text}!important;border:1px solid {T.outline}!important;min-height:{T.touch_min}px!important;font-size:{T.font_base}px!important;font-weight:600!important;padding:12px 18px!important;box-shadow:{T.shadow_sm}!important;transition:transform .08s ease,box-shadow .15s ease!important;}}
.stButton>button:hover{{border-color:{T.violet}!important;box-shadow:{T.shadow_md}!important;}}
.stButton>button:active{{transform:translateY(1px)!important;}}
.stButton>button[kind="primary"]{{background:linear-gradient(180deg,{T.violet},{T.violet_strong})!important;color:#fff!important;border:none!important;box-shadow:0 6px 18px rgba(108,74,182,.30)!important;}}
.stButton>button[kind="primary"]:hover{{box-shadow:0 10px 26px rgba(108,74,182,.42)!important;}}
.ds-mobile-frame{{max-width:{T.mobile_width}px;margin:0 auto;padding:0 12px 24px;animation:ds-fade-in .25s ease-out;}}
@keyframes ds-fade-in{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
/* Reusable elevated card + consistent section header. */
.ds-card{{background:#fff;border:1px solid {T.outline};border-radius:{T.radius_md}px;box-shadow:{T.shadow_sm};padding:16px;margin:12px 0;}}
.ds-section-title{{font-size:{T.font_xs}px;letter-spacing:.08em;text-transform:uppercase;color:{T.text_muted};font-weight:700;margin:0 0 8px;}}
.ds-viewfinder{{position:relative;width:100%;max-width:300px;aspect-ratio:1/1;margin:0 auto;border-radius:{T.radius_md}px;background:{T.bg_elev};border:1px solid {T.outline};overflow:hidden;}}
.ds-viewfinder-slot{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;}}
.ds-viewfinder-slot img{{width:100%;height:100%;object-fit:cover;}}
.ds-disclaimer{{font-size:{T.font_xs}px;letter-spacing:.08em;text-transform:uppercase;color:{T.text_muted};text-align:center;}}
.ds-disclaimer-sub{{font-size:{T.font_sm}px;color:{T.text_muted};text-align:center;}}
.ds-app-bar-time{{font-size:{T.font_xs}px;color:{T.text_muted};text-align:right;}}
.ds-prob-track{{height:14px;background:{T.surface};border-radius:999px;flex:1;overflow:hidden;}}
.ds-prob-fill{{height:100%;background:linear-gradient(90deg,{T.violet},{T.violet_strong});}}
.ds-rec-card{{background:rgba(108,74,182,.10);border:1px solid {T.outline};border-radius:{T.radius_md}px;padding:16px;margin:16px 0;}}
.ds-empty{{text-align:center;color:{T.text_muted};padding:24px;}}
@keyframes pulse-shutter{{0%{{box-shadow:inset 0 0 0 rgba(108,74,182,0)}}50%{{box-shadow:inset 0 0 40px 8px rgba(108,74,182,.45)}}100%{{box-shadow:inset 0 0 0 rgba(108,74,182,0)}}}}
.ds-shutter-pulse{{animation:pulse-shutter .6s ease-out;}}
.ds-history-scroll{{display:flex;flex-wrap:nowrap;gap:12px;overflow-x:auto;padding:8px 0 16px;-webkit-overflow-scrolling:touch;}}
.ds-history-scroll .ds-folder-card{{min-width:120px;flex:0 0 auto;}}
.ds-folder-card{{border:1px solid {T.outline};border-radius:{T.radius_sm}px;padding:12px 14px;margin-bottom:4px;font-size:{T.font_sm}px;}}
.ds-case-row,.ds-scan-row{{padding:12px 0;border-bottom:1px solid {T.outline};font-size:{T.font_sm}px;}}
.ds-viewfinder .stImage img,.ds-viewfinder [data-testid="stCameraInput"]{{border-radius:{T.radius_md}px;}}
/* Never clip the browser camera widget — its video + "Take photo" button must stay
   fully visible. If it ever ends up inside a viewfinder square, undo the clipping. */
[data-testid="stCameraInput"]{{overflow:visible!important;}}
.ds-viewfinder:has([data-testid="stCameraInput"]){{aspect-ratio:auto;max-width:100%;overflow:visible;}}
.ds-viewfinder [data-testid="stCameraInput"] video{{width:100%;height:auto;object-fit:contain;}}
.ds-prob-fill{{transition:width .5s cubic-bezier(.4,0,.2,1);}}
/* Brand wordmark in the app bar. */
.ds-brand{{display:flex;align-items:center;gap:8px;font-weight:800;font-size:{T.font_md}px;color:{T.violet};letter-spacing:-.01em;}}
.ds-brand-dot{{width:12px;height:12px;border-radius:999px;background:linear-gradient(135deg,{T.violet},{T.teal});box-shadow:0 0 0 3px {T.violet_tint};}}
.ds-brand-sub{{font-size:{T.font_xs}px;color:{T.text_muted};font-weight:500;}}
.ds-offline-pill{{margin-left:8px;padding:3px 10px;border-radius:999px;background:{T.violet_tint};color:{T.violet};font-size:{T.pill_font}px;font-weight:700;letter-spacing:.08em;vertical-align:middle;}}
/* Icon buttons in the app bar: keep the click target, hide the text label. */
.ds-iconbtn{{display:flex;justify-content:center;color:{T.text_muted};}}
/* Risk ring hero. */
.ds-ring-wrap{{display:flex;flex-direction:column;align-items:center;margin:8px 0 4px;}}
.ds-ring{{width:170px;height:170px;border-radius:999px;display:flex;align-items:center;justify-content:center;}}
.ds-ring-inner{{width:130px;height:130px;border-radius:999px;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;box-shadow:{T.shadow_sm};}}
.ds-ring-label{{font-size:{T.font_lg}px;font-weight:800;line-height:1.05;letter-spacing:-.01em;}}
.ds-ring-conf{{font-size:{T.font_sm}px;color:{T.text_muted};margin-top:2px;}}
/* Advice/recommendation card with a risk-tinted left accent. */
.ds-advice{{display:flex;gap:12px;align-items:flex-start;background:{T.violet_tint};border:1px solid {T.outline};border-left:4px solid {T.violet};border-radius:{T.radius_md}px;padding:14px 16px;margin:16px 0;}}
.ds-advice svg{{flex:0 0 auto;margin-top:2px;}}
/* ABCDE tier dot. */
.ds-tier-dot{{width:10px;height:10px;border-radius:999px;display:inline-block;margin-right:4px;vertical-align:middle;}}
/* --- Responsive: narrow windows stack columns; wide keeps the framed column --- */
@media (max-width:{_STACK_BP}px){{
  .stApp{{font-size:18px!important;}}
  .ds-mobile-frame{{max-width:100%;padding:0 8px 16px;}}
  .stButton>button{{min-height:{max(T.touch_min, 52)}px!important;font-size:18px!important;}}
  /* Stack Streamlit's horizontal columns so 4/5-col grids don't crush. */
  [data-testid="stHorizontalBlock"]{{flex-direction:column!important;}}
  [data-testid="stHorizontalBlock"]>div{{width:100%!important;}}
  .ds-app-bar-time{{font-size:13px;}}
  .ds-ring{{width:150px;height:150px;}}
  .ds-ring-inner{{width:116px;height:116px;}}
}}
@media (min-width:{_STACK_BP + 1}px){{
  .stApp{{font-size:{T.font_base if _IS_7IN else 16}px;}}
  .ds-mobile-frame{{max-width:{T.mobile_width}px;}}
}}
</style>""",
        unsafe_allow_html=True,
    )
