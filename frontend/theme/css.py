"""Inject global DermaScan theme CSS into Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from theme.tokens import TOKENS as T

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
[data-testid="stAppViewContainer"],.stApp{{background:{T.bg}!important;color:{T.text}!important;font-family:{_FONT_STACK}!important;}}
[data-testid="stSidebar"]{{background:{T.bg_elev}!important;border-right:1px solid {T.outline}!important;}}
.stButton>button{{border-radius:999px!important;background:{T.surface}!important;color:{T.text}!important;border:1px solid {T.outline}!important;}}
.stButton>button[kind="primary"]{{background:linear-gradient(180deg,{T.violet},{T.violet_strong})!important;color:#fff!important;border:none!important;box-shadow:0 8px 24px rgba(108,74,182,.4)!important;}}
.ds-mobile-frame{{max-width:{T.mobile_width}px;margin:0 auto;padding:0 12px 32px;box-shadow:0 12px 40px rgba(0,0,0,.35);border-radius:{T.radius_md}px;}}
.ds-viewfinder{{position:relative;width:280px;height:280px;margin:0 auto;border-radius:{T.radius_md}px;background:{T.bg_elev};overflow:hidden;}}
.ds-viewfinder-slot{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;}}
.ds-viewfinder-slot img{{width:100%;height:100%;object-fit:cover;}}
.ds-disclaimer{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{T.text_muted};text-align:center;}}
.ds-disclaimer-sub{{font-size:13px;color:{T.text_muted};text-align:center;}}
.ds-app-bar-time{{font-size:11px;color:{T.text_muted};text-align:right;}}
.ds-prob-track{{height:10px;background:{T.surface};border-radius:999px;flex:1;overflow:hidden;}}
.ds-prob-fill{{height:100%;background:linear-gradient(90deg,{T.violet},{T.violet_strong});}}
.ds-rec-card{{background:rgba(108,74,182,.25);border:1px solid {T.outline};border-radius:{T.radius_md}px;padding:16px;margin:16px 0;}}
.ds-empty{{text-align:center;color:{T.text_muted};padding:24px;}}
@keyframes pulse-shutter{{0%{{box-shadow:inset 0 0 0 rgba(181,140,240,0)}}50%{{box-shadow:inset 0 0 40px 8px rgba(181,140,240,.55)}}100%{{box-shadow:inset 0 0 0 rgba(181,140,240,0)}}}}
.ds-shutter-pulse{{animation:pulse-shutter .6s ease-out;}}
.ds-history-scroll{{display:flex;flex-wrap:nowrap;gap:12px;overflow-x:auto;padding:8px 0 16px;-webkit-overflow-scrolling:touch;}}
.ds-history-scroll .ds-folder-card{{min-width:120px;flex:0 0 auto;}}
.ds-folder-card{{border:1px solid {T.outline};border-radius:{T.radius_sm}px;padding:12px 14px;margin-bottom:4px;font-size:13px;}}
.ds-case-row,.ds-scan-row{{padding:8px 0;border-bottom:1px solid {T.outline};font-size:13px;}}
.ds-viewfinder .stImage img,.ds-viewfinder [data-testid="stCameraInput"]{{border-radius:{T.radius_md}px;}}
</style>""",
        unsafe_allow_html=True,
    )
