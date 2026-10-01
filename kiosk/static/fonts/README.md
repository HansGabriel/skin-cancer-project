# Vendored fonts

Served by Streamlit's static file server at `app/static/fonts/` (enabled via
`enableStaticServing` in `.streamlit/config.toml`) and declared in
`frontend/theme/css.py`.

They are committed rather than downloaded because the kiosk runs fully offline
(AGENTS.md, Agent default 6: no runtime external downloads or CDN-only assets on
demo-critical paths). They are served as files rather than inlined as base64
because Streamlit re-sends the stylesheet on every rerun; as files the browser
caches them once.

| File | Family | Weight | Used for |
|---|---|---|---|
| `archivo-400.woff2` | Archivo | 400 | body text, data |
| `archivo-600.woff2` | Archivo | 600 | buttons, labels |
| `archivo-700.woff2` | Archivo | 700 | headlines, verdict |
| `sourceserif-400.woff2` | Source Serif 4 | 400 | the one sentence of advice |

These are Latin subsets from Google Fonts.

## Licence

Both families are licensed under the **SIL Open Font License, Version 1.1**,
which requires the licence to accompany the font files when they are
redistributed — which committing them here does.

- **Archivo** — Copyright (c) Omnibus-Type. https://github.com/Omnibus-Type/Archivo
- **Source Serif 4** — Copyright (c) Adobe. https://github.com/adobe-fonts/source-serif

Full licence text: <https://openfontlicense.org/open-font-license-official-text/>

Neither font is sold, and neither is distributed under a reserved font name, so
the OFL's remaining conditions are satisfied by this notice.
