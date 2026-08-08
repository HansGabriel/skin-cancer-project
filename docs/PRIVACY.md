# Privacy & data handling (kiosk deployments)

DermaScan/E.P.I.V.U.E. photographs skin. On a shared walk-up device that makes
privacy a first-class feature, not an afterthought. This document is the policy
the app enforces and the paragraph the research paper's ethics section can cite.

## What the app stores, and when

- **Nothing persists by default.** A scan lives only in the session; leaving the
  Results screen (Clear / new scan / reboot) discards the photo.
- **"Save to case" requires explicit consent.** The save dialog contains a
  consent confirmation; `Storage.save_scan` refuses (raises `PermissionError`)
  unless the consent flag is passed. Saved data: the analysis JPEG, the
  segmentation mask, body site, scores — under `~/.dermascan/` (or
  `DERMASCAN_DATA_DIR`). No names — cases are identified by body site + date.
- **The inference log** (`logs/scans.sqlite`) records label/confidence/latency
  per scan for the research evaluation. It contains **no images and no
  identifiers**.

## Who can see saved scans

History, Folder, Case, and Settings are **staff-only** when a passcode is
configured: set `DERMASCAN_PASSCODE` (env) or `.streamlit/secrets.toml`:

```toml
[dermascan]
passcode = "1234"
```

The kiosk shows a numeric keypad for it. Scanning never requires the code —
only reviewing stored data does. The supervising staff member/adviser holds the
passcode and acts as data steward.

## Retention

Default policy for events with student participants: **erase at the end of the
event** via Settings → "End event — erase all saved scans" (staff area). Images,
masks, and all case rows are deleted. Keep data longer only when the signed
consent for the study covers it.

## Before photographing students for the study

Capturing students' skin as research data is human-participants research under
ISEF rules (which Philippine NSTF fairs follow). Before the first photo:
constituted IRB/SRC approval, written parental permission plus minor assent,
and the current forms (1, 1A, 1B, 4 + informed-consent statement). See
`docs/FINAL_IMPROVEMENTS_RESEARCH.md`, Track D, for sources.
