# Deploying DermaScan AI

## Streamlit Community Cloud (recommended first hop)

Tested host for the public smoke-test deployment.

### 1. Prepare the repo

- Branch with the v2 UI (e.g. `feat/new-design`) pushed to GitHub.
- Confirm the slim deps file exists at repo root: `streamlit-requirements.txt`.
- Confirm `.streamlit/config.toml` caps uploads at 8 MB.

### 2. Create the app

1. Open <https://share.streamlit.io> → **New app**.
2. Pick repo + branch.
3. **Main file path:** `frontend/app.py`.
4. **Advanced settings → Python version:** 3.11.
5. **Requirements file:** `streamlit-requirements.txt`.

### 3. Configure secrets

Under **Advanced settings → Secrets**, paste:

```toml
[dermascan]
passcode = "<pick-a-strong-passcode>"
```

Do **not** set `ENABLE_PI_BACKEND` or `SKIN_KERAS_PATH` — the Pi is unreachable
from the cloud, and full TensorFlow (needed for Grad-CAM) would blow the image
size limit. The Results screen already renders a friendly fallback when no
Keras model is configured.

### 4. (Optional) Seed demo data

After the first successful boot, open the Streamlit Cloud **Manage app → Shell**
and run:

```bash
python scripts/seed_demo_data.py
```

This creates one demo folder + case with two pre-scored scans so the History
screen is populated for visitors.

### 5. Smoke test

1. Visit the public URL.
2. Passcode prompt appears → enter the secret → home screen.
3. Settings → backend selector lists **only** local + mock (no Pi option).
4. Upload a sample JPEG from `samples/` → results render with ABCDE row, CNN
   probability bars, and a recommendation card. Grad-CAM panel shows the
   "unavailable" fallback.
5. Try uploading a >8 MB file → friendly error, no crash.
6. Save scan to a new case → appears in History.

### What's intentionally turned off on the hosted demo

- **Grad-CAM** — needs full TensorFlow (~1 GB of wheels). Re-enable on a larger
  host (Fly.io / HF Spaces) by installing `tensorflow-cpu` + `tf-keras-vis` and
  setting `SKIN_KERAS_PATH`.
- **Pi camera backend** — gated by `ENABLE_PI_BACKEND=1`.
- **Telemetry** — `gatherUsageStats = false` in `.streamlit/config.toml`.

---

## Other deployment targets

| Option | When to use | Extra files needed |
|---|---|---|
| **Fly.io / Render / Railway** | Always-on, persistent volume for the SQLite DB, custom domain. | `Dockerfile`, `fly.toml` (or `render.yaml`), volume mount for `~/.dermascan`. |
| **Hugging Face Spaces** | Free GPU tier for the Grad-CAM (7-class Keras) path. | Larger requirements file with `tensorflow-cpu` + `tf-keras-vis`; HF Space metadata. |

---

## Follow-up TODOs (deferred from this deployment pass)

These are tracked in the parent Notion v2 plan and were intentionally not
shipped with the first online deployment:

- ~~B.1 Fit `models/temperature.json`~~ **Done** — T=1.54 fitted and applied to *displayed* confidence only (`backend/tflite_shared.apply_temperature`); decisions stay on raw probs since the 0.11 threshold was tuned uncalibrated.
- ~~B.2 Wire test-time augmentation~~ **Done** — 4-view TTA in `backend/tflite_shared.run_inference_on_rgb` (`SKIN_TTA`, **on everywhere, including the Pi**). This line previously said "off on Pi", which was both wrong for the kiosk (it runs the `local` backend in-process, so TTA was on) and wrong as an intention: `docs/METRICS.md` measured the deployed sensitivity of 0.911 *with* TTA, so running without it serves an unvalidated configuration. The extra passes cost ~0.6 s.
- B.5 INT8 re-export: `scripts/export_tflite_int8.py` must add a `representative_dataset` (currently missing — the "int8" file is dynamic-range only). Re-export, check `scripts/eval_threshold.py --model models/skin_classifier_int8.tflite` parity (screening sensitivity within 1 pt of float), benchmark on the Pi, then switch via the existing `SKIN_MODEL` env in `scripts/pi_server.py`.
- Tier-2: implement `frontend/services/uncertainty.py` (MC-Dropout) and `frontend/services/skintone.py` (Fitzpatrick/ITA).
- Tier-3: implement `frontend/services/report.py` (PDF export).

---

## 7" touchscreen kiosk (MVP 2)

Target panel: **7" 1024x600 HDMI IPS, 5-point capacitive touch, drive-free**
(touch is a USB HID device — plug HDMI + USB, no driver install).

- `launch_kiosk.sh` exports `SKIN_DISPLAY=7in` (larger type/touch targets),
  `SKIN_NUM_THREADS=4`, and opens surf at zoom **1.0** on a local splash page
  that redirects itself once Streamlit answers.
- **No signal on the panel?** Add the panel's timing to `/boot/firmware/config.txt`:
  `hdmi_cvt=1024 600 60 6 0 0 0` plus `hdmi_group=2` and `hdmi_mode=87`, then reboot.
- **Typing is designed out on the kiosk** (`SKIN_KIOSK=1`): numeric on-screen
  keypad for the passcode, preset chips for save-to-case, search hidden, and the
  assistant uses tappable suggested questions. If free-text entry is ever needed,
  install a system on-screen keyboard as an optional extra:
  `sudo apt install wvkbd` and bind `wvkbd-mobintl` to a hot corner/gesture —
  not required for any core flow.

---

## Autostart & crash recovery (kiosk)

**Canonical Pi install path: `~/Documents/skin-cancer-project`.** `launch_kiosk.sh`
self-resolves its own location, but the desktop launcher (`EPIVUE.desktop`),
the systemd unit (`deploy/dermascan-kiosk.service`), and AGENTS.md all assume
this path — clone the repo there (or update those two files if you deviate).

Current Raspberry Pi OS (images from **2024-10-28** onward, all Pi models) boots
into the **labwc** Wayland compositor by default. `surf` is an X11 browser; it
runs under **XWayland**, which labwc provides automatically — no X11 setup needed.
Both autostart options below require **desktop autologin**
(`sudo raspi-config` → System Options → Boot / Auto Login → **Desktop Autologin**),
otherwise nothing graphical runs at boot.

### Option 1 — labwc autostart (simplest, no crash recovery)

Append one line to `~/.config/labwc/autostart` (create the file if missing):

```sh
~/Documents/skin-cancer-project/launch_kiosk.sh &
```

labwc runs this when the desktop session starts. If the kiosk crashes or
Streamlit fails to boot, nothing restarts it — fine for supervised demos.

### Option 2 — systemd user unit (autostart + restart on failure)

```bash
mkdir -p ~/.config/systemd/user
cp ~/Documents/skin-cancer-project/deploy/dermascan-kiosk.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable dermascan-kiosk.service
```

- `Restart=on-failure` + `RestartSec=5` relaunch the kiosk if it exits nonzero
  (e.g. `launch_kiosk.sh`'s 40s Streamlit watchdog) — but not after a clean exit
  via the on-screen Exit button.
- The unit is `WantedBy=graphical-session.target`, so it starts with the desktop
  and `PartOf=` stops it when the session ends. **That target has to be
  activated by the session**, via labwc's documented bridge line. labwc runs
  `~/.config/labwc/autostart`, but it does not add the line for you — check it
  is there, or the unit stays enabled-but-never-started:

  ```bash
  grep -q labwc-session.target ~/.config/labwc/autostart 2>/dev/null || \
    echo 'systemctl --user --no-block start labwc-session.target' >> ~/.config/labwc/autostart
  systemctl --user status dermascan-kiosk   # should be "active" after a reboot
  ```

  This line is not a second supervisor — it only wires the session to systemd,
  so it does not conflict with the "pick one option" rule below.
- Logs: `journalctl --user -u dermascan-kiosk` plus `/tmp/dermascan_kiosk.log`.

**Pick one option, not both** — two supervisors fighting over the same
Streamlit port and pidfiles is worse than either alone. (The bridge line above
is part of Option 2, not Option 1's `launch_kiosk.sh &` entry.)

### Staff passcode

`launch_kiosk.sh` reads the staff code from `~/.dermascan_passcode` (never
committed) and exports `DERMASCAN_PASSCODE`. If the file is missing it prints a
warning and the kiosk keeps History, saved scans and Settings **closed** — an
unattended kiosk must not expose participant photos. Create it with:

```bash
printf '%s' '<choose-4-digits>' > ~/.dermascan_passcode && chmod 600 ~/.dermascan_passcode
```

A staff unlock expires after `DERMASCAN_STAFF_TIMEOUT` seconds (default 300),
and Settings has a "Lock the staff area" button — the kiosk holds one browser
session for the whole event, so an unlock would otherwise never end.

### Disable screen blanking

An unattended kiosk goes black without this:

- **Desktop session:** Raspberry Pi Control Centre (Preferences → Raspberry Pi
  Configuration) → **Display** → **Screen Blanking** → off.
- **Console (text) blanking:** append `consoleblank=0` to
  `/boot/firmware/cmdline.txt` (keep everything on the single existing line) —
  covers the brief console phase before the desktop and any non-desktop fallback.
