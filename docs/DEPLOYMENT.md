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

- B.1 Fit `models/temperature.json` via `scripts/fit_temperature.py` (currently `T=1.0` placeholder).
- B.2 Wire test-time augmentation in `backend/tflite_shared.py`.
- B.5 Add runtime `SKIN_MODEL_PATH` switch for the INT8-only re-export.
- Tier-2: implement `frontend/services/uncertainty.py` (MC-Dropout) and `frontend/services/skintone.py` (Fitzpatrick/ITA).
- Tier-3: implement `frontend/services/report.py` (PDF export).
