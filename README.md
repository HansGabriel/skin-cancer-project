# E.P.I.V.U.E. — skin check kiosk

A Raspberry Pi 4 with a camera module photographs one spot on skin, a small on-device
model sorts it into *benign / pre-cancerous / malignant*, and the screen says in plain
words what to do next. It runs with no internet. **It is a screening aid for a science
project, not a diagnosis.**

## Read the code in this order

Everything that decides anything is in `dermascan/` (about 550 lines). The rest is the
screen around it.

| file | what it does |
|---|---|
| `dermascan/config.py` | every number you can tune, each with a one-line reason |
| `dermascan/scan.py` | `run_scan(jpeg_bytes)`: decode → gate → model → words. The one function the UI calls |
| `dermascan/gate.py` | "is this a readable photo of one spot on skin?" — skin colour, focus and light, one compact spot, a real edge |
| `dermascan/classifier.py` | loads the TFLite model, averages 4 flipped views, applies the screening threshold |
| `dermascan/verdict.py` | label + confidence → the headline, body and advice a visitor reads |
| `kiosk/server.py` | Flask: serves the page, streams the camera, answers `/scan` |
| `kiosk/camera.py` | owns the Pi camera for the life of the process; plain pixels, no enhancement |
| `kiosk/static/index.html`, `kiosk.css`, `kiosk.js` | the one page, five states |
| `cloud/streamlit_app.py` | the same core behind an upload box, for Streamlit Community Cloud |
| `training/` | the notebook that makes the model, and `eval.py` that scores it |

## Run it on a laptop (no Pi)

```bash
uv venv --python 3.12 venv && source venv/bin/activate   # or: python3.12 -m venv venv
pip install -r requirements.txt
scripts/run_dev.sh          # then open http://127.0.0.1:8080
pytest -q                   # 80+ tests, under a minute, no dataset needed
```

The page offers your laptop's camera and a file picker. Every photo is scanned by the
exact code the Pi runs.

## Run it on the Pi

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Short version: clone to
`~/Documents/skin-cancer-project`, make a venv with `--system-site-packages` so apt's
`picamera2` is visible, `pip install -r requirements.txt`, double-click `EPIVUE.desktop`.

## How a scan works

```
tap "Take the photo"        kiosk/camera.py      1024 px centre crop, JPEG, ~0.1 s
tap "Check this spot"       dermascan/scan.py
   decode + shrink          ~20 ms
   gate                     dermascan/gate.py    ~50 ms   refuses walls, blur, plain skin, shadows
   model (4 views)          dermascan/classifier.py       ~0.6 s on a Pi 4
   verdict                  dermascan/verdict.py          words on screen
```

The gate is what stops the model being asked about a photo of nothing: the model has
only three labels and will always pick one. Each refusal says what to point the camera
at instead; the "spot" refusals can be overridden with "Check it anyway".

## Change a threshold, swap a model

* Any number: edit `dermascan/config.py`, restart. That is the whole procedure.
* A new model: run the notebook on the PC, then `python training/eval.py`, then copy
  the four files it wrote into `models/` and commit. The threshold and temperature are
  fitted to the model they came with, so always swap all four together.

## Documents

* [docs/HARDWARE_CHECKLIST.md](docs/HARDWARE_CHECKLIST.md) — the panel ribbon, the case, power
* [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Pi setup, autostart, Streamlit Cloud
* [docs/METRICS.md](docs/METRICS.md) — how good the model is, honestly
* [docs/PRIVACY.md](docs/PRIVACY.md) — nothing is stored; what that means for the fair
* [AGENTS.md](AGENTS.md) — the technical plan and the decisions behind it

The earlier, much larger version of this app (ABCDE measurements, saved cases, an
offline assistant, heat maps) is on the `main` branch if the paper needs it.
