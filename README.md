# Skin lesion screening prototype (HAM10000 → TFLite → Pi)

AI-assisted **screening** demo: EfficientNet-style classifier on PC, TensorFlow Lite on a Raspberry Pi 4 with Camera Module 2, and a Streamlit UI that can call the Pi over the LAN.

**Technical source of truth:** see [AGENTS.md](AGENTS.md) (dataset, stack, offline-first constraints, Notion links).

**Not medical advice.** Outputs are for research and education only.

## Quick start (PC)

1. Python **3.11+** recommended (project venv may use 3.12; see `requirements.txt`).
2. Create a venv and install PC dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **WSL / desktop:** do not rely on `pip install tflite-runtime` here — Google often ships **no wheel** for Linux x86_64 + Python 3.12, so pip shows *“from versions: none”*. Use `requirements.txt` on the PC, or a lighter TFLite-only option: `pip install ai-edge-litert` (same interpreter path Streamlit uses when `tflite-runtime` is missing). Reserve `tflite-runtime` for the **Raspberry Pi** venv when a wheel exists for that platform.

3. Place `models/skin_classifier.tflite` and `models/labels.txt` (or set `SKIN_MODEL_PATH` / `SKIN_LABELS_PATH`).
4. Run the demo:

   ```bash
   streamlit run frontend/app.py
   ```

5. Run unit tests (no GPU required):

   ```bash
   pytest tests -q
   ```

   (`tests/conftest.py` adds `frontend/` to `PYTHONPATH`; use the project venv.)

## Raspberry Pi (inference + camera)

1. Copy `skin_classifier.tflite`, `labels.txt`, and `scripts/pi_server.py` to the same directory on the Pi (see [AGENTS.md](AGENTS.md) layout), or adjust paths inside `pi_server.py`.
2. Install Picamera2 from **apt** where possible (`python3-picamera2` metapackage on Raspberry Pi OS).
3. Create a venv and install Pi-only wheels (unpinned so pip can pick any wheel your platform supports; then `pip freeze > requirements-pi.txt` if you want a pin file):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy opencv-python-headless tflite-runtime flask
   ```

   If **`tflite-runtime` fails with “No matching distribution”**, your Python/OS combo may have no published wheel (common on **PC or WSL with Python 3.12**). Install on a **Raspberry Pi** venv instead, or on the PC use the full project `requirements.txt` / `ai-edge-litert` for Streamlit local inference (see `backend/streamlit_resources.py`).

4. Start the server: `python pi_server.py` (default `http://0.0.0.0:5000`).
5. On the PC, set Streamlit sidebar Pi URL or env `PI_BASE_URL` (default `http://raspberrypi.local:5000`).

**Debug:** `POST /scan` accepts optional multipart field `image` (JPEG/PNG) to bypass the camera.

## Repository layout (short)

| Path | Role |
|------|------|
| `notebooks/train_skin_classifier.ipynb` | Training + export |
| `backend/` | Inference backends (mock / local TFLite / Pi HTTP) |
| `frontend/app.py` | Streamlit UI |
| `scripts/pi_server.py` | Flask API on the Pi |
| `requirements.txt` | PC training + Streamlit (full pin set) |
| `streamlit-requirements.txt` | Slim deps for Streamlit Community Cloud (no CUDA / no full TF) |
| `requirements-pi.txt` | Raspberry Pi runtime only |

## Deploy online (Streamlit Community Cloud)

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full recipe. In short: push
the branch, point share.streamlit.io at `frontend/app.py` with
`streamlit-requirements.txt`, and set a `[dermascan] passcode` secret. Grad-CAM
is intentionally disabled on the hosted demo; classification + ABCDE +
E-Evolving + risk scoring all run via the TFLite path.
