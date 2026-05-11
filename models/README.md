# Model artifacts (`models/`)

Training is set up so exports land **here** (repo root `models/`), as long as you run the notebook with the project root as the working directory (typical in Jupyter / VS Code).

After **`notebooks/train_skin_classifier.ipynb`** export cell runs, you should see files such as:

- **`skin_classifier_full.keras`** — full Keras model (gitignored if present).
- **`skin_classifier.tflite`** — quantized TFLite for Streamlit + Pi (**gitignored**; required for the demo UI).
- **`labels.txt`** — one class label per line (`benign`, `pre_cancerous`, `malignant`). Usually committed.
- **`test_split.csv`** — optional holdout metadata from the notebook.

The Streamlit app defaults to **`models/skin_classifier.tflite`** and **`models/labels.txt`**. If you moved files, set `SKIN_MODEL_PATH` / `SKIN_LABELS_PATH`.

**Streamlit deps (same venv as training is fine):**

```bash
pip install streamlit opencv-python-headless numpy pandas pillow requests
```

On **Python 3.12+**, `tflite-runtime` often has no wheel; use:

```bash
pip install ai-edge-litert
```

The Raspberry Pi usually uses `tflite-runtime` only.
