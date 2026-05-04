# Agent instructions — skin-cancer-project

End-to-end technical plan for the **AI-Assisted Skin Lesion Classifier**: dataset choice, TensorFlow/Keras training (PC GPU or Colab), TFLite on Raspberry Pi 4, camera integration, PC frontend, and offline-first operation.

## Source of truth (Notion)

| Doc | URL |
|-----|-----|
| **AI Model Development & Deployment Plan** (this sync) | https://www.notion.so/AI-Model-Development-Deployment-Plan-9c33d9f582ef41f88306084b43d65b6c |
| **Parent — AI-assisted skin cancer detection platform** | https://www.notion.so/AI-assisted-skin-cancer-detection-platform-0a1108390ba2473c9906bf0a47471c68 |

**Audience:** Mentor (Hans) + research team (Maria Beatriz, Kumar, Tyeisha, Cristine Eve).

**Hardware:** Windows PC with NVIDIA **RTX 4060** + **Raspberry Pi 4 Model B** + **Pi Camera Module 2** + **32GB SD card**.

If chat instructions conflict with Notion, follow Notion unless the user overrides.

---

## Stack decisions (do not change without explicit approval)

| Area | Choice | Rationale (short) |
|------|--------|-------------------|
| **Primary dataset** | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (~10k, 7 dx classes) | Standard benchmark, Kaggle-friendly, fits 4060 training time |
| **Validation / skin-tone angle** | **MSKCC Skin Tone Labeling** + **DDI** (secondary) | Filipino / Fitzpatrick diversity story; not the sole training set for v1 |
| **Avoid for v1** | ISIC 2024 SLICE-3D at full scale | Size, compute, distribution mismatch vs dermoscopic device |
| **Framework** | **TensorFlow 2.16+** + **Keras** | **TFLite** + `tflite-runtime` on Pi is the deployment target |
| **Architecture** | **EfficientNetB0** (ImageNet), **int8 TFLite** for Pi | Accuracy/size tradeoff; **MobileNetV2** fallback if Pi latency too high |
| **Training location** | **Primary: local RTX 4060** | Faster than free Colab T4; Kaggle Notebooks acceptable for teaching |
| **PC demo UI** | **Streamlit** (`frontend/app.py`) | Simple, browser-based |
| **Pi service** | **Flask** `POST /scan` (e.g. `pi_server.py` on Pi) | PC app calls Pi over LAN |
| **Offline** | No runtime `pip`, no cloud model fetch, no telemetry; LAN only PC↔Pi | Barangay / field use case |

### Class strategy

- HAM10000 **7 dx labels** map to **3 display groups**: **malignant** (mel, bcc), **pre_cancerous** (akiec), **benign** (nv, bkl, df, vasc).
- **Design note:** The written plan argues for a **7-class softmax + post-hoc collapse** for defensibility; the reference notebook cells in Notion use a **3-class head** end-to-end. When implementing, pick one approach per milestone and keep **labels.txt**, metrics, and UI in sync.

### Preprocessing contract

- **Training (Keras):** `efficientnet.preprocess_input` on RGB `224×224`.
- **TFLite / OpenCV on Pi:** match training — typically `(img / 127.5) - 1.0` on resized RGB unless quantization changes input dtype (document if full int8 I/O is enabled).

---

## Repository layout (PC — align with this repo)

Project root: `skin-cancer-project` (e.g. `c:\Users\Hans_\codes\skin-cancer-project`).

```
datasets/ham10000/          # HAM10000_metadata.csv, HAM10000_images_part_1|2, test_samples/
notebooks/train_skin_classifier.ipynb   # main training notebook (per plan)
models/                     # skin_classifier_full.keras, skin_classifier.tflite, labels.txt
frontend/app.py             # Streamlit demo (calls Pi)
scripts/test_inference_pc.py
requirements.txt
venv/
```

**Pi (`~/skin-classifier/`):** `skin_classifier.tflite`, `labels.txt`, `classify.py`, `pi_server.py`, `test_with_files.py`, `samples/`, `captures/`, optional `predictions_log.csv`, `venv/`.

---

## Phased work checklist (from plan)

1. **Data** — Download HAM10000 (Kaggle or [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)); place under `datasets/ham10000/`; stratified train/val/test.
2. **Train** — Notebook: `tf.data` pipeline, augmentation, class weights, EfficientNetB0 two-phase (frozen then partial unfreeze), early stopping + checkpoint; evaluate with report + confusion matrix.
3. **Export** — Save `.keras`; convert **TFLite** with representative dataset for quantization; ship `labels.txt`.
4. **PC sanity check** — `scripts/test_inference_pc.py` on held-out images before Pi.
5. **Pi** — 64-bit Pi OS, `python3-picamera2`, `venv`, **`tflite-runtime`** (not full TF on Pi); copy `.tflite` + labels; `test_with_files.py` then `classify.py` with camera.
6. **Integration** — Flask on Pi (`/scan`); Streamlit on PC; verify LAN path (`raspberrypi.local` or static IP when offline Ethernet / hotspot).
7. **Hardware milestone (week ~6 plan)** — LED ring, enclosure, macro lens; tune capture for dermoscopic-like quality.
8. **Offline verification** — Disconnect internet; PC↔Pi only; run full demo per plan checklist (Part 8.4).

### Suggested timeline (8-week cadence from plan)

| Week | Goal | Output |
|------|------|--------|
| **2** | Train v1, baseline metrics | `skin_classifier_full.keras` (~85%+ test acc target on 3-class formulation) |
| **4** | Pi camera E2E | `classify.py` live capture → TFLite |
| **6** | Imaging hardware complete | Stable LED + enclosure + macro captures |
| **8** | Demo + paper + poster | Science-fair ready |

---

## CUDA / Python note (PC)

Plan recommends **Python 3.11** and **TF 2.16.x** with **CUDA 12.3 + cuDNN 8.9** for GPU on Windows. This repo’s `requirements.txt` may differ; when upgrading TF/CUDA, update pins and document in README.

---

## Agent defaults

1. **Phase alignment** — Implement the next unchecked milestone above; avoid unrelated refactors.
2. **Clinical safety** — Screening only; never imply diagnosis or treatment; keep disclaimers in CLI, Streamlit, and Pi output.
3. **Reproducibility** — Seeds, stratified splits, versioned deps, documented preprocessing.
4. **Evaluation** — Classification report, confusion matrix, discussion of imbalance and skin-tone generalization limits.
5. **Secrets** — No API keys or Kaggle tokens in git; use env vars or local config ignored by VCS.
6. **Offline** — No runtime external downloads, analytics, or CDN-only assets for demo-critical paths.

---

## References (from plan)

- EfficientNet: [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- TF vs PyTorch (medical imaging): [arXiv:2507.14587](https://arxiv.org/abs/2507.14587)
- MDPI edge / TFLite on Pi: [MDPI Applied Sciences 2025](https://www.mdpi.com/2076-3417/15/6/3077)
- HAM10000: Tschandl et al., *Scientific Data* 2018
- MSKCC skin tone dataset: [ISIC DOI 10.34970/962049](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/)
- TensorFlow Lite Python: https://www.tensorflow.org/lite/guide/python  
- Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf  
- ISIC Archive: https://www.isic-archive.com/

---

## Summary line

**HAM10000** + **TensorFlow/Keras EfficientNetB0** → train on **RTX 4060** → **TFLite** → **Pi 4 + Camera Module 2** + **`tflite-runtime`** → **Streamlit on PC** + **Flask on Pi** → verify **fully offline** LAN operation for field demos.
