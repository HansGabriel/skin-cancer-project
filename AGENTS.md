# Agent instructions — skin-cancer-project

**AI-Assisted Skin Lesion Classifier**: HAM10000 → TensorFlow/Keras EfficientNetB0 trained
on an RTX 4060 → TFLite on a Raspberry Pi 4 with Camera Module 2 → a Flask kiosk on a 7"
1024×600 touch panel → fully offline.

## Source of truth (Notion)

| Doc | URL |
|-----|-----|
| **AI Model Development & Deployment Plan** | https://www.notion.so/AI-Model-Development-Deployment-Plan-9c33d9f582ef41f88306084b43d65b6c |
| **Parent — AI-assisted skin cancer detection platform** | https://www.notion.so/AI-assisted-skin-cancer-detection-platform-0a1108390ba2473c9906bf0a47471c68 |

**Audience:** Mentor (Hans) + research team (Maria Beatriz, Kumar, Tyeisha, Cristine Eve).
**Hardware:** Windows PC with RTX 4060 (training, WSL) · Raspberry Pi 4 Model B · Pi Camera
Module 2 · 7" 1024×600 HDMI touch panel · Mac for development.
**Demo:** 13 October 2026.

If chat instructions conflict with Notion, follow Notion unless the user overrides.

---

## Stack decisions (do not change without explicit approval)

| Area | Choice | Why |
|------|--------|-----|
| Dataset | HAM10000, 7 dx → 3 labels (`benign`, `pre_cancerous`, `malignant`) | standard, fits the 4060 |
| Split | **by `lesion_id`** (GroupShuffleSplit), never by image | image-level split leaked 40% of test lesions |
| Model | EfficientNetB0, 3-class softmax head, TFLite, float32 in/out, raw `[0,255]` input. **Deployed: the June 2026 model (`fec7993`), restored 2026-10-03** — the 1 Oct retrain (`e1579a0`) missed 3/17 melanomas the June model caught on photos neither had seen (`docs/METRICS.md`) | fast on the Pi CPU; EfficientNet rescales internally — never add `/255` |
| Decision | flag if `p(pre)+p(mal) >= thresholds.json` (0.11 for the June model). A new model's threshold is fitted by `training/calibrate.py` on the exported TFLite with 4-view TTA for 0.90 val sensitivity | the guarantee describes the device, not the Keras model |
| Confidence | temperature-scaled for display only; decision on raw probs | honest numbers, same labels |
| Kiosk UI | **Flask + one static HTML page** (`kiosk/`) — *changed from Streamlit on 2026-10-01 with the mentor's approval* | Streamlit re-ran everything per tap and took 40 s to boot on the Pi |
| Kiosk look | the **EPIVUE Interface** design from Claude Design (project `36f54779…`, file `EPIVUE Interface.dc.html`): navy instrument band + white page, sizes in `clamp(…cqh…)`, stacked under 720 px | restored 2026-10-01; match the design file, not memory |
| Web demo | Streamlit Community Cloud, `cloud/streamlit_app.py`, upload only | same `dermascan/` core |
| Camera | picamera2 opened once per process, never torn down; plain pixels to the model | reopening stalled the preview; CLAHE before the model hurt accuracy |
| Offline | no runtime pip, no downloads, no telemetry; nothing written to disk (saved scans live in RAM until "End event") | barangay / field use |

### What was removed on 2026-10-01 (old app at commit `246c072`, `pi/last-known-good`)

GrabCut segmentation, E-evolving, saved cases on disk, the offline LLM assistant
(Ollama), Grad-CAM/Eigen-CAM, the feature-distance OOD gate, pixels-per-mm scale, PC-to-Pi
HTTP mode, ~40 `SKIN_*` env vars. Reason: 30–60 s scans, a gate nobody could calibrate,
and a codebase the students could not read. Do not bring them back into the kiosk path;
cite them from `246c072` in the paper if needed.

**Brought back, lighter, the same day (mentor's call):** A B C on the gate's own outline
(`dermascan/signs.py`, ~10 ms, display only — never feeds the verdict; D shows "no
scale"), saving in server RAM only (`kiosk/saved.py`), and a Questions tab that
matches written answers (`dermascan/answers.json`, TF-IDF, no LLM).

**E (evolving), 2026-10-03:** saved scans go into a visitor-chosen group (one group = one
spot); a group's history starts at its first scan, and "Compare" shows the first and the
current photo side by side for a PERSON to judge. No automatic "changed / not changed":
on 71 same-mole HAM10000 pairs, any limit that kept false "changed" under 15% caught under
5% of a simulated 35% darkening. Revisit only with real Pi captures of the same spot.

---

## Repository layout

```
dermascan/      config.py scan.py gate.py classifier.py verdict.py signs.py answers.py answers.json   — the core, no UI framework
kiosk/          server.py camera.py saved.py static/{index.html,kiosk.css,kiosk.js,fonts/}
cloud/          streamlit_app.py requirements.txt
models/         skin_classifier.tflite labels.txt thresholds.json temperature.json  — swap all four together
training/       train_skin_classifier.ipynb calibrate.py eval.py split_scores.py requirements.txt splits.csv (*.keras: gitignored)
scripts/        launch_kiosk.sh run_dev.sh
tests/          pytest, < 1 min, no dataset
docs/           HARDWARE_CHECKLIST DEPLOYMENT METRICS PRIVACY
```

Pi install path: `~/Documents/skin-cancer-project` (`EPIVUE.desktop` assumes it).

---

## Agent defaults

1. **Keep it readable.** Every module starts with a docstring a student can follow; every
   tunable lives in `dermascan/config.py` with a one-line reason; no env vars except
   `DERMASCAN_PASSCODE` (the Exit code, also read from `~/.dermascan_passcode`).
2. **Clinical safety.** Screening only. Visitor-facing risk copy lives only in
   `dermascan/verdict.py` and `dermascan/answers.json`, plain words, no jargon, no
   percentages, never a green "safe". `tests/test_verdict.py` and `tests/test_answers.py`
   enforce it. Answers stay "not yet reviewed by a doctor" until `reviewed_by` is filled.
3. **Calibrate on the real path.** Thresholds are measured through a real JPEG encode
   (`tests/conftest.py`), never on raw arrays, never on `samples/` style patches.
   Real Pi captures (`SAVE_CAPTURES_DIR`) beat synthetic frames whenever they exist.
4. **Measure speed from the log line** (`scan status=... ms decode= gate= model= signs=`),
   target < 5 s on the Pi.
5. **Reproducibility.** Seeds, lesion-grouped split, versioned deps, `training/eval.py`
   numbers only in `docs/METRICS.md`. Before replacing the deployed model, compare old vs
   new on photos neither model trained on, through the kiosk path; ship the new one only
   if it flags at least as many cancers.
6. **Secrets.** None needed; keep it that way.
7. **Pi work:** give copy-paste commands rather than running installs unprompted; confirm
   the branch before pulling on the Pi.

---

## CUDA / Python note (PC)

Training uses WSL + the venv's NVIDIA wheels (`training/requirements.txt`); cell 0 of the
notebook sets `LD_LIBRARY_PATH` and must import TensorFlow in the same cell. Python 3.12.
On the Mac, Homebrew's python@3.12 is broken; use `uv venv --python 3.12`.

## References

- EfficientNet: https://arxiv.org/abs/1905.11946
- HAM10000: Tschandl et al., *Scientific Data* 2018
- Split leakage in ISIC-derived sets: Cassidy et al. 2022, *Medical Image Analysis* 75:102305
- TensorFlow Lite Python: https://www.tensorflow.org/lite/guide/python
- Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
