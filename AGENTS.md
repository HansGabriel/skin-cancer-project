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
| Dataset | HAM10000 **+ ISIC archive collections BCN20000, MSK-1..5, MILK10k, PAD-UFES-20** (`training/prepare_datasets.py`, labels from the archive's `diagnosis_*` tree → `benign`, `pre_cancerous`, `malignant`; UDA-1/2 are test-only) — *mentor's call 2026-10-18 after the June model missed a quarter of the cancers from other hospitals* | HAM10000 alone is one hospital's dermoscopy; the Pi sees close-up photos |
| Split | **by `lesion_id`** (GroupShuffleSplit), never by image | image-level split leaked 40% of test lesions |
| Model | EfficientNetB0, 3-class softmax head, TFLite, float32 in/out, raw `[0,255]` input. **Deployed: the October 2026 multi-source model** (threshold 0.47) — it flags 121/149 external cancers vs 90/149 for the June model (`fec7993`, restored 2026-10-03, now history in `docs/METRICS.md`); the trade is lower benign specificity (0.59 vs 0.71), accepted by the mentor 2026-10-18 | fast on the Pi CPU; EfficientNet rescales internally — never add `/255` |
| Decision | flag if `p(pre)+p(mal) >= thresholds.json` (0.47 for the October model; 0.11 was the June model's). The threshold is fitted by `training/calibrate.py` on the exported TFLite with 4-view TTA for 0.90 val sensitivity. **"Read anyway" on `no_spot` / `plain_skin` / `several_marks` (`gate.NOT_A_SPOT_CODES`) gives no verdict** (`verdict.cannot_judge`, `judged: false`); the staff drawer shows the numbers marked unreliable — *mentor's call 2026-10-29 after the 24 Oct Pi session: 8 of 9 forced reads of hair, plain skin and a pimple came out "flagged"*. Framing refusals (`gate.FRAMING_CODES`: `not_in_ring`, `fills_frame`, `soft_edge`, `ragged_spot`) still answer with the caveat and with A B C measured. **The check screen labels every photo** (`verdict.check_label`: SPOT FOUND · NOT A SPOT · PLAIN SKIN · SPOT · NOT IN THE RING · CAN'T READ · BLURRY …) and draws the gate's outline on it (`gate.outline_polygon`, ~1 ms) — *2026-10-29*. Gate limits marked `GUESSED` in `config.py` wait for real Pi captures (`tests/fixtures/gate/` is empty) | the guarantee describes the device, not the Keras model; the model has no "not a lesion" class, so on a non-spot its numbers mean nothing |
| Confidence | temperature-scaled for display only; decision on raw probs | honest numbers, same labels |
| Heatmap | **Eigen-CAM** (`dermascan/cam.py`), staff drawer: photo and heatmap side by side, tap to put one in the ring; computed on demand (~1 s), cached per photo, never on the scan path or the web demo. **Read from the deployed classifier itself** (a second interpreter on `models/skin_classifier.tflite` with `experimental_preserve_all_tensors`, the 7×7×1280 `top_activation` tensor dequantised) — *2026-10-29, replacing the fifth-file plan: no `.keras` of the October model exists*. A `models/skin_classifier_cam.tflite` + `.json` (`scripts/export_cam_tflite.py`) is still used first if present and its sha matches. Built on a background thread after boot (`warm_in_background`). Display only — *mentor's override 2026-10-29 of the 2026-10-01 removal* | LiteRT has no gradients, so Grad-CAM is impossible on the Pi; Eigen-CAM needs one forward pass; the paper must say "Eigen-CAM", not "Grad-CAM" (`docs/PAPER_ALIGNMENT.md`) |
| Gemma 4 | **Build-time only.** `scripts/draft_answers_gemma.py` on the PC (Ollama, `gemma4:e4b`) rewords the team's approved `answers.json` entries in plainer words, keeping every timeframe; drafts carry `drafted_by` and the badge "DRAFTED BY GEMMA 4 · NOT YET REVIEWED BY A DOCTOR" until `reviewed_by` is filled. **Nothing runs on the Pi, nothing is generated at the kiosk** — *mentor's call 2026-10-29* | the Pi has 4 GB; Gemma 4 E2B needs 8; the old on-Pi Ollama assistant was removed 2026-10-01 for speed; the paper's title names Gemma 4, so it must take part somewhere true |
| Kiosk UI | **Flask + one static HTML page** (`kiosk/`) — *changed from Streamlit on 2026-10-01 with the mentor's approval* | Streamlit re-ran everything per tap and took 40 s to boot on the Pi |
| Kiosk look | the **EPIVUE Interface** design from Claude Design (project `36f54779…`, file `EPIVUE Interface.dc.html`): navy instrument band + white page, sizes in `clamp(…cqh…)`, stacked under 720 px | restored 2026-10-01; match the design file, not memory |
| Web demo | **Railway**, Docker (`Dockerfile`, `railway.json`, GitHub auto-deploy from `main`), running **the kiosk page in web mode** (`kiosk.server --web`: nothing kept, no Saved, no staff routes) — *mentor's call 2026-10-13, replacing the plain Streamlit page*; `cloud/streamlit_app.py` stays as a Streamlit Community Cloud fallback | the site looks like the Pi; same `dermascan/` core; a real container that does not sleep |
| Camera | picamera2 opened once per process, never torn down; plain pixels to the model | reopening stalled the preview; CLAHE before the model hurt accuracy |
| Offline | no runtime pip, no downloads, no telemetry; nothing leaves the device. **One exception, staff-only and outside the server process:** "Update from GitHub" in Settings runs `scripts/update.sh` (fetch of the **public HTTPS URL by name** with `GIT_TERMINAL_PROMPT=0` — the Pi's SSH `origin` has no usable key from the launcher, which was the false "not reachable" of 2026-10-29 — fast-forward of `main`, `pip install` only when `requirements.txt` changed, pytest, rollback; "Show the update log" in Settings) while the kiosk is stopped — *mentor's call 2026-10-24 so the students can take Hans's pushes without a keyboard*; the scan path stays offline. Saved scans persist in `~/.dermascan/saved/` (two files per scan) until "End event" — *mentor's decision 2026-10-11, replacing RAM-only saves that vanished on every reboot* | barangay / field use |

### What was removed on 2026-10-01 (old app at commit `246c072`, `pi/last-known-good`)

GrabCut segmentation, E-evolving, saved cases on disk, the offline LLM assistant
(Ollama), Grad-CAM/Eigen-CAM (**Eigen-CAM returns 2026-10-29 by the mentor's override: staff
drawer button only, computed on demand, never on the scan path**), the feature-distance OOD gate, pixels-per-mm scale, PC-to-Pi
HTTP mode, ~40 `SKIN_*` env vars. Reason: 30–60 s scans, a gate nobody could calibrate,
and a codebase the students could not read. Do not bring them back into the kiosk path;
cite them from `246c072` in the paper if needed.

**Brought back, lighter, the same day (mentor's call):** A B C on the gate's own outline
(`dermascan/signs.py`, ~10 ms, display only — never feeds the verdict; D shows "no
scale"), saving (`kiosk/saved.py`, in RAM at the time), and a Questions tab that
matches written answers (`dermascan/answers.json`, TF-IDF, no LLM).
**Saves moved to the SD card on 2026-10-11** (`kiosk/saved.py`, `~/.dermascan/saved/`,
`<id>.jpg` + `<id>.json`, atomic writes, reload at boot): still no names, no database, no
network; "End event" empties the folder, "Exit kiosk" no longer does.

**E (evolving), 2026-10-03:** saved scans go into a visitor-chosen group (one group = one
spot); a group's history starts at its first scan, and "Compare" shows the first and the
current photo side by side for a PERSON to judge. No automatic "changed / not changed":
on 71 same-mole HAM10000 pairs, any limit that kept false "changed" under 15% caught under
5% of a simulated 35% darkening. Revisit only with real Pi captures of the same spot.

---

## Repository layout

```
dermascan/      config.py scan.py gate.py classifier.py cam.py verdict.py signs.py answers.py answers.json   — the core, no UI framework
kiosk/          server.py camera.py saved.py static/{index.html,kiosk.css,kiosk.js,fonts/}
cloud/          streamlit_app.py requirements.txt
models/         skin_classifier.tflite labels.txt thresholds.json temperature.json (+ skin_classifier_cam.tflite/.json)  — swap them together
training/       train_skin_classifier.ipynb calibrate.py eval.py split_scores.py isic_fetch.py prepare_datasets.py
                build_external_bench.py external_eval.py external_bench.csv dataset_index.csv splits.csv requirements.txt (*.keras: gitignored)
scripts/        launch_kiosk.sh update.sh run_dev.sh bench_scan.py gate_report.py export_cam_tflite.py draft_answers_gemma.py (PC, Ollama)
tests/          pytest, < 1 min, no dataset; tests/fixtures/gate/{refuse,pass}: meant for a few dozen <= 512 px real photos (no faces) — EMPTY until the 30 Oct 2026 captures
docs/           HARDWARE_CHECKLIST DEPLOYMENT METRICS PRIVACY HANDOFF PAPER_ALIGNMENT (what the students' paper must say differently)
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
   numbers only in `docs/METRICS.md`. Before replacing the deployed model, run
   `training/external_eval.py` (the frozen `training/external_bench.csv`, through the kiosk
   path) for old and new; ship the new one only if it flags at least as many cancers **and**
   is at least as specific on benign spots — or the mentor accepts the trade in writing in
   `docs/METRICS.md` (done once, 2026-10-18, for the October model). Never judge a model by the drawer's bars: the
   decision is `cancer score >= limit`, and the drawer now prints that line.
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
