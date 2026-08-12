# DermaScan displayed metrics

## CNN (3-class)

- **Benign / Pre-Cancerous / Malignant %** — model softmax, temperature-scaled for display (`models/temperature.json`, T = 1.54; `backend/tflite_shared.apply_temperature`). The screening *decision* always uses raw probabilities — the 0.11 screen threshold was tuned uncalibrated, so calibration only affects how confident the UI reports being, never the label. (Label metrics below are therefore temperature-independent.)
- **Confidence** — calibrated probability of the predicted class × 100.

### Model quality — deployed model, production configuration

Measured 2026-08-08 with `scripts/eval_threshold.py --production` (and
`--production --clean-subset`): the deployed `models/skin_classifier.tflite`,
**4-view TTA on** exactly as the Streamlit backend serves scans
(`backend/tflite_shared.run_inference_on_rgb`, `SKIN_TTA=1` default), decisions
on raw probabilities at the deployed 0.11 screen threshold
(`models/thresholds.json`). All 1503 test images readable in both runs; no
TTA-off fallback was needed.

| Metric (production config) | Full test (n=1503) | Leakage-free subset (n=897) |
|---|---|---|
| **Deployed screening (0.11 threshold)** | | |
| Cancer sensitivity | **0.911** (267/293) | **0.903** (102/113) |
| Benign specificity | 0.736 (891/1210) | 0.855 (670/784) |
| Cancers missed (called benign) | 26 (pre 4/49, mal 22/244) | 11 (pre 1/25, mal 10/88) |
| 3-class accuracy at threshold | 0.754 | 0.843 |
| **Argmax (no screening threshold)** | | |
| 3-class accuracy | 0.857 | 0.903 |
| Benign recall | 0.917 | 0.952 |
| Pre-cancerous recall | 0.429 (21/49) | 0.360 (9/25) |
| Malignant recall | 0.643 (157/244) | 0.625 (55/88) |
| Cancer sensitivity / benign specificity | 0.652 / 0.917 | 0.619 / 0.952 |

**Read the columns per-row, not the accuracy row across:** the leakage-free
subset has a different class mix (multi-image lesions are disproportionately
malignant, so 64% of malignant test images are excluded vs 35% of benign —
subset counts: benign 784, pre-cancerous 25, malignant 88). Aggregate accuracy
*rises* on the clean subset purely because it is more benign-heavy; the honest
signal is the per-class rows, where cancer-class recall **drops** once lesions
the model may have partially memorized are removed (malignant 0.643 → 0.625,
pre-cancerous 0.429 → 0.360 at argmax). The deployed sensitivity-first screen
holds up: **~0.90 cancer sensitivity on the leakage-free subset** (0.903 vs
0.911 full), with specificity 0.855.

### Split leakage limitation

The checked-in train/val/test split was made **per image, not per lesion**.
HAM10000 deliberately contains multiple photographs of the same lesion (linked
by `lesion_id` — different magnifications, angles, cameras; Tschandl,
Rosendahl & Kittler 2018, *Scientific Data* 5:180161). As a result, **606 of
the 1503 test images (40%) share a lesion with at least one training/validation
image** — the model has effectively seen those lesions before, which inflates
naive test metrics. Duplicate and near-duplicate leakage across ISIC-derived
splits is a documented benchmark-inflation problem (Cassidy et al. 2022,
*Medical Image Analysis* 75:102305). The "leakage-free subset" column above
scores only test images whose entire lesion sits inside the test set (897
images); cite those numbers when honesty matters. Full background and the
lesion-grouped retrain plan: `docs/FINAL_IMPROVEMENTS_RESEARCH.md` (Track A).
Any retrain must split by `lesion_id` and re-tune the 0.11 threshold.

**Known limitation:** `pre_cancerous` recall is weak (0.429 full / 0.360 clean
at argmax — `akiec` is rare in HAM10000, only 49 full / 25 clean test samples).
The sensitivity-first screening threshold partially compensates: most missed
pre-cancerous cases are still *flagged* (only 4/49 full, 1/25 clean end up
called benign), just labelled malignant. A retrain with more `akiec`-like data /
mixup / focal-alpha re-weighting is planned future work.

> **Historical numbers (do not cite):** accuracy 0.887 / macro-F1 0.746 /
> screening ROC-AUC 0.937, and threshold sensitivity 0.90 / specificity 0.77
> "(validation)". These came from the training notebook's **Keras float model**
> evaluated single-view on the **leaky image-level split** (the validation pair
> is the threshold-tuning target from `models/thresholds.json`, also leaky).
> They describe neither the deployed TFLite artifact nor the production serving
> path and are kept only so old citations can be recognised.

## Composite risk (0–100)

`0.55 × P(malignant) + 0.35 × (tier_A + tier_B + tier_C + tier_D + tier_E) / 10 + 0.10 × evolution_weight`

- **Bands:** &lt;34 low, 34–66 moderate, &gt;66 high.

## ABCDE (educational)

| Letter | Meaning | Units |
|--------|---------|-------|
| A | Asymmetry score | 0–1 |
| B | Border irregularity | score |
| C | Colour clusters | count |
| D | Diameter | mm (needs `pixels_per_mm`) |
| E | Evolving | mm diameter growth vs earliest scan; colour drift = mean LAB ΔE between matched cluster centers |

## E-Evolving tiers

- **Stable:** Δ diameter ≤ 0.5 mm and colour drift ≤ 8.
- **Watch:** 0.5–2.0 mm or drift 8–18.
- **Changing:** &gt;2.0 mm or drift &gt;18 or large border change.

## Storage

- Case DB: `~/.dermascan/dermascan.db` (override: `DERMASCAN_DATA_DIR`).
- Backend scan log: `logs/scans.sqlite` (separate inference history).

## Segmentation cost (GrabCut skip margin)

`frontend/services/segmentation.py` builds several candidate masks and keeps the
one whose foreground fraction is closest to `_IDEAL_FRAC = 0.18`. GrabCut was
the 7th candidate and ran on every scan. Measured at the 1024×1024 frame the Pi
captures: the six threshold candidates cost **~78 ms together**, GrabCut alone
**~2200 ms** — 99% of segmentation and roughly 85% of a 30-second Pi scan.

It is now skipped when the best cheap candidate already scores within
`_GRABCUT_SKIP_SCORE`. That bound is exact rather than empirical, because
`_score_mask` depends *only* on foreground fraction and ties break toward the
cheap candidate — so GrabCut can only change the outcome by scoring strictly
lower. Chosen against the old always-run behaviour on 20 real HAM10000 images:

| margin | GrabCut skipped | mask differs | ABCD tier flips |
|---|---|---|---|
| 0.02 | 30% | 0 | 0 |
| 0.05 | 40% | 0 | 0 |
| **0.10 (shipped)** | **60%** | **0** | **0** |
| 0.20 | 90% | 1 | 1 |

Result on 20 real images (x86, warm interpreter): **median scan 158 ms**, down
from ~2300 ms. The mean is ~1000 ms because the 40% that still need GrabCut
remain expensive — that tail is where any further optimisation has to go.

**Two ways of speeding up the remaining 40% were measured and rejected**, both
because they move a displayed ABCDE tier. Border score is `P²/(4πA)`, so it
reacts to small changes in contour roughness far more than mask overlap
suggests — a high IoU is *not* evidence that the tiers survived.

- *GrabCut downscaled to 384px + linear upscale*: 16× faster, but mask IoU
  against the full-resolution result was 0.756 (min 0.383) and it flipped the
  border tier on **4 of 10 images**.
- *Fewer GrabCut iterations* (`cv2.grabCut(..., iters, ...)`, currently 5):
  even 5→3 flipped an ABCD tier on **6 of 8 images** for a 30% speedup. IoU
  stayed at 0.930, which is exactly the trap above.

  | iters | ms/img | IoU vs 5 | ABCD tier flips |
  |---|---|---|---|
  | 5 (shipped) | baseline | — | — |
  | 3 | −30% | 0.930 | 6/8 |
  | 2 | −42% | 0.904 | 6/8 |
  | 1 | −58% | 0.835 | 5/8 |

Anything further should change *which mask is chosen* (as the skip margin does)
rather than *how that mask is computed*, or else re-validate the ABCDE tiers
end-to-end and update this file.

## Capture quality thresholds

> **Recalibrated 2026-08-13 against real dermoscopy.** The thresholds below were
> previously tuned against `samples/*.jpg`, which `samples/README.md` documents
> as **synthetic placeholders (simple colour patches)**. Measured on 200 random
> HAM10000 images, that mistake was costing almost every real scan:
>
> | | focus score |
> |---|---|
> | `samples/*.jpg` (synthetic) | 449 / 435 / 469 |
> | Real HAM10000 (n=200) | **median 79**, p25 55, p10 42 |
> | Old soft threshold | 120 → **72.5% of real lesions flagged** |
> | New soft threshold | **40** → 9.5% flagged |
>
> A flagged photo is *refused outright* whenever strict mode is on, and strict
> mode was defaulted on (`app.py`) despite `settings_view.py` documenting it as
> off. Together those two facts refused roughly 72% of genuine dermoscopic
> images. The hard stop moved 25 → 20 because a real HAM10000 image measured
> 24.8 — losing by 0.2 is not "unusable".
>
> The same audit found the skin gate rejecting 11% of real images: the YCrCb
> box's `Cb <= 127` ceiling excludes polarized and oil-immersion dermoscopy,
> which sits at Cb 133–138. Raising it to 140 dropped that to 2.0% with no
> measured junk surface crossing over. `_SKIN_MIN` is 0.08, chosen for junk
> margin rather than for real images — between 0.06 and 0.12 the real rejection
> rate is unchanged at 2.0%, while 0.08 sits well above the worst neutral
> surface in `tests/test_gate_robustness.py` (textured dark grey, 0.057).
>
> `tests/test_gate_real_images.py` holds these lines against real images and is
> skipped when `datasets/` is absent. **Do not re-tune any of these against
> `samples/`** — that is the error this note exists to stop repeating.

The focus check is a Laplacian variance, but two normalisations are applied
first (`frontend/services/quality.py`), because the raw number lies in two ways:

- **Size** — variance grows with resolution, so the same scene graded
  differently from the 4056px Pi camera and a 640px webcam. Downscaled to at
  most 512px, never upscaled (upscaling interpolates detail away and makes a
  sharp photo look soft).
- **Contrast** — variance scales with the square of contrast, so a dim but
  perfectly sharp photo read as "blurry". Stretched to full range first;
  exposure is reported separately by the light meter and is only advisory.

Measured on the dermoscopic images in `samples/` (regression-tested in
`tests/test_quality_levels.py`):

| Image | Focus score |
|---|---|
| Real dermoscopic sample | ~450 |
| Same sample, Gaussian blur 5×5 | ~52 |
| Same sample, Gaussian blur 11×11 | ~17 |
| Featureless frame | 0 |

Thresholds sit in that gap: **advisory below 120**, **hard stop below 25**.

This matters more than it sounds. A lesion close-up is smooth skin with a
soft-edged mark in it, so its variance is inherently low, while a synthetic
checkerboard scores in the thousands. The previous default of 35 was tuned on
synthetic patterns and sat *above* every real sample — which is why genuine
lesions were rejected with "image too blurry".

### Skin detection

`frontend/services/lesion_gate.py` identifies skin from YCrCb chroma, with
exposure normalised first. Two failure modes drove that design, both invisible
to tests on raw arrays:

- **JPEG chroma quantisation** nudges near-neutral pixels toward the skin box: a
  grey desk measured 5% skin as an array and 15% after a quality-92 JPEG, over
  the 12% threshold. Fixed by the saturation floor of 30.
- **Low light crushes chroma**, so a real lesion shot in a dim hall came back as
  "no skin was found". Fixed by lifting the frame to a standard brightness
  before judging colour.

Both are regression-tested through the real encode in
`tests/test_gate_robustness.py`, across seven skin tones and five exposures.
