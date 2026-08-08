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
