# DermaScan displayed metrics

## CNN (3-class)

- **Benign / Pre-Cancerous / Malignant %** — model softmax, temperature-scaled for display (`models/temperature.json`, T = 1.54; `backend/tflite_shared.apply_temperature`). The screening *decision* always uses raw probabilities — the 0.11 screen threshold was tuned uncalibrated, so calibration only affects how confident the UI reports being, never the label.
- **Confidence** — calibrated probability of the predicted class × 100.

### Model quality (held-out test set, EfficientNetB0, HAM10000 → 3-class)

- Accuracy **0.887**, macro-F1 **0.746**, screening ROC-AUC **0.937**.
- Deployed sensitivity-first threshold (0.11 on p(pre)+p(mal)): cancer sensitivity **0.90**, benign specificity **0.77** (validation).
- Per-class sensitivity: benign 0.937, pre_cancerous **0.469**, malignant 0.721.

**Known limitation:** `pre_cancerous` recall is weak (0.469, only 49 test samples —
`akiec` is rare in HAM10000). The sensitivity-first screening threshold partially
compensates (most missed pre-cancerous cases are still *flagged*, just labelled
malignant), but a retrain with more `akiec`-like data / mixup / focal-alpha
re-weighting is planned future work. Re-tune the 0.11 threshold after any retrain.

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
