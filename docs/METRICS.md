# DermaScan displayed metrics

## CNN (3-class)

- **Benign / Pre-Cancerous / Malignant %** — softmax after optional temperature scaling (`models/temperature.json`, T applied to logits).
- **Confidence** — probability of the predicted class × 100.

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
