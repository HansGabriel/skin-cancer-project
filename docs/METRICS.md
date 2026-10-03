# How good is the model, honestly

## The deployed model

`models/skin_classifier.tflite` is the **first model (June 2026, commit `fec7993`)**,
EfficientNetB0, 3 classes, float32 in/out, decision at `p(pre)+p(mal) >= 0.11`
(`models/thresholds.json`), temperature 1.54 for the displayed numbers. It was
**restored on 3 October 2026** after the 1 October retrain read real lesions worse.

**Why it was restored: a head-to-head on photos neither model ever saw.** 80 HAM10000
images whose lesions are wholly inside BOTH models' test splits (the old image-level one
and the new lesion-grouped one), fetched from the ISIC archive, run through the kiosk's
own path (JPEG decode, 1024 px, 4-view TTA, each model with its own threshold):

| | First model (deployed) | 1 Oct retrain (rolled back) |
|---|---|---|
| Melanomas flagged | **17 / 17** | 14 / 17 (3 called benign) |
| Cancer sensitivity (pre + mal flagged) | 1.00 | 0.87 |
| Benign specificity | 0.81 | 0.86 |

57 benign, 6 pre-cancerous, 17 malignant: small, so read it as "the retrain misses
melanomas the first model catches", not as precise rates. On 199 images from the new test
split the gap is the same direction (malignant 67/68 vs 49/68), but the first model may
have trained on some of those lesions, so that set flatters it.

**The 1 October retrain, for the record.** Lesion-grouped split, camera-style
augmentation, threshold 0.235 fitted on the exported TFLite. Test split (n=1494):
sensitivity 0.874, specificity 0.698, malignant recall at argmax 0.447 (its own Keras
model scored 0.620 - the int8 export lost most of it). Kept in git at `e1579a0`; the
lesion-grouped split and the augmentation are still the right recipe, the export is what
needs fixing (float16 or dynamic-range is the obvious next try).

**The first model's own figures are optimistic.** It was trained and tested on an
image-level split that put 40% of test images' lesions into training: sensitivity 0.911
at 0.11 and specificity 0.736 on that test set, 0.903 / 0.855 on its leakage-free subset.
Quote the head-to-head above, not those. Its threshold was fitted on validation
probabilities from training, not on the exported model with TTA as `calibrate.py` does.

**What is not measured at all.** Every number above is on dermoscopy images. The kiosk
is a bare camera module, and the students demo by photographing images on a phone
screen. No accuracy figure exists for that input. The camera-style augmentation is a
mitigation, not a measurement.

To refresh this table after a retrain: run `python training/calibrate.py`, then
`python training/eval.py`, and paste what it prints. Do not mix numbers from two models.

## Displayed confidence

The percentages in the staff drawer are temperature-scaled (`T` in
`models/temperature.json`, fitted on validation data). The decision never uses them; it
uses the raw probabilities, so calibration can change how sure the screen sounds but
never which label it shows. `tests/test_classifier.py` holds that invariant.

## The photo gate (dermascan/gate.py)

Four checks, every one on a small copy, every threshold in `dermascan/config.py`:

| check | refuses | measured on | threshold |
|---|---|---|---|
| skin fraction | walls, desks, screens showing non-skin | YCrCb + HSV colour box, exposure-normalised | ≥ 0.08 |
| focus / brightness | blur, black frames, glare | Laplacian variance at 512 px; mean V | ≥ 20; 12–246 |
| one spot | plain skin, speckle, whole-frame dark | Otsu outline (hairs filled in, dark frame edges set aside): area 0.4–75%, solidity ≥ 0.6, Lab contrast ≥ 5 | |
| real edge | shadows, lighting gradients | lightness drop ÷ edge steepness, % of diagonal | ≤ 5.5 (overridable) |

**On 267 real HAM10000 lesions through JPEG (3 Oct 2026): 91% pass** (was 69%). The
refusals fixed: hairs across a mole outlined instead of the mole, a dark vignette at the
frame edge outlined instead of a pale mole, and real edges (median 3.1, max 4.9) refused
as shadows by the old 4.0 limit. Bare skin, walls, desks and shadows (6–13) are still
refused. The remaining refusals are mostly faint pink marks and blurred photos.

Measured on synthetic frames through a real JPEG encode (`tests/test_gate.py`): every
lesion fixture passes (dark, pale, hairy, irregular, vignetted, inside a shadow, blurred
to 10 px); bare skin, gradients, shadows, walls are refused. **No real Pi captures have
been measured.** Setting `SAVE_CAPTURES_DIR` in `config.py` keeps each capture with its
gate numbers so the thresholds can be checked against the real camera.

The edge check is the one the plan had listed as dropped. It came back because without it
a bare forearm under a lamp or in a soft shadow passes the other three checks and gets a
verdict (measured: edge width 6–12 for those, 1–3 for every lesion fixture, including one
blurred by 14 px). It is the least proven of the four, so it is overridable and its number
is in every log line.

What was dropped from the earlier gate, and why: moiré/screen detection (the demo input
*is* a screen), dark-blob dominance, contrast-variation z-score, on-skin geometry, scale
check, feature-distance OOD (its statistics file was never built), GrabCut (20 s on a Pi).
