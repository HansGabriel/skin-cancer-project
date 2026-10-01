# How good is the model, honestly

## The deployed model (until the retrain lands)

`models/skin_classifier.tflite`, EfficientNetB0, 3 classes, trained June 2026 on HAM10000
with an **image-level** split. Measured through the production path (4-view TTA, decision
at the 0.11 threshold):

| | Full test (n=1503) | Leakage-free subset (n=897) |
|---|---|---|
| Cancer sensitivity (pre-cancerous + malignant flagged) | 0.911 | 0.903 |
| Benign specificity | 0.736 | 0.855 |
| 3-class accuracy at the threshold | 0.754 | 0.843 |
| Argmax accuracy (no threshold) | 0.857 | 0.903 |
| Malignant recall at argmax | 0.643 | 0.625 |
| Pre-cancerous recall at argmax | 0.429 | 0.360 |

**Why two columns.** HAM10000 has several photos of many lesions. The split was made per
image, so 606 of 1,503 test images share a lesion with a training image — the model has
seen them. The right-hand column keeps only lesions entirely inside the test set. Cite
that one.

**What the threshold does.** `p(pre_cancerous) + p(malignant) >= 0.11` flags the spot.
0.11 was chosen so 90% of cancers in the validation set are flagged. The cost is that
about one in four benign spots is also flagged ("better to get checked"). For a screening
tool that is the right trade: a missed cancer costs more than a precautionary visit.

**What is not measured at all.** Every number above is on dermoscopy images. The kiosk
is a bare camera module, and the students demo by photographing images on a phone
screen. No accuracy figure exists for that input. The retrain (below) adds augmentation
that imitates it; it is a mitigation, not a measurement.

## The retrain (training/train_skin_classifier.ipynb)

* Split by `lesion_id` (`GroupShuffleSplit`), so the two columns above become one.
* Camera-style augmentation: blur, JPEG re-compression, colour cast, exposure drift,
  glare, a faint screen grid.
* Threshold and temperature fitted by `training/calibrate.py` on the **exported TFLite model
  with 4-view TTA** — the device path — for 90% cancer sensitivity on the new validation set.
  (The first model's threshold was fitted on Keras single-view probabilities.)
* `python training/eval.py` prints the table to paste here.

Replace the table above when it has run. Do not mix numbers from the two models.

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
| one spot | plain skin, speckle, whole-frame dark | Otsu outline: area 0.4–75%, solidity ≥ 0.6, Lab contrast ≥ 5 | |
| real edge | shadows, lighting gradients | lightness drop ÷ edge steepness, % of diagonal | ≤ 4.0 (overridable) |

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
