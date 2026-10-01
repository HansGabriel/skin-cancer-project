# How good is the model, honestly

## The deployed model

`models/skin_classifier.tflite`, EfficientNetB0, 3 classes, retrained 1 October 2026 on
HAM10000. The split is **by lesion** (`GroupShuffleSplit` on `lesion_id`, recorded in
`training/splits.csv`): train 7,002, validation 1,519, test 1,494 images, and no lesion
appears on both sides of any line. Training images get camera-style augmentation (blur,
JPEG re-compression, colour cast, exposure drift, glare, a faint screen grid); validation
and test images are clean dermoscopy.

Measured on the **test split** through the production path (`python training/eval.py`: the
exported int8 TFLite model, 4-view TTA, decision at the 0.235 threshold):

| | Test (n=1494) |
|---|---|
| Cancer sensitivity (pre-cancerous + malignant flagged) | 0.874 |
| Benign specificity | 0.698 |
| 3-class accuracy at the threshold | 0.700 |
| Argmax accuracy (no threshold) | 0.824 |
| Benign recall at argmax | 0.920 |
| Pre-cancerous recall at argmax | 0.556 |
| Malignant recall at argmax | 0.447 |
| Cancer sensitivity / benign specificity at argmax | 0.579 / 0.920 |

Test set: 1,176 benign, 63 pre-cancerous, 255 malignant. With only 63 pre-cancerous
images, any figure for that class is rough.

**What the threshold does.** `p(pre_cancerous) + p(malignant) >= 0.235` flags the spot.
0.235 is the highest threshold that flags 90% of cancers in the validation set (it
flagged 0.906, with validation specificity 0.681). On the test set it flagged 0.874: a
little under the target, as expected for a threshold fitted on other images. The cost is
that about three in ten benign spots are also flagged ("better to get checked"). For a
screening tool that is the right trade: a missed cancer costs more than a precautionary
visit. The threshold is fitted by `training/calibrate.py` on the exported TFLite model with
4-view TTA, so it describes the device, not the Keras model.

**Quantization costs accuracy.** The Keras model scored better than the exported one on
the same test split (single view, argmax):

| | Keras | Exported TFLite, 4-view TTA |
|---|---|---|
| Accuracy | 0.849 | 0.824 |
| Pre-cancerous recall | 0.524 | 0.556 |
| Malignant recall | 0.620 | 0.447 |

Malignant recall drops most. The threshold recovers sensitivity at the price of
specificity. The cause has not been isolated (int8 quantization is the first suspect;
trying float16 or dynamic-range export is the obvious experiment).

**Not comparable to the first model.** The first model (June 2026) was trained and tested
on an image-level split that put 40% of test images' lesions into training. Its figures
(sensitivity 0.911 at threshold 0.11, specificity 0.736 on the full test set; 0.903 / 0.855
on a leakage-free subset) came from a different, easier test set. Do not quote them next
to the table above, and do not cite either set of numbers as measured on this model.

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
