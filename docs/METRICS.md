# How good is the model, honestly

## The deployed model

`models/skin_classifier.tflite` is the **first model (June 2026, commit `fec7993`)**,
EfficientNetB0, 3 classes, float32 in/out with int8 weights and activations inside
(the export quantised it; the kiosk never sees that), decision at `p(pre)+p(mal) >= 0.11`
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
model scored 0.620, so the export lost a lot of it). Kept in git at `e1579a0`; the
lesion-grouped split and the augmentation are still the right recipe. Note that the
deployed June model went through the same full-int8 export and did not suffer the same
way, so "int8 lost it" is a hypothesis, not the explanation; a float16 export of the
retrain is the cheap way to test it.

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

## Open items on the model (final check, 11 October 2026)

Found while auditing the inference path before the demo. None changes what the kiosk
does today; each is a decision for after 13 October, with the measurement that settles it.

1. **The threshold was not fitted the way AGENTS.md says.** 0.11 and T = 1.54 come from the
   June model's Keras validation probabilities, single view. The rule is "fit on the exported
   TFLite with 4-view TTA" (`training/calibrate.py`). To settle it: run `calibrate.py` against
   the June `.tflite` on a lesion-grouped validation split, then score **both** thresholds on
   the 80 never-seen photos above. Keep 0.11 unless the refit flags as many melanomas (17/17)
   with better specificity. Do not change `thresholds.json` before the demo.
2. **The deployed model is int8 inside too** (348 int8 tensors, QUANTIZE/DEQUANTIZE at the
   edges, 5,083,480 bytes, identical to the retrain's size). The paragraph above used to
   blame the retrain's loss on the int8 export; the June model contradicts that. A float16
   export of the 1 Oct Keras model, scored with `eval.py`, is the one-hour experiment.
3. **Train/serve preprocessing differ.** Training squashed each 600×450 photo to 224×224
   (bilinear, no antialias). The kiosk centre-crops a square from the camera and resizes with
   INTER_AREA, so lesions arrive at a different aspect ratio and sharpness than in training.
   For the next retrain: train on centre-cropped squares with area interpolation, or add the
   kiosk's exact resize to the augmentation. Not a change to make on the kiosk side.
4. **The notebook in HEAD is the 1 Oct recipe,** not the one that made the deployed model.
   The June recipe is at commit `fec7993`. Re-running `calibrate.py` from HEAD would overwrite
   the June threshold and temperature; do it on a copy.
5. **The retrain on the WSL PC has not been evaluated.** It is not in the repo on any branch.
   Procedure, in this order, none of it on the kiosk: copy its four files (`.tflite`,
   `labels.txt`, `thresholds.json`, `temperature.json`) into a scratch folder; point
   `calibrate.py` at it to fit the threshold on the exported TFLite with TTA; `eval.py` on the
   lesion-grouped test split; then the 80-photo head-to-head through the kiosk path exactly as
   above. Ship it only if it flags at least as many melanomas as the June model, and move all
   four files together (AGENTS.md rule 5).
6. **4-view TTA is four sequential invokes.** The views could be batched as one
   `[4,224,224,3]` tensor, but the squeeze-excitation blocks carry dynamic shapes that already
   split XNNPACK into 18 partitions, so the gain is not predictable. Measure with
   `scripts/bench_scan.py` on the Pi before and after if anyone tries it. The "~0.5 s extra on
   the Pi" in `config.py` has never been measured; the Pi's real numbers belong below.

## Speed on the device

No timing on the Pi has been recorded yet. The illustrative `model=640` in the log examples
is a guess. To record it: on the Pi, `source venv/bin/activate && python scripts/bench_scan.py`
and paste the three lines here with the date. On an Apple-silicon Mac (11 Oct 2026, three
synthetic 1024 px photos, medians): check 14 ms, scan after check 15 ms (model 13), scan from
scratch 29 ms. Before the check was reused in the scan, a visitor's check + scan cost 58 ms.

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
| focus / brightness | blur, black frames, glare | Laplacian variance at 512 px after a 1–99 percentile stretch; mean V | ≥ 13; 12–246 |
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

## Focus limit (3 Oct 2026)

Set from what the model tolerates. On the 80 never-seen photos the model flags every
cancer up to Gaussian blur σ = 2 px (at 1024 px) and slips only from σ = 3 (22/23, then
21/23 at σ = 6). The old check (min/max stretch, limit 20) refused 45% of photos at σ = 1
and 46% of sharp photos after camera-style noise reduction. The new one (1–99 percentile
stretch, limit 13) refuses 0–5% of sharp photos with glare, dim light or noise reduction,
and about half of σ 3–6 blur. Real-lesion pass rate 91% → 93%.

## E (evolving) is not automatic

71 pairs of photos of the same unchanged HAM10000 mole, compared on colour (Lab ΔE of
the spot), asymmetry and border: the noise between two photos of one mole (median ΔE 7,
p95 22; zoom differs by ×2.3 median, so size is unusable) is larger than a simulated 35%
darkening or a grown lobe. Limits that kept false "changed" at 15% caught 4% of the
darkening and 1% of the lobes. So the kiosk shows the first and latest photo of a saved
group side by side and leaves the judgement to a person. Pi captures through the cone
(fixed distance and light) may be far steadier; measure them before automating E.
