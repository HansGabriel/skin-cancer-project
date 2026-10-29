# How good is the model, honestly

## The deployed model

`models/skin_classifier.tflite` is the **October 2026 multi-source model** (branch
`feat/external-bench-retrain`, 18 October 2026): EfficientNetB0 at 224 px, 3 classes,
float32 in/out with int8 inside, trained on 48,481 photos from HAM10000, BCN20000,
MSK-1..5, MILK10k and PAD-UFES-20 (`training/dataset_index.csv`, lesion-grouped split
70/15/15, `training/splits.csv`). Decision at `p(pre)+p(mal) >= 0.47` and T = 1.84, both
fitted by `training/calibrate.py` on the exported TFLite with 4-view TTA for 90 %
validation sensitivity. It replaced the June 2026 model (`fec7993`, threshold 0.11), which
is kept at `git show fec7993:models/skin_classifier.tflite` for comparisons.

**Why: the June model could not see cancer in close-up photos.** On the frozen external
benchmark (next section) it flagged 7 of 31 MILK10k cancers photographed as ordinary
close-ups — the kind of photo the Pi camera takes — and half of Barcelona's dermoscopy.
(Earlier versions of this file said "0 of 20 MILK melanomas"; `training/external_bench.csv`
holds **one** MILK melanoma. The 31 MILK cancers are 30 actinic keratoses and that one
melanoma, which June missed and October flags.)

### October model vs June model, external benchmark (602 photos, kiosk path)

| | June (threshold 0.11) | **October (threshold 0.47)** |
|---|---|---|
| ROC-AUC, cancer vs benign | 0.712 | **0.754** |
| Cancer sensitivity (pre + mal flagged) | 0.60 (90/149) | **0.81 (121/149)** |
| Malignant recall | 0.76 | 0.78 |
| Benign specificity | **0.71** | 0.59 |
| Students' 96 malignant photos flagged | 72 | 75 |
| MILK10k clinical close-ups: cancers flagged | 7/31 (the one melanoma missed) | **28/31 (the one melanoma flagged)** |
| MILK10k clinical close-ups: benign specificity (118 scored) | 0.81 | **0.37** |
| BCN20000 dermoscopy: cancers flagged | 18/38 | **33/38** |
| UDA (2016) dermoscopy: cancers flagged | 46/55 | 40/55 |

Per-source AUC: BCN 0.65 → 0.78, MILK close-ups 0.58 → 0.77, UDA 0.81 → 0.78,
MSK 0.94 → 0.88 (all printed by `external_eval.py`, which also prints the threshold sweep). So: a clearly better model where the June model was blind, slightly
worse on the two sources closest to HAM10000, and a threshold that trades benign
specificity for sensitivity. A threshold of 0.60 would match June's specificity (0.71)
with sensitivity 0.70 on the benchmark, but drops HAM10000-type validation sensitivity to
0.62 — rejected. The 0.47 operating point is the 90 % validation-sensitivity rule the
figures promise; it sends roughly 4 benign spots in 10 to "better to get checked" on
these sources, against 3 in 10 for June. **Rule 5 in AGENTS.md ("at least as specific")
was not met; the mentor accepted the trade on 18 October 2026 because the device's job
is to catch cancers in camera photos.**

### October model, its own test split (n = 7,288, lesion-grouped, 9 sources)

| | value |
|---|---|
| Cancer sensitivity (at threshold 0.47) | 0.907 |
| Benign specificity (at threshold 0.47) | 0.633 |
| 3-class accuracy (at threshold) | 0.684 |
| Argmax accuracy | 0.700 |
| benign / pre_cancerous / malignant recall (argmax) | 0.690 / 0.569 / 0.731 |
| Screening ROC-AUC (Keras, single view) | 0.894 |

Validation, per source, at 0.47: HAM10000 sens 0.80 / spec 0.80; BCN20000 0.89 / 0.48;
MILK10k 0.95 / 0.44; PAD-UFES-20 0.99 / 0.58. The test split is far harder than HAM10000
alone (BCN20000 "in the wild" includes nails, mucosa and lesions larger than the scope),
so do not compare these to the June model's HAM-only figures.

Training: Oct recipe (balanced resampling, focal loss, camera-style augmentation),
phase 1 frozen 20 epochs, phase 2 full fine-tune with early stopping on validation
macro-F1 (best 0.705; validation malignant recall 0.79 at argmax). ~1 h 40 min on the
RTX 4060. The 512 px cache (`datasets/cache512/`) makes the epochs fast.

## The students' external test (October 2026) and how to read the drawer

The students scored 100 photos from the ISIC archive through the web demo and read the
staff drawer's three bars: 43/100 "correct" by the tallest bar. Those bars are the
**temperature-scaled** percentages (T = 1.54), shown for display. The device never decides
by them: it flags a spot when the **raw** cancer score `p(pre)+p(mal)` is at or above the
threshold (0.11 for the June model; on the drawer's scale that line sits near 20 %).
Re-scored under the device's own rule, the same table gives **74/100 flagged**. Since
18 October the drawer prints that decision line itself (`cancer score · limit → flagged`).

The archive's diagnosis, not the students' list, is the ground truth: two of the 100 were
not malignant (`ISIC_0056469` actinic keratosis — flagged; `ISIC_0000027` nevus — the
model's "benign" was right), and one id was a duplicate, so 99 photos remain. None are in
HAM10000. They come from Barcelona (BCN20000), the 2016 UDA set, MILK study close-ups,
Vienna, MSKCC and Buenos Aires; every miss is from a source the June model never saw.

Those 99, plus benign and pre-cancerous controls from the same kinds of source, are now
**`training/external_bench.csv`** (602 photos, `training/build_external_bench.py`), scored
by **`training/external_eval.py`** through the kiosk path. Every lesion in it is excluded
from training (`training/prepare_datasets.py`), so it stays external for every model.
**June model on the external benchmark (18 October 2026, kiosk path, threshold 0.11)** — the
October model's per-source numbers are in the table at the top:

| source | n | not scored | cancers | flagged | sensitivity | malignant recall | benign | specificity |
|---|---|---|---|---|---|---|---|---|
| BCN20000 · dermoscopy | 165 | 28 | 38 | 18 | 0.47 | 0.50 | 99 | 0.74 |
| MSK / ISIC 2017-18 · dermoscopy | 111 | 3 | 11 | 11 | 1.00 | 1.00 | 97 | 0.67 |
| UDA (ISIC 2016) · dermoscopy | 161 | 7 | 55 | 46 | 0.84 | 0.84 | 99 | 0.62 |
| MILK10k · clinical close-up | 151 | 2 | 31 | 7 | 0.23 | **0.00** | 118 | 0.81 |
| Vienna, HIBA, MSKCC, TBP tiles (13 photos) | 13 | 0 | 13 | 8 | 0.62 | 0.62 | 0 | – |
| **students' list (99)** | 99 | 2 | 96 | 72 | **0.75** | 0.76 | 1 | 1.00 |
| **all** | 602 | 40 | 149 | 90 | **0.60** | 0.76 | 413 | **0.71** |

Read it as: fine on the kinds of dermoscopy it was trained on, half-blind on Barcelona's,
and **nearly blind to cancer in ordinary close-up photos** (7 of 31 MILK cancers flagged,
its one melanoma missed) — the kind of photo the Pi camera takes. "Not scored"
are hard refusals by the photo check (`no_skin`, mostly dark-vignetted dermoscopy); they
are listed by the script, not counted either way.


## The June model, for the record

**Why it was restored in October (over the 1 Oct retrain): a head-to-head on photos neither model ever saw.** 80 HAM10000
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

**What is measured on camera-style photos.** The benchmark's 151 MILK10k clinical
close-ups are the closest thing to the Pi camera. There the October model flags 28 of 31
cancers but its benign specificity is **0.37** (118 scored): six harmless close-ups in ten
are sent to "get checked", against two to three in ten on the dermoscopy sources (0.72-0.78).
The cause is in the training recipe: close-ups (MILK10k, PAD-UFES-20) are 75 % cancer in
the train split, and the equal-per-class sampler makes 87 % of the close-ups the model
saw during training cancer, so a well-lit camera photo reads as "cancer" by default.
The students demo by photographing images on a phone screen; no accuracy figure exists
for that input. The camera-style augmentation is a mitigation, not a measurement.

**Known limitation: nothing in the model says "this is not a lesion".** There is no
non-lesion photo anywhere in training, validation or the benchmark, and no "not a lesion"
label; the probabilities always sum to 1 across benign / pre-cancerous / malignant. The
photo gate is the only defence. On 20 October 2026 the web demo (same code and model as
the Pi) passed a webcam photo of a face (one nostril as the "spot", 99 % malignant) and
of a palm (its shadowed half, cancer score 0.75) and flagged both. The gate now also
refuses a spot that is not in the ring or shares the frame with several dark marks (see
"The photo gate" below); the model-side fix is a retrain with image-type-stratified
sampling and a `no_lesion` class from the team's own photos of normal skin (parked;
open item 4).

To refresh this table after a retrain: run `python training/calibrate.py`, then
`python training/eval.py`, and paste what it prints. Do not mix numbers from two models.

## Figure facts (keep the design figures in step with the model)

The paper/slide figures (`EPIVUE Diagrams` in Claude Design, figure B "Screening flow")
print these values. Whoever edits the figures copies them from here, not from memory.

| fact | value today (October model) | where it comes from |
|---|---|---|
| backbone, input size | EfficientNetB0, 224 × 224 | notebook Cell 6 `BACKBONE` / `IMG_SIZE` |
| views averaged | 4 (as-is, mirrored, upside down, rotated 180°) | `dermascan/classifier.py` |
| decision | flag if `p(pre)+p(mal) >= 0.47` | `models/thresholds.json` |
| threshold fitted for | 90 % cancer sensitivity on validation | `training/calibrate.py` |
| low-confidence wording below | 45 % | `dermascan/config.py` |
| power supply | official Raspberry Pi USB-C, **5.1 V** 3 A (figures currently say 5 V) | supply label |

Pending edits for the design owner (18 October 2026): "5 V · 3 A" → "5.1 V · 3 A" on figures
A, C, D and the paper's Figure 3; figure B's decision box must now say **0.47** (it says 0.11) and takes the threshold and
backbone from this table whenever `models/` changes.

## Open items on the model

**24 October 2026: the image-type-stratified sampler was trained and NOT shipped.**
Cell 6 printed `Cancer share within each type: {'clinical: close-up': 0.67, 'dermoscopic':
0.67}` (was 0.87 / 0.61). Same recipe otherwise (B0, 224 px, int8 export, `calibrate.py`
at 0.90 val sensitivity → threshold **0.44**, T from the same script). Run on the PC,
1 h 45 min; log and the four model files kept off git in
`datasets/models_2026-10-24_stratified/`. External benchmark, kiosk path, force on:

| source | n | skip | cancers | flagged | sens | benign | spec |
|---|--:|--:|--:|--:|--:|--:|--:|
| BCN20000 · dermoscopic | 165 | 28 | 38 | 31 | 0.816 | 99 | 0.545 |
| Anonymous (ISIC 2017-18) · dermoscopic | 111 | 3 | 11 | 8 | 0.727 | 97 | 0.567 |
| MILK10k · clinical: close-up | 151 | 2 | 31 | **28** | 0.903 | 118 | **0.525** |
| UDA (ISIC 2016) · dermoscopic | 161 | 7 | 55 | 37 | 0.673 | 99 | 0.788 |
| students' list | 99 | 2 | 96 | 68 | 0.708 | 1 | 1.000 |
| **ALL** | 602 | 40 | 149 | **113** | 0.758 | 413 | 0.603 |

External ROC-AUC 0.740 (October model 0.78); MILK10k AUC 0.810. Own test split (n = 7,288):
sensitivity 0.900, benign specificity 0.656, argmax accuracy 0.738.

Against the ship rule in item 4: MILK benign specificity **0.37 → 0.525** (toward 0.7, yes);
MILK cancers flagged **28/31** (yes); total cancers flagged **113/149, below 121** (no — the
dermoscopy rows lost 8 cancers, UDA 40 → 37, Vienna 4 → 3, ISIC 2017-18 7 → 8 is the only
gain). Mentor's rule says all three must hold, so `models/` stays the 18 October model.
What it tells us: showing close-ups at the dermoscopy cancer rate fixes the "close-up →
cancer" shortcut (MILK specificity +0.15 at the same 28/31) but the dermoscopy sources
pay for it at a single global threshold. Next tries, in order: (a) a per-image-type
threshold (`calibrate.py` fitting one limit per `image_type`, the kiosk only ever sees
close-ups, so the Pi would run at the close-up limit); (b) the `no_lesion` class once
photos exist. The `no_lesion` class still waits for photos the team may use (`docs/PRIVACY.md`).

Items 1–4 from the 11 October audit are closed by the 18 October retrain: the threshold
and temperature are now fitted by `calibrate.py` on the exported TFLite with 4-view TTA
(1), the int8 export is the recipe for both models and was not the cause of the 1 Oct
loss (2, measured by the external benchmark: the October model, same export, beats June),
train and serve both squash to 224 px (3), and the notebook in HEAD is the recipe that
produced `models/` (4). The further WSL retrain mentioned in item 5 was never evaluated
and is superseded.

Still open:

1. **TTA is four sequential invokes** (`dermascan/classifier.py`). Batching the four views
   in one `invoke()` would cut model time on the Pi; measure with `scripts/bench_scan.py`.
2. **Benign specificity is 0.59 on the external benchmark** at the 90 % sensitivity
   operating point. More benign close-up photos in training (MILK10k and PAD-UFES-20 are
   cancer-heavy) is the likely fix; the threshold sweep printed by `external_eval.py`
   shows what any other operating point would cost.
3. **The photo check refuses dark-vignetted dermoscopy** (40 of 602 benchmark photos,
   `no_skin`). Irrelevant for the Pi camera, but it shrinks the dermoscopy rows of the
   benchmark; the table's `n` counts them, the rates do not.
4. **No "not a lesion" answer, and close-up specificity 0.37** (see "Known limitation"
   above). The stratified sampler was tried on 24 October and not shipped (numbers at the
   top of this section: MILK specificity 0.525 and 28/31, but 113/149 overall). Next: a
   per-image-type threshold, then a `no_lesion` class from team phone / Pi photos of
   normal skin. Ship a model only if MILK benign close-up specificity rises from 0.37 toward
   0.7 **and** MILK cancers flagged stay >= 28/31 **and** total cancers flagged stay
   >= 121/149, all measured by `external_eval.py` per source.
5. **The new gate limits cost more lesion photos than budgeted, and the bench cannot
   set them.** Measured 24 Oct 2026, `python training/external_eval.py --no-force`,
   old gate (`acf90fd`) vs new gate (`main`), same 602 photos:

   | gate | not scored | of which soft refusals | MILK10k close-ups not scored |
   |------|-----------:|-----------------------:|-----------------------------:|
   | old (before PR #9) | 123 | 83 (plain_skin 57, soft_edge 22, fills_frame 2, no_spot 2) | 45 of 151 |
   | new (`SPOT_MAX_FRACTION` 0.40, `SPOT_MAX_OFFSET` 0.25) | 272 | 232 (not_in_ring 101, fills_frame 62, plain_skin 57, several_marks 7, soft_edge 3, no_spot 2) | 97 of 151 |

   The rise is +149 soft refusals, 26 % of the 562 the old gate scored, against a budget
   of ~5 %. It is not the two limits: `scripts/gate_report.py --full` on the bench shows
   93 lesion photos *touch the frame border* (dermoscopy fills the field by design), and
   that rule has no knob. Loosening to 0.50 / 0.30 gives +127; the old 0.75 with no
   offset limit still gives +86. Decision (mentor, 24 Oct): keep 0.40 / 0.25 — the bench
   was never framed for the ring, the Pi photos will be, and loosening lets the palm back
   in (it passed at 0.46). The hard-refusal count is unchanged at 40. Still owed: one Pi
   session with `SAVE_CAPTURES_DIR` on, through `scripts/gate_report.py --full`, to set
   `SPOT_MAX_OFFSET`, `SPOT_MAX_FRACTION` (0.30 target) and `SPOT_MIN_FRACTION` (0.01
   candidate) from the cone — steps in `docs/DEPLOYMENT.md` and `docs/HANDOFF.md`. Also
   worth a look then: a spot that touches the border but is still centred (the
   dermoscopy case) could be measured rather than refused.

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
| one spot | plain skin, speckle, whole-frame dark | Otsu outline (hairs filled in, dark frame edges set aside): area 0.4–40%, solidity ≥ 0.6, Lab contrast ≥ 5 | |
| in the ring | a mark running off the frame, a hand's shadow, an off-centre upload | spot touches the frame edge, or its centre is > 0.25 of the short side from the middle | overridable |
| several marks | faces, hands with creases, scenes | more than 1 other dark mark clear of the edge and the spot holds < 0.65 of the dark area (old app's `dark_structure`, 246c072) | overridable |
| real edge | shadows, lighting gradients | lightness drop ÷ edge steepness, % of diagonal | ≤ 5.5 (overridable) |

**20 October 2026: "in the ring" and "several marks" added** after a webcam face and a
palm passed the four checks (numbers in "Known limitation" above). The spot area ceiling
went from 75 % to 40 % (the palm's shadow was 46 %). All three limits are guesses until
real captures are measured; `scripts/gate_report.py FOLDER --full` prints every number
per photo for that. `tests/fixtures/gate/{refuse,pass}` holds small real photos the gate
must get right (no faces in the repo; those live in the gitignored `captures/`), and
`tests/test_gate_real.py` runs them through the real decode path.

**On 267 real HAM10000 lesions through JPEG (3 Oct 2026): 91% pass** (was 69%) - measured
on the four checks as they stood then, before the 20 October additions below; the re-run
is owed (open item 5). The
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
*is* a screen), contrast-variation z-score, on-skin geometry, scale check,
feature-distance OOD (its statistics file was never built), GrabCut (20 s on a Pi).
Dark-blob dominance was dropped with them and came back on 20 October as "several marks".

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

## Forced reads of non-spots (24 Oct 2026 Pi session)

Nine photos through the Pi camera: the one the gate passed scored 0.24 (not flagged); the
eight it refused as "not one spot" (hair, plain skin, a pimple) were read anyway and all
scored 0.52–1.00, "flagged for a doctor". Cause: no "not a lesion" class and 87% cancer
among the close-ups the model trained on (above). From 29 October a forced read on
`no_spot` / `plain_skin` / `several_marks` returns no verdict ("WE CAN'T JUDGE THIS PHOTO");
the staff drawer keeps the numbers under a "numbers unreliable" line. Framing refusals
(`not_in_ring`, `fills_frame`, `soft_edge`) still answer with the caveat. The drawer's
"null" row on those scans was a page bug, fixed the same day.

Still no Pi capture measured. First capture session: 30 October 2026 (`SAVE_CAPTURES_DIR`
on for the event, plus ~50 deliberate non-lesion frames at the end), to seed a Pi bench
and a `no_lesion` class for the next retrain.

## Photo check labels, outline and two gate fixes (29 Oct 2026, `feat/epivue-align`)

No limit moved. What changed in `dermascan/gate.py` / `verdict.py`:

- Every check outcome has a short label on the check screen (`verdict.check_label`):
  SPOT FOUND · NOT A SPOT · PLAIN SKIN / SEVERAL MARKS / NONE FOUND · SKIN NOT RECOGNISED ·
  SPOT · NOT IN THE RING / TOO CLOSE / LOOKS LIKE A SHADOW / EDGE HARD TO FOLLOW ·
  CAN'T READ · BLURRY / DARK / GLARE. The gate's outline is drawn on the photo
  (`gate.outline_polygon`, ≤ 64 points, ~1 ms: the only addition to the check path).
- `ragged_spot` split out of `plain_skin`: low **contrast** is still "plain skin" (no verdict
  when forced); low **solidity** with real contrast is now a framing refusal, so a forced
  read of a scattered or multi-lobed mark answers, with the caveat and with A B C measured.
  On the synthetic suite a thick "C" (solidity 0.50) is `ragged_spot`; bare skin stays
  `plain_skin`. A four-lobed mark still reads as `several_marks` (dominance 0.23) — a known
  weak spot for the capture session.
- The light-spot path (a pale mark on dark skin) never counted other marks, so such a mark
  passed "several marks" whatever else was in the frame. It now counts on the light binary.
- Signs are measured on every judged read, forced framing reads included.
- Still zero real photos in `tests/fixtures/gate/`; the limits marked GUESSED in `config.py`
  wait for the 30 Oct captures.
