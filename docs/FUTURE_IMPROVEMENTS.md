# DermaScan — Future Improvements

Notes on performance and analysis-quality improvements worth considering. None of
these block a demo — the app works as-is. These are "make it genuinely better"
items, ranked by impact. No code has been changed for these yet.

---

## Performance (the Pi is the bottleneck)

### 1. Reduce camera warmup frames (easy latency win)
`frontend/services/pi_camera.py` `capture_still_jpeg()` discards **8 full-res
`main` frames** on every "Take Photo", then captures a 9th. On the Pi that's
~1–2s of dead time per capture. The camera already runs continuously with AE/AWB
settled (the live preview proves it), so **1–2 warmup frames is likely enough**.
Lowering `_WARMUP_FRAMES` from 8 → 2 should noticeably speed up capture.

### 2. Keep TTA off on the Pi (already correct)
Test-time augmentation runs the model 4× and averages — the single biggest
inference cost. It's already disabled for the Pi backend by default. Leave it off.

### 3. Profile CLAHE + unsharp mask if captures feel slow
`pi_camera.py` `_enhance()` runs CLAHE + unsharp mask on the 1024px crop on every
capture (pure OpenCV on an ARM core). Cheap-ish, but a candidate to profile if
capture latency matters.

### 4. Benchmark and possibly switch to the int8 model
`models/skin_classifier_int8.tflite` exists but is **unused** — the kiosk runs
`skin_classifier.tflite`. A fully-int8 model is typically **2–4× faster on ARM**
with minimal accuracy loss; its latency was never benchmarked. Action: run both
on the Pi, compare inference ms *and* whether predictions shift. If int8 holds
up, switch `SKIN_MODEL_PATH` to it.

---

## Analysis quality (where the real value is)

### 5. The CNN sees the raw JPEG, not the enhanced image (by design)
`frontend/services/pipeline.py` feeds the **raw** image to the classifier and the
**enhanced** image (hair removal, color constancy, CLAHE) only to the ABCDE
heuristics. This is the correct call for not breaking the trained weights, but it
means the CNN never benefits from the image cleanup. If the model is ever
retrained, baking that preprocessing into training could help.

### 6. ⭐ Apply temperature calibration (highest-value, free)
`models/temperature.json` has `T ≈ 1.54` fitted on validation data, but it is
**not applied during inference**. The confidence numbers shown to users are
therefore raw softmax — **overconfident**. Applying temperature scaling
(`logits / T` before softmax) makes "87% confidence" actually mean ~87%. The
value is already computed; this is essentially free, and for a *medical
screening* tool, honest confidence matters a lot. **Do this first.**

### 7. Add an uncertainty / "not sure" output
Every scan currently gets a confident label, even on garbage input. The deferred
`frontend/services/uncertainty.py` (MC-Dropout) would let the app say "low
confidence — retake or see a professional" instead of guessing. For a screening
demo this is arguably more clinically honest than another decimal of accuracy.

### 8. Hardware: fixed-focus IMX219 is the accuracy ceiling
The Camera Module 2 can't focus close, so lesion close-ups may be soft — this
hurts the ABCDE border/diameter measurements more than the CNN. A cheap clip-on
**macro lens** would do more for analysis quality than any code change. The
CLAHE/unsharp enhancement is partly compensating for this hardware limit.

---

## Recommended order
1. **#6 — Apply temperature scaling.** Free, makes confidence numbers truthful.
2. **#4 — Benchmark int8.** Potential 2–4× speedup on the Pi.
3. **#1 — Cut warmup frames** (8 → 2). Faster captures.
4. Everything else is polish.
