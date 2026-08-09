# DermaScan — Future Improvements

Notes on performance and analysis-quality improvements worth considering. None of
these block a demo — the app works as-is. These are "make it genuinely better"
items, ranked by impact. No code has been changed for these yet.

---

## Performance (the Pi is the bottleneck)

### 1. ✅ Done — camera warmup frames reduced
Both capture paths now discard only **2** warmup frames
(`frontend/services/pi_camera.py` and `scripts/pi_server.py`, `_WARMUP_FRAMES = 2`).
`pi_server.py` was still on 8 when this item was first marked done, so the
standalone Flask path kept the ~1–2s dead time; that is fixed.

### 2. Keep TTA off on the Pi (already correct)
Test-time augmentation runs the model 4× and averages — the single biggest
inference cost. It's already disabled for the Pi backend by default. Leave it off.

### 3. Profile CLAHE + unsharp mask if captures feel slow
`pi_camera.py` `_enhance()` runs CLAHE + unsharp mask on the 1024px crop on every
capture (pure OpenCV on an ARM core). Cheap-ish, but a candidate to profile if
capture latency matters.

### 4. ❌ Dropped — int8 model offers no speedup
Measured: `skin_classifier_int8.tflite` is only +24 bytes vs the deployed file
with bit-identical confusion matrices — the deployed `skin_classifier.tflite` is
already integer-quantized, so there is no 2–4× win to collect (a true speedup
would need a full-int8 re-export with a representative dataset; see
docs/DEPLOYMENT.md follow-up B.5).

---

## Analysis quality (where the real value is)

### 5. The CNN sees the raw JPEG, not the enhanced image (by design)
`frontend/services/pipeline.py` feeds the **raw** image to the classifier and the
**enhanced** image (hair removal, color constancy, CLAHE) only to the ABCDE
heuristics. This is the correct call for not breaking the trained weights, but it
means the CNN never benefits from the image cleanup. If the model is ever
retrained, baking that preprocessing into training could help.

### 6. ✅ Done — temperature calibration applied
`models/temperature.json` (`T ≈ 1.54`) **is applied** to all displayed
confidences (`backend/tflite_shared.apply_temperature`, called from
`compose_scan_result`; mirrored in `scripts/pi_server.py`). Decisions still run
on raw probabilities so the validated 0.11 screen threshold is preserved.
Nothing left to do.

### 7. ✅ Done — uncertainty and a "not a skin spot" stop
Two separate things now stop a confident-sounding answer on bad input:
- **Low confidence** → `services/verdict.py` returns "NOT SURE" (or "BETTER TO
  GET CHECKED" when the screen flagged it) instead of a clean label.
- **Not a skin spot at all** → `services/lesion_gate.py` checks for skin (tone-
  robust YCrCb, so dark skin is not rejected), then for a distinct spot on it,
  and stops before the classifier. Without it a photo of a wall came back as a
  confident "benign", because a 3-class softmax always sums to 1.

Still open: MC-Dropout in `frontend/services/uncertainty.py`, and the
feature-distance stage in `lesion_gate.py`, which needs
`models/feature_stats.json` built from the training set on the GPU machine
(`scripts/build_feature_stats.py`).

### 8. Hardware: fixed-focus IMX219 is the accuracy ceiling
The Camera Module 2 can't focus close, so lesion close-ups may be soft — this
hurts the ABCDE border/diameter measurements more than the CNN. A cheap clip-on
**macro lens** would do more for analysis quality than any code change. The
CLAHE/unsharp enhancement is partly compensating for this hardware limit.

---

## Recommended order
(#1, #6, #7 are done; #4 was dropped after measurement — see items above.)
1. **Retrain with a lesion-grouped split + augmentation.** The only real fix for
   angle/lighting robustness; everything at inference time is mitigation.
   Recipe in docs/FINAL_IMPROVEMENTS_RESEARCH.md Track A.
2. **#8 — Macro lens.** Hardware fix for the fixed-focus accuracy ceiling.
3. Everything else (#3, #5) is profiling/retrain-time polish.
