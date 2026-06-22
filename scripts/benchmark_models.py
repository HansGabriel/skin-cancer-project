"""Benchmark TFLite models head-to-head on the Pi (latency + prediction agreement).

Run this ON THE PI to decide whether to switch SKIN_MODEL_PATH to the int8 model.
It times each model over N runs and reports whether their decisions agree on the
same inputs, so a faster model that changes predictions is caught before deploying.

Usage (from the project root, with the venv active):
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --runs 100 --image samples/malignant_demo.jpg
    python scripts/benchmark_models.py --models models/skin_classifier.tflite models/skin_classifier_int8.tflite

If --image is omitted, every image under samples/ is used (or a random tensor as a
last resort). TTA is OFF (matches the Pi default).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import preprocessing  # noqa: E402
from backend.tflite_shared import (  # noqa: E402
    apply_temperature,
    decide_index,
    decode_image_bytes_to_rgb,
    load_labels,
)

DEFAULT_MODELS = [
    str(ROOT / "models" / "skin_classifier.tflite"),
    str(ROOT / "models" / "skin_classifier_int8.tflite"),
]


def _interpreter_class():
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]

        return tflite.Interpreter
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]

        return Interpreter


def _load_interpreter(model_path: str):
    Interpreter = _interpreter_class()
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _infer_once(interp, rgb: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    tensor = preprocessing.to_input_tensor(rgb, inp)
    interp.set_tensor(inp["index"], tensor)
    interp.invoke()
    raw = interp.get_tensor(out["index"])[0]
    logits = preprocessing.dequantize_output(raw, out)
    if logits.max() <= 1.0 and logits.sum() > 0.9:
        return logits.astype(np.float32)
    x = logits.astype(np.float64) - np.max(logits)
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _load_images(image_arg: str | None) -> list[tuple[str, np.ndarray]]:
    if image_arg:
        p = Path(image_arg)
        return [(p.name, decode_image_bytes_to_rgb(p.read_bytes()))]
    samples = sorted((ROOT / "samples").glob("*.jpg"))
    if samples:
        return [(p.name, decode_image_bytes_to_rgb(p.read_bytes())) for p in samples]
    print("No samples found — using a random 224x224 tensor.")
    return [("random", (np.random.rand(224, 224, 3) * 255).astype(np.uint8))]


def benchmark(model_path: str, images, runs: int, labels):
    interp = _load_interpreter(model_path)
    # Warm up (first invoke allocates/JITs and is not representative).
    _infer_once(interp, images[0][1])

    times_ms: list[float] = []
    decisions: dict[str, tuple[str, float]] = {}
    for name, rgb in images:
        probs = _infer_once(interp, rgb)  # capture a representative decision
        idx = decide_index(probs)
        cal = apply_temperature(probs)
        decisions[name] = (labels[idx], float(cal[idx]) * 100.0)
        for _ in range(runs):
            t0 = time.perf_counter()
            _infer_once(interp, rgb)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms, decisions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--image", default=None, help="Single image; default = all of samples/")
    ap.add_argument("--runs", type=int, default=50, help="Timed runs per image per model")
    ap.add_argument("--labels", default=str(ROOT / "models" / "labels.txt"))
    args = ap.parse_args()

    labels = load_labels(args.labels)
    images = _load_images(args.image)
    print(f"Images: {[n for n, _ in images]}  |  runs/image: {args.runs}  |  TTA: off\n")

    results = {}
    for model_path in args.models:
        if not Path(model_path).is_file():
            print(f"SKIP (not found): {model_path}")
            continue
        times_ms, decisions = benchmark(model_path, images, args.runs, labels)
        results[model_path] = decisions
        name = Path(model_path).name
        print(f"=== {name} ===")
        print(
            f"  latency: mean {statistics.mean(times_ms):6.1f} ms  "
            f"median {statistics.median(times_ms):6.1f} ms  "
            f"min {min(times_ms):6.1f}  max {max(times_ms):6.1f}"
        )
        for img_name, (lbl, conf) in decisions.items():
            print(f"  {img_name:<28} -> {lbl:<14} ({conf:4.1f}% calibrated)")
        print()

    # Agreement check: does swapping models change any decision?
    if len(results) >= 2:
        paths = list(results.keys())
        base = results[paths[0]]
        disagreements = []
        for other in paths[1:]:
            for img_name, (lbl, _) in results[other].items():
                if img_name in base and base[img_name][0] != lbl:
                    disagreements.append(
                        f"  {img_name}: {Path(paths[0]).name}={base[img_name][0]} "
                        f"vs {Path(other).name}={lbl}"
                    )
        if disagreements:
            print("⚠ PREDICTION DISAGREEMENTS (do NOT switch blindly):")
            print("\n".join(disagreements))
        else:
            print("✓ All models agree on every sample decision — a faster one is safe to deploy.")


if __name__ == "__main__":
    main()
