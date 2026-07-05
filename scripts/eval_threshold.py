"""Evaluate TFLite model on the held-out test split at the deployed decision threshold.

Usage:
    ./venv/bin/python scripts/eval_threshold.py
    ./venv/bin/python scripts/eval_threshold.py --model models/skin_classifier_int8.tflite
    ./venv/bin/python scripts/eval_threshold.py --out docs/

Reads models/test_split.csv (columns: path, label_idx, label_3).
Applies the same decide_index() logic used by backend/tflite_shared.py so the
metrics reflect exactly what the Streamlit / Pi backends do, not a post-hoc
re-analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.tflite_shared import decide_index, load_labels  # noqa: E402
from backend.preprocessing import to_input_tensor, dequantize_output  # noqa: E402


LABELS_PATH = ROOT / "models" / "labels.txt"
TEST_CSV = ROOT / "models" / "test_split.csv"
DEFAULT_MODEL = ROOT / "models" / "skin_classifier.tflite"
DEFAULT_OUT = ROOT / "models"

CANCER_CLASSES = {"pre_cancerous", "malignant"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL), help=".tflite model path")
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--test_csv", default=str(TEST_CSV))
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Directory for PNG output")
    p.add_argument("--no_png", action="store_true", help="Skip PNG output (no matplotlib needed)")
    return p.parse_args()


def load_interpreter(model_path: str):
    import os

    num_threads = int(os.environ.get("SKIN_NUM_THREADS", "4"))
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            return tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        except ImportError:
            import tensorflow as tf
            return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def infer(interp, image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    tensor = to_input_tensor(rgb, input_details)
    interp.set_tensor(input_details["index"], tensor)
    interp.invoke()
    raw = interp.get_tensor(output_details["index"])[0]
    probs = dequantize_output(raw, output_details)
    # model output is softmax; normalise just in case of tiny float drift
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return probs.astype(np.float32)


def print_confusion(matrix: np.ndarray, labels: list[str], title: str) -> None:
    print(f"\n{title}")
    w = max(len(l) for l in labels) + 2
    header = " " * (w + 2) + "  ".join(f"{l:>{w}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{matrix[i, j]:>{w}d}" for j in range(len(labels)))
        print(f"  {row_label:>{w}}  {row}")


def save_confusion_png(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [skip PNG] matplotlib not installed; saving to {path} skipped.")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax)
    thresh = matrix.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Need pandas: pip install pandas"); sys.exit(1)

    labels = load_labels(args.labels)
    n_cls = len(labels)
    cancer_indices = {i for i, l in enumerate(labels) if l in CANCER_CLASSES}
    benign_idx = next(i for i, l in enumerate(labels) if l not in CANCER_CLASSES)

    df = pd.read_csv(args.test_csv)
    missing = [c for c in ("path", "label_idx") if c not in df.columns]
    if missing:
        print(f"test_split.csv missing columns: {missing}"); sys.exit(1)

    model_name = Path(args.model).name
    print(f"\nModel : {args.model}")
    print(f"Labels: {labels}")
    print(f"Test rows: {len(df)}")
    print(f"Cancer classes: {[labels[i] for i in sorted(cancer_indices)]}")
    print(f"Cancer threshold: {__import__('json').loads((ROOT / 'models' / 'thresholds.json').read_text())['screen_cancer_threshold']:.3f}")

    interp = load_interpreter(args.model)
    interp.allocate_tensors()

    y_true = df["label_idx"].to_numpy(dtype=int)
    y_argmax = np.zeros(len(df), dtype=int)
    y_thresh = np.zeros(len(df), dtype=int)
    probs_all = np.zeros((len(df), n_cls), dtype=np.float32)

    skipped = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 200 == 0:
            print(f"  {i}/{len(df)}...", end="\r", flush=True)
        try:
            probs = infer(interp, row.path)
        except FileNotFoundError:
            skipped += 1
            y_argmax[i] = y_true[i]  # don't penalise missing images
            y_thresh[i] = y_true[i]
            continue
        probs_all[i] = probs
        y_argmax[i] = int(np.argmax(probs))
        y_thresh[i] = decide_index(probs)

    print(f"  Done. Skipped {skipped} missing images.")

    # ── 3-class confusion matrices ──────────────────────────────────────────
    def confusion(y_t, y_p):
        m = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_t, y_p):
            m[t, p] += 1
        return m

    cm_argmax = confusion(y_true, y_argmax)
    cm_thresh = confusion(y_true, y_thresh)

    print_confusion(cm_argmax, labels, "3-class confusion — argmax (what you saw in the matrix)")
    print_confusion(cm_thresh, labels, "3-class confusion — deployed threshold (0.11)")

    # ── Cancer-vs-benign binary view ────────────────────────────────────────
    def binary_cancer(y):
        return np.array([1 if v in cancer_indices else 0 for v in y])

    bc_true = binary_cancer(y_true)
    bc_argmax = binary_cancer(y_argmax)
    bc_thresh = binary_cancer(y_thresh)

    def sens_spec(bt, bp):
        tp = int(((bt == 1) & (bp == 1)).sum())
        fn = int(((bt == 1) & (bp == 0)).sum())
        tn = int(((bt == 0) & (bp == 0)).sum())
        fp = int(((bt == 0) & (bp == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sens, spec, tp, fn, tn, fp

    sens_a, spec_a, tp_a, fn_a, tn_a, fp_a = sens_spec(bc_true, bc_argmax)
    sens_t, spec_t, tp_t, fn_t, tn_t, fp_t = sens_spec(bc_true, bc_thresh)

    total_cancer = int(bc_true.sum())
    total_benign = int((bc_true == 0).sum())

    print("\n=== Cancer-vs-benign screening (TEST set) ===")
    print(f"{'':30s}  argmax   deployed(0.11)")
    print(f"{'Cancer sensitivity':30s}  {sens_a:.3f}    {sens_t:.3f}")
    print(f"{'Benign specificity':30s}  {spec_a:.3f}    {spec_t:.3f}")
    print(f"{'Cancers caught (TP)':30s}  {tp_a}/{total_cancer}    {tp_t}/{total_cancer}")
    print(f"{'Cancers missed (FN — benign call)':30s}  {fn_a}         {fn_t}")
    print(f"{'Benign correctly cleared (TN)':30s}  {tn_a}/{total_benign}    {tn_t}/{total_benign}")
    print(f"{'Benign flagged cancer (FP)':30s}  {fp_a}         {fp_t}")

    # ── Per-class false-negative breakdown ──────────────────────────────────
    print("\n=== False-negatives at deployed threshold (predicted benign, actually cancer) ===")
    for ci in sorted(cancer_indices):
        fn_count = int(((y_true == ci) & (y_thresh == benign_idx)).sum())
        total_ci = int((y_true == ci).sum())
        print(f"  {labels[ci]:15s}: {fn_count}/{total_ci} missed as benign")

    # ── PNG output ──────────────────────────────────────────────────────────
    if not args.no_png:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = model_name.replace(".tflite", "")
        save_confusion_png(cm_argmax, labels, f"Confusion (argmax) — {stem}",
                           out_dir / f"confusion_argmax_{stem}.png")
        save_confusion_png(cm_thresh, labels, f"Confusion (threshold=0.11) — {stem}",
                           out_dir / f"confusion_threshold_{stem}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
