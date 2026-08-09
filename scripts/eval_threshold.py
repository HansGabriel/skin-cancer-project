"""Evaluate TFLite model on the held-out test split at the deployed decision threshold.

Usage:
    ./venv/bin/python scripts/eval_threshold.py
    ./venv/bin/python scripts/eval_threshold.py --production               # serving path: 4-view TTA
    ./venv/bin/python scripts/eval_threshold.py --production --clean-subset
    ./venv/bin/python scripts/eval_threshold.py --model models/skin_classifier_int8.tflite
    ./venv/bin/python scripts/eval_threshold.py --out docs/

Reads models/test_split.csv (columns: path, label_idx, label_3, lesion_id).
Applies the same decide_index() logic used by backend/tflite_shared.py so the
metrics reflect exactly what the Streamlit / Pi backends do, not a post-hoc
re-analysis.

--production evaluates through the deployed serving configuration: the same
4-view TTA averaging as backend.tflite_shared.run_inference_on_rgb (SKIN_TTA
defaults ON in the Streamlit backend). Without it, a single identity view is
scored (TTA off).

--clean-subset keeps only test images whose HAM10000 lesion_id has NO images
outside the test split (HAM10000 has multiple photos per lesion; the checked-in
split is image-level, so most test lesions leak into train/val).

Unreadable/missing test images are NOT scored; if any exist the script prints
them and exits nonzero so a broken dataset can never inflate the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.tflite_shared import decide_index, load_labels, run_inference_on_rgb  # noqa: E402
from backend.preprocessing import to_input_tensor, dequantize_output  # noqa: E402
from backend.recommendations import CANCER_LABELS  # noqa: E402


LABELS_PATH = ROOT / "models" / "labels.txt"
TEST_CSV = ROOT / "models" / "test_split.csv"
METADATA_CSV = ROOT / "datasets" / "ham10000" / "HAM10000_metadata.csv"
DEFAULT_MODEL = ROOT / "models" / "skin_classifier.tflite"
DEFAULT_OUT = ROOT / "models"

# One definition, shared with services.verdict — AGENTS.md requires labels,
# metrics and UI to stay in sync, and separate copies is how they drift apart.
CANCER_CLASSES = set(CANCER_LABELS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL), help=".tflite model path")
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--test_csv", default=str(TEST_CSV))
    p.add_argument("--metadata", default=str(METADATA_CSV),
                   help="HAM10000_metadata.csv (lesion_id per image; used by --clean-subset)")
    p.add_argument("--production", action="store_true",
                   help="Evaluate the deployed serving path: 4-view TTA averaging exactly as "
                        "backend.tflite_shared.run_inference_on_rgb with SKIN_TTA on (the "
                        "Streamlit backend default)")
    p.add_argument("--clean-subset", action="store_true",
                   help="Score only test images whose lesion_id has no images outside the test "
                        "split (leakage-free lesions)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Directory for PNG output")
    p.add_argument("--no_png", action="store_true", help="Skip PNG output (no matplotlib needed)")
    return p.parse_args()


def filter_clean_subset(df, metadata_csv: str):
    """Keep rows whose lesion_id has ALL of its HAM10000 images inside the test split.

    HAM10000 ships multiple images of the same lesion (linked by lesion_id). The
    checked-in split is image-level, so a test image whose lesion has other images
    in train/val is contaminated. Uses stdlib csv for the metadata read.
    """
    import csv

    if "lesion_id" not in df.columns:
        print("test_split.csv has no lesion_id column — cannot build --clean-subset")
        sys.exit(1)
    total_by_lesion: dict[str, int] = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid = row["lesion_id"]
            total_by_lesion[lid] = total_by_lesion.get(lid, 0) + 1
    in_test = df["lesion_id"].value_counts()
    clean_lesions = {lid for lid, n in in_test.items() if total_by_lesion.get(lid, 0) == int(n)}
    return df[df["lesion_id"].isin(clean_lesions)].reset_index(drop=True)


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


def infer(interp, image_path: str, production: bool = False) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if production:
        # Deployed serving path: identical 4-view TTA average (identity/hflip/vflip/
        # rot180) as the Streamlit backend runs with its SKIN_TTA=1 default.
        probs, _ms = run_inference_on_rgb(rgb, interp, use_tta=True)
        return probs.astype(np.float32)
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

    df_full = pd.read_csv(args.test_csv)
    missing = [c for c in ("path", "label_idx") if c not in df_full.columns]
    if missing:
        print(f"test_split.csv missing columns: {missing}"); sys.exit(1)

    df = df_full
    if args.clean_subset:
        df = filter_clean_subset(df_full, args.metadata)
        print("\n=== Clean (leakage-free) subset ===")
        print(f"Test images total: {len(df_full)}; clean (whole lesion inside test): {len(df)}; "
              f"excluded (lesion also has train/val images): {len(df_full) - len(df)}")
        if "label_3" in df_full.columns:
            for lbl in labels:
                tot = int((df_full["label_3"] == lbl).sum())
                cln = int((df["label_3"] == lbl).sum())
                print(f"  {lbl:>15}: {cln}/{tot} kept")

    model_name = Path(args.model).name
    print(f"\nModel : {args.model}")
    print(f"Labels: {labels}")
    print(f"Test rows: {len(df)}")
    print(f"Inference mode: "
          f"{'PRODUCTION (4-view TTA, deployed serving path)' if args.production else 'single view (TTA off)'}")
    print(f"Cancer classes: {[labels[i] for i in sorted(cancer_indices)]}")
    print(f"Cancer threshold: {__import__('json').loads((ROOT / 'models' / 'thresholds.json').read_text())['screen_cancer_threshold']:.3f}")

    interp = load_interpreter(args.model)
    interp.allocate_tensors()

    y_true = df["label_idx"].to_numpy(dtype=int)
    y_argmax = np.zeros(len(df), dtype=int)
    y_thresh = np.zeros(len(df), dtype=int)
    probs_all = np.zeros((len(df), n_cls), dtype=np.float32)

    unreadable: list[str] = []
    valid = np.ones(len(df), dtype=bool)
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 200 == 0:
            print(f"  {i}/{len(df)}...", end="\r", flush=True)
        try:
            probs = infer(interp, row.path, production=args.production)
        except FileNotFoundError:
            # Never score an image we could not read — collected and FAILED below.
            unreadable.append(str(row.path))
            valid[i] = False
            continue
        probs_all[i] = probs
        y_argmax[i] = int(np.argmax(probs))
        y_thresh[i] = decide_index(probs)

    print(f"  Done. {int(valid.sum())}/{len(df)} images readable.")
    y_true, y_argmax, y_thresh = y_true[valid], y_argmax[valid], y_thresh[valid]

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
        # Distinct filenames per config so serving-path / clean-subset runs never
        # overwrite the historical single-view full-set PNGs.
        stem += ("_production" if args.production else "") + ("_clean" if args.clean_subset else "")
        save_confusion_png(cm_argmax, labels, f"Confusion (argmax) — {stem}",
                           out_dir / f"confusion_argmax_{stem}.png")
        save_confusion_png(cm_thresh, labels, f"Confusion (threshold=0.11) — {stem}",
                           out_dir / f"confusion_threshold_{stem}.png")

    if unreadable:
        print(f"\nERROR: {len(unreadable)} test image(s) could not be read; the metrics above "
              f"cover only the {int(valid.sum())} readable images and MUST NOT be quoted:",
              file=sys.stderr)
        for path in unreadable:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
