#!/usr/bin/env python3
"""Build ``models/feature_stats.json`` — the reference the not-a-lesion gate
compares against.

Stages 1 and 2 of ``frontend/services/lesion_gate.py`` ask "is this skin?" and
"is there a spot on it?" using colour and shape. They stop a wall or a keyboard,
but not something that merely *looks* skin-like. This adds the third stage: how
far the model's own features are from what it saw during training.

Run this once on the machine that holds the HAM10000 images (it is not needed at
runtime, and the gate degrades to stages 1-2 if the output file is absent):

    python scripts/build_feature_stats.py --images datasets/ham10000/HAM10000_images_part_1

It reads the two-output CAM export (``models/skin_classifier_cam.tflite``, from
``scripts/export_cam_tflite.py``), mean-pools each image's conv feature map, and
writes the centroid plus the inverse covariance of that cloud. At inference the
gate takes a Mahalanobis distance against it — microseconds, no second model.

Pick the threshold from the printed percentiles: ``SKIN_GATE_MAX_OOD`` should sit
just above the 99th percentile of the training distances, so ~1% of real lesion
photos are questioned and genuinely foreign input lands far outside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

from backend.preprocessing import to_input_tensor  # noqa: E402

DEFAULT_CAM = ROOT / "models" / "skin_classifier_cam.tflite"
DEFAULT_OUT = ROOT / "models" / "feature_stats.json"


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover - depends on the host
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def _features(it, rgb: np.ndarray) -> np.ndarray | None:
    """Mean-pooled conv feature vector for one image."""
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], to_input_tensor(rgb, inp))
    it.invoke()
    for od in it.get_output_details():
        t = it.get_tensor(od["index"])
        if t.ndim == 4:
            return t[0].mean(axis=(0, 1)).astype(np.float64)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="directory of training images")
    ap.add_argument("--cam-model", default=str(DEFAULT_CAM))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=2000, help="cap images for speed")
    args = ap.parse_args()

    cam = Path(args.cam_model)
    if not cam.is_file():
        raise SystemExit(f"missing {cam} — run scripts/export_cam_tflite.py first")

    paths = sorted(
        p for p in Path(args.images).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    it = _interpreter(cam)
    vectors = []
    for i, p in enumerate(paths, 1):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        v = _features(it, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if v is not None:
            vectors.append(v)
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}")

    if len(vectors) < 50:
        raise SystemExit(f"only {len(vectors)} usable images — need at least 50 for a covariance")

    x = np.stack(vectors)
    mean = x.mean(axis=0)
    # Shrinkage keeps the covariance invertible when features outnumber images.
    cov = np.cov(x, rowvar=False)
    cov += np.eye(cov.shape[0]) * (1e-3 * np.trace(cov) / cov.shape[0])
    inv_cov = np.linalg.pinv(cov)

    d = x - mean
    dist = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", d, inv_cov, d)))
    pct = {str(q): float(np.percentile(dist, q)) for q in (50, 90, 95, 99, 100)}

    Path(args.out).write_text(
        json.dumps(
            {
                "mean": mean.tolist(),
                "inv_cov": inv_cov.tolist(),
                "n_images": int(x.shape[0]),
                "train_distance_percentiles": pct,
            }
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} from {x.shape[0]} images, {x.shape[1]} features")
    print("training-set distance percentiles:")
    for q, v in pct.items():
        print(f"  p{q:>3}: {v:8.2f}")
    print(f"\nSuggested: export SKIN_GATE_MAX_OOD={pct['99'] * 1.1:.0f}")


if __name__ == "__main__":
    main()
