"""Score photos no model of ours trained on, exactly as the kiosk would read them.

    python training/external_eval.py                            # deployed models/, training/external_bench.csv
    python training/external_eval.py --models-dir /tmp/new_model   # another model's four files
    python training/external_eval.py --json /tmp/june.json        # keep per-photo results to diff two models
    python training/external_eval.py --no-force                   # do not override "spot"/"edge" refusals

Every photo goes through dermascan.scan.run_scan: the real JPEG decode, the 1024 px
shrink, the photo check, the 4-view TTA model and the screening threshold. The answer
counted is the one the visitor would get - flagged or not - never the tallest bar in the
staff drawer (those bars are temperature-scaled for display; the decision is made on the
raw cancer score, p(pre_cancerous) + p(malignant), against the threshold).

Photos the check refuses outright are reported as "not scored" and listed, never
silently dropped. Soft refusals ("one spot?", "a real edge?") are overridden the way a
visitor can tap "Check it anyway", unless --no-force.

The CSV needs: isic_id (or image_id), source, label_3, label_idx; build_external_bench.py
writes one. Images are <isic_id>.jpg under --images-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dermascan import config, scan  # noqa: E402
from dermascan.classifier import FLAGGED_LABELS, Classifier  # noqa: E402

BENCH_CSV = ROOT / "training" / "external_bench.csv"
BENCH_DIR = ROOT / "datasets" / "external_bench"
STUDENTS = "students' list"  # the summary row for group == "students" in the CSV
SWEEP = (0.11, 0.2, 0.3, 0.4, 0.47, 0.5, 0.6, 0.7)  # thresholds shown in the sweep, so two models can be compared


def load_classifier(models_dir: Path | None) -> Classifier:
    if models_dir is None:
        return Classifier(use_tta=True)
    d = Path(models_dir)
    return Classifier(
        d / config.MODEL_PATH.name, d / config.LABELS_PATH.name,
        d / config.THRESHOLDS_PATH.name, d / config.TEMPERATURE_PATH.name, use_tta=True,
    )


def score_photo(path: Path, clf: Classifier, force: bool) -> dict:
    """One row of results: what the kiosk said about this file."""
    out = scan.run_scan(path.read_bytes(), force=force, classifier=clf)
    if out.status != "ok" or out.prediction is None:
        code = out.refusal.code if out.refusal else out.status
        return {"status": out.status, "reason": str(code), "flagged": None}
    p = out.prediction
    return {
        "status": "ok",
        "flagged": bool(p.flagged),
        "label": p.label,
        "argmax": clf.labels[int(np.argmax(p.raw_probs))],
        "cancer_score": round(p.cancer_score, 4),
        "forced": bool(out.forced),
    }


def summarise(rows: list[dict], results: dict[str, dict]) -> dict[str, dict]:
    """Per-source and overall counts. Keys: source name, then STUDENTS and 'ALL'.

    `n` counts every photo of the source; the rates are over the scored ones (n - not_scored).
    """
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for r in rows:
        res = results[r["isic_id"]]
        groups[r["source"]].append((r, res))
        groups["ALL"].append((r, res))
        if r.get("group") == "students":
            groups[STUDENTS].append((r, res))

    def stats(pairs: list[tuple[dict, dict]]) -> dict:
        scored = [(r, s) for r, s in pairs if s["status"] == "ok"]
        cancers = [(r, s) for r, s in scored if r["label_3"] in FLAGGED_LABELS]
        malignant = [(r, s) for r, s in scored if r["label_3"] == "malignant"]
        benign = [(r, s) for r, s in scored if r["label_3"] not in FLAGGED_LABELS]
        flagged_c = sum(s["flagged"] for _, s in cancers)
        return {
            "n": len(pairs),
            "not_scored": len(pairs) - len(scored),
            "cancers": len(cancers),
            "flagged": flagged_c,
            "missed": len(cancers) - flagged_c,
            "sensitivity": flagged_c / len(cancers) if cancers else None,
            "malignant_recall": sum(s["flagged"] for _, s in malignant) / len(malignant) if malignant else None,
            "benign": len(benign),
            "specificity": sum(not s["flagged"] for _, s in benign) / len(benign) if benign else None,
            "argmax_accuracy": sum(s["argmax"] == r["label_3"] for r, s in scored) / len(scored) if scored else None,
        }

    return {k: stats(v) for k, v in groups.items()}


def fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def roc_auc(rows: list[dict], results: dict[str, dict]) -> float | None:
    """Area under the ROC curve of the cancer score, cancer vs benign, over the scored photos.

    Threshold-free, so it compares two models even when their thresholds differ. Plain
    rank statistic (the probability a random cancer scores above a random benign spot).
    """
    cancer = [results[r["isic_id"]]["cancer_score"] for r in rows if results[r["isic_id"]]["status"] == "ok" and r["label_3"] in FLAGGED_LABELS]
    benign = [results[r["isic_id"]]["cancer_score"] for r in rows if results[r["isic_id"]]["status"] == "ok" and r["label_3"] not in FLAGGED_LABELS]
    if not cancer or not benign:
        return None
    c, b = np.asarray(cancer)[:, None], np.asarray(benign)[None, :]
    return float((c > b).mean() + 0.5 * (c == b).mean())


def sweep(rows: list[dict], results: dict[str, dict]) -> list[tuple[float, float, float]]:
    """(threshold, sensitivity, specificity) over the scored photos, for each value in SWEEP."""
    scored = [(r["label_3"] in FLAGGED_LABELS, results[r["isic_id"]]["cancer_score"]) for r in rows if results[r["isic_id"]]["status"] == "ok"]
    is_cancer = np.array([c for c, _ in scored])
    score = np.array([s for _, s in scored])
    out = []
    for t in SWEEP:
        flagged = score >= t
        out.append((t, float(flagged[is_cancer].mean()), float((~flagged)[~is_cancer].mean())))
    return out


def print_table(summary: dict[str, dict], model_name: str, threshold: float) -> None:
    print(f"\n{model_name} · threshold {threshold:.3f} · flagged = cancer score >= threshold")
    print("n counts every photo; skip = refused by the photo check; the rates are over the scored ones")
    print(f"{'source':44s} {'n':>4} {'skip':>4} {'cancers':>7} {'flagged':>7} {'missed':>6} {'sens':>6} {'mal-rec':>7} {'benign':>6} {'spec':>6} {'argmax':>6}")
    sources = [k for k in summary if k not in ("ALL", STUDENTS)]
    order = sources + [k for k in (STUDENTS, "ALL") if k in summary]
    for k in order:
        s = summary[k]
        print(
            f"{k[:44]:44s} {s['n']:4d} {s['not_scored']:4d} {s['cancers']:7d} {s['flagged']:7d} {s['missed']:6d} "
            f"{fmt(s['sensitivity']):>6} {fmt(s['malignant_recall']):>7} {s['benign']:6d} {fmt(s['specificity']):>6} {fmt(s['argmax_accuracy']):>6}"
        )
    a = summary["ALL"]
    print("\nFor docs/METRICS.md:")
    print(f"| External sensitivity (pre+mal, at threshold) | {fmt(a['sensitivity'])} ({a['flagged']}/{a['cancers']}) |")
    print(f"| External malignant recall (at threshold) | {fmt(a['malignant_recall'])} |")
    print(f"| External specificity (benign, at threshold) | {fmt(a['specificity'])} (n={a['benign']}) |")
    print(f"| Not scored (hard refusals) | {a['not_scored']} of {a['n']} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=BENCH_CSV)
    ap.add_argument("--images-dir", type=Path, default=BENCH_DIR)
    ap.add_argument("--models-dir", type=Path, default=None, help="folder with the four model files (default: models/)")
    ap.add_argument("--json", type=Path, default=None, help="write per-photo results here")
    ap.add_argument("--no-force", action="store_true", help="count soft refusals as not scored")
    args = ap.parse_args()
    logging.getLogger("dermascan").setLevel(logging.WARNING)  # one log line per scan is noise here

    if not args.csv.is_file():
        sys.exit(f"{args.csv} does not exist. Run training/build_external_bench.py first.")
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("isic_id", r.get("image_id", ""))
        r.setdefault("source", "all")
    missing = [r["isic_id"] for r in rows if not (args.images_dir / f"{r['isic_id']}.jpg").is_file()]
    if missing:
        sys.exit(f"{len(missing)} images missing under {args.images_dir} (first: {missing[:3]}). Run build_external_bench.py.")

    clf = load_classifier(args.models_dir)
    print(f"Scoring {len(rows)} photos with {clf.model_path} through the kiosk path, 4-view TTA...")
    results: dict[str, dict] = {}
    for k, r in enumerate(rows, 1):
        results[r["isic_id"]] = score_photo(args.images_dir / f"{r['isic_id']}.jpg", clf, force=not args.no_force)
        if k % 50 == 0:
            print(f"  {k}/{len(rows)}", file=sys.stderr)

    summary = summarise(rows, results)
    print_table(summary, clf.model_path.name, float(clf.thresholds.get("screen_cancer_threshold", float("nan"))))
    auc = roc_auc(rows, results)
    print(f"| External ROC-AUC (cancer vs benign, threshold-free) | {fmt(auc)} |")
    by_source = {s: roc_auc([r for r in rows if r["source"] == s], results) for s in set(r["source"] for r in rows)}
    print("per-source AUC: " + " · ".join(f"{s[:24]} {fmt(a)}" for s, a in sorted(by_source.items()) if a is not None))
    print("\nThreshold sweep (sensitivity / specificity over all scored photos):")
    print("  " + "   ".join(f"{t:.2f}: {s:.3f}/{p:.3f}" for t, s, p in sweep(rows, results)))
    skipped = [(i, s["reason"]) for i, s in results.items() if s["status"] != "ok"]
    if skipped:
        print(f"\nNot scored ({len(skipped)}): " + ", ".join(f"{i} ({why})" for i, why in skipped))
    missed = [r["isic_id"] for r in rows if r["label_3"] in FLAGGED_LABELS and results[r["isic_id"]].get("flagged") is False]
    if missed:
        print(f"\nMissed cancers ({len(missed)}): " + " ".join(missed))
    if args.json:
        args.json.write_text(json.dumps({"model": str(clf.model_path), "summary": summary, "results": results}, indent=1))
        print(f"\nper-photo results -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
