"""Assemble the training data from HAM10000 + the ISIC archive, labelled and indexed.

    python training/prepare_datasets.py                   # download everything, then index
    python training/prepare_datasets.py --skip-download   # index only what is already on disk
    python training/prepare_datasets.py --collections 249 425   # a subset, for a quick try

Writes training/dataset_index.csv, which the notebook reads instead of HAM10000's own
metadata. One row per photo: source, image_id, lesion_id, dx, label_3, label_idx,
image_type, path. The notebook splits it by lesion_id (several photos of one spot stay on
one side of the line) and never sees anything that is not in this file.

Sources (ISIC collection ids in isic_fetch.COLLECTIONS):
  HAM10000 (local copy)      dermoscopy, Vienna + Queensland     - what the June model knew
  BCN20000 (249)             dermoscopy, Hospital Clinic Barcelona
  MSK-1..5 (286-290)         dermoscopy, Memorial Sloan Kettering
  MILK10k (425)              dermoscopy AND clinical close-ups of the same lesions
  PAD-UFES-20 (406)          smartphone close-ups, Brazil         - the closest thing to the Pi camera
UDA-1/2 (the 2016 challenge set) is never trained on: it has no lesion ids, so it could
not be split safely. It is test-only, through training/external_bench.csv.

Labels come from isic_fetch.three_class_label (the archive's diagnosis tree); HAM's own
dx column is mapped the way the June model's notebook did. Photos the archive calls
Indeterminate (other than actinic keratosis) are dropped.

Every isic_id and lesion_id in training/external_bench.csv is removed, so the benchmark
stays external for every model trained from this index. The script refuses to finish if
any slipped through.

Each photo is also shrunk once to <= 512 px on its longest edge into datasets/cache512/
(JPEG quality 95). Training reads those: six times fewer bytes per epoch, no change to
what the model sees after its 224 px resize.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import isic_fetch as isic  # noqa: E402

DATASETS = ROOT / "datasets"
HAM_DIR = DATASETS / "ham10000"
ISIC_DIR = DATASETS / "isic"
CACHE_DIR = DATASETS / "cache512"
CACHE_PX = 512
INDEX_CSV = ROOT / "training" / "dataset_index.csv"
BENCH_CSV = ROOT / "training" / "external_bench.csv"

TRAIN_COLLECTIONS = [249, 286, 287, 288, 289, 290, 406, 425]

# HAM10000's seven dx codes -> our three labels (the June notebook's table, unchanged).
HAM_CLASS_MAP = {
    "mel": "malignant", "bcc": "malignant", "akiec": "pre_cancerous",
    "nv": "benign", "bkl": "benign", "df": "benign", "vasc": "benign",
}
INDEX_FIELDS = ["source", "image_id", "lesion_id", "dx", "label_3", "label_idx", "image_type", "path"]


def ham_rows() -> list[dict]:
    """HAM10000 from the local Kaggle copy, as index rows (path = original file for now)."""
    meta = HAM_DIR / "HAM10000_metadata.csv"
    if not meta.is_file():
        sys.exit(f"{meta} not found. The notebook's Cell 2 says where HAM10000 lives.")
    paths: dict[str, Path] = {}
    for folder in HAM_DIR.iterdir():
        if folder.is_dir() and folder.name.lower().startswith("ham10000_images"):
            paths.update({p.stem: p for p in folder.glob("*.jpg")})
    rows = []
    with open(meta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "source": "HAM10000", "image_id": r["image_id"], "lesion_id": r["lesion_id"], "dx": r["dx"],
                "label_3": HAM_CLASS_MAP[r["dx"]], "image_type": "dermoscopic", "path": str(paths.get(r["image_id"], "")),
            })
    missing = [r["image_id"] for r in rows if not r["path"]]
    if missing:
        sys.exit(f"{len(missing)} HAM10000 images missing on disk (first: {missing[:3]})")
    return rows


def isic_rows(collection: int, download: bool) -> list[dict]:
    """One archive collection: metadata (cached as metadata.csv) and, if asked, the JPEGs."""
    folder = ISIC_DIR / isic.collection_name(collection)
    meta = folder / "metadata.csv"
    if meta.is_file():
        found = isic.read_csv(meta)
    else:
        found = isic.search(collection)
        isic.write_csv(found, meta)
    if download:
        isic.download(found, folder)
    rows = []
    for r in found:
        label = isic.three_class_label(r)
        if label is None:
            continue
        rows.append({
            "source": r["source"], "image_id": r["isic_id"], "lesion_id": r["lesion_id"] or r["isic_id"],
            "dx": r["diagnosis_3"], "label_3": label, "image_type": r["image_type"], "path": str(folder / f"{r['isic_id']}.jpg"),
        })
    return rows


def shrink_to_cache(src: Path) -> Path:
    """<= 512 px copy in datasets/cache512/, made once. Returns the cached path."""
    dst = CACHE_DIR / src.name
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {src}")
    h, w = img.shape[:2]
    s = CACHE_PX / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    tmp = dst.with_name(dst.stem + ".part.jpg")  # cv2 picks the format from the extension
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    tmp.replace(dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", type=int, nargs="*", default=TRAIN_COLLECTIONS)
    ap.add_argument("--skip-download", action="store_true", help="index only what is already on disk")
    ap.add_argument("--no-cache", action="store_true", help="index the original files instead of 512 px copies")
    args = ap.parse_args()
    for cid in args.collections:
        assert cid not in (291, 292), "UDA is test-only (see the docstring)"

    rows = ham_rows()
    print(f"HAM10000: {len(rows)} rows")
    for cid in args.collections:
        got = isic_rows(cid, download=not args.skip_download)
        print(f"{isic.collection_name(cid)}: {len(got)} labelled rows")
        rows.extend(got)

    # Keep one row per image id (the archive collections do not overlap, but be sure).
    seen: set[str] = set()
    unique = []
    for r in rows:
        if r["image_id"] not in seen:
            seen.add(r["image_id"])
            unique.append(r)
    rows = unique

    # The external benchmark is sacred: nothing in it, and no other photo of its lesions, trains.
    if not BENCH_CSV.is_file():
        sys.exit(f"{BENCH_CSV} missing. Run training/build_external_bench.py first.")
    bench = isic.read_csv(BENCH_CSV)
    held_ids = {r["isic_id"] for r in bench}
    held_lesions = {r["lesion_id"] for r in bench if r["lesion_id"]}
    before = len(rows)
    rows = [r for r in rows if r["image_id"] not in held_ids and r["lesion_id"] not in held_lesions]
    print(f"removed {before - len(rows)} rows that share an image or a lesion with the external benchmark")
    assert not ({r["image_id"] for r in rows} & held_ids) and not ({r["lesion_id"] for r in rows} & held_lesions)

    if not args.skip_download or not args.no_cache:
        on_disk = [r for r in rows if Path(r["path"]).is_file()]
        if len(on_disk) != len(rows):
            print(f"warning: {len(rows) - len(on_disk)} images are not on disk yet; they are left out of the index")
        rows = on_disk
    if not args.no_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"shrinking {len(rows)} photos to <= {CACHE_PX} px in {CACHE_DIR} (skips what is done)...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            cached = list(pool.map(lambda r: shrink_to_cache(Path(r["path"])), rows))
        for r, p in zip(rows, cached):
            r["path"] = str(p)

    for r in rows:
        r["label_idx"] = isic.LABELS.index(r["label_3"])
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows, {len({r['lesion_id'] for r in rows})} lesions -> {INDEX_CSV}\n")
    counts = Counter((r["source"], r["label_3"]) for r in rows)
    sources = sorted({s for s, _ in counts})
    print(f"{'source':14s}" + "".join(f"{l:>15}" for l in isic.LABELS) + f"{'total':>9}")
    for s in sources:
        print(f"{s:14s}" + "".join(f"{counts[(s, l)]:15d}" for l in isic.LABELS) + f"{sum(counts[(s, l)] for l in isic.LABELS):9d}")
    print(f"{'ALL':14s}" + "".join(f"{sum(counts[(s, l)] for s in sources):15d}" for l in isic.LABELS) + f"{len(rows):9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
