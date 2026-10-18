"""Talk to the ISIC Archive: list images by collection or query, download them, label them.

    python training/isic_fetch.py --collections 249 --query 'diagnosis_1:Malignant' \
        --out datasets/isic/bcn20000 --csv /tmp/bcn_malignant.csv
    python training/isic_fetch.py --collections 406 --metadata-only --csv /tmp/pad.csv

Used by prepare_datasets.py (training data) and build_external_bench.py (the held-out
benchmark), so both read the archive the same way and label it with the same table.

The archive is public and needs no login (https://api.isic-archive.com/api/v2/). Every
image carries a three-level diagnosis (`diagnosis_1` Benign / Malignant / Indeterminate,
`diagnosis_2` a family, `diagnosis_3` the named diagnosis), usually a `lesion_id` (several
photos of one spot share it), and an `image_type` ("dermoscopic", "clinical: close-up", ...).
Nothing here is clever: paginate, write rows, fetch JPEGs, skip what is already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.isic-archive.com/api/v2"
PAGE = 100  # the largest page the API serves
HEADERS = {"User-Agent": "skin-cancer-project/1.0 (research; training/isic_fetch.py)"}

# The archive collections this project uses, by id. Names are what the API returns.
COLLECTIONS = {
    212: "HAM10000",
    249: "BCN20000",
    286: "MSK-5",
    287: "MSK-4",
    288: "MSK-3",
    289: "MSK-1",
    290: "MSK-2",
    291: "UDA-2",
    292: "UDA-1",
    406: "PAD-UFES-20",
    425: "MILK10k",
}

# Columns of every CSV this module writes. `source` is the collection name.
FIELDS = [
    "isic_id", "source", "lesion_id", "diagnosis_1", "diagnosis_2", "diagnosis_3",
    "image_type", "pixels_x", "pixels_y", "attribution", "license", "url",
]

# --- The ISIC diagnosis tree -> our three labels --------------------------------------
# Same meaning as the HAM10000 map in the notebook (mel, bcc -> malignant; akiec -> pre_cancerous;
# nv, bkl, df, vasc -> benign). HAM's "akiec" holds actinic keratoses AND Bowen's disease
# (squamous cell carcinoma in situ), so both named diagnoses go to pre_cancerous here too.
# Anything else labelled Indeterminate, or without a diagnosis, gets no label and is dropped.
PRE_CANCEROUS_DX3 = {"Solar or actinic keratosis", "Squamous cell carcinoma in situ"}
LABELS = ["benign", "pre_cancerous", "malignant"]


def three_class_label(row: dict) -> str | None:
    """'benign' / 'pre_cancerous' / 'malignant', or None when the archive is not sure."""
    if row.get("diagnosis_3") in PRE_CANCEROUS_DX3:
        return "pre_cancerous"
    d1 = row.get("diagnosis_1")
    if d1 == "Benign":
        return "benign"
    if d1 == "Malignant":
        return "malignant"
    return None


# --- HTTP ---------------------------------------------------------------------------


def _get(url: str, tries: int = 5) -> bytes:
    """GET with a few patient retries: the archive rate-limits and sometimes 5xx's."""
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def collection_name(collection_id: int) -> str:
    if collection_id in COLLECTIONS:
        return COLLECTIONS[collection_id]
    return _get_json(f"{API}/collections/{collection_id}/")["name"]


def _row(result: dict, source: str) -> dict:
    clinical = result.get("metadata", {}).get("clinical", {})
    acq = result.get("metadata", {}).get("acquisition", {})
    return {
        "isic_id": result["isic_id"],
        "source": source,
        "lesion_id": clinical.get("lesion_id", "") or "",
        "diagnosis_1": clinical.get("diagnosis_1", "") or "",
        "diagnosis_2": clinical.get("diagnosis_2", "") or "",
        "diagnosis_3": clinical.get("diagnosis_3", "") or "",
        "image_type": acq.get("image_type", "") or "",
        "pixels_x": acq.get("pixels_x", "") or "",
        "pixels_y": acq.get("pixels_y", "") or "",
        "attribution": result.get("attribution", "") or "",
        "license": result.get("copyright_license", "") or "",
        "url": result.get("files", {}).get("full", {}).get("url", "") or "",
    }


def search(
    collection: int | None = None,
    query: str | None = None,
    limit: int | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Every image matching the filters, as rows of FIELDS. Pages until `next` is null.

    `query` is the archive's own search syntax, e.g. 'diagnosis_1:Malignant' or
    'image_type:"clinical: close-up"'. `limit` stops early (handy for tests).
    """
    params = {"limit": PAGE}
    if collection is not None:
        params["collections"] = collection
    if query:
        params["query"] = query
    source = collection_name(collection) if collection is not None else "ISIC"
    url: str | None = f"{API}/images/search/?{urllib.parse.urlencode(params)}"
    rows: list[dict] = []
    total = "?"
    while url:
        page = _get_json(url)
        total = page.get("count") or total  # only the first page carries the count
        rows.extend(_row(r, source) for r in page["results"])
        if not quiet:
            print(f"  {source}: {len(rows)}/{total}", file=sys.stderr, end="\r")
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break
        url = page.get("next")
    if not quiet:
        print(file=sys.stderr)
    return rows


def fetch_one(isic_id: str) -> dict:
    """One image's row by id (for the students' lists, which come as bare ids).

    The single-image endpoint does not say which collection the image is in, so `source`
    is filled with the attribution; build_external_bench.source_name() tidies it.
    """
    result = _get_json(f"{API}/images/{isic_id}/")
    return _row(result, result.get("attribution", "") or "ISIC")


# --- Files --------------------------------------------------------------------------


def download(rows: list[dict], out_dir: Path, workers: int = 32, quiet: bool = False) -> dict[str, Path]:
    """Fetch every row's full-size JPEG into out_dir/<isic_id>.jpg. Returns id -> path.

    Files already present (and not empty) are kept, so an interrupted run just resumes.
    Each file is written to a .part name first and renamed when complete.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    todo: list[dict] = []
    for r in rows:
        p = out_dir / f"{r['isic_id']}.jpg"
        paths[r["isic_id"]] = p
        if not (p.is_file() and p.stat().st_size > 0):
            todo.append(r)

    def grab(r: dict) -> str:
        p = paths[r["isic_id"]]
        part = p.with_suffix(".part")
        part.write_bytes(_get(r["url"]))
        part.replace(p)
        return r["isic_id"]

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed(pool.submit(grab, r) for r in todo):
            fut.result()  # raise on a failed download instead of pretending
            done += 1
            if not quiet and (done % 50 == 0 or done == len(todo)):
                print(f"  downloaded {done}/{len(todo)} into {out_dir}", file=sys.stderr, end="\r")
    if not quiet and todo:
        print(file=sys.stderr)
    return paths


def write_csv(rows: list[dict], path: Path, extra_fields: list[str] | None = None) -> None:
    fields = FIELDS + [f for f in (extra_fields or []) if f not in FIELDS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- CLI ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collections", type=int, nargs="+", required=True, help="archive collection ids")
    ap.add_argument("--query", default=None, help="archive search syntax, e.g. diagnosis_1:Malignant")
    ap.add_argument("--csv", type=Path, required=True, help="where to write the metadata rows")
    ap.add_argument("--out", type=Path, default=None, help="folder for the JPEGs (omit with --metadata-only)")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many rows per collection")
    args = ap.parse_args()
    if not args.metadata_only and args.out is None:
        ap.error("--out is required unless --metadata-only")

    rows: list[dict] = []
    for cid in args.collections:
        rows.extend(search(cid, args.query, args.limit))
    for r in rows:
        r["label_3"] = three_class_label(r) or ""
    write_csv(rows, args.csv, extra_fields=["label_3"])
    print(f"{len(rows)} rows -> {args.csv}")
    if not args.metadata_only:
        download(rows, args.out)
        print(f"images in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
