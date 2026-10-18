"""Build the frozen external benchmark: training/external_bench.csv + datasets/external_bench/.

    python training/build_external_bench.py              # metadata + images
    python training/build_external_bench.py --no-download

The benchmark is the set of photos no model of ours ever trains on. It has two parts:

1. The malignant photos the students tested in October 2026 (99 distinct ids below, from
   their list of 100; one typo corrected, one duplicate dropped, both noted). They come from hospitals HAM10000 never saw:
   Barcelona (BCN20000), the 2016 UDA set, MILK study clinical close-ups, Vienna, MSKCC,
   Buenos Aires. The archive's own diagnosis is the ground truth (two turned out not to be
   malignant; they stay, with the archive's label).
2. Benign and pre-cancerous controls from the same kinds of source, drawn with a fixed seed,
   one photo per lesion, so specificity can be measured next to sensitivity.

prepare_datasets.py excludes every lesion_id and isic_id listed here from training, so
the benchmark stays external for every model that follows. The CSV is committed; the
images are not (datasets/ is gitignored) - rerun this script to fetch them again.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import isic_fetch as isic  # noqa: E402

BENCH_CSV = ROOT / "training" / "external_bench.csv"
BENCH_DIR = ROOT / "datasets" / "external_bench"
SEED = 42

# The students' list, 18 October 2026, 100 lines. "ISIC_000170" was a typo for ISIC_0000170.
# Row 37, "ISIC_587936", is not an archive id; the next row is ISIC_5987936 with near-identical
# scores, so it was the same photo listed twice. 99 distinct photos remain.
STUDENT_MALIGNANT = """
ISIC_0053456 ISIC_0056469 ISIC_0053481 ISIC_0053485 ISIC_0053503 ISIC_0053507 ISIC_0053510
ISIC_0053518 ISIC_0053520 ISIC_0053533 ISIC_0053582 ISIC_0053615 ISIC_0053628 ISIC_0053644
ISIC_0053670 ISIC_0015088 ISIC_0015120 ISIC_0015121 ISIC_0015169 ISIC_0015228 ISIC_0015441
ISIC_0015478 ISIC_0015596 ISIC_0015606 ISIC_0016438 ISIC_0016911 ISIC_0017350 ISIC_0017389
ISIC_0017406 ISIC_0017409 ISIC_0017411 ISIC_2700242 ISIC_6444803 ISIC_5149698 ISIC_1540514
ISIC_0211418 ISIC_5987936 ISIC_0024294 ISIC_0023763 ISIC_0021954 ISIC_0000002 ISIC_0000004
ISIC_0000022 ISIC_0000026 ISIC_0000027 ISIC_0000029 ISIC_0000030 ISIC_0000031 ISIC_0000035
ISIC_0000036 ISIC_0000043 ISIC_0000049 ISIC_0000054 ISIC_0000056 ISIC_0000070 ISIC_0000076
ISIC_0000077 ISIC_0000078 ISIC_0000139 ISIC_0000140 ISIC_0000141 ISIC_0000142 ISIC_0000143
ISIC_0000144 ISIC_0000145 ISIC_0000146 ISIC_0000147 ISIC_0000148 ISIC_0000149 ISIC_0000150
ISIC_0000151 ISIC_0000152 ISIC_0000153 ISIC_0000154 ISIC_0000155 ISIC_0000156 ISIC_0000157
ISIC_0000158 ISIC_0000159 ISIC_0000160 ISIC_0000161 ISIC_0000162 ISIC_0000163 ISIC_0000164
ISIC_0000165 ISIC_0000166 ISIC_0000167 ISIC_0000168 ISIC_0000169 ISIC_0000170 ISIC_0000171
ISIC_0000172 ISIC_0000173 ISIC_0000174 ISIC_0000175 ISIC_0000176 ISIC_0000280 ISIC_0000281
ISIC_9367522
""".split()

# Controls: (collection id, archive query, how many lesions). One photo per lesion.
CONTROLS = [
    (249, "diagnosis_1:Benign", 120),  # BCN20000 dermoscopy
    (249, 'diagnosis_3:"Solar or actinic keratosis"', 30),
    (292, "diagnosis_1:Benign", 80),  # UDA-1, the 2016 set
    (291, "diagnosis_1:Benign", 23),  # UDA-2 (all of them)
    (425, 'image_type:"clinical: close-up" AND diagnosis_1:Benign', 120),  # MILK10k phone-like photos
    (425, 'image_type:"clinical: close-up" AND diagnosis_3:"Solar or actinic keratosis"', 30),
    (287, "diagnosis_1:Benign", 50),  # MSK-4 dermoscopy
    (286, "diagnosis_1:Benign", 50),  # MSK-5 dermoscopy
]


def source_name(row: dict) -> str:
    """A short, readable origin for the per-source table (attribution + image kind)."""
    a = row["attribution"]
    if a == "Anonymous":
        # Two anonymous donors share the label. The 2016 UDA set has the archive's first ids
        # (ISIC_0000000-ISIC_0001xxx); the later anonymous dermoscopy (MSK-type, 2017-18) starts
        # at ISIC_0012xxx. A string compare on the zero-padded id separates them.
        a = "UDA (ISIC 2016)" if row["isic_id"] < "ISIC_0010000" else "Anonymous (ISIC 2017-18)"
    a = {
        "Hospital Clínic de Barcelona": "BCN20000",
        "MILK study team": "MILK10k",
        "Department of Dermatology, Medical University of Vienna": "Vienna",
        "Memorial Sloan Kettering Cancer Center": "MSKCC",
        "Hospital Italiano de Buenos Aires": "HIBA",
        "Federal University of Espírito Santo (UFES)": "PAD-UFES-20",
    }.get(a, a)
    kind = row["image_type"] or "unknown"
    return f"{a} · {kind}"


def one_per_lesion(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_lesion: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lesion[r["lesion_id"] or r["isic_id"]].append(r)
    lesions = sorted(by_lesion)
    rng.shuffle(lesions)
    return [rng.choice(by_lesion[l]) for l in lesions[:n]]


def build() -> list[dict]:
    rng = random.Random(SEED)
    assert len(STUDENT_MALIGNANT) == 99 and len(set(STUDENT_MALIGNANT)) == 99
    print("fetching the students' 99 ids...", file=sys.stderr)
    rows = []
    for i, isic_id in enumerate(STUDENT_MALIGNANT, 1):
        r = isic.fetch_one(isic_id)
        r["group"] = "students"
        rows.append(r)
        print(f"  {i}/99", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    # The archive, not the students' list, is the ground truth. On 18 Oct 2026 two of the
    # 99 were not malignant: ISIC_0056469 (actinic keratosis) and ISIC_0000027 (nevus).
    for r in rows:
        if isic.three_class_label(r) != "malignant":
            print(f"  note: {r['isic_id']} is {r['diagnosis_3']} in the archive, kept with that label", file=sys.stderr)
    taken = {r["isic_id"] for r in rows}
    taken_lesions = {r["lesion_id"] for r in rows if r["lesion_id"]}

    for collection, query, n in CONTROLS:
        found = isic.search(collection, query)
        found = [r for r in found if r["isic_id"] not in taken and r["lesion_id"] not in taken_lesions]
        picked = one_per_lesion(found, n, rng)
        for r in picked:
            r["group"] = "control"
            taken.add(r["isic_id"])
            if r["lesion_id"]:
                taken_lesions.add(r["lesion_id"])
        rows.extend(picked)

    for r in rows:
        r["label_3"] = isic.three_class_label(r)
        r["label_idx"] = isic.LABELS.index(r["label_3"])
        r["source"] = source_name(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    rows = build()
    isic.write_csv(rows, BENCH_CSV, extra_fields=["group", "label_3", "label_idx"])
    print(f"{len(rows)} rows -> {BENCH_CSV}")
    for (src, lab), n in sorted(Counter((r["source"], r["label_3"]) for r in rows).items()):
        print(f"  {n:4d}  {lab:14s} {src}")
    if not args.no_download:
        isic.download(rows, BENCH_DIR)
        print(f"images in {BENCH_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
