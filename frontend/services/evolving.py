"""E (Evolving) from case scan history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import cv2
import numpy as np

from services.abcde import LetterResult, lab_cluster_centers, mean_color_drift_delta_e
from services.storage import Scan, Storage, get_storage
from theme.tokens import EVOLUTION

LetterAbcd = dict[str, LetterResult]


@dataclass
class EvolutionResult:
    delta_days: int
    diameter_growth_mm: float
    diameter_growth_pct: float
    color_drift_delta_e: float
    border_change: float
    asymmetry_change: float
    tier: Literal[0, 1, 2]
    verdict: str
    history_count: int


def _val(abcd: LetterAbcd, letter: str) -> float | None:
    v = abcd.get(letter, {}).get("value")
    return float(v) if isinstance(v, (int, float)) else None


def _parse_ts(iso: str) -> datetime:
    s = iso.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)


def _load_scan_rgb_mask(store: Storage, scan: Scan) -> tuple[np.ndarray | None, np.ndarray | None]:
    img_path = store.root / scan.image_path
    if not img_path.is_file():
        return None, None
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask: np.ndarray | None = None
    if scan.mask_path:
        mp = store.root / scan.mask_path
        if mp.is_file():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = m
    if mask is None:
        from services.segmentation import segment_safe

        mask = segment_safe(rgb)
    return rgb, mask


def compute(
    case_id: str,
    current_abcd: LetterAbcd,
    *,
    current_rgb: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
    taken_at: datetime | None = None,
) -> EvolutionResult:
    store = get_storage()
    scans = store.list_scans(case_id)
    history_count = len(scans) + 1
    if len(scans) < 1:
        return EvolutionResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "needs history", history_count)

    earliest = scans[0]
    prev = json.loads(earliest.abcd_json)
    t0 = _parse_ts(earliest.taken_at)
    t1 = taken_at or datetime.now(timezone.utc)
    delta_days = max(1, (t1 - t0).days)

    d0 = _val(prev, "D") or 0.1
    d1 = _val(current_abcd, "D") or d0
    diam_g = d1 - d0
    diam_pct = (diam_g / max(d0, 0.1)) * 100.0
    border_ch = abs((_val(current_abcd, "B") or 0) - (_val(prev, "B") or 0))
    asym_ch = abs((_val(current_abcd, "A") or 0) - (_val(prev, "A") or 0))

    color_drift = 0.0
    if current_rgb is not None and current_mask is not None:
        rgb0, mask0 = _load_scan_rgb_mask(store, earliest)
        if rgb0 is not None and mask0 is not None:
            c0 = lab_cluster_centers(rgb0, mask0)
            c1 = lab_cluster_centers(current_rgb, current_mask)
            color_drift = mean_color_drift_delta_e(c0, c1)
    else:
        color_drift = abs((_val(current_abcd, "C") or 0) - (_val(prev, "C") or 0))

    tier: Literal[0, 1, 2] = 0
    verdict = "stable"
    if (
        diam_g > EVOLUTION.diam_watch_mm
        or color_drift > EVOLUTION.de_watch
        or border_ch > EVOLUTION.border_change_suspicious
    ):
        tier, verdict = 2, "changing"
    elif diam_g > EVOLUTION.diam_stable_mm or color_drift > EVOLUTION.de_stable:
        tier, verdict = 1, "watch"

    return EvolutionResult(
        delta_days,
        diam_g,
        diam_pct,
        color_drift,
        border_ch,
        asym_ch,
        tier,
        verdict,
        history_count,
    )


def to_letter_result(ev: EvolutionResult) -> LetterResult:
    if ev.verdict == "needs history":
        return {"value": None, "tier": 0, "verdict": "needs history", "detail": "NEEDS HISTORY"}
    return {
        "value": round(ev.diameter_growth_mm, 2),
        "tier": ev.tier,
        "verdict": ev.verdict,
        "detail": f"Δ over {ev.delta_days} d",
    }


def apply_to_abcde(
    case_id: str,
    abcde: dict[str, LetterResult] | None,
    *,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, LetterResult] | None:
    if abcde is None:
        return None
    abcd = {k: abcde[k] for k in "ABCD" if k in abcde}
    if not abcd:
        return abcde
    ev = compute(case_id, abcd, current_rgb=rgb, current_mask=mask)
    out = dict(abcde)
    out["E"] = to_letter_result(ev)
    return out
