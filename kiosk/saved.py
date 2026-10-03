"""Saved scans: kept in the server's memory for one event, never written to disk.

A visitor taps "Save this scan", picks where the spot is (left arm, back...),
and then the GROUP it belongs to: one group is one spot, so "Left arm" and
"Left arm · 2" are two different moles on the same arm. A group's history starts
at the first scan saved into it, and the page can put the first and the latest
photo side by side - that is how E (evolving) is shown: a person compares the
photos. Nothing here decides "changed" or "not changed" on its own: tested on
same-mole photo pairs (docs/METRICS.md) that guess was wrong too often.

The kiosk keeps a small copy of each photo and the scan's outcome. All of it lives in this process:

    * "End event - erase all" (staff) empties it,
    * Exit, a crash, or switching the Pi off empties it,
    * past SAVED_MAX scans the oldest is dropped.

Nothing here touches the SD card, so docs/PRIVACY.md's "nothing stored on disk"
still holds. No names, no accounts: a scan is known only by its group.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2

from dermascan import config, gate
from dermascan.scan import decode_jpeg

# Where on the body. The page shows these as buttons; anything else is refused.
SITES = (
    "Face", "Scalp", "Neck", "Chest", "Stomach", "Back",
    "Left arm", "Right arm", "Left hand", "Right hand",
    "Left leg", "Right leg", "Left foot", "Right foot", "Other",
)


@dataclass
class SavedScan:
    id: int
    site: str
    group: str  # "Left arm", "Left arm · 2": one group is one spot
    saved_at: float  # time.time()
    jpeg: bytes  # shrunk to SAVED_PHOTO_PX
    outcome: dict  # ScanOutcome.to_dict(), exactly what the page rendered

    def summary(self) -> dict:
        v = self.outcome.get("verdict") or {}
        return {
            "id": self.id, "site": self.site, "group": self.group, "saved_at": self.saved_at,
            "headline": v.get("headline", ""), "tone": v.get("tone", ""), "state": v.get("state", ""),
        }


def _shrink_jpeg(jpeg: bytes) -> bytes:
    rgb = gate.shrink(decode_jpeg(jpeg), config.SAVED_PHOTO_PX)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, config.SAVED_JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode the saved photo")
    return buf.tobytes()


class SavedScans:
    """The event's saved scans, oldest first. Thread-safe: Flask serves requests in threads."""

    def __init__(self) -> None:
        self._scans: OrderedDict[int, SavedScan] = OrderedDict()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, site: str, jpeg: bytes, outcome: dict, group: str | None = None) -> SavedScan:
        """Keep a small copy of this photo and its outcome; drop the oldest past SAVED_MAX.

        `group` names an existing group to add this scan to (its history); None
        starts a new group for `site`, named "Left arm", then "Left arm · 2", ...
        """
        if site not in SITES:
            raise ValueError(f"unknown body site {site!r}")
        photo = _shrink_jpeg(jpeg)
        with self._lock:
            if group is not None:
                if not any(s.group == group and s.site == site for s in self._scans.values()):
                    raise ValueError(f"no group {group!r} on {site!r}")
            else:
                taken = {s.group for s in self._scans.values() if s.site == site}
                group = next(name for n in itertools.count(1) if (name := site if n == 1 else f"{site} · {n}") not in taken)
            scan = SavedScan(0, site, group, time.time(), photo, outcome)
            scan.id = next(self._ids)
            self._scans[scan.id] = scan
            while len(self._scans) > config.SAVED_MAX:
                self._scans.popitem(last=False)
        return scan

    def get(self, scan_id: int) -> SavedScan | None:
        with self._lock:
            return self._scans.get(scan_id)

    def spots(self) -> list[dict]:
        """One entry per group (one spot), most recently used first, each with its scans newest first."""
        with self._lock:
            scans = list(self._scans.values())
        by_group: dict[str, list[SavedScan]] = {}
        for s in reversed(scans):
            by_group.setdefault(s.group, []).append(s)
        return [{"group": g, "site": group[0].site, "scans": [s.summary() for s in group]} for g, group in by_group.items()]

    def history(self, scan_id: int) -> dict | None:
        """Where this scan sits in its group: the group's first scan, how many, and which number this is."""
        with self._lock:
            scan = self._scans.get(scan_id)
            if scan is None:
                return None
            group = [s for s in self._scans.values() if s.group == scan.group]
        first = group[0]
        return {
            "group": scan.group, "count": len(group), "number": group.index(scan) + 1,
            "first": first.summary(), "latest": group[-1].summary(),
        }

    def stats(self) -> dict:
        with self._lock:
            return {"count": len(self._scans), "bytes": sum(len(s.jpeg) for s in self._scans.values())}

    def delete(self, scan_id: int) -> bool:
        """Drop one saved scan. False if it was not there (already deleted, or dropped past SAVED_MAX)."""
        with self._lock:
            return self._scans.pop(scan_id, None) is not None

    def erase(self) -> None:
        with self._lock:
            self._scans.clear()


saved = SavedScans()
