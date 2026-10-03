"""Saved scans: kept in the server's memory for one event, never written to disk.

A visitor taps "Save this scan" and picks where the spot is (left arm, back...).
The kiosk keeps a small copy of the photo and the scan's outcome so the Saved
tab can list spots and reopen a result. All of it lives in this process:

    * "End event - erase all" (staff) empties it,
    * Exit, a crash, or switching the Pi off empties it,
    * past SAVED_MAX scans the oldest is dropped.

Nothing here touches the SD card, so docs/PRIVACY.md's "nothing stored on disk"
still holds. No names, no accounts: a scan is known only by its body site.
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
    saved_at: float  # time.time()
    jpeg: bytes  # shrunk to SAVED_PHOTO_PX
    outcome: dict  # ScanOutcome.to_dict(), exactly what the page rendered

    def summary(self) -> dict:
        v = self.outcome.get("verdict") or {}
        return {
            "id": self.id, "site": self.site, "saved_at": self.saved_at,
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

    def add(self, site: str, jpeg: bytes, outcome: dict) -> SavedScan:
        """Keep a small copy of this photo and its outcome; drop the oldest past SAVED_MAX."""
        if site not in SITES:
            raise ValueError(f"unknown body site {site!r}")
        scan = SavedScan(0, site, time.time(), _shrink_jpeg(jpeg), outcome)
        with self._lock:
            scan.id = next(self._ids)
            self._scans[scan.id] = scan
            while len(self._scans) > config.SAVED_MAX:
                self._scans.popitem(last=False)
        return scan

    def get(self, scan_id: int) -> SavedScan | None:
        with self._lock:
            return self._scans.get(scan_id)

    def spots(self) -> list[dict]:
        """One entry per body site, newest site first, each with its scans newest first."""
        with self._lock:
            scans = list(self._scans.values())
        by_site: dict[str, list[SavedScan]] = {}
        for s in reversed(scans):
            by_site.setdefault(s.site, []).append(s)
        return [{"site": site, "scans": [s.summary() for s in group]} for site, group in by_site.items()]

    def stats(self) -> dict:
        with self._lock:
            return {"count": len(self._scans), "bytes": sum(len(s.jpeg) for s in self._scans.values())}

    def erase(self) -> None:
        with self._lock:
            self._scans.clear()


saved = SavedScans()
