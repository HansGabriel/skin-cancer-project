"""Saved scans: kept on this kiosk, in one folder on the SD card, until staff end the event.

A visitor taps "Save this scan", picks where the spot is (left arm, back...),
and then the GROUP it belongs to: one group is one spot, so "Left arm" and
"Left arm · 2" are two different moles on the same arm. A group's history starts
at the first scan saved into it, and the page can put the first and the latest
photo side by side - that is how E (evolving) is shown: a person compares the
photos. Nothing here decides "changed" or "not changed" on its own: tested on
same-mole photo pairs (docs/METRICS.md) that guess was wrong too often.

Each saved scan is two files in config.SAVED_DIR (default ~/.dermascan/saved):

    <id>.jpg    the photo, shrunk to SAVED_PHOTO_PX
    <id>.json   site, group, when, and the outcome exactly as the page rendered it

The JSON is written last, through a temporary file and an atomic rename, so a
power cut mid-save loses at most the scan being saved and never the others. At
start-up the folder is read back and ids continue from the highest one found.
The scans survive Exit, a crash, a reboot and a power cut; they go when:

    * "End event - erase all" (staff) empties the folder,
    * a visitor deletes their own scan,
    * past SAVED_MAX scans the oldest is dropped.

If the card cannot be written (full, read-only) the scan stays in memory for
this run and a warning is logged; a visitor is never refused a save for that.
No names, no accounts: a scan is known only by its group. docs/PRIVACY.md says
what this means for whoever holds the SD card.

Mentor's decision, 2026-10-11: saves used to live in memory only and vanished on
every reboot, which lost a day's groups when the Pi was switched off.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2

from dermascan import config, gate
from dermascan.scan import decode_jpeg

log = logging.getLogger("dermascan.saved")

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

    def record(self) -> dict:
        """What goes into <id>.json (the photo has its own file)."""
        return {"id": self.id, "site": self.site, "group": self.group, "saved_at": self.saved_at, "outcome": self.outcome}


def _shrink_jpeg(jpeg: bytes) -> bytes:
    rgb = gate.shrink(decode_jpeg(jpeg), config.SAVED_PHOTO_PX)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, config.SAVED_JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode the saved photo")
    return buf.tobytes()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a sibling temp file, then rename over `path`: readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _unlink_quietly(path: Path) -> None:
    """Remove a file if it is there; a card error is logged, never raised at a visitor."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove %s", path, exc_info=True)


class SavedScans:
    """The event's saved scans, oldest first. Thread-safe: Flask serves requests in threads.

    `folder` is where they live between runs; None keeps them in memory only
    (unit tests that do not care about disk).
    """

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = Path(folder) if folder is not None else None
        self._scans: OrderedDict[int, SavedScan] = OrderedDict()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._write_ok = True  # False after a failed write: /health shows the card is not keeping saves
        self._load()

    # --- the folder ---------------------------------------------------------------

    def _paths(self, scan_id: int) -> tuple[Path, Path]:
        assert self.folder is not None
        return self.folder / f"{scan_id}.jpg", self.folder / f"{scan_id}.json"

    def _load(self) -> None:
        """Read the folder back, oldest id first; tidy what a power cut may have left behind."""
        if self.folder is None or not self.folder.is_dir():
            return
        for stray in self.folder.glob("*.tmp"):
            stray.unlink(missing_ok=True)
        jsons = {int(p.stem): p for p in self.folder.glob("*.json") if p.stem.isdigit()}
        for orphan in self.folder.glob("*.jpg"):
            if not orphan.stem.isdigit() or int(orphan.stem) not in jsons:
                orphan.unlink(missing_ok=True)  # photo written, JSON never made it: not a saved scan
        for scan_id in sorted(jsons):
            jpg, js = self._paths(scan_id)
            try:
                rec = json.loads(js.read_text(encoding="utf-8"))
                scan = SavedScan(scan_id, str(rec["site"]), str(rec["group"]), float(rec["saved_at"]), jpg.read_bytes(), dict(rec["outcome"]))
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("saved scan %d unreadable, skipped", scan_id, exc_info=True)
                continue
            if scan.site not in SITES:
                log.warning("saved scan %d has unknown site %r, skipped", scan_id, scan.site)
                continue
            self._scans[scan_id] = scan
        if jsons:
            # Continue after every id on the card, skipped ones included, so a new
            # scan never overwrites an unreadable record's files.
            self._ids = itertools.count(max(jsons) + 1)
        self._drop_past_max()
        log.info("saved scans: %d loaded from %s", len(self._scans), self.folder)

    def _store(self, scan: SavedScan) -> None:
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            jpg, js = self._paths(scan.id)
            _write_atomic(jpg, scan.jpeg)
            _write_atomic(js, json.dumps(scan.record(), ensure_ascii=False).encode("utf-8"))
            self._write_ok = True
        except OSError:
            self._write_ok = False
            log.warning("saved scan %d kept in memory only: could not write %s", scan.id, self.folder, exc_info=True)

    def _discard(self, scan_id: int) -> None:
        if self.folder is None:
            return
        for p in self._paths(scan_id):
            _unlink_quietly(p)

    def _drop_past_max(self) -> None:
        while len(self._scans) > config.SAVED_MAX:
            oldest, _ = self._scans.popitem(last=False)
            self._discard(oldest)

    # --- the API the routes use ---------------------------------------------------

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
            scan = SavedScan(next(self._ids), site, group, time.time(), photo, outcome)
            self._scans[scan.id] = scan
            self._store(scan)
            self._drop_past_max()
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
            return {
                "count": len(self._scans),
                "bytes": sum(len(s.jpeg) for s in self._scans.values()),
                "folder": str(self.folder) if self.folder is not None else None,
                "on_disk": self.folder is not None and self._write_ok,  # False: the card is not keeping saves
            }

    def delete(self, scan_id: int) -> bool:
        """Drop one saved scan. False if it was not there (already deleted, or dropped past SAVED_MAX)."""
        with self._lock:
            gone = self._scans.pop(scan_id, None) is not None
            if gone:
                self._discard(scan_id)
        return gone

    def erase(self) -> None:
        """End event: forget every scan and remove every file in the folder."""
        with self._lock:
            ids = list(self._scans)
            self._scans.clear()
            for scan_id in ids:
                self._discard(scan_id)
            if self.folder is not None and self.folder.is_dir():
                for leftover in self.folder.iterdir():  # files from an older run, or a skipped unreadable scan
                    if leftover.suffix in (".jpg", ".json", ".tmp"):
                        _unlink_quietly(leftover)


saved = SavedScans(config.SAVED_DIR)
