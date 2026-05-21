"""SQLite: folders → cases → scans."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_VERSION = "0.4.0"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, color TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
  name TEXT NOT NULL, body_site TEXT, notes TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans (
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  taken_at TEXT NOT NULL, image_path TEXT NOT NULL, mask_path TEXT,
  pixels_per_mm REAL NOT NULL, label TEXT NOT NULL, confidence REAL NOT NULL,
  probs_json TEXT NOT NULL, seven_probs_json TEXT, abcd_json TEXT NOT NULL, e_json TEXT,
  composite_risk REAL NOT NULL, risk_band TEXT NOT NULL, quality_json TEXT NOT NULL,
  app_version TEXT NOT NULL, model_sha256 TEXT);
CREATE INDEX IF NOT EXISTS idx_scans_case_taken ON scans(case_id, taken_at);
"""


@dataclass(frozen=True)
class Folder:
    id: str
    name: str
    color: str | None
    created_at: str


@dataclass(frozen=True)
class Case:
    id: str
    folder_id: str
    name: str
    body_site: str | None
    notes: str | None
    created_at: str


@dataclass(frozen=True)
class Scan:
    id: str
    case_id: str
    taken_at: str
    image_path: str
    mask_path: str | None
    pixels_per_mm: float
    label: str
    confidence: float
    probs_json: str
    seven_probs_json: str | None
    abcd_json: str
    e_json: str | None
    composite_risk: float
    risk_band: str
    quality_json: str
    app_version: str
    model_sha256: str | None


def data_dir() -> Path:
    if os.environ.get("DERMASCAN_DATA_DIR"):
        return Path(os.environ["DERMASCAN_DATA_DIR"]).expanduser().resolve()
    return Path.home() / ".dermascan"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: str | None) -> str | None:
    p = Path(path) if path else None
    if not p or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Storage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or data_dir()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "masks").mkdir(exist_ok=True)
        (self.root / "cache").mkdir(exist_ok=True)
        self.db_path = self.root / "dermascan.db"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_folder(self, name: str, color: str | None = "#B58CF0") -> Folder:
        fid, created = str(uuid.uuid4()), _utc_now()
        with self._connect() as conn:
            conn.execute("INSERT INTO folders VALUES (?,?,?,?)", (fid, name, color, created))
        return Folder(fid, name, color, created)

    def list_folders(self) -> list[Folder]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM folders ORDER BY created_at DESC").fetchall()
        return [Folder(**dict(r)) for r in rows]

    def get_folder(self, folder_id: str) -> Folder | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM folders WHERE id=?", (folder_id,)).fetchone()
        return Folder(**dict(r)) if r else None

    def create_case(self, folder_id: str, name: str, *, body_site: str | None = None, notes: str | None = None) -> Case:
        cid, created = str(uuid.uuid4()), _utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cases VALUES (?,?,?,?,?,?)",
                (cid, folder_id, name, body_site, notes, created),
            )
        return Case(cid, folder_id, name, body_site, notes, created)

    def list_cases(self, folder_id: str) -> list[Case]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cases WHERE folder_id=? ORDER BY created_at DESC", (folder_id,)).fetchall()
        return [Case(**dict(r)) for r in rows]

    def get_case(self, case_id: str) -> Case | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return Case(**dict(r)) if r else None

    def list_scans(self, case_id: str) -> list[Scan]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM scans WHERE case_id=? ORDER BY taken_at ASC", (case_id,)).fetchall()
        return [Scan(**dict(r)) for r in rows]

    def save_scan(self, case_id: str, pipeline: dict[str, Any], image_bytes: bytes, *, model_path: str | None = None) -> Scan:
        sr = pipeline["scan_result"]
        sid, taken = str(uuid.uuid4()), _utc_now()
        rel_img = f"images/{sid}.jpg"
        (self.root / rel_img).write_bytes(image_bytes)
        mask_rel = None
        mask = pipeline.get("mask")
        if mask is not None:
            mask_rel = f"masks/{sid}.png"
            cv2.imwrite(str(self.root / mask_rel), (mask > 0).astype(np.uint8) * 255)
        abcde = dict(pipeline.get("abcde") or {})
        rgb = pipeline.get("rgb")
        mask = pipeline.get("mask")
        from services.evolving import apply_to_abcde
        from services.risk import composite_risk_score, risk_band

        abcde = apply_to_abcde(case_id, abcde, rgb=rgb, mask=mask) or abcde
        p_mal = float(sr.probs.get("malignant", 0.0))
        composite = float(composite_risk_score(p_mal, abcde))
        risk_band_val = risk_band(composite)
        row = Scan(
            id=sid,
            case_id=case_id,
            taken_at=taken,
            image_path=rel_img,
            mask_path=mask_rel,
            pixels_per_mm=float(pipeline.get("pixels_per_mm", 10.0)),
            label=sr.label,
            confidence=float(sr.confidence),
            probs_json=json.dumps(sr.probs),
            seven_probs_json=json.dumps(pipeline["seven_class_probs"]) if pipeline.get("seven_class_probs") else None,
            abcd_json=json.dumps({k: abcde[k] for k in "ABCD" if k in abcde}),
            e_json=json.dumps(abcde.get("E")),
            composite_risk=composite,
            risk_band=risk_band_val,
            quality_json=json.dumps(pipeline.get("quality") or {}),
            app_version=APP_VERSION,
            model_sha256=_sha256_file(model_path),
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(getattr(row, f) for f in Scan.__dataclass_fields__),
            )
        return row

    def delete_folder(self, folder_id: str) -> None:
        paths = [(s.mask_path, s.image_path) for c in self.list_cases(folder_id) for s in self.list_scans(c.id)]
        with self._connect() as conn:
            conn.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        self._unlink(paths)

    def delete_case(self, case_id: str) -> None:
        paths = [(s.mask_path, s.image_path) for s in self.list_scans(case_id)]
        with self._connect() as conn:
            conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
        self._unlink(paths)

    def _unlink(self, paths: list[tuple[str | None, str]]) -> None:
        for mask_p, img_p in paths:
            for rel in (mask_p, img_p):
                if rel:
                    p = self.root / rel
                    if p.is_file():
                        p.unlink()

    def export_case_csv(self, case_id: str) -> str:
        import csv
        import io

        buf, w = io.StringIO(), csv.writer(buf)
        w.writerow(["id", "taken_at", "label", "confidence", "composite_risk", "risk_band"])
        for s in self.list_scans(case_id):
            w.writerow([s.id, s.taken_at, s.label, s.confidence, s.composite_risk, s.risk_band])
        return buf.getvalue()

    def reset_all(self) -> None:
        with self._connect() as conn:
            conn.executescript("DELETE FROM scans; DELETE FROM cases; DELETE FROM folders;")
        for sub in ("images", "masks"):
            d = self.root / sub
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file():
                        p.unlink()

    def storage_size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def clear_cache(self) -> None:
        cache = self.root / "cache"
        if cache.is_dir():
            for p in cache.iterdir():
                if p.is_file():
                    p.unlink()

    def latest_scans_global(self, limit: int = 6) -> list[tuple[Scan, Case, Folder]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.*, c.name AS case_name, c.folder_id, f.name AS folder_name, f.color, f.created_at AS f_created
                   FROM scans s JOIN cases c ON c.id=s.case_id JOIN folders f ON f.id=c.folder_id
                   ORDER BY s.taken_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            scan = Scan(**{k: d[k] for k in Scan.__dataclass_fields__})
            case = Case(id=d["case_id"], folder_id=d["folder_id"], name=d["case_name"], body_site=None, notes=None, created_at="")
            folder = Folder(id=d["folder_id"], name=d["folder_name"], color=d["color"], created_at=d["f_created"])
            out.append((scan, case, folder))
        return out


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
