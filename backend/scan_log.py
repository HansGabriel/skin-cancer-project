"""Append-only SQLite log for local Streamlit backends (mock / local)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / "logs" / "scans.sqlite"


def init_db(db_path: Path | None = None) -> None:
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_iso TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                probs_json TEXT NOT NULL,
                inference_ms INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def insert_scan(
    *,
    backend_id: str,
    label: str,
    confidence: float,
    probs: dict[str, float],
    inference_ms: int,
    ts_iso: str,
    db_path: Path | None = None,
) -> None:
    path = db_path or default_db_path()
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO scans (ts_iso, backend_id, label, confidence, probs_json, inference_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts_iso, backend_id, label, confidence, json.dumps(probs), inference_ms),
        )
        conn.commit()


def fetch_recent(limit: int = 200, db_path: Path | None = None) -> list[dict[str, Any]]:
    path = db_path or default_db_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT ts_iso, backend_id, label, confidence, probs_json, inference_ms "
            "FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for row in cur.fetchall():
            rows.append(
                {
                    "ts_iso": row["ts_iso"],
                    "backend_id": row["backend_id"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "probs": json.loads(row["probs_json"]),
                    "inference_ms": row["inference_ms"],
                }
            )
        return rows
    finally:
        conn.close()
