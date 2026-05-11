"""HTTP client for Raspberry Pi Flask ``pi_server`` (capture + classify on device)."""

from __future__ import annotations

import base64
import json
from typing import Any

import requests

from backend.contracts import ScanResult


class PiHttpBackend:
    """POST ``/scan`` on the Pi; optional ``/health`` and ``/log``."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.backend_id = "pi"

    def health(self) -> dict:
        url = f"{self.base_url}/health"
        try:
            r = requests.get(url, timeout=min(5, self.timeout))
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "reason": f"Cannot reach Pi at {self.base_url}. Is pi_server.py running?",
            }
        except requests.exceptions.Timeout:
            return {"status": "error", "reason": "Pi health check timed out."}
        if r.status_code != 200:
            return {
                "status": "error",
                "reason": f"HTTP {r.status_code}: {(r.text or '')[:200]}",
            }
        try:
            return dict(r.json())
        except json.JSONDecodeError:
            return {"status": "error", "reason": "Invalid JSON from /health"}

    def scan(self, image_jpg_bytes: bytes | None = None) -> ScanResult:
        del image_jpg_bytes  # capture happens on the Pi
        url = f"{self.base_url}/scan"
        try:
            r = requests.post(url, timeout=self.timeout)
            r.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Can't reach Pi at {self.base_url}. Is pi_server.py running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Pi took too long to respond. Check it is not overloaded or retry."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            body = (resp.text if resp is not None else "")[:200]
            code = resp.status_code if resp is not None else "?"
            raise RuntimeError(f"Pi returned HTTP {code}: {body}") from exc
        data = r.json()
        probs_raw = data.get("probs") or {}
        probs = {str(k): float(v) for k, v in probs_raw.items()}
        b64 = data.get("image") or ""
        try:
            image_jpg_bytes_out = base64.b64decode(b64, validate=False) if b64 else b""
        except (ValueError, TypeError):
            image_jpg_bytes_out = b""
        return ScanResult(
            label=str(data.get("label", "")),
            confidence=float(data.get("confidence", 0.0)),
            probs=probs,
            image_jpg_bytes=image_jpg_bytes_out,
            timestamp=str(data.get("timestamp", "")),
            inference_ms=int(data.get("inference_ms", 0)),
            urgency=str(data.get("urgency", "")),
            icon=str(data.get("icon", "")),
            action=str(data.get("action", "")),
            backend_id=self.backend_id,
        )

    def fetch_log(self) -> tuple[list[dict[str, Any]], str | None]:
        url = f"{self.base_url}/log"
        try:
            r = requests.get(url, timeout=min(10, self.timeout))
        except requests.exceptions.ConnectionError:
            return [], f"Cannot reach Pi at {self.base_url} for log export."
        except requests.exceptions.Timeout:
            return [], "Pi log request timed out."
        if r.status_code == 404:
            return [], "Pi server has no /log endpoint (older pi_server.py)."
        if r.status_code != 200:
            return [], f"Pi /log returned HTTP {r.status_code}: {(r.text or '')[:200]}"
        try:
            payload = r.json()
        except json.JSONDecodeError:
            return [], "Invalid JSON from Pi /log."
        if isinstance(payload, list):
            return payload, None
        return [], "Unexpected /log JSON shape."
