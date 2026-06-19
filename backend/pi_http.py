"""HTTP client for Raspberry Pi Flask ``pi_server`` (capture + classify on device)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import requests

from backend.contracts import ScanResult

logger = logging.getLogger("dermascan.pi")


class PiHttpBackend:
    """POST ``/scan`` on the Pi; optional ``/health`` and ``/log``."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.backend_id = "pi"

    def health(self) -> dict:
        url = f"{self.base_url}/health"
        logger.debug("GET %s", url)
        try:
            r = requests.get(url, timeout=min(5, self.timeout))
        except requests.exceptions.ConnectionError:
            reason = f"Cannot reach Pi at {self.base_url}. Is pi_server.py running?"
            logger.error("GET /health failed: %s", reason)
            return {"status": "error", "reason": reason}
        except requests.exceptions.Timeout:
            reason = "Pi health check timed out."
            logger.error("GET /health failed: %s", reason)
            return {"status": "error", "reason": reason}
        if r.status_code != 200:
            reason = f"HTTP {r.status_code}: {(r.text or '')[:200]}"
            logger.error("GET /health failed: %s", reason)
            return {"status": "error", "reason": reason}
        try:
            payload = dict(r.json())
        except json.JSONDecodeError:
            reason = "Invalid JSON from /health"
            logger.error("GET /health failed: %s", reason)
            return {"status": "error", "reason": reason}
        if payload.get("status") == "error":
            logger.error("GET /health failed: %s", payload.get("reason", "unknown"))
        else:
            logger.info(
                "GET /health ok model=%s labels=%s",
                payload.get("model", "?"),
                payload.get("labels", "?"),
            )
        return payload

    def scan(self, image_jpg_bytes: bytes | None = None) -> ScanResult:
        """POST ``/scan``. If ``image_jpg_bytes`` is set, sends multipart ``image`` (Pi debug path)."""
        url = f"{self.base_url}/scan"
        if image_jpg_bytes:
            logger.info("POST %s (multipart upload, %d bytes)", url, len(image_jpg_bytes))
        else:
            logger.info("POST %s (Pi camera)", url)
        try:
            if image_jpg_bytes:
                r = requests.post(
                    url,
                    files={"image": ("probe.jpg", image_jpg_bytes, "image/jpeg")},
                    timeout=self.timeout,
                )
            else:
                r = requests.post(url, timeout=self.timeout)
            r.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            msg = f"Can't reach Pi at {self.base_url}. Is pi_server.py running?"
            logger.error("POST /scan connection error: %s", msg)
            raise RuntimeError(msg) from exc
        except requests.exceptions.Timeout as exc:
            msg = "Pi took too long to respond. Check it is not overloaded or retry."
            logger.error("POST /scan timeout: %s", msg)
            raise RuntimeError(msg) from exc
        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            code = resp.status_code if resp is not None else "?"
            reason = ""
            if resp is not None and resp.text:
                try:
                    payload = resp.json()
                    if isinstance(payload, dict) and payload.get("status") == "error":
                        reason = str(payload.get("reason", "")).strip()
                except json.JSONDecodeError:
                    reason = (resp.text or "")[:200]
            if not reason:
                reason = (resp.text if resp is not None else "")[:200]
            msg = f"Pi returned HTTP {code}: {reason}"
            logger.error("POST /scan HTTP %s: %s", code, reason or "(no body)")
            raise RuntimeError(msg) from exc
        data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            reason = str(data.get("reason", "Pi scan failed"))
            logger.error("POST /scan error response: %s", reason)
            raise RuntimeError(reason)
        probs_raw = data.get("probs") or {}
        probs = {str(k): float(v) for k, v in probs_raw.items()}
        b64 = data.get("image") or ""
        try:
            # validate=True rejects malformed payloads instead of silently storing garbage.
            image_jpg_bytes_out = base64.b64decode(b64, validate=True) if b64 else b""
        except (ValueError, TypeError) as exc:
            logger.warning("POST /scan returned a malformed image payload: %s", exc)
            image_jpg_bytes_out = b""
        label = str(data.get("label", ""))
        confidence = float(data.get("confidence", 0.0))
        inference_ms = int(data.get("inference_ms", 0))
        if not label:
            logger.warning("POST /scan success but label is empty")
        if not image_jpg_bytes_out:
            logger.warning("POST /scan success but image bytes are empty")
        logger.info(
            "POST /scan ok label=%s confidence=%.1f inference_ms=%d image_bytes=%d",
            label or "(empty)",
            confidence,
            inference_ms,
            len(image_jpg_bytes_out),
        )
        return ScanResult(
            label=label,
            confidence=confidence,
            probs=probs,
            image_jpg_bytes=image_jpg_bytes_out,
            timestamp=str(data.get("timestamp", "")),
            inference_ms=inference_ms,
            urgency=str(data.get("urgency", "")),
            icon=str(data.get("icon", "")),
            action=str(data.get("action", "")),
            backend_id=self.backend_id,
        )

    def fetch_log(self) -> tuple[list[dict[str, Any]], str | None]:
        url = f"{self.base_url}/log"
        logger.debug("GET %s", url)
        try:
            r = requests.get(url, timeout=min(10, self.timeout))
        except requests.exceptions.ConnectionError:
            warning = f"Cannot reach Pi at {self.base_url} for log export."
            logger.warning("GET /log failed: %s", warning)
            return [], warning
        except requests.exceptions.Timeout:
            warning = "Pi log request timed out."
            logger.warning("GET /log failed: %s", warning)
            return [], warning
        if r.status_code == 404:
            warning = "Pi server has no /log endpoint (older pi_server.py)."
            logger.warning("GET /log failed: %s", warning)
            return [], warning
        if r.status_code != 200:
            warning = f"Pi /log returned HTTP {r.status_code}: {(r.text or '')[:200]}"
            logger.warning("GET /log failed: %s", warning)
            return [], warning
        try:
            payload = r.json()
        except json.JSONDecodeError:
            warning = "Invalid JSON from Pi /log."
            logger.warning("GET /log failed: %s", warning)
            return [], warning
        if isinstance(payload, list):
            logger.debug("GET /log ok rows=%d", len(payload))
            return payload, None
        warning = "Unexpected /log JSON shape."
        logger.warning("GET /log failed: %s", warning)
        return [], warning
