"""Shared Pi Camera Module (picamera2) owner: live MJPEG preview + still capture.

Only one process can own the IMX219 sensor at a time, so this module keeps a
single Picamera2 instance open for the lifetime of the Streamlit process. It
serves an MJPEG preview over a tiny background HTTP server (so the browser can
embed a live <img> feed) and grabs full-resolution stills on demand from the
same camera — no open/close churn that caused 'Camera in Configured state'
acquire errors.

Import-guarded: on machines without picamera2 (Mac/PC), AVAILABLE is False and
the camera_view falls back to the browser webcam / upload flow.
"""

from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_log = logging.getLogger("dermascan.pi_camera")

try:
    import cv2
    from picamera2 import Picamera2
    from libcamera import controls as _libcontrols

    AVAILABLE = True
except ImportError:  # pragma: no cover - Mac/PC path
    AVAILABLE = False

# Preview stream runs on localhost only — embedded by Streamlit running on the same Pi.
PREVIEW_HOST = "127.0.0.1"
PREVIEW_PORT = 8555
PREVIEW_PATH = "/preview.mjpg"

# Capture geometry. Sensor still is large; we center-crop to the sharpest region
# (fixed-focus IMX219 is sharpest in the middle) then resize for the model.
_STILL_SIZE = (1640, 1232)  # native-ish mode, good detail
_CROP_SIZE = 1024  # center square pulled from the still and sent on for analysis
_PREVIEW_SIZE = (640, 480)  # lores stream for the live feed

# Number of frames to discard so auto-exposure / auto-white-balance settle.
# The camera runs continuously (the live preview proves AE/AWB are already settled),
# so only a couple of fresh frames are needed — 8 added ~1-2s of dead time per shot.
_WARMUP_FRAMES = 2


class _PiCamera:
    """Singleton wrapper holding the one Picamera2 instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cam: Picamera2 | None = None
        self._server: ThreadingHTTPServer | None = None
        self._latest_jpeg: bytes | None = None
        self._stream_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Open the camera and start preview + MJPEG server. Idempotent."""
        with self._lock:
            if self._cam is not None:
                return
            cam = Picamera2()
            config = cam.create_still_configuration(
                main={"size": _STILL_SIZE, "format": "RGB888"},
                lores={"size": _PREVIEW_SIZE, "format": "YUV420"},
                display="lores",
            )
            cam.configure(config)
            # Continuous auto-exposure + auto-white-balance for a bright, settled image.
            cam.set_controls(
                {
                    "AeEnable": True,
                    "AwbEnable": True,
                    "AeExposureMode": _libcontrols.AeExposureModeEnum.Normal,
                    # Slight positive brightness bias; lesions photographed close are often dark.
                    "Brightness": 0.1,
                }
            )
            cam.start()
            self._cam = cam
            self._stop.clear()
            self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._stream_thread.start()
            self._start_server()

    def _start_server(self) -> None:
        if self._server is not None:
            return
        cam_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence access logs
                return

            def do_GET(self):  # noqa: N802
                if self.path != PREVIEW_PATH:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while not cam_ref._stop.is_set():
                        frame = cam_ref._latest_jpeg
                        if frame is None:
                            time.sleep(0.03)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)  # ~20 fps cap
                except (BrokenPipeError, ConnectionResetError):
                    return

        self._server = ThreadingHTTPServer((PREVIEW_HOST, PREVIEW_PORT), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _stream_loop(self) -> None:
        """Continuously refresh the latest preview JPEG from the lores stream."""
        assert self._cam is not None
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                # Capture the lores (preview) stream. YUV420 planar -> RGB; fall back to
                # treating it as already-RGB if the channel layout doesn't match.
                frame = self._cam.capture_array("lores")
                if frame.ndim == 2:  # planar YUV420 comes back as a tall single-channel array
                    rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
                elif frame.shape[2] == 3:
                    rgb = frame
                else:  # 4-channel XBGR
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                rgb = cv2.resize(rgb, _PREVIEW_SIZE)
                # Draw a center guide box matching the capture crop region.
                self._draw_guide(rgb)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._latest_jpeg = buf.tobytes()
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 — preview must not crash the thread
                consecutive_failures += 1
                # Log the first failure and then only periodically, so a persistently
                # broken camera surfaces in the terminal without spamming it.
                if consecutive_failures == 1 or consecutive_failures % 50 == 0:
                    _log.warning(
                        "Preview capture failed (%d in a row): %s",
                        consecutive_failures,
                        exc,
                    )
                time.sleep(0.1)
            time.sleep(0.04)

    @staticmethod
    def _draw_guide(rgb) -> None:
        """Overlay a center square showing the region that will be sent for analysis.

        The side must match ``_center_crop`` exactly. It used to be drawn at
        0.7x while the capture took the full centre square — so the box claimed
        an area barely half of what was actually kept. People framed the lesion
        to fill the box and got a photo with twice the field of view, which made
        the lesion smaller on the result screen *and* smaller inside the 224px
        model tensor than they intended. Two constants describing one crop is
        how that drifts, so this reads the crop geometry rather than repeating it.
        """
        h, w = rgb.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        # Inset by a pixel so a full-bleed rectangle is still visible on-screen.
        cv2.rectangle(rgb, (x0 + 1, y0 + 1), (x0 + side - 2, y0 + side - 2), (80, 200, 120), 2)

    # -- still capture -----------------------------------------------------
    def capture_still_jpeg(self) -> bytes | None:
        """Grab a settled, center-cropped still and return JPEG bytes."""
        with self._lock:
            if self._cam is None:
                return None
            # Warmup: discard a few frames so AE/AWB have fully settled at capture time.
            for _ in range(_WARMUP_FRAMES):
                self._cam.capture_array("main")
            frame = self._cam.capture_array("main")

        # picamera2's "RGB888" format actually delivers BGR byte order, which is
        # exactly what cv2.imencode expects — do NOT swap channels here or the
        # red/blue channels invert and the image turns blue.
        bgr = self._center_crop(frame, _CROP_SIZE)
        # Mild sharpening + auto contrast helps the fixed-focus, sometimes-soft IMX219.
        bgr = self._enhance(bgr)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None
        return buf.tobytes()

    @staticmethod
    def _center_crop(rgb, target: int):
        h, w = rgb.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        square = rgb[y0 : y0 + side, x0 : x0 + side]
        if side != target:
            square = cv2.resize(square, (target, target))
        return square

    @staticmethod
    def _enhance(bgr):
        # CLAHE on the L channel: lifts dark, low-contrast lesion shots without blowing highlights.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        # Light unsharp mask for crispness.
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        return cv2.addWeighted(out, 1.4, blur, -0.4, 0)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._server is not None:
                self._server.shutdown()
                self._server = None
            if self._cam is not None:
                try:
                    self._cam.stop()
                    self._cam.close()
                except Exception as exc:  # noqa: BLE001 — shutdown is best-effort
                    _log.warning("Error closing camera: %s", exc)
                self._cam = None


_INSTANCE: _PiCamera | None = None


def get_camera() -> _PiCamera | None:
    """Return the shared camera, starting it on first use. None if picamera2 absent."""
    global _INSTANCE
    if not AVAILABLE:
        return None
    if _INSTANCE is None:
        _INSTANCE = _PiCamera()
    _INSTANCE.start()
    return _INSTANCE


def preview_url() -> str:
    return f"http://{PREVIEW_HOST}:{PREVIEW_PORT}{PREVIEW_PATH}"
