"""The one owner of the Pi Camera Module.

Only one process can hold the IMX219 sensor, and opening it takes about a
second, so this module opens it ONCE when the server starts and keeps it open
until the process exits. Nothing ever stops and restarts it between screens.
(An earlier version tore the camera down on every screen change; the preview
stalled, the browser kept a dead stream, and the port stayed bound.)

Two things come out of it:

    preview_frames()   a generator of small JPEGs for the live view (~15 fps)
    capture_jpeg()     one full-size centre-cropped still for the scan

The still is NOT enhanced. No CLAHE, no sharpening: the model was trained on
plain photographs, and anything done to the pixels here is something the
model never saw.

picamera2's "RGB888" format actually delivers BGR byte order. Both outputs
here are produced with cv2, which expects BGR, so no channel swap is needed.
On a machine without picamera2 (a laptop) `AVAILABLE` is False and the web
page falls back to the browser's own camera or a file upload.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

import cv2

log = logging.getLogger("dermascan.camera")

try:
    from picamera2 import Picamera2  # type: ignore[import-not-found]

    AVAILABLE = True
except ImportError:  # laptop
    AVAILABLE = False

STILL_SIZE = (1640, 1232)  # full field of view, 2x2 binned: sharp and fast on the IMX219
PREVIEW_SIZE = (640, 480)
CROP_PX = 1024  # centre square sent for analysis; the preview draws the same square
PREVIEW_FPS = 15


class Camera:
    def __init__(self) -> None:
        self._cam = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._cam is not None or not AVAILABLE:
            return
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": STILL_SIZE, "format": "RGB888"},
            lores={"size": PREVIEW_SIZE, "format": "YUV420"},
            buffer_count=4,
        )
        cam.configure(config)
        # Keep the sensor at ~30 fps even on a dark subject, so the preview never
        # drops to one frame a second while auto-exposure hunts.
        cam.set_controls({"AeEnable": True, "AwbEnable": True, "FrameDurationLimits": (33333, 33333)})
        cam.start()
        time.sleep(0.5)  # let exposure and white balance settle once, at boot
        self._cam = cam
        log.info("camera started")

    def stop(self) -> None:
        with self._lock:
            if self._cam is not None:
                self._cam.stop()
                self._cam.close()
                self._cam = None

    @property
    def running(self) -> bool:
        return self._cam is not None

    def preview_frames(self) -> Iterator[bytes]:
        """JPEG after JPEG, for as long as the browser keeps the connection open."""
        period = 1.0 / PREVIEW_FPS
        while self._cam is not None:
            t = time.perf_counter()
            try:
                with self._lock:
                    yuv = self._cam.capture_array("lores")
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                _draw_guide(bgr)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield buf.tobytes()
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the stream
                log.warning("preview frame failed: %s", exc)
                time.sleep(0.2)
            time.sleep(max(0.0, period - (time.perf_counter() - t)))

    def capture_jpeg(self) -> bytes:
        """One still: centre square, 1024 px, plain pixels, JPEG quality 92."""
        if self._cam is None:
            raise RuntimeError("The camera is not running.")
        with self._lock:
            frame = self._cam.capture_array("main")  # BGR, see module docstring
        square = _centre_square(frame, CROP_PX)
        ok, buf = cv2.imencode(".jpg", square, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Could not encode the photo.")
        return buf.tobytes()


def _centre_square(frame, target: int):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = frame[y0 : y0 + side, x0 : x0 + side]
    return square if side == target else cv2.resize(square, (target, target), interpolation=cv2.INTER_AREA)


def _draw_guide(bgr) -> None:
    """A thin square showing exactly what capture_jpeg will keep."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cv2.rectangle(bgr, (x0 + 1, y0 + 1), (x0 + side - 2, y0 + side - 2), (120, 200, 80), 2)


camera = Camera()
