"""The camera owner, with a fake sensor standing in for the Pi Camera Module."""

from __future__ import annotations

import numpy as np

from kiosk import camera


class _FakeSensor:
    """Stands in for Picamera2: a grey I420 preview frame and a grey still."""

    def capture_array(self, stream: str):
        if stream == "lores":
            w, h = camera.config.CAMERA_PREVIEW_SIZE
            return np.full((h * 3 // 2, w), 128, np.uint8)
        w, h = camera.config.CAMERA_STILL_SIZE
        return np.full((h, w, 3), 128, np.uint8)


def test_only_the_newest_live_view_keeps_streaming(monkeypatch) -> None:
    """A browser can leave old streams open; enough of them froze every tap on the kiosk."""
    monkeypatch.setattr(camera.camera, "_cam", _FakeSensor())
    monkeypatch.setattr(camera.config, "CAMERA_PREVIEW_FPS", 1000)
    old, new = camera.camera.preview_frames(), camera.camera.preview_frames()
    assert next(old)
    assert next(new)  # opening a second live view...
    assert next(old, None) is None  # ...ends the first one
    assert next(new)  # while the newest keeps going
    camera.camera.capture_jpeg()  # a still ends the live view behind the photo too
    assert next(new, None) is None
