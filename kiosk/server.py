"""The kiosk web server. One page, a few routes, and one remembered photo.

    GET  /               the page (static/index.html)
    GET  /health         is the model loaded, is the camera running, does Exit need a code
    GET  /preview.mjpg   live camera stream, when a Pi camera is present
    POST /capture        take a photo; returns the JPEG and remembers it
    POST /scan           scan the uploaded `image` file, or the remembered photo
                         (`force=1` reads a photo the spot check refused)
    POST /forget         drop the remembered photo (the page calls this on "Done")
    POST /quit           ask scripts/launch_kiosk.sh to shut down (`code` if a passcode is set)

Every answer from /capture and /scan that is not a photo has the same shape as a
scan outcome, so the page renders errors with the same code as results, in words
from dermascan/verdict.py.

Run it:  python -m kiosk.server            (Pi, or a laptop with upload/webcam)
         python -m kiosk.server --debug    (auto-reload while editing)
"""

from __future__ import annotations

import argparse
import hmac
import logging
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from dermascan import config
from dermascan.classifier import get_classifier
from dermascan.scan import ScanOutcome, run_scan
from dermascan.verdict import error_verdict
from kiosk import camera as cam

STATIC = Path(__file__).resolve().parent / "static"

log = logging.getLogger("dermascan.server")
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")


class LastPhoto:
    """The one photo the kiosk holds: the last capture or upload, until "Done"."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()

    def put(self, jpeg: bytes) -> bytes:
        with self._lock:
            self._jpeg = jpeg
        return jpeg

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def forget(self) -> None:
        with self._lock:
            self._jpeg = None


last_photo = LastPhoto()


def _error(kind: str, http_status: int):
    return jsonify(ScanOutcome("error", error_verdict(kind)).to_dict()), http_status


@app.get("/")
def index() -> Response:
    return send_from_directory(STATIC, "index.html")


@app.get("/health")
def health():
    status = {"camera": cam.camera.running, "exit_needs_code": config.staff_passcode() is not None}
    try:
        clf = get_classifier()
    except Exception as exc:  # noqa: BLE001
        log.exception("model failed to load")
        return jsonify({**status, "ok": False, "error": f"model: {exc}"}), 500
    return jsonify({**status, "ok": True, "model": {"file": clf.model_path.name, "labels": clf.labels, "tta": clf.use_tta}})


@app.get("/preview.mjpg")
def preview():
    if not cam.camera.running:
        return _error("camera", 404)

    def stream():
        for frame in cam.camera.preview_frames():
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"

    return Response(
        stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},  # never let the browser reuse an old stream
    )


@app.post("/capture")
def capture():
    if not cam.camera.running:
        return _error("camera", 404)
    try:
        jpeg = last_photo.put(cam.camera.capture_jpeg())
    except Exception:  # noqa: BLE001
        log.exception("capture failed")
        return _error("camera", 500)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/scan")
def scan():
    upload = request.files.get("image")
    # This request scans exactly these bytes, even if another request replaces the
    # remembered photo while it runs.
    jpeg = last_photo.put(upload.read()) if upload is not None else last_photo.get()
    if not jpeg:
        return _error("no_photo", 400)
    force = request.form.get("force", request.args.get("force", "0")) == "1"
    return jsonify(run_scan(jpeg, force=force).to_dict())


@app.post("/forget")
def forget():
    last_photo.forget()
    return jsonify({"ok": True})


@app.post("/quit")
def quit_kiosk():
    expected = config.staff_passcode()
    if expected is not None and not hmac.compare_digest(request.form.get("code", ""), expected):
        return jsonify({"ok": False}), 403
    try:
        config.QUIT_FLAG.write_text("quit")
    except OSError:
        log.warning("could not write the quit flag", exc_info=True)
    return jsonify({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="DermaScan kiosk server")
    parser.add_argument("--port", type=int, default=config.KIOSK_PORT)
    parser.add_argument("--debug", action="store_true", help="auto-reload on edits (laptop only)")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens it to the network")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    get_classifier()  # load the model before the first visitor, not during their scan
    cam.camera.start()
    log.info(
        "model %s, camera %s, exit %s, http://%s:%d",
        config.MODEL_PATH.name,
        "on" if cam.camera.running else "off",
        "needs the staff code" if config.staff_passcode() else "is two taps",
        args.host,
        args.port,
    )
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=args.debug)
    finally:
        cam.camera.stop()


if __name__ == "__main__":
    main()
