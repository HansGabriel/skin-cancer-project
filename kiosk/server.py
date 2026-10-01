"""The kiosk web server. One page, five routes, no state beyond the last photo.

    GET  /               the page (static/index.html)
    GET  /preview.mjpg   live camera stream, when a Pi camera is present
    POST /capture        take a photo; returns the JPEG and remembers it as "last"
    POST /scan           scan the uploaded `image` file, or the last capture
                         (`force=1` reads a photo the spot check refused)
    GET  /health         is the model loaded, is the camera running
    POST /quit           ask launch_kiosk.sh to shut the kiosk down

Run it:  python -m kiosk.server            (Pi, or laptop with upload/webcam)
         python -m kiosk.server --debug    (auto-reload while editing)
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from dermascan import config
from dermascan.classifier import get_classifier
from dermascan.scan import run_scan
from kiosk import camera as cam

STATIC = Path(__file__).resolve().parent / "static"
QUIT_FLAG = Path("/tmp/dermascan_quit")
PORT = 8080

log = logging.getLogger("dermascan.server")
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

_last_capture: bytes | None = None
_scan_lock = threading.Lock()  # the TFLite interpreter is not thread-safe


@app.get("/")
def index() -> Response:
    return send_from_directory(STATIC, "index.html")


@app.get("/health")
def health():
    try:
        clf = get_classifier()
        model = {"file": clf.model_path.name, "labels": clf.labels, "tta": clf.use_tta}
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"model: {exc}", "camera": cam.camera.running}), 500
    return jsonify({"ok": True, "model": model, "camera": cam.camera.running})


@app.get("/preview.mjpg")
def preview():
    if not cam.camera.running:
        return jsonify({"error": "no camera"}), 404

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
    global _last_capture
    if not cam.camera.running:
        return jsonify({"error": "no camera"}), 404
    try:
        _last_capture = cam.camera.capture_jpeg()
    except Exception as exc:  # noqa: BLE001
        log.exception("capture failed")
        return jsonify({"error": f"The camera did not take the photo ({exc})."}), 500
    return Response(_last_capture, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/scan")
def scan():
    global _last_capture
    upload = request.files.get("image")
    if upload is not None:
        _last_capture = upload.read()
    if not _last_capture:
        return jsonify({"error": "Take a photo first."}), 400
    force = request.form.get("force", request.args.get("force", "0")) == "1"
    with _scan_lock:
        outcome = run_scan(_last_capture, force=force)
    return jsonify(outcome.to_dict())


@app.post("/quit")
def quit_kiosk():
    try:
        QUIT_FLAG.write_text("quit")
    except OSError:
        pass
    return jsonify({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="DermaScan kiosk server")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--debug", action="store_true", help="auto-reload on edits (laptop only)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open on the LAN")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    get_classifier()  # load the model before the first visitor, not during their scan
    cam.camera.start()
    log.info("model %s, camera %s, http://%s:%d", config.MODEL_PATH.name, "on" if cam.camera.running else "off", args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=args.debug)
    finally:
        cam.camera.stop()


if __name__ == "__main__":
    main()
