"""The kiosk web server. One page, a few routes, one remembered photo, and the event's saved scans.

    GET  /               the page (static/index.html)
    GET  /health         model, camera, staff settings, saved-scan count
    GET  /preview.mjpg   live camera stream, when a Pi camera is present
    POST /capture        take a photo; returns the JPEG and remembers it
    POST /check          the photo check only (no model): light, focus, spot
    POST /scan           scan the uploaded `image` file, or the remembered photo
                         (`force=1` reads a photo the spot check refused)
    POST /forget         drop the remembered photo (the page calls this on "Done")

    GET  /questions      questions to tap (`state` = the verdict being asked about)
    POST /ask            {question, state, sign_lines} -> a written answer

    GET  /saved          saved spots, grouped by body site, plus the site list
    POST /saved          save the last scan: `site`, and `group` to add it to an existing
                         group's history (omitted: start a new group)
    GET  /saved/<id>     one saved scan's outcome;  /saved/<id>.jpg its photo
    POST /saved/<id>/delete   drop one saved scan (no staff code: it is the visitor's own photo)

    POST /staff          check the staff code (`code`); every staff route below needs it
    POST /settings       change ALLOW_READ_ANYWAY / SHOW_STAFF_DETAILS until restart
    POST /erase          "End event - erase all": every saved scan, now
    POST /quit           ask scripts/launch_kiosk.sh to shut down

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
from dermascan.answers import BAND_FOR_STATE, get_bank
from dermascan.classifier import get_classifier
from dermascan.scan import ScanOutcome, run_check, run_scan
from dermascan.verdict import EVOLVING_COMPARE, SIGN_LINE_TEXTS, error_verdict
from kiosk import camera as cam
from kiosk.saved import SITES, saved

STATIC = Path(__file__).resolve().parent / "static"

log = logging.getLogger("dermascan.server")
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")


class LastPhoto:
    """The one photo the kiosk holds: the last capture or upload, until "Done".

    Also the outcome of the last scan of exactly that photo, so "Save this scan"
    saves what the visitor saw, never a newer photo's result.
    """

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._outcome: dict | None = None
        self._lock = threading.Lock()

    def put(self, jpeg: bytes) -> bytes:
        with self._lock:
            self._jpeg, self._outcome = jpeg, None
        return jpeg

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def scanned(self, jpeg: bytes, outcome: dict) -> None:
        with self._lock:
            if self._jpeg is jpeg:  # still the same photo
                self._outcome = outcome

    def last_scan(self) -> tuple[bytes, dict] | None:
        with self._lock:
            return (self._jpeg, self._outcome) if self._jpeg and self._outcome else None

    def forget(self) -> None:
        with self._lock:
            self._jpeg, self._outcome = None, None


last_photo = LastPhoto()

# What staff changed in Settings. Back to config.py's values on every restart.
settings = {"allow_read_anyway": config.ALLOW_READ_ANYWAY, "show_staff_details": config.SHOW_STAFF_DETAILS}


def _error(kind: str, http_status: int):
    return jsonify(ScanOutcome("error", error_verdict(kind)).to_dict()), http_status


def _staff_ok() -> bool:
    """True when no staff code is set, or the request carries the right one."""
    expected = config.staff_passcode()
    if expected is None:
        return True
    sent = request.form.get("code") or (request.get_json(silent=True) or {}).get("code") or ""
    return hmac.compare_digest(str(sent), expected)


def _apply_read_anyway_setting(result: dict) -> dict:
    """Staff switched "read anyway" off: no refusal can be overridden from the page."""
    if result.get("refusal") and not settings["allow_read_anyway"]:
        result["refusal"]["can_override"] = False
    return result


def _photo_from_request() -> bytes | None:
    upload = request.files.get("image")
    return last_photo.put(upload.read()) if upload is not None else last_photo.get()


@app.get("/")
def index() -> Response:
    return send_from_directory(STATIC, "index.html")


@app.get("/health")
def health():
    status = {
        "camera": cam.camera.running,
        "exit_needs_code": config.staff_passcode() is not None,
        "settings": settings,
        "saved": saved.stats(),
    }
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


@app.post("/check")
def check():
    jpeg = _photo_from_request()
    if not jpeg:
        return _error("no_photo", 400)
    return jsonify(_apply_read_anyway_setting(run_check(jpeg)))


@app.post("/scan")
def scan():
    # This request scans exactly these bytes, even if another request replaces the
    # remembered photo while it runs.
    jpeg = _photo_from_request()
    if not jpeg:
        return _error("no_photo", 400)
    force = request.form.get("force", request.args.get("force", "0")) == "1"
    outcome = _apply_read_anyway_setting(run_scan(jpeg, force=force and settings["allow_read_anyway"]).to_dict())
    last_photo.scanned(jpeg, outcome)
    return jsonify(outcome)


@app.post("/forget")
def forget():
    last_photo.forget()
    return jsonify({"ok": True})


# --- Questions ------------------------------------------------------------------


@app.get("/questions")
def questions():
    band = BAND_FOR_STATE.get(request.args.get("state", ""), "general")
    return jsonify({"suggestions": get_bank().suggestions(band)})


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()[: config.QUESTION_MAX_CHARS]
    if not question:
        return jsonify({"ok": False}), 400
    # Only lines verdict.py itself writes: the answer carries the project's badge,
    # so the page must not be able to append text of its own to it.
    lines = [x for x in (body.get("sign_lines") or []) if x in SIGN_LINE_TEXTS]
    return jsonify(get_bank().ask(question, body.get("state"), lines).to_dict())


# --- Saved scans ----------------------------------------------------------------


@app.get("/saved")
def saved_list():
    return jsonify({"spots": saved.spots(), "sites": SITES, "stats": saved.stats()})


@app.post("/saved")
def saved_add():
    last = last_photo.last_scan()
    if last is None:
        return _error("no_photo", 400)
    jpeg, outcome = last
    if outcome.get("status") != "ok":
        return jsonify({"ok": False}), 400  # only a read result is worth keeping
    try:
        scan = saved.add(request.form.get("site", ""), jpeg, outcome, group=request.form.get("group") or None)
    except ValueError:
        return jsonify({"ok": False}), 400
    return jsonify({"ok": True, "scan": scan.summary(), "history": _history(scan.id)})


def _history(scan_id: int) -> dict | None:
    h = saved.history(scan_id)
    return {**h, "compare_advice": EVOLVING_COMPARE} if h else None


@app.get("/saved/<int:scan_id>")
def saved_one(scan_id: int):
    scan = saved.get(scan_id)
    if scan is None:
        return jsonify({"ok": False}), 404
    return jsonify({"ok": True, "scan": scan.summary(), "outcome": scan.outcome, "history": _history(scan_id)})


@app.get("/saved/<int:scan_id>.jpg")
def saved_photo(scan_id: int):
    scan = saved.get(scan_id)
    if scan is None:
        return Response(status=404)
    return Response(scan.jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/saved/<int:scan_id>/delete")
def saved_delete(scan_id: int):
    if not saved.delete(scan_id):
        return jsonify({"ok": False}), 404
    log.info("saved scan %d deleted", scan_id)
    return jsonify({"ok": True, "saved": saved.stats()})


# --- Staff ----------------------------------------------------------------------


@app.post("/staff")
def staff_unlock():
    return (jsonify({"ok": True}), 200) if _staff_ok() else (jsonify({"ok": False}), 403)


@app.post("/settings")
def change_settings():
    if not _staff_ok():
        return jsonify({"ok": False}), 403
    for key in settings:
        if key in request.form:
            settings[key] = request.form[key] == "1"
    return jsonify({"ok": True, "settings": settings})


@app.post("/erase")
def erase():
    if not _staff_ok():
        return jsonify({"ok": False}), 403
    saved.erase()
    last_photo.forget()
    log.info("saved scans erased by staff")
    return jsonify({"ok": True, "saved": saved.stats()})


@app.post("/quit")
def quit_kiosk():
    if not _staff_ok():
        return jsonify({"ok": False}), 403
    saved.erase()
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
    code = config.staff_passcode()
    if code is not None and not (len(code) == 4 and code.isdigit()):
        raise SystemExit("The staff code must be exactly 4 digits (the on-screen keypad sends 4). Fix ~/.dermascan_passcode.")
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
