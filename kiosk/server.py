"""The kiosk web server. One page, a few routes, one remembered photo, and the event's saved scans.

    GET  /               the page (static/index.html)
    GET  /health         model, camera, staff settings, saved-scan count (web mode: model and settings only)
    GET  /preview.mjpg   live camera stream, when a Pi camera is present
    POST /capture        take a photo; returns the JPEG and remembers it
    POST /check          the photo check only (no model): light, focus, spot
    POST /scan           scan the uploaded `image` file, or the remembered photo
                         (`force=1` reads a photo the spot check refused)
    POST /forget         drop the remembered photo (the page calls this on "Done")
    GET  /cam.jpg        "Where it looked" over the last checked or scanned photo (web: POST the photo)

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
    POST /update         ask the launcher to run scripts/update.sh;  POST /update/log  its log's tail

Every answer from /capture and /scan that is not a photo has the same shape as a
scan outcome, so the page renders errors with the same code as results, in words
from dermascan/verdict.py.

Web mode (`--web`, the public demo on Railway): the same page for many strangers at
once, so the server holds nothing between requests. Every /check and /scan carries its
own photo, nothing is remembered or saved, and the saved-scan and staff routes are
switched off (404). Staff settings stay at config.py's values.

Run it:  python -m kiosk.server            (Pi, or a laptop with upload/webcam)
         python -m kiosk.server --debug    (auto-reload while editing)
         python -m kiosk.server --web --host 0.0.0.0 --port 8080   (the web demo)
"""

from __future__ import annotations

import argparse
import hmac
import logging
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from dermascan import config, gate
from dermascan.answers import BAND_FOR_STATE, get_bank
from dermascan.cam import alignment, get_cam, warm_in_background
from dermascan.classifier import get_classifier
from dermascan.scan import CheckedPhoto, ScanOutcome, check_photo, check_result, decode_jpeg, run_scan
from dermascan.verdict import ERROR_ADVICE_WEB, EVOLVING_COMPARE, SIGN_LINE_TEXTS, error_verdict
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
        self._checked: CheckedPhoto | None = None  # the "check the photo" result, so /scan does not redo it
        self._outcome: dict | None = None
        self._cam: tuple[bytes, dict | None] | None = None  # "Where it looked" (JPEG, alignment), built once on request
        self._lock = threading.Lock()

    def put(self, jpeg: bytes) -> bytes:
        with self._lock:
            self._jpeg, self._checked, self._outcome, self._cam = jpeg, None, None, None
        return jpeg

    def cam_cached(self, jpeg: bytes) -> tuple[bytes, dict | None] | None:
        with self._lock:
            return self._cam if self._jpeg is jpeg else None

    def cam_built(self, jpeg: bytes, heat: bytes, align: dict | None) -> None:
        with self._lock:
            if self._jpeg is jpeg:
                self._cam = (heat, align)

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def remember_check(self, jpeg: bytes, checked: CheckedPhoto) -> None:
        with self._lock:
            if self._jpeg is jpeg:  # still the same photo
                self._checked = checked

    def last_check(self, jpeg: bytes) -> CheckedPhoto | None:
        with self._lock:
            return self._checked if self._jpeg is jpeg else None

    def scanned(self, jpeg: bytes, outcome: dict) -> None:
        with self._lock:
            if self._jpeg is jpeg:  # still the same photo
                self._outcome = outcome

    def last_scan(self) -> tuple[bytes, dict] | None:
        with self._lock:
            return (self._jpeg, self._outcome) if self._jpeg and self._outcome else None

    def last_checked_or_scanned(self) -> bytes | None:
        """The photo the visitor is looking at: checked (/check) or scanned (/scan). None before either."""
        with self._lock:
            return self._jpeg if self._jpeg and (self._checked is not None or self._outcome is not None) else None

    def forget(self) -> None:
        with self._lock:
            self._jpeg, self._checked, self._outcome, self._cam = None, None, None, None


last_photo = LastPhoto()

# What staff changed in Settings. Back to config.py's values on every restart.
settings = {"allow_read_anyway": config.ALLOW_READ_ANYWAY, "show_staff_details": config.SHOW_STAFF_DETAILS}

# Set by `--web` (use_web_mode). In web mode only these routes answer; anything else,
# including a route added later, is 404 until it is listed here on purpose.
app.config["WEB_DEMO"] = False
WEB_ROUTES = {"index", "health", "check", "scan", "cam_jpg", "forget", "questions", "ask", "static"}


def use_web_mode() -> None:
    """Switch this server to the public web demo: nothing kept, no camera, no staff routes."""
    if config.SAVE_CAPTURES_DIR is not None:
        raise SystemExit("--web with SAVE_CAPTURES_DIR set would keep strangers' photos. Set it to None in config.py.")
    app.config["WEB_DEMO"] = True
    app.config["MAX_CONTENT_LENGTH"] = config.WEB_MAX_UPLOAD_MB * 1024 * 1024


@app.before_request
def _web_demo_guard():
    if app.config["WEB_DEMO"] and request.endpoint not in WEB_ROUTES:
        return jsonify({"ok": False}), 404
    return None


def _error(kind: str, http_status: int):
    return jsonify(_web_words(ScanOutcome("error", error_verdict(kind)).to_dict())), http_status


def _web_words(result: dict) -> dict:
    """On the web there is no staff member to ask, so an error carries the web demo's advice."""
    if app.config["WEB_DEMO"] and result.get("status") == "error" and result.get("verdict"):
        result["verdict"]["advice"] = ERROR_ADVICE_WEB
    return result


@app.errorhandler(413)
def _too_big(_exc):  # web demo: an upload over WEB_MAX_UPLOAD_MB
    return _error("too_big", 413)


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


def _read_line(path) -> str | None:
    """First line of a small text file the launcher or updater wrote, or None."""
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None


def _photo_from_request() -> bytes | None:
    upload = request.files.get("image")
    if app.config["WEB_DEMO"]:  # never remember a stranger's photo; it comes with every request
        return upload.read() if upload is not None else None
    return last_photo.put(upload.read()) if upload is not None else last_photo.get()


@app.get("/")
def index() -> Response:
    return send_from_directory(STATIC, "index.html")


@app.get("/health")
def health():
    status = {"web": True, "camera": False, "settings": settings} if app.config["WEB_DEMO"] else {
        "web": False,
        "camera": cam.camera.running,
        "exit_needs_code": config.staff_passcode() is not None,
        "settings": settings,
        "saved": saved.stats(),
        "update": {"version": _read_line(config.VERSION_FILE), "last": _read_line(config.LAST_UPDATE_FILE)},
    }
    try:
        clf = get_classifier()
    except Exception as exc:  # noqa: BLE001
        log.exception("model failed to load")
        return jsonify({**status, "ok": False, "error": f"model: {exc}"}), 500
    # get_cam() builds the heatmap interpreter on first call; warm_in_background() has usually done it by now.
    heat = get_cam() is not None
    return jsonify({
        **status, "ok": True,
        # cam: the server can build "Where it looked"; cam_on_check: the page shows it on the check screen too.
        "model": {"file": clf.model_path.name, "labels": clf.labels, "tta": clf.use_tta,
                  "cam": heat, "cam_on_check": heat and config.CAM_ON_CHECK},
        # The limit each gate number is held to, so the staff drawer prints value and limit side by side.
        "gate_limits": gate.limits(),
    })


@app.route("/cam.jpg", methods=["GET", "POST"])
def cam_jpg():
    """"Where it looked" (dermascan/cam.py) blended over the photo: the check screen after a
    passing photo, the result, and the staff drawer. On the kiosk, GET gives the last checked
    or scanned photo's heatmap, built once per photo. On the web the server keeps nothing, so
    the page POSTs the photo again and nothing is remembered. 404 when there is none to give.

    Two headers say whether the heat sits on the spot the check outlined (cam.alignment):
    X-Cam-Aligned 1|0 and X-Cam-Inside 0-1 (absent when the check drew no outline).
    """
    cam_model = get_cam()
    web = app.config["WEB_DEMO"]
    if web:
        upload = request.files.get("image")
        jpeg = upload.read() if upload is not None else None
        checked = check_photo(jpeg) if jpeg else None  # nothing was kept, so the outline is drawn again (~50 ms)
    else:
        jpeg = last_photo.last_checked_or_scanned() if request.method == "GET" else None
        checked = last_photo.last_check(jpeg) if jpeg else None
    if cam_model is None or not jpeg:
        return _error("no_photo", 404)
    cached = None if web else last_photo.cam_cached(jpeg)
    if cached is None:
        t = time.perf_counter()
        try:
            rgb = checked.rgb if checked is not None and checked.rgb is not None else decode_jpeg(jpeg, config.MAX_WORK_PX)
            heat, cam01 = cam_model.render(rgb)
        except Exception:  # noqa: BLE001 - a heatmap failure is a 404, never a crash
            log.exception("cam failed")
            return _error("scanner", 404)
        outline = gate.outline_polygon(checked.report.outline) if checked is not None and checked.report is not None else []
        align = alignment(cam01, outline)
        if not web:
            last_photo.cam_built(jpeg, heat, align)
        log.info("cam ms=%d aligned=%s inside=%s", (time.perf_counter() - t) * 1000,
                 "-" if align is None else int(align["aligned"]), "-" if align is None else align["inside"])
    else:
        heat, align = cached
    headers = {"Cache-Control": "no-store"}
    if align is not None:
        headers["X-Cam-Aligned"] = "1" if align["aligned"] else "0"
        headers["X-Cam-Inside"] = f"{align['inside']:.2f}"
    return Response(heat, mimetype="image/jpeg", headers=headers)


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
    checked = check_photo(jpeg)
    last_photo.remember_check(jpeg, checked)  # /scan on the same photo skips decode + gate
    return jsonify(_web_words(_apply_read_anyway_setting(check_result(checked))))


@app.post("/scan")
def scan():
    # This request scans exactly these bytes, even if another request replaces the
    # remembered photo while it runs.
    jpeg = _photo_from_request()
    if not jpeg:
        return _error("no_photo", 400)
    force = request.form.get("force", request.args.get("force", "0")) == "1"
    outcome = _apply_read_anyway_setting(
        run_scan(jpeg, force=force and settings["allow_read_anyway"], checked=last_photo.last_check(jpeg)).to_dict()
    )
    last_photo.scanned(jpeg, outcome)
    return jsonify(_web_words(outcome))


@app.post("/forget")
def forget():
    last_photo.forget()
    return jsonify({"ok": True})


# --- Questions ------------------------------------------------------------------


@app.get("/questions")
def questions():
    band = BAND_FOR_STATE.get(request.args.get("state", ""), "general")
    return jsonify({"suggestions": get_bank().suggestions(band), "drafters": get_bank().drafters()})


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()[: config.QUESTION_MAX_CHARS]
    if not question:
        return jsonify({"ok": False}), 400
    # Only lines verdict.py itself writes: the answer carries the project's badge,
    # so the page must not be able to append text of its own to it.
    lines = [x for x in (body.get("sign_lines") or []) if x in SIGN_LINE_TEXTS]
    return jsonify(get_bank().ask(question, body.get("state"), lines, web=app.config["WEB_DEMO"]).to_dict())


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
    # Saved scans stay on the card: only "End event - erase all" removes them.
    try:
        config.QUIT_FLAG.write_text("quit")
    except OSError:
        log.warning("could not write the quit flag", exc_info=True)
    return jsonify({"ok": True})


@app.post("/update")
def update_kiosk():
    """Update from GitHub: the launcher sees the flag, stops the kiosk, runs
    scripts/update.sh and starts the kiosk again. Nothing runs inside this process."""
    if not _staff_ok():
        return jsonify({"ok": False}), 403
    try:
        config.UPDATE_FLAG.write_text("update")
    except OSError:
        log.warning("could not write the update flag", exc_info=True)
        return jsonify({"ok": False}), 500
    log.info("update from GitHub requested by staff")
    return jsonify({"ok": True})


@app.post("/update/log")
def update_log():
    """The tail of ~/.dermascan/update.log, so staff can read why an update failed without a keyboard."""
    if not _staff_ok():
        return jsonify({"ok": False}), 403
    try:
        lines = config.UPDATE_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    return jsonify({"ok": True, "lines": lines[-config.UPDATE_LOG_LINES :]})


def main() -> None:
    parser = argparse.ArgumentParser(description="DermaScan kiosk server")
    parser.add_argument("--port", type=int, default=config.KIOSK_PORT)
    parser.add_argument("--debug", action="store_true", help="auto-reload on edits (laptop only)")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens it to the network")
    parser.add_argument("--web", action="store_true", help="public web demo: no camera, nothing kept, no staff routes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    code = config.staff_passcode()
    if code is not None and not (len(code) == 4 and code.isdigit()):
        raise SystemExit("The staff code must be exactly 4 digits (the on-screen keypad sends 4). Fix ~/.dermascan_passcode.")
    get_classifier()  # load the model before the first visitor, not during their scan
    if args.web:
        use_web_mode()
    else:
        cam.camera.start()
    warm_in_background()  # the "Where it looked" interpreter, off the boot path and off the scan path
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
