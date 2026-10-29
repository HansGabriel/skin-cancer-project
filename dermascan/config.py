"""Every number the kiosk can be tuned with, in one place.

Change a value here, restart the app, and that is the whole procedure. Nothing
else in the project reads environment variables for tuning; the only outside
input is the staff passcode (see the end of this file).

Each number carries a one-line reason. If you change one, change the reason.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# --- Model files --------------------------------------------------------------
# Produced by training/ on the PC. Swap all four together: the threshold and the
# temperature were fitted to the model they shipped with (training/calibrate.py).
MODEL_PATH = MODELS_DIR / "skin_classifier.tflite"
LABELS_PATH = MODELS_DIR / "labels.txt"  # one label per line, same order as the model output
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"  # screen_cancer_threshold + class indexes
TEMPERATURE_PATH = MODELS_DIR / "temperature.json"  # T for honest displayed confidence
# Optional fifth file, staff heatmap only (dermascan/cam.py). Exported from the same run as
# the classifier by scripts/export_cam_tflite.py; absent or mismatched -> no heatmap button.
CAM_PATH = MODELS_DIR / "skin_classifier_cam.tflite"
CAM_META_PATH = MODELS_DIR / "skin_classifier_cam.json"  # which classifier (sha256) it belongs to
CAM_ALPHA = 0.5  # half photo, half heat: the spot stays visible under the colours
CAM_JPEG_QUALITY = 85

# --- Inference ----------------------------------------------------------------
NUM_THREADS = 4  # the Pi 4 has four cores
INPUT_SIZE = 224  # EfficientNetB0 was trained at 224x224
USE_TTA = True  # average 4 flipped views; ~0.5 s extra on a Pi 4, steadier answers
CONFIDENCE_FLOOR_PCT = 45.0  # below this the verdict says "not sure" instead of a label

# --- Photo size ---------------------------------------------------------------
MAX_WORK_PX = 1024  # longest edge any photo is reduced to before anything looks at it
GATE_WORK_PX = 384  # the gate measures a 384 px copy: fast on a Pi, same answer
FOCUS_WORK_PX = 512  # focus is measured at one fixed size, so a webcam and the Pi score alike
# A small file can claim a huge picture (a 1 MB PNG of 20000x20000 decodes to 1.2 GB).
# OpenCV refuses anything larger. It reads the limit once, when cv2 is first imported,
# so dermascan/__init__.py loads this file first; that holds for kiosk.server,
# cloud/streamlit_app.py and scripts/bench_scan.py (not for training/ or the tests,
# which import cv2 themselves and never open strangers' uploads).
MAX_DECODE_PIXELS = 50_000_000  # a 50 MP phone photo still opens (~150 MB while decoding)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(MAX_DECODE_PIXELS)  # OpenCV's own switch, set here, never from outside
MAX_PARALLEL_DECODES = 2  # photos opened at once; the rest wait, so memory stays bounded

# --- Gate 1: is there skin? ---------------------------------------------------
# Fraction of the frame that must look like skin. 0.08 sits above the worst
# grey surface measured through JPEG (0.057) and far below real skin (0.8+).
SKIN_MIN_FRACTION = 0.08
# Dim frames are brightened to this mean value before their colour is judged:
# JPEG crushes colour in low light, and real skin then reads as grey.
SKIN_EXPOSURE_TARGET = 110.0

# --- Gate 2: can the photo be read? -------------------------------------------
# Focus = variance of the Laplacian after a 1-99 percentile contrast stretch.
# Set from what the MODEL tolerates, not from what looks sharp (2026-10-03, 80
# never-seen HAM10000 photos): it still flags every cancer at Gaussian blur
# sigma 2 px and slips only from sigma 3. 267 real photos score median ~124;
# at 13, sharp photos with glare, dim light or camera noise reduction are refused
# 0-5% of the time (the old min/max stretch at 20 refused 46% after noise
# reduction), and about half of sigma 3-6 blur is still refused.
FOCUS_MIN = 13.0
# Mean brightness (HSV value, 0-255). These are the old app's measured HARD limits:
# outside them the photo is near-black or blown out. (25-235 was only ever advice.)
BRIGHTNESS_MIN = 12.0
BRIGHTNESS_MAX = 246.0

# --- Gate 3: is there one spot on the skin? -----------------------------------
# "GUESSED" marks a limit nobody has measured on a Pi capture yet (scripts/gate_report.py
# prints the same word). The learned "is this one spot?" check that would replace these
# waits for the 30 Oct 2026 capture session (docs/METRICS.md, open item 5).
# The region the gate outlines must be this share of the frame...
# GUESSED — TO BE MEASURED on Pi cone captures: a webcam face passed at 0.0045 (one nostril).
# 0.01 is the candidate floor once real moles under the cone measure 0.02 or more.
SPOT_MIN_FRACTION = 0.004  # smaller is noise; "move closer"
# Set 2026-10-20 because a palm's shadowed half passed at 0.46 under the old 0.75.
# MEASURED 2026-10-24 on the frozen bench (training/external_eval.py --no-force):
# 62 of 602 lesion photos are refused "fills_frame" at 0.40 (spot_fraction median
# 0.49); 0.50 would refuse 29, 0.75 only 2. Kept at 0.40 on purpose: the bench photos
# were not framed for the ring, and loosening it lets the palm back in. 0.30 is the
# target for the cone once Pi captures show what a real spot covers (docs/METRICS.md 5).
SPOT_MAX_FRACTION = 0.40  # larger is the whole frame; "move back"
# ...differ from the skin around it by this much (CIE Lab distance; ~2.3 is
# the smallest difference a person can see, 5 means a real mark, not noise).
# GUESSED — a rule of thumb; below it the photo is "plain skin" (no verdict even when forced).
SPOT_MIN_CONTRAST = 5.0
# ...and be one compact blob rather than a scatter. Solidity = blob area / area of
# its convex hull: a mole scores 0.85+, speckle on plain skin scores under 0.4.
# GUESSED — below it the photo is "ragged_spot" since 2026-10-29: a real mark whose edge is
# hard to follow (scattered or multi-lobed), so "Check it anyway" still answers, with the caveat.
SPOT_MIN_SOLIDITY = 0.6
# Hairs thinner than this share of the frame are filled in before the spot is
# outlined (a morphological close). 3% of 384 px = 11 px: wider than a hair at
# dermoscope or cone distance, far narrower than any mole worth reading.
HAIR_KERNEL_FRACTION = 0.03
# Dark regions touching the frame edge (a vignette) are set aside and the outline
# redone only while they cover less than this share of the frame: past it, the
# "vignette" IS the frame - a spot too big for it, which check 3 refuses anyway.
VIGNETTE_MAX_FRACTION = 0.9
# The ring of skin a spot is compared against: this share of the frame's width.
SPOT_RING_FRACTION = 0.06

# --- Gate 3b: is it ONE spot, in the ring? ----------------------------------
# Added 2026-10-20 after a webcam face (one nostril as the "spot", 99% malignant)
# and a palm (its shadowed half, cancer score 0.75) passed every check above.
# Rejected on the way: "spot pixels must be skin-coloured" (a black melanoma fails
# it exactly like a pupil), a higher skin floor (face 0.53, palm 0.98) and a
# higher focus floor (a sharp palm passes anyway).
# How far the spot's centre may sit from the frame's centre, as a share of the
# short side. The ring the visitor sees has a radius of about 0.40 of the side.
# MEASURED 2026-10-24 on the frozen bench: 101 of 602 lesion photos are refused
# "not_in_ring"; 48 of those touch the frame border (the rule below, which no limit
# moves) and 53 sit off-centre (offset median 0.40, so 0.30 would still refuse 43).
# Kept at 0.25: the bench is dermoscopy framing, not the ring. Set it from Pi cone
# captures (docs/DEPLOYMENT.md, "Setting the gate limits from real photos").
SPOT_MAX_OFFSET = 0.25
# A spot that touches the frame edge is refused with the same words: the gate
# cannot see where it ends, and a hand's shadow always runs off the frame.
# Other dark marks in the frame, besides the spot. Carried over from the old
# app's dark_structure check (commit 246c072, measured on a real portrait: 10
# marks, dominance 0.50; tight face crop 5 / 0.57; a lesion with one satellite
# 1 / 0.92). Refuse when there are MORE than this many others AND the spot holds
# less than this share of all the dark-mark area (one satellite is fine).
SPOT_MAX_OTHER_MARKS = 1  # GUESSED — transferred, not yet measured on this gate's outline
SPOT_MIN_DOMINANCE = 0.65  # GUESSED — transferred: spot area / (spot + other marks)
# `ring_skin` (how much of the skin ring around the spot looks like skin) is
# measured and logged but has NO limit: deep skin tones sit at the edge of the
# skin colour box, and refusing on it unmeasured would hurt the people the
# kiosk is for. Set a limit only from real captures.

# --- Gate 4: does the spot have a real edge? ----------------------------------
# Edge width = (lightness drop across the outline) / (steepness at the outline),
# as a percentage of the frame diagonal. 267 real HAM10000 lesions through JPEG
# (2026-10-03) measure median 3.1, max 4.9; a shadow or lighting gradient measures
# 6-13. 4.0 refused one real lesion in eight. GUESSED for the Pi camera (measured on
# dermoscopy only): switch on SAVE_CAPTURES_DIR and check before trusting it further.
SPOT_MAX_EDGE_WIDTH_PCT = 5.5

# --- The photo readings on the "check the photo" screen (verdict.photo_readings)
# Display only: the gate's limits above still decide. These split a passing photo
# into "good" and "could be better" so the visitor sees why a retake might help.
FOCUS_SHARP = 30.0  # sigma-1 blur scores ~38, sigma-1.5 ~22: under 30 reads, but shows as soft
BRIGHTNESS_GOOD = (40.0, 215.0)  # inside this the light is even; outside, dim or glary

# --- The A B C D E signs (dermascan/signs.py) ---------------------------------
# Measured on the gate's outline of the spot. Shown to help a person look, never
# used to decide the verdict: the model and thresholds.json do that alone.
# Each pair is (borderline from, stands out above), taken from the old app's
# Notion-agreed tiers so the paper's numbers stay comparable.
ASYMMETRY_TIERS = (0.15, 0.30)  # share of the outline that does not match when folded
BORDER_TIERS = (1.8, 2.5)  # perimeter^2 / (4 pi area): a circle is 1.0
COLOUR_TIERS = (2, 2)  # colour groups: one is normal, two borderline, three or more stand out
COLOUR_DISTINCT_DE = 15.0  # two colour groups closer than this (CIE Lab) count as one
COLOUR_GROUPS_K = 5  # colour groups k-means looks for inside the spot
COLOUR_RIM_PX = 3  # outline pixels skipped before counting colours: spot and skin blend there
# D is not measured in millimetres: there is no scale calibration on this camera.
# E is a saved group's history shown side by side (kiosk/saved.py), not a measurement.

# --- Questions (dermascan/answers.py) -----------------------------------------
ANSWER_MATCH_MIN = 0.25  # TF-IDF cosine below this gets the "ask a health worker" answer
QUESTION_MAX_CHARS = 200  # a typed question is cut here; the page's text box stops at the same
SUGGESTED_QUESTIONS = 3  # tappable questions under each answer; three fit the 600 px panel

# --- Saved scans (kiosk/saved.py) ---------------------------------------------
# Kept on the SD card in SAVED_DIR, two small files per scan, so they survive
# Exit, a reboot and a power cut (mentor's decision, 2026-10-11: before this they
# lived in memory only and a day's groups vanished when the Pi was switched off).
# Nothing leaves the device. "End event - erase all" empties the folder.
SAVED_DIR = Path.home() / ".dermascan" / "saved"  # next to ~/.dermascan_passcode; never inside the repo
SAVED_MAX = 60  # oldest is dropped past this: a day's event, a few MB on the card
SAVED_PHOTO_PX = 512  # saved photos are shrunk to this; enough for the aperture
SAVED_JPEG_QUALITY = 85  # small on the card, still sharp at aperture size

# --- Kiosk camera (kiosk/camera.py) -------------------------------------------
CAMERA_STILL_SIZE = (1640, 1232)  # IMX219 full field of view, 2x2 binned: sharp and fast
CAMERA_PREVIEW_SIZE = (640, 480)  # the live view; small so the Pi encodes it cheaply
CAMERA_PREVIEW_FPS = 15  # smooth enough to aim with, light enough for the Pi's browser
CAMERA_FRAME_US = 33_333  # 30 fps sensor timing even on dark skin, so the preview never stalls
CAMERA_SETTLE_S = 0.5  # once at boot: let exposure and white balance settle
CAPTURE_JPEG_QUALITY = 92  # the photo that is scanned
PREVIEW_JPEG_QUALITY = 70  # the live view only

# --- Kiosk server (kiosk/server.py) -------------------------------------------
KIOSK_PORT = 8080  # scripts/launch_kiosk.sh uses the same number
# Staff can flip these two in the Settings tab; they reset to these on restart.
ALLOW_READ_ANYWAY = True  # offer "Check it anyway" when the spot check refuses a photo
SHOW_STAFF_DETAILS = True  # show "Details for staff" (numbers, timings) under a result
QUIT_FLAG = Path("/tmp/dermascan_quit")  # scripts/launch_kiosk.sh watches for this file
UPDATE_FLAG = Path("/tmp/dermascan_update")  # "Update from GitHub": the launcher runs scripts/update.sh, then restarts
VERSION_FILE = Path.home() / ".dermascan" / "version.txt"  # written by launch_kiosk.sh at each start: "hash · date"
LAST_UPDATE_FILE = Path.home() / ".dermascan" / "last_update.txt"  # one line from scripts/update.sh, shown in Settings
UPDATE_LOG_FILE = Path.home() / ".dermascan" / "update.log"  # git's and pytest's own words; Settings shows the tail
UPDATE_LOG_LINES = 20  # lines of it that fit the Settings screen
WEB_MAX_UPLOAD_MB = 8  # web demo only: a phone photo fits; a bigger upload is refused before it is read

# --- Optional: keep raw captures for later calibration ------------------------
# Set to a directory (Path) to save every capture as a JPEG next to its gate
# numbers. Off by default: captures are photos of people's skin.
SAVE_CAPTURES_DIR: Path | None = None


# --- Staff passcode -----------------------------------------------------------
def staff_passcode() -> str | None:
    """The 4-digit code that "Exit kiosk" asks for, or None (then two taps exit).

    Read from DERMASCAN_PASSCODE, else from ~/.dermascan_passcode (one line, kept
    off git, chmod 600). Read each time, so changing the file needs no restart.
    The on-screen keypad sends exactly four digits, so any other code would lock
    staff out for good: kiosk/server.py refuses to start with one.
    """
    code = os.environ.get("DERMASCAN_PASSCODE", "").strip()
    if not code:
        try:
            code = (Path.home() / ".dermascan_passcode").read_text(encoding="utf-8").strip()
        except OSError:
            code = ""
    return code or None
