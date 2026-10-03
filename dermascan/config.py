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

# --- Inference ----------------------------------------------------------------
NUM_THREADS = 4  # the Pi 4 has four cores
INPUT_SIZE = 224  # EfficientNetB0 was trained at 224x224
USE_TTA = True  # average 4 flipped views; ~0.5 s extra on a Pi 4, steadier answers
CONFIDENCE_FLOOR_PCT = 45.0  # below this the verdict says "not sure" instead of a label

# --- Photo size ---------------------------------------------------------------
MAX_WORK_PX = 1024  # longest edge any photo is reduced to before anything looks at it
GATE_WORK_PX = 384  # the gate measures a 384 px copy: fast on a Pi, same answer
FOCUS_WORK_PX = 512  # focus is measured at one fixed size, so a webcam and the Pi score alike

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
# The region the gate outlines must be this share of the frame...
SPOT_MIN_FRACTION = 0.004  # smaller is noise; "move closer"
SPOT_MAX_FRACTION = 0.75  # larger is the whole frame; "move back"
SPOT_IDEAL_FRACTION = 0.18  # a well-framed lesion; used to pick between candidate outlines
# ...be one compact blob rather than a scatter. Solidity = blob area / area of
# its convex hull: a mole scores 0.85+, speckle on plain skin scores under 0.4.
SPOT_MIN_SOLIDITY = 0.6
# ...and differ from the skin around it by this much (CIE Lab distance; ~2.3 is
# the smallest difference a person can see, 5 means a real mark, not noise).
SPOT_MIN_CONTRAST = 5.0
# Hairs thinner than this share of the frame are filled in before the spot is
# outlined (a morphological close). 3% of 384 px = 11 px: wider than a hair at
# dermoscope or cone distance, far narrower than any mole worth reading.
HAIR_KERNEL_FRACTION = 0.03
# The ring of skin a spot is compared against: this share of the frame's width.
SPOT_RING_FRACTION = 0.06

# --- Gate 4: does the spot have a real edge? ----------------------------------
# Edge width = (lightness drop across the outline) / (steepness at the outline),
# as a percentage of the frame diagonal. 267 real HAM10000 lesions through JPEG
# (2026-10-03) measure median 3.1, max 4.9; a shadow or lighting gradient measures
# 6-13. 4.0 refused one real lesion in eight. NOT yet measured on Pi captures:
# switch on SAVE_CAPTURES_DIR and check before trusting it further.
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
# E needs an earlier photo of the same spot, which this kiosk does not keep.

# --- Questions (dermascan/answers.py) -----------------------------------------
ANSWER_MATCH_MIN = 0.25  # TF-IDF cosine below this gets the "ask a health worker" answer
QUESTION_MAX_CHARS = 200  # a typed question is cut here; the page's text box stops at the same
SUGGESTED_QUESTIONS = 3  # tappable questions under each answer; three fit the 600 px panel

# --- Saved scans (kiosk/saved.py) ---------------------------------------------
# Kept in the server's memory only, never on the SD card. They vanish on
# "End event - erase all", on Exit, and on any restart.
SAVED_MAX = 60  # oldest is dropped past this: a day's event, ~10 MB of RAM
SAVED_PHOTO_PX = 512  # saved photos are shrunk to this; enough for the aperture
SAVED_JPEG_QUALITY = 85  # small in memory, still sharp at aperture size

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

# --- Optional: keep raw captures for later calibration ------------------------
# Set to a directory (Path) to save every capture as a JPEG next to its gate
# numbers. Off by default: captures are photos of people's skin.
SAVE_CAPTURES_DIR: Path | None = None


# --- Staff passcode -----------------------------------------------------------
def staff_passcode() -> str | None:
    """The 4-digit code that "Exit kiosk" asks for, or None (then two taps exit).

    Read from DERMASCAN_PASSCODE, else from ~/.dermascan_passcode (one line, kept
    off git, chmod 600). Read each time, so changing the file needs no restart.
    The on-screen keypad sends exactly four digits.
    """
    code = os.environ.get("DERMASCAN_PASSCODE", "").strip()
    if not code:
        try:
            code = (Path.home() / ".dermascan_passcode").read_text(encoding="utf-8").strip()
        except OSError:
            code = ""
    return code or None
