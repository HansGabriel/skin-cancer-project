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
# Focus = variance of the Laplacian after a contrast stretch.
# Real dermoscopy photos score median 79; 20 is "genuinely unusable".
FOCUS_MIN = 20.0
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
# The ring of skin a spot is compared against: this share of the frame's width.
SPOT_RING_FRACTION = 0.06

# --- Gate 4: does the spot have a real edge? ----------------------------------
# Edge width = (lightness drop across the outline) / (steepness at the outline),
# as a percentage of the frame diagonal. Synthetic moles measure 1-3 even blurred
# by 14 px; a shadow or a lighting gradient measures 6-12. NOT yet measured on Pi
# captures: switch on SAVE_CAPTURES_DIR and check before trusting it further.
SPOT_MAX_EDGE_WIDTH_PCT = 4.0

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
