"""Every number the kiosk can be tuned with, in one place.

Change a value here, restart the app, and that is the whole procedure. Nothing
else in the project reads environment variables for tuning.

Each number carries a one-line reason. If you change one, change the reason.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# --- Model files --------------------------------------------------------------
# Produced by training/ on the PC. Swap all four together: the threshold and the
# temperature were fitted to the model they shipped with.
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

# --- Gate: is there skin? -----------------------------------------------------
# Fraction of the frame that must look like skin. 0.08 sits above the worst
# grey surface measured through JPEG (0.057) and far below real skin (0.8+).
SKIN_MIN_FRACTION = 0.08

# --- Gate: can the photo be read? ---------------------------------------------
# Focus = variance of the Laplacian at <=512 px after a contrast stretch.
# Real dermoscopy photos score median 79; 20 is "genuinely unusable".
FOCUS_MIN = 20.0
# Mean brightness (HSV value, 0-255). Outside this the photo is black or glare.
BRIGHTNESS_MIN = 12.0
BRIGHTNESS_MAX = 246.0

# --- Gate: is there one spot on the skin? -------------------------------------
# The dark region the gate outlines must be this share of the frame...
SPOT_MIN_FRACTION = 0.004  # smaller is noise; "move closer"
SPOT_MAX_FRACTION = 0.75  # larger is the whole frame; "move back"
SPOT_IDEAL_FRACTION = 0.18  # a well-framed lesion; used to pick between candidate outlines
# ...be one compact blob rather than a scatter. Solidity = blob area / area of
# its convex hull: a mole scores 0.85+, speckle on plain skin scores under 0.4.
SPOT_MIN_SOLIDITY = 0.6
# ...have an edge that stops the way pigment does, not fades the way a shadow
# does. Edge width = (lightness drop across the outline) / (steepness at the
# outline), as a percentage of the frame diagonal. Moles measure 1-2 even when
# soft-focused; a shadow or a lighting gradient measures 6 and up.
SPOT_MAX_EDGE_WIDTH_PCT = 4.0
# ...and differ from the skin around it by this much (CIE Lab distance; ~2.3 is
# the smallest difference a person can see, 5 means a real mark, not noise).
SPOT_MIN_CONTRAST = 5.0

# --- Optional: keep raw captures for later calibration ------------------------
# Set to a directory (Path) to save every capture as a JPEG next to its gate
# numbers. Off by default: captures are photos of people's skin.
SAVE_CAPTURES_DIR: Path | None = None
