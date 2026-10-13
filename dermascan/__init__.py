"""DermaScan core: gate -> classifier -> verdict. No web framework in here.

Read the files in this order:

    config.py       every number the kiosk can be tuned with, and why
    scan.py         run_scan(jpeg_bytes): the one function the UI calls
    gate.py         "is this a readable photo of one spot on skin?"
    classifier.py   the TFLite model, 4-view TTA, the screening threshold
    verdict.py      label + confidence -> plain-language words on screen
"""

from dermascan import config  # noqa: F401 - sets OpenCV's picture-size limit before cv2 is imported
