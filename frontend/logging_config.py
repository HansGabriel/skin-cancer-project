"""Terminal logging for DermaScan (Pi HTTP errors, pipeline failures)."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Attach a stderr handler to the ``dermascan`` logger (once per process)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("DERMASCAN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger = logging.getLogger("dermascan")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
