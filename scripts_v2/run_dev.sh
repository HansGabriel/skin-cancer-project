#!/bin/bash
# Run the kiosk on a laptop (no Pi camera): the page offers the browser's own
# camera and a file picker. Open http://127.0.0.1:8080 afterwards.
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
exec python -m kiosk.server --debug "$@"
