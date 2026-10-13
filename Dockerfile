# The public web demo: the kiosk page (kiosk/) in web mode, for Railway or any Docker host.
# Same dermascan/ core and models/ as the Pi; no camera, nothing kept, no staff routes.
#   docker build -t epivue-demo . && docker run --rm -p 8080:8080 epivue-demo
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Dependencies first, so a code change does not reinstall them. The kiosk's own pins.
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Only what the demo runs: the core, the four model files, the kiosk server and page.
COPY dermascan/ dermascan/
COPY models/ models/
COPY kiosk/ kiosk/

EXPOSE 8080

# Shell form so Railway's $PORT is expanded.
CMD python -m kiosk.server --web --host 0.0.0.0 --port "${PORT}"
