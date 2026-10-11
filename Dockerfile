# The public web demo (cloud/streamlit_app.py) for Railway, or any Docker host.
# Same dermascan/ core and models/ as the kiosk; no camera, nothing stored.
#   docker build -t epivue-demo . && docker run --rm -p 8501:8501 epivue-demo
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY cloud/requirements.txt cloud/requirements.txt
RUN pip install -r cloud/requirements.txt

# Only what the demo runs: the core, the four model files, the page, Streamlit's config.
COPY dermascan/ dermascan/
COPY models/ models/
COPY cloud/ cloud/
COPY .streamlit/ .streamlit/

EXPOSE 8501

# Shell form so Railway's $PORT is expanded. Run from /app so .streamlit/config.toml is read.
CMD streamlit run cloud/streamlit_app.py --server.port "${PORT}" --server.address 0.0.0.0
