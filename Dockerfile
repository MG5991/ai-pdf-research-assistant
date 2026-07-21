# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    CHROMA_DIRECTORY=/app/chroma_db \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Run the application as a non-root user.
RUN groupadd --system appuser \
    && useradd \
        --system \
        --gid appuser \
        --create-home \
        --no-log-init \
        appuser

# Copy dependencies separately so Docker can reuse
# this layer when application code changes.
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        -r requirements.txt

COPY . .

# Create writable directories for ChromaDB
# and the embedding-model cache.
RUN mkdir -p \
        /app/chroma_db \
        /home/appuser/.cache/huggingface \
    && chown -R \
        appuser:appuser \
        /app \
        /home/appuser

USER appuser

EXPOSE 8501

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]