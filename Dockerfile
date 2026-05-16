# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/st

WORKDIR /app

# System deps for lxml + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so cold start doesn't pay for it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy code AFTER deps (cache-friendly)
COPY app ./app
COPY scripts ./scripts
COPY tests ./tests

# Catalog: bake it in at build time. If you want to rebuild it without
# rebuilding the image, mount data/ at runtime instead.
COPY data ./data

# Render injects $PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
