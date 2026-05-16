FROM python:3.10-slim

# Install system dependencies + gosu for privilege de-escalation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    libsndfile1 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root application user
RUN groupadd -r supertonic && \
    useradd -r -g supertonic -d /app -s /sbin/nologin supertonic

WORKDIR /app

# Copy requirements first — this layer is cached unless dependencies change
COPY py/requirements.txt py/requirements-api.txt /app/py/

# Install all Python dependencies in a single layer
RUN pip install --no-cache-dir \
    -r /app/py/requirements.txt \
    -r /app/py/requirements-api.txt

# Copy application code (changes more often, placed after deps for caching)
COPY py/ /app/py/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Ensure the app user owns the working directory
RUN mkdir -p /app/assets && chown -R supertonic:supertonic /app

# Environment configuration
ENV ONNX_DIR="/app/assets/onnx" \
    VOICE_STYLES_DIR="/app/assets/voice_styles" \
    USE_GPU="0" \
    PORT="8032" \
    DEFAULT_VOICE="M1" \
    LOG_LEVEL="INFO"

EXPOSE ${PORT}

# Docker-native health check — used by orchestrators (Dokploy, Kubernetes, etc.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

WORKDIR /app/py

ENTRYPOINT ["/app/entrypoint.sh"]

# Production server — single worker is appropriate for CPU-bound inference
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
