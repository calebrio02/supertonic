FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python code and requirements
COPY py/ /app/py/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Install standard dependencies
RUN pip install --no-cache-dir -r /app/py/requirements.txt
RUN pip install --no-cache-dir -r /app/py/requirements-api.txt
RUN pip install --no-cache-dir onnxruntime

# Set environment variables for the API
ENV ONNX_DIR="/app/assets/onnx"
ENV VOICE_STYLES_DIR="/app/assets/voice_styles"
ENV USE_GPU="0"
ENV PORT="8032"

EXPOSE ${PORT}

WORKDIR /app/py

ENTRYPOINT ["/app/entrypoint.sh"]

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
