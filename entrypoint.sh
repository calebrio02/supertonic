#!/bin/bash
set -e

ASSETS_DIR="/app/assets"
ONNX_DIR="${ASSETS_DIR}/onnx"

# ---------------------------------------------------------------
# 1. Download models from Hugging Face if not already present
# ---------------------------------------------------------------
if [ ! -d "${ONNX_DIR}" ] || [ -z "$(ls -A "${ONNX_DIR}" 2>/dev/null)" ]; then
    echo "========================================================="
    echo "  Models not found in ${ASSETS_DIR}."
    echo "  Downloading from Hugging Face (approx. 380 MB) ..."
    echo "========================================================="

    git lfs install --skip-smudge 2>/dev/null || true
    git lfs install

    TEMP_DIR=$(mktemp -d)
    if ! git clone https://huggingface.co/Supertone/supertonic-3 "${TEMP_DIR}"; then
        echo "ERROR: Failed to download models from Hugging Face." >&2
        rm -rf "${TEMP_DIR}"
        exit 1
    fi

    cp -r "${TEMP_DIR}"/* "${ASSETS_DIR}/"
    rm -rf "${TEMP_DIR}"

    # Verify critical files exist
    if [ ! -f "${ONNX_DIR}/vocoder.onnx" ]; then
        echo "ERROR: Download completed but vocoder.onnx is missing." >&2
        echo "       The model files may be corrupted. Try again." >&2
        exit 1
    fi

    echo "Download complete. Models saved to ${ASSETS_DIR}."
else
    echo "Models found in ${ASSETS_DIR}. Skipping download."
fi

# ---------------------------------------------------------------
# 2. Fix ownership so the non-root user can read the assets
# ---------------------------------------------------------------
chown -R supertonic:supertonic "${ASSETS_DIR}" 2>/dev/null || true

# ---------------------------------------------------------------
# 3. Drop privileges and exec the main process
# ---------------------------------------------------------------
echo "Starting Supertonic API server ..."
exec gosu supertonic "$@"
