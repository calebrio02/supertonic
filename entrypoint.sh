#!/bin/bash
set -e

# Check if the models are missing
if [ ! -d "/app/assets/onnx" ] || [ -z "$(ls -A /app/assets/onnx 2>/dev/null)" ]; then
    echo "========================================================="
    echo "Modelos no encontrados en /app/assets."
    echo "Descargándolos automáticamente de Hugging Face..."
    echo "Esto puede tardar unos minutos (aprox ~99MB)."
    echo "========================================================="
    
    # Setup git lfs
    git lfs install
    
    # Clone to a temporary directory to avoid conflicts with existing mounted folder
    git clone https://huggingface.co/Supertone/supertonic-3 /tmp/supertonic-models
    
    # Move the contents to the mounted assets folder
    cp -r /tmp/supertonic-models/* /app/assets/
    rm -rf /tmp/supertonic-models
    
    echo "¡Descarga completada y guardada en volumen persistente!"
else
    echo "Modelos encontrados en /app/assets. Omitiendo descarga."
fi

echo "Iniciando servidor API..."
# Execute the command passed to the container (uvicorn)
exec "$@"
