from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import io
import soundfile as sf
import os
from helper import load_text_to_speech, load_voice_style, AVAILABLE_LANGS

app = FastAPI(title="Supertonic OpenAI-Compatible API")

# Configuration
ONNX_DIR = os.getenv("ONNX_DIR", "/app/assets/onnx")
VOICE_STYLES_DIR = os.getenv("VOICE_STYLES_DIR", "/app/assets/voice_styles")
USE_GPU = os.getenv("USE_GPU", "0").lower() in ("1", "true", "yes")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "M1")

# Load TTS model once at startup
print(f"Loading Supertonic TTS from {ONNX_DIR} (GPU: {USE_GPU})")
try:
    tts = load_text_to_speech(ONNX_DIR, use_gpu=USE_GPU)
except Exception as e:
    print(f"Error loading TTS: {e}")
    tts = None

class SpeechRequest(BaseModel):
    model: str = "supertonic-3"
    input: str
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.05

@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded properly.")
    
    # Load voice style
    voice_path = os.path.join(VOICE_STYLES_DIR, f"{request.voice}.json")
    if not os.path.exists(voice_path):
        raise HTTPException(status_code=400, detail=f"Voice style '{request.voice}' not found in {VOICE_STYLES_DIR}.")
    
    try:
        style = load_voice_style([voice_path])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load voice style: {e}")

    # Use language-agnostic mode ("na") by default to auto-detect and support all 31 languages
    lang = "na" 
    total_step = 8
    
    try:
        wav, duration = tts(request.input, lang, style, total_step, request.speed)
        
        # Extract actual audio up to the given duration
        wav_audio = wav[0, : int(tts.sample_rate * duration[0].item())]
        
        # Write to in-memory buffer
        buf = io.BytesIO()
        sf.write(buf, wav_audio, tts.sample_rate, format='WAV', subtype='PCM_16')
        buf.seek(0)
        
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_enabled": USE_GPU, "model_loaded": tts is not None}
