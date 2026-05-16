"""
Supertonic TTS — OpenAI-Compatible API Server

Provides a drop-in replacement for OpenAI's /v1/audio/speech endpoint,
serving Supertonic's neural TTS models via ONNX Runtime inference.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8032
"""

import asyncio
import io
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from helper import chunk_text, load_text_to_speech, load_voice_style

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("supertonic")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ONNX_DIR = os.getenv("ONNX_DIR", "/app/assets/onnx")
VOICE_STYLES_DIR = os.getenv("VOICE_STYLES_DIR", "/app/assets/voice_styles")
USE_GPU = os.getenv("USE_GPU", "0").lower() in ("1", "true", "yes")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "M1")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "4096"))

SUPPORTED_FORMATS = frozenset({"wav", "flac", "pcm", "mp3"})

# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------
tts_engine = None
available_voices: dict[str, str] = {}  # voice_id -> absolute file path


def _discover_voices(styles_dir: str) -> dict[str, str]:
    """Scan the voice styles directory and return an ``{id: path}`` mapping."""
    voices: dict[str, str] = {}
    d = Path(styles_dir)
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            voices[f.stem] = str(f)
    return voices


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global tts_engine, available_voices

    logger.info("Loading TTS model from %s (GPU=%s) …", ONNX_DIR, USE_GPU)
    try:
        tts_engine = load_text_to_speech(ONNX_DIR, use_gpu=USE_GPU)
        logger.info(
            "TTS model loaded (sample_rate=%d)", tts_engine.sample_rate
        )
    except Exception:
        logger.exception("Failed to load TTS model — API will return 503")
        tts_engine = None

    available_voices = _discover_voices(VOICE_STYLES_DIR)
    logger.info(
        "Discovered %d voice(s): %s",
        len(available_voices),
        sorted(available_voices.keys()),
    )

    yield  # ---- application is running ----

    logger.info("Supertonic API shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Supertonic TTS API",
    description="OpenAI-compatible Text-to-Speech API powered by Supertonic",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    """Schema matching the OpenAI ``/v1/audio/speech`` request body."""

    model: str = "supertonic-3"
    input: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description="The text to generate audio for.",
    )
    voice: str = Field(
        default=DEFAULT_VOICE,
        description="The voice to use for synthesis.",
    )
    response_format: str = Field(
        default="mp3",
        description="Audio output format: wav, mp3, flac, or pcm.",
    )
    speed: float = Field(
        default=1.05,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier (0.25–4.0).",
    )


# ---------------------------------------------------------------------------
# Audio encoding helpers
# ---------------------------------------------------------------------------
def _encode_wav(samples: np.ndarray, sample_rate: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read(), "audio/wav"


def _encode_flac(samples: np.ndarray, sample_rate: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="FLAC")
    buf.seek(0)
    return buf.read(), "audio/flac"


def _encode_pcm(samples: np.ndarray, _sample_rate: int) -> tuple[bytes, str]:
    pcm = (samples * 32767).astype(np.int16)
    return pcm.tobytes(), "audio/pcm"


def _encode_mp3(samples: np.ndarray, sample_rate: int) -> tuple[bytes, str]:
    try:
        import lameenc

        pcm = (samples * 32767).astype(np.int16).tobytes()
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)  # 2 = near-best quality
        data = encoder.encode(pcm) + encoder.flush()
        return data, "audio/mpeg"
    except ImportError:
        logger.warning(
            "lameenc is not installed — falling back to WAV. "
            "Install it with: pip install lameenc"
        )
        return _encode_wav(samples, sample_rate)


ENCODERS = {
    "wav": _encode_wav,
    "flac": _encode_flac,
    "pcm": _encode_pcm,
    "mp3": _encode_mp3,
}

FORMAT_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
def _wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """Build a WAV header with an unknown data size (for streaming).

    Uses ``0x7FFFFFFF`` as the data-size placeholder.  Most audio players
    simply read until EOF, so the oversized header is harmless.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = 0x7FFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36,
        b"WAVE",
        b"fmt ",
        16,           # PCM sub-chunk size
        1,            # PCM format tag
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )


async def _audio_stream(
    text: str,
    tts_engine,
    style,
    speed: float,
    fmt: str,
):
    """Async generator — yields encoded audio bytes chunk-by-chunk.

    Each text chunk is synthesised independently and sent immediately,
    dramatically reducing time-to-first-byte for long inputs.
    """
    lang = "na"
    total_step = 8
    silence_duration = 0.3
    sample_rate = tts_engine.sample_rate
    max_len = 300  # characters per chunk ("na" is not ko/ja)

    chunks = chunk_text(text, max_len=max_len)
    t0 = time.perf_counter()

    # For WAV streaming, send the header before any audio data
    if fmt == "wav":
        yield _wav_header(sample_rate)

    total_audio_duration = 0.0

    for i, text_chunk in enumerate(chunks):
        # --- Synthesise this chunk in a thread pool ---
        wav, duration = await asyncio.to_thread(
            tts_engine._infer,
            [text_chunk],
            [lang],
            style,
            total_step,
            speed,
        )

        chunk_dur = duration[0].item()
        total_audio_duration += chunk_dur
        audio = wav[0, : int(sample_rate * chunk_dur)]

        # Insert silence between chunks (not after the last one)
        if i < len(chunks) - 1:
            silence = np.zeros(
                int(silence_duration * sample_rate), dtype=np.float32
            )
            audio = np.concatenate([audio, silence])
            total_audio_duration += silence_duration

        # --- Encode and yield ---
        if fmt in ("wav", "pcm"):
            # Raw PCM16-LE bytes (WAV header already sent for wav)
            pcm = (audio * 32767).astype(np.int16)
            yield pcm.tobytes()
        else:
            encoder = ENCODERS[fmt]
            audio_bytes, _ = encoder(audio, sample_rate)
            yield audio_bytes

    elapsed = time.perf_counter() - t0
    logger.info(
        "Streamed %.2fs audio in %.2fs (%d chunk%s) — fmt=%s",
        total_audio_duration,
        elapsed,
        len(chunks),
        "s" if len(chunks) != 1 else "",
        fmt,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Liveness / readiness probe for container orchestrators."""
    return {
        "status": "ok" if tts_engine is not None else "degraded",
        "model_loaded": tts_engine is not None,
        "gpu_enabled": USE_GPU,
        "available_voices": sorted(available_voices.keys()),
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible ``GET /v1/models``)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "supertonic-3",
                "object": "model",
                "created": 1747000000,
                "owned_by": "supertone",
            },
            {
                "id": "tts-1",
                "object": "model",
                "created": 1747000000,
                "owned_by": "supertone",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1747000000,
                "owned_by": "supertone",
            },
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate speech from text (OpenAI-compatible ``POST /v1/audio/speech``).

    Audio is streamed chunk-by-chunk as it is synthesised, so the client
    can begin playback before the full text has been processed.
    """

    # --- Guard: model loaded? ---
    if tts_engine is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model is not loaded. Check server logs.",
        )

    # --- Validate voice ---
    if request.voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown voice '{request.voice}'. "
                f"Available: {sorted(available_voices.keys())}"
            ),
        )

    # --- Validate format ---
    fmt = request.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported response_format '{fmt}'. "
                f"Supported: {sorted(SUPPORTED_FORMATS)}"
            ),
        )

    # --- Load voice style ---
    try:
        style = load_voice_style([available_voices[request.voice]])
    except Exception:
        logger.exception("Failed to load voice style '%s'", request.voice)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load voice style '{request.voice}'.",
        )

    chunks = chunk_text(request.input, max_len=300)

    if len(chunks) == 1:
        logger.info(
            "Starting single-chunk synthesis — voice=%s, chars=%d, fmt=%s",
            request.voice, len(request.input), fmt
        )
        wav, duration = await asyncio.to_thread(
            tts_engine._infer,
            [chunks[0]],
            ["na"],
            style,
            8,
            request.speed,
        )
        sample_rate = tts_engine.sample_rate
        audio = wav[0, : int(sample_rate * duration[0].item())]
        
        encoder = ENCODERS[fmt]
        audio_bytes, media_type = encoder(audio, sample_rate)
        
        return Response(
            content=bytes(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{fmt}"',
            },
        )

    logger.info(
        "Starting streaming synthesis — voice=%s, chars=%d, chunks=%d, fmt=%s",
        request.voice,
        len(request.input),
        len(chunks),
        fmt,
    )

    return StreamingResponse(
        _audio_stream(
            text=request.input,
            tts_engine=tts_engine,
            style=style,
            speed=request.speed,
            fmt=fmt,
        ),
        media_type=FORMAT_MEDIA_TYPES[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="speech.{fmt}"',
        },
    )
