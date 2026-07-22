"""SenseVoice ASR (DashScope, for the mic-input trigger) + edge-tts (local,
free) for speaking the final advice out loud."""
from __future__ import annotations

import asyncio
import tempfile

import dashscope
from dashscope.audio.asr import Recognition

import config

dashscope.api_key = config.DASHSCOPE_API_KEY

DEFAULT_VOICE = "en-US-AriaNeural"


def transcribe(audio_path: str) -> str | None:
    """Best-effort local-file transcription via SenseVoice. Verify against a
    real DASHSCOPE_API_KEY during the H0-H1 smoke test — this wraps the
    synchronous Recognition.call() path, not the async file_urls task API."""
    try:
        result = Recognition.call(model="sensevoice-v1", file=audio_path, format="wav", sample_rate=16000)
        if result.status_code == 200 and result.output:
            return result.output.get("text")
    except Exception:
        pass
    return None


async def _speak_async(text: str, voice: str, out_path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def speak_text(text: str, voice: str = DEFAULT_VOICE) -> str | None:
    """Synthesizes text to an mp3 file, returns the file path (or None on failure)."""
    try:
        out_path = tempfile.mktemp(suffix=".mp3")
        asyncio.run(_speak_async(text, voice, out_path))
        return out_path
    except Exception:
        return None
