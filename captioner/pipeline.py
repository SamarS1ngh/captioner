"""Video -> romanized Hinglish transcript, via Groq (whisper-large-v3 + LLM cleanup).

Requires a Groq API key (passed in, or $GROQ_API_KEY). Cloud-only: audio is sent
to Groq. This gives far better romanized Hinglish than any local option tried.
"""
from __future__ import annotations

import os
import tempfile

from captioner import groq_engine
from captioner.export import extract_audio
from captioner.models import Transcript


class MissingKeyError(ValueError):
    """Raised when no Groq API key is available."""


def process_video(video_path: str, api_key: str | None = None,
                  language: str = "hi", romanize: bool = True) -> Transcript:
    key = (api_key or groq_engine.key_from_env() or "").strip()
    if not key:
        raise MissingKeyError(
            "A Groq API key is required. Get a free one at console.groq.com "
            "and paste it in the app."
        )
    fd, audio_path = tempfile.mkstemp(suffix=".flac")
    os.close(fd)
    try:
        extract_audio(video_path, audio_path)
        return groq_engine.transcribe_and_romanize(
            audio_path, key, language=language, romanize=romanize
        )
    finally:
        os.remove(audio_path)
