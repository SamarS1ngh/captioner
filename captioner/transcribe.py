"""faster-whisper transcription adapter: audio -> raw (pre-romanize) Transcript."""
from __future__ import annotations

from faster_whisper import WhisperModel

from captioner.models import Segment, Transcript, Word

# Biases toward keeping English words in Latin and against translating Hindi
# function words (aur -> and).
PROMPT = (
    "Namaste! Ye ek Hinglish video hai, Hindi aur English dono mix karte hain. "
    "Jaise: reel, video, phone, happy, matlab, yaar."
)

_models: dict = {}


def _get_model(model_name: str) -> WhisperModel:
    if model_name not in _models:
        _models[model_name] = WhisperModel(model_name, device="auto", compute_type="int8")
    return _models[model_name]


def transcribe(audio_path: str, model_name: str = "large-v3-turbo") -> Transcript:
    model = _get_model(model_name)
    # VAD is mandatory: without it music-only audio yields garbage.
    raw_segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        word_timestamps=True,
        language="hi",
        task="transcribe",
        initial_prompt=PROMPT,
    )
    segments = [
        Segment(
            text=s.text,
            start=s.start,
            end=s.end,
            words=[Word(text=w.word.strip(), start=w.start, end=w.end) for w in (s.words or [])],
        )
        for s in raw_segments
    ]
    return Transcript(segments=segments, language=info.language)
