"""Video -> romanized transcript pipeline."""
import os
import tempfile

from captioner.export import extract_audio
from captioner.models import Transcript
from captioner.romanize import romanize_transcript
from captioner.transcribe import transcribe


def process_video(video_path: str, model_name: str = "large-v3-turbo") -> Transcript:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        extract_audio(video_path, wav_path)
        transcript = transcribe(wav_path, model_name)
    finally:
        os.remove(wav_path)
    return romanize_transcript(transcript)
