import subprocess
from pathlib import Path

import pytest

from captioner.transcribe import transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "sample_hinglish.wav"


@pytest.mark.slow
def test_transcribes_known_clip():
    t = transcribe(str(FIXTURE))
    joined = " ".join(s.text for s in t.segments).lower()
    assert "happy" in joined
    assert "namaste" in joined
    with_words = [s for s in t.segments if s.words]
    assert with_words
    starts = [w.start for w in with_words[0].words]
    assert starts == sorted(starts) and len(set(starts)) > 1


@pytest.mark.slow
def test_silence_returns_empty_transcript(tmp_path):
    wav = tmp_path / "silence.wav"
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=16000:cl=mono", "-t", "2", str(wav)],
        check=True,
    )
    t = transcribe(str(wav))
    assert t.segments == []
