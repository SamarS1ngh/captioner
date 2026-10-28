import re

import pytest

from captioner.models import Transcript
from captioner.pipeline import process_video

VIDEO = "tests/fixtures/sample_hinglish.mp4"


@pytest.mark.slow
def test_process_video_end_to_end():
    t = process_video(VIDEO)
    assert isinstance(t, Transcript)
    first = t.segments[0]
    assert not re.search("[ऀ-ॿ]", first.text)
    assert "happy" in first.text.lower()
    assert first.words
    for w in first.words:
        assert w.end >= w.start
