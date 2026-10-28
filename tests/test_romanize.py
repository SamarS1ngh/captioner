import pytest

from captioner.models import Segment, Transcript, Word
from captioner.romanize import romanize_transcript


def _t(text):
    words = [Word("w", 0.0, 0.5), Word("x", 0.5, 1.0)]
    return Transcript(segments=[Segment(text, 0.0, 1.0, words)], language="hi")


@pytest.mark.slow
def test_devanagari_line_romanizes():
    out = romanize_transcript(_t("मैं ऑफिस जा रहा हूँ")).segments[0].text.lower()
    assert "office" in out
    assert out.startswith("main")


@pytest.mark.slow
def test_pure_english_preserved():
    out = romanize_transcript(_t("happy diwali let's go")).segments[0].text
    assert out.lower() == "happy diwali let's go"


@pytest.mark.slow
def test_timings_unchanged():
    t = _t("मैं ऑफिस जा रहा हूँ")
    out = romanize_transcript(t)
    s, o = t.segments[0], out.segments[0]
    assert (o.start, o.end, o.words) == (s.start, s.end, s.words)
    assert out.language == t.language
