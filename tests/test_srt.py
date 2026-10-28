from captioner.models import Segment, Transcript, Word
from captioner.srt import build_srt


def test_srt_cue_format():
    seg = Segment("hello", 1.0, 2.5, [Word("hello", 1.0, 2.5)])
    t = Transcript([seg], "hi")
    assert build_srt(t) == "1\n00:00:01,000 --> 00:00:02,500\nhello"


def test_multiple_cues_numbered():
    segs = [
        Segment("one", 0.0, 1.0, [Word("one", 0.0, 1.0)]),
        Segment("two", 1.0, 2.0, [Word("two", 1.0, 2.0)]),
    ]
    out = build_srt(Transcript(segs, "hi"))
    indices = [l for l in out.splitlines() if l.isdigit()]
    assert indices == ["1", "2"]
