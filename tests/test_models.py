from captioner.models import Segment, Transcript, Word


def test_transcript_roundtrips_through_dict():
    t = Transcript(
        segments=[
            Segment(
                text="namaste duniya",
                start=0.0,
                end=1.5,
                words=[Word(text="नमस्ते", start=0.0, end=0.7), Word(text="दुनिया", start=0.8, end=1.5)],
            )
        ],
        language="hi",
    )
    assert Transcript.from_dict(t.to_dict()) == t


def test_segment_dict_keys():
    seg = Segment(text="x", start=0.0, end=1.0, words=[Word(text="x", start=0.0, end=1.0)])
    assert set(seg.to_dict().keys()) == {"text", "start", "end", "words", "word_styles"}


def test_word_styles_roundtrip_with_string_keys():
    seg = Segment(text="a b", start=0.0, end=1.0,
                  words=[Word("a", 0.0, 0.5), Word("b", 0.5, 1.0)],
                  word_styles={1: {"font": "Impact", "size": 90}})
    # simulate the JSON trip: keys become strings, from_dict coerces back to int
    d = seg.to_dict()
    d["word_styles"] = {"1": {"font": "Impact", "size": 90}}
    assert Segment.from_dict(d).word_styles == {1: {"font": "Impact", "size": 90}}
