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
    assert set(seg.to_dict().keys()) == {"text", "start", "end", "words"}
