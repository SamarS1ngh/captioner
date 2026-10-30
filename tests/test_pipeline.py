import pytest

from captioner import pipeline
from captioner.models import Segment, Transcript, Word


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(pipeline.MissingKeyError):
        pipeline.process_video("tests/fixtures/sample_hinglish.mp4", api_key="")


def test_routes_to_groq_with_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    calls = {}
    monkeypatch.setattr(pipeline, "extract_audio",
                        lambda v, w: open(w, "wb").close())
    fake = Transcript([Segment("main office ja raha hu", 0.0, 1.0,
                               [Word("x", 0.0, 1.0)])], "hi")

    def fake_groq(wav, key, **kw):
        calls["key"] = key
        calls.update(kw)
        return fake

    monkeypatch.setattr(pipeline.groq_engine, "transcribe_and_romanize", fake_groq)
    out = pipeline.process_video("tests/fixtures/sample_hinglish.mp4", api_key="gsk_test")
    assert out is fake
    assert calls["key"] == "gsk_test"


def test_env_key_used_when_arg_empty(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    monkeypatch.setattr(pipeline, "extract_audio", lambda v, w: open(w, "wb").close())
    seen = {}
    monkeypatch.setattr(pipeline.groq_engine, "transcribe_and_romanize",
                        lambda wav, key, **kw: seen.setdefault("key", key) or
                        Transcript([], "hi"))
    pipeline.process_video("tests/fixtures/sample_hinglish.mp4")
    assert seen["key"] == "gsk_env"


def test_language_and_romanize_passed_through(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    monkeypatch.setattr(pipeline, "extract_audio", lambda v, w: open(w, "wb").close())
    got = {}
    monkeypatch.setattr(pipeline.groq_engine, "transcribe_and_romanize",
                        lambda wav, key, **kw: got.update(kw) or Transcript([], "ta"))
    pipeline.process_video("tests/fixtures/sample_hinglish.mp4",
                           language="ta", romanize=False)
    assert got == {"language": "ta", "romanize": False}
