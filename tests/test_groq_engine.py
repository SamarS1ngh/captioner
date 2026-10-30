import json

import pytest

from captioner import groq_engine


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


TRANSCRIBE_PAYLOAD = {
    "language": "hi",
    "segments": [
        {"text": "मैं office जा रहा हूँ", "start": 0.0, "end": 1.0},
        {"text": "और happy diwali", "start": 1.0, "end": 2.0},
    ],
    "words": [
        {"word": "मैं", "start": 0.0, "end": 0.3},
        {"word": "office", "start": 0.3, "end": 0.6},
        {"word": "जा", "start": 0.6, "end": 1.0},
        {"word": "और", "start": 1.0, "end": 1.4},
        {"word": "happy", "start": 1.4, "end": 2.0},
    ],
}


def _patch(monkeypatch, cleanup_map):
    def fake_post(url, **kw):
        if url.endswith("/audio/transcriptions"):
            return FakeResp(200, TRANSCRIBE_PAYLOAD)
        if url.endswith("/chat/completions"):
            text = kw["json"]["messages"][-1]["content"]
            return FakeResp(200, {"choices": [{"message": {"content": cleanup_map[text]}}]})
        raise AssertionError(url)
    monkeypatch.setattr(groq_engine.requests, "post", fake_post)


def test_transcribe_and_romanize(monkeypatch, tmp_path):
    cleanup = {"मैं office जा रहा हूँ": "main office ja raha hu",
               "और happy diwali": "aur happy diwali"}
    _patch(monkeypatch, cleanup)
    wav = tmp_path / "a.wav"; wav.write_bytes(b"x")
    t = groq_engine.transcribe_and_romanize(str(wav), "gsk_test")
    assert t.language == "hi"
    assert [s.text for s in t.segments] == ["main office ja raha hu", "aur happy diwali"]
    # words bucketed into the right segments by timestamp, spans preserved
    assert [w.text for w in t.segments[0].words] == ["मैं", "office", "जा"]
    assert [w.text for w in t.segments[1].words] == ["और", "happy"]
    assert t.segments[0].words[0].start == 0.0


def test_no_key_raises(tmp_path):
    wav = tmp_path / "a.wav"; wav.write_bytes(b"x")
    with pytest.raises(groq_engine.GroqError):
        groq_engine.transcribe_and_romanize(str(wav), "")


def test_api_error_surfaces(monkeypatch, tmp_path):
    monkeypatch.setattr(groq_engine.requests, "post",
                        lambda url, **kw: FakeResp(401, {"error": "bad key"}))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"x")
    with pytest.raises(groq_engine.GroqError):
        groq_engine.transcribe_and_romanize(str(wav), "gsk_bad")
