from pathlib import Path

from fastapi.testclient import TestClient

from captioner import webapp

client = TestClient(webapp.app)


def test_root_serves_page():
    r = client.get("/")
    assert r.status_code == 200
    assert 'type="file"' in r.text


def test_export_builds_from_edited_segments(tmp_path, monkeypatch):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    webapp.JOBS["stub"] = {"dir": tmp_path, "video": video}
    captured = {}

    def fake_burn(video_path, ass_path, out_path):
        captured["ass"] = Path(ass_path).read_text()
        Path(out_path).write_bytes(b"mp4")

    monkeypatch.setattr(webapp, "burn_captions", fake_burn)
    segments = [
        {"text": "Namaste dosto", "start": 0.0, "end": 1.0,
         "words": [{"text": "Namaste", "start": 0.0, "end": 0.5},
                   {"text": "dosto", "start": 0.5, "end": 1.0}]},
        {"text": "EDITED line here", "start": 1.0, "end": 2.0,
         "words": [{"text": "orig", "start": 1.0, "end": 2.0}]},
    ]
    r = client.post("/export", json={
        "job": "stub", "segments": segments,
        "style": {"font_size": 60, "highlight_color": "&H0000E5FF"},
        "want_srt": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["download"] == "/download/stub"
    assert "EDITED" in captured["ass"] and "orig" not in captured["ass"]
    assert "Fontname" in captured["ass"] and ",60," in captured["ass"]
    assert body["srt"] == "/download/stub?kind=srt"
    srt = client.get(body["srt"])
    assert srt.status_code == 200 and srt.text.endswith("\n")
    assert "EDITED line here" in srt.text
    assert client.get(body["download"]).content == b"mp4"


def test_export_unknown_job():
    r = client.post("/export", json={"job": "nope", "segments": []})
    assert r.status_code == 404
