import json
import subprocess

import pytest

from captioner.ass import build_ass
from captioner.export import _ffmpeg, _ffprobe, burn_captions
from captioner.models import Segment, Transcript, Word

LANDSCAPE = "tests/fixtures/sample_hinglish.mp4"


def _size(path):
    out = subprocess.run(
        [_ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def _transcript():
    return Transcript(
        [
            Segment("hello world", 0.0, 1.0, [Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)]),
            Segment("kya hal", 1.0, 2.0, [Word("kya", 1.0, 1.5), Word("hal", 1.5, 2.0)]),
        ],
        "hi",
    )


@pytest.fixture
def vertical_clip(tmp_path):
    p = tmp_path / "vertical.mp4"
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=2",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "2",
         "-c:v", "libx264", "-c:a", "aac", str(p)],
        capture_output=True, check=True,
    )
    return str(p)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["landscape", "vertical"])
def test_burn_produces_playable_output(kind, tmp_path, vertical_clip):
    src = LANDSCAPE if kind == "landscape" else vertical_clip
    # space + apostrophe + colon + comma exercise filter-path escaping
    ass_dir = tmp_path / "my dir's: a,b"
    ass_dir.mkdir()
    ass = ass_dir / "c.ass"
    ass.write_text(build_ass(_transcript()), encoding="utf-8")
    out = tmp_path / "out.mp4"

    burn_captions(src, str(ass), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert _size(out) == _size(src)


@pytest.mark.slow
def test_burn_raises_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        burn_captions(str(tmp_path / "missing.mp4"), str(tmp_path / "x.ass"), str(tmp_path / "o.mp4"))
