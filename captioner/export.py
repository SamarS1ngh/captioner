"""Media export helpers (ffmpeg)."""
import os
import shutil
import subprocess
from pathlib import Path

# run.sh (Task 9) provides the vendored binaries under vendor/ffmpeg/ if missing.
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "ffmpeg"


def _resolve(env_var: str, name: str) -> str:
    env = os.environ.get(env_var)
    if env:
        return env
    vendored = _VENDOR / name
    if vendored.exists():
        return str(vendored)
    return shutil.which(name) or f"/opt/homebrew/bin/{name}"


def _ffmpeg() -> str:
    return _resolve("CAPTIONER_FFMPEG", "ffmpeg")


def _ffprobe() -> str:
    return _resolve("CAPTIONER_FFPROBE", "ffprobe")


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract 16 kHz mono WAV audio from a video file."""
    cmd = [_ffmpeg(), "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", out_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) extracting audio from "
            f"{video_path}:\n{result.stderr[-2000:]}"
        )


def _escape_filter_path(path: str) -> str:
    """Escape a filename for the ``subtitles`` filter option (two parse levels).

    Level 1 (option value): escape backslash, ':' and "'". Level 2 (filtergraph):
    escape backslash, "'", ',', '[', ']', ';'. Applying both in order yields a
    value that unescapes back to the literal path.
    """
    def esc(text: str, chars: str) -> str:
        text = text.replace("\\", "\\\\")
        for ch in chars:
            text = text.replace(ch, "\\" + ch)
        return text

    return esc(esc(path, ":'"), "',[];")


def burn_captions(video_path: str, ass_path: str, out_path: str) -> None:
    """Burn an ASS subtitle file onto a video, preserving resolution."""
    vf = f"subtitles=filename={_escape_filter_path(ass_path)}"
    cmd = [
        _ffmpeg(), "-y", "-i", video_path, "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) burning captions into "
            f"{video_path}:\n{result.stderr[-2000:]}"
        )
