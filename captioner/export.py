"""Media export helpers (ffmpeg)."""
import shutil
import subprocess


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract 16 kHz mono WAV audio from a video file."""
    cmd = [_ffmpeg(), "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", out_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) extracting audio from "
            f"{video_path}:\n{result.stderr[-2000:]}"
        )
