"""SRT subtitle export for a romanized Transcript."""
from __future__ import annotations

from captioner.models import Transcript


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timecode HH:MM:SS,mmm (comma before milliseconds)."""
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(t: Transcript) -> str:
    """One numbered SRT cue per segment, cues separated by a blank line."""
    cues = []
    for index, seg in enumerate(t.segments, start=1):
        cues.append(
            f"{index}\n"
            f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n"
            f"{seg.text.strip()}"
        )
    return "\n\n".join(cues)
