"""Groq cloud engine: whisper-large-v3 transcription + LLM Hinglish cleanup.

Far better romanized Hinglish than the local whisper-turbo + IndicXlit path,
because a large LLM cleans the transcript with context (fixing English words
whisper spelled phonetically, romanizing Hindi naturally, leaving real English
alone). Used when the user supplies a Groq API key; otherwise the local
pipeline is the fallback.

Not offline: audio is sent to Groq. The key is never logged or persisted here.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile

import requests

from captioner.export import _ffmpeg, _ffprobe
from captioner.models import Segment, Transcript, Word

API_ROOT = "https://api.groq.com/openai/v1"
ASR_MODEL = "whisper-large-v3"          # full model: cleaner Devanagari than turbo
CLEANUP_MODEL = "openai/gpt-oss-120b"   # large model: no hallucination/translation
# Groq caps the upload per request; long audio is split into chunks and stitched
# back together with each chunk's start time added to its timestamps.
CHUNK_SECS = 600          # ~10 min of 16 kHz mono FLAC stays well under the cap
_CHUNK_SLOP = 15          # don't chunk something only a little over the limit
# urllib's default UA is 403'd by Groq's edge; requests' UA is fine, but be explicit.
_HEADERS_UA = {"User-Agent": "captioner/1.0"}

CLEANUP_SYSTEM = (
    "You convert auto-transcribed Hindi subtitles into natural ROMANIZED HINGLISH, "
    "the way Indian creators type on Instagram/YouTube. Rules: Hindi words -> casual "
    "Roman Latin (hai, nahi, matlab, kyunki, karna). English words -> correct English "
    "spelling (the transcript may spell English phonetically in Devanagari - fix them: "
    "विटनेस्ट -> witnessed, आईफोन -> iPhone). Keep the meaning and word order. Never "
    "translate Hindi into English. If a line is already correct English, leave it "
    "unchanged. Output ONLY the cleaned line - no quotes, no notes, no explanation."
)


class GroqError(RuntimeError):
    pass


def _transcribe(audio_path: str, api_key: str) -> dict:
    """Groq whisper-large-v3 -> verbose_json with word + segment timestamps."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{API_ROOT}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}", **_HEADERS_UA},
            data={
                "model": ASR_MODEL,
                "language": "hi",
                "response_format": "verbose_json",
                "temperature": "0",
                "timestamp_granularities[]": ["word", "segment"],
            },
            files={"file": ("audio.flac", f, "audio/flac")},
            timeout=300,
        )
    if resp.status_code != 200:
        raise GroqError(f"transcription failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _probe_duration(path: str) -> float:
    """Audio duration in seconds, or 0.0 if it can't be determined."""
    try:
        out = subprocess.run(
            [_ffprobe(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _split(path: str, secs: int):
    """Split audio into <=secs FLAC chunks. Returns (tmpdir, [(offset, path), ...])."""
    tmpdir = tempfile.mkdtemp(prefix="capchunk_")
    pattern = os.path.join(tmpdir, "chunk_%04d.flac")
    subprocess.run(
        [_ffmpeg(), "-y", "-i", path, "-vn", "-ar", "16000", "-ac", "1",
         "-f", "segment", "-segment_time", str(secs), "-c:a", "flac", pattern],
        capture_output=True, text=True,
    )
    files = sorted(glob.glob(os.path.join(tmpdir, "chunk_*.flac")))
    # audio has no keyframes, so each segment is exactly `secs` long
    return tmpdir, [(i * secs, f) for i, f in enumerate(files)]


def _offset(items: list, offset: float, out: list) -> None:
    for it in items:
        it = dict(it)
        if it.get("start") is not None:
            it["start"] = float(it["start"]) + offset
        if it.get("end") is not None:
            it["end"] = float(it["end"]) + offset
        out.append(it)


def _collect(audio_path: str, key: str):
    """Transcribe, chunking long audio. Returns (words, segments, language)."""
    dur = _probe_duration(audio_path)
    if dur <= CHUNK_SECS + _CHUNK_SLOP:
        d = _transcribe(audio_path, key)
        return d.get("words") or [], d.get("segments") or [], d.get("language", "hi")
    tmpdir, parts = _split(audio_path, CHUNK_SECS)
    try:
        if not parts:  # split produced nothing -> try the whole file once
            d = _transcribe(audio_path, key)
            return d.get("words") or [], d.get("segments") or [], d.get("language", "hi")
        words_all, segs_all, lang = [], [], "hi"
        for offset, cpath in parts:
            d = _transcribe(cpath, key)
            lang = d.get("language", lang)
            _offset(d.get("words") or [], offset, words_all)
            _offset(d.get("segments") or [], offset, segs_all)
        return words_all, segs_all, lang
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _cleanup(text: str, api_key: str, model: str = CLEANUP_MODEL) -> str:
    resp = requests.post(
        f"{API_ROOT}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", **_HEADERS_UA},
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": CLEANUP_SYSTEM},
                {"role": "user", "content": text},
            ],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise GroqError(f"cleanup failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def _bucket_words(words: list[dict], start: float, end: float) -> list[Word]:
    """Words (flat, timestamped) that fall inside [start, end)."""
    out = []
    for w in words:
        ws = w.get("start")
        if ws is None:
            continue
        if start <= ws < end or (ws <= start and w.get("end", ws) > start):
            out.append(Word(text=(w.get("word") or "").strip(),
                            start=float(ws), end=float(w.get("end", ws))))
    return out


def transcribe_and_romanize(audio_path: str, api_key: str) -> Transcript:
    """Full Groq path: audio -> cleaned, word-timestamped Hinglish Transcript.

    Segment.text is the romanized Hinglish (from the LLM); Segment.words keep the
    original word timestamps so karaoke timing (distribute_timing) still works
    after the text is reworded.
    """
    key = (api_key or "").strip()
    if not key:
        raise GroqError("no API key")
    words, raw_segments, language = _collect(audio_path, key)
    segments: list[Segment] = []
    for s in raw_segments:
        raw_text = (s.get("text") or "").strip()
        if not raw_text:
            continue
        start, end = float(s["start"]), float(s["end"])
        cleaned = _cleanup(raw_text, key) or raw_text
        seg_words = _bucket_words(words, start, end)
        segments.append(Segment(text=cleaned, start=start, end=end, words=seg_words))
    return Transcript(segments=segments, language=language)


def key_from_env() -> str:
    return os.environ.get("GROQ_API_KEY", "")
