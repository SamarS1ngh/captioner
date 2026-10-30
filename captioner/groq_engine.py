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
import re
import shutil
import subprocess
import tempfile
import time

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
# Romanize all segments together in one LLM request per batch, not one per
# segment: a short clip has dozens of segments and per-segment calls blow past
# Groq's requests-per-minute cap (429).
_BATCH = 40
_NUM_PREFIX = re.compile(r"^\s*\d+\s*[\|.):]\s*")
# urllib's default UA is 403'd by Groq's edge; requests' UA is fine, but be explicit.
_HEADERS_UA = {"User-Agent": "captioner/1.0"}

# Groq whisper-large-v3 language codes we surface in the UI (code -> display name).
# "" / None = auto-detect. Any ISO code whisper supports also works if passed through.
LANGUAGES: dict[str, str] = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu",
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "ru": "Russian", "ar": "Arabic", "tr": "Turkish",
    "id": "Indonesian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
}


# The language menu shown in the UI. Each entry picks a whisper language code and
# whether to romanize the output to Latin ("-glish") or keep the native script.
LANG_MENU: list[dict] = [
    {"id": "hinglish", "label": "Hinglish", "code": "hi", "romanize": True},
    {"id": "hindi", "label": "Hindi", "code": "hi", "romanize": False},
    {"id": "english", "label": "English", "code": "en", "romanize": False},
    {"id": "tanglish", "label": "Tanglish (Tamil)", "code": "ta", "romanize": True},
    {"id": "tamil", "label": "Tamil", "code": "ta", "romanize": False},
    {"id": "tenglish", "label": "Tenglish (Telugu)", "code": "te", "romanize": True},
    {"id": "telugu", "label": "Telugu", "code": "te", "romanize": False},
    {"id": "banglish", "label": "Banglish (Bengali)", "code": "bn", "romanize": True},
    {"id": "bengali", "label": "Bengali", "code": "bn", "romanize": False},
    {"id": "punglish", "label": "Punjabi (Latin)", "code": "pa", "romanize": True},
    {"id": "punjabi", "label": "Punjabi", "code": "pa", "romanize": False},
    {"id": "marathi", "label": "Marathi", "code": "mr", "romanize": False},
    {"id": "gujarati", "label": "Gujarati", "code": "gu", "romanize": False},
    {"id": "kannada", "label": "Kannada", "code": "kn", "romanize": False},
    {"id": "malayalam", "label": "Malayalam", "code": "ml", "romanize": False},
    {"id": "urdu", "label": "Urdu", "code": "ur", "romanize": False},
    {"id": "spanish", "label": "Spanish", "code": "es", "romanize": False},
    {"id": "french", "label": "French", "code": "fr", "romanize": False},
    {"id": "german", "label": "German", "code": "de", "romanize": False},
    {"id": "portuguese", "label": "Portuguese", "code": "pt", "romanize": False},
    {"id": "italian", "label": "Italian", "code": "it", "romanize": False},
    {"id": "russian", "label": "Russian", "code": "ru", "romanize": False},
    {"id": "arabic", "label": "Arabic", "code": "ar", "romanize": False},
    {"id": "japanese", "label": "Japanese", "code": "ja", "romanize": False},
    {"id": "korean", "label": "Korean", "code": "ko", "romanize": False},
    {"id": "chinese", "label": "Chinese", "code": "zh", "romanize": False},
    {"id": "auto", "label": "Auto-detect", "code": "", "romanize": False},
]


def _cleanup_system(language: str, batch: bool = False) -> str:
    """System prompt that romanizes a given language into natural Latin script,
    keeping code-switched English correct (the Hinglish approach, generalized)."""
    name = LANGUAGES.get(language or "", "the source language")
    base = (
        f"You convert auto-transcribed {name} subtitles into natural ROMANIZED text, "
        "the way creators type on Instagram/YouTube. Rules: native-language words -> "
        "casual Latin letters, written phonetically and naturally. English (or other "
        "already-Latin) words -> correct English spelling (the transcript may spell them "
        "phonetically in the native script - fix those). Keep the meaning and word order. "
        "Never translate into English. If a line is already correct English, leave it "
        "unchanged."
    )
    if batch:
        return base + (
            " The input is numbered lines, each like `3| text`. Return EXACTLY the same "
            "number of lines, each prefixed with its own number and `| `, in the same "
            "order. Clean each line's text per the rules above. Never merge, split, add, "
            "reorder, or drop lines. Output ONLY the numbered lines - no notes."
        )
    return base + " Output ONLY the cleaned line - no quotes, no notes, no explanation."


class GroqError(RuntimeError):
    pass


def _post_with_retry(url: str, *, max_tries: int = 5, **kw):
    """POST, retrying on HTTP 429 (rate limit) with exponential backoff."""
    delay = 2.0
    resp = None
    for attempt in range(max_tries):
        resp = requests.post(url, **kw)
        if resp.status_code != 429 or attempt == max_tries - 1:
            return resp
        time.sleep(delay)
        delay = min(delay * 2, 20)
    return resp


def _transcribe(audio_path: str, api_key: str, language: str = "hi") -> dict:
    """Groq whisper-large-v3 -> verbose_json with word + segment timestamps.

    `language` is an ISO code; falsy means let whisper auto-detect.
    """
    data = {
        "model": ASR_MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language:
        data["language"] = language
    with open(audio_path, "rb") as f:
        resp = _post_with_retry(
            f"{API_ROOT}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}", **_HEADERS_UA},
            data=data,
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


def _collect(audio_path: str, key: str, language: str = "hi"):
    """Transcribe, chunking long audio. Returns (words, segments, language)."""
    dur = _probe_duration(audio_path)
    if dur <= CHUNK_SECS + _CHUNK_SLOP:
        d = _transcribe(audio_path, key, language)
        return d.get("words") or [], d.get("segments") or [], d.get("language") or language or "hi"
    tmpdir, parts = _split(audio_path, CHUNK_SECS)
    try:
        if not parts:  # split produced nothing -> try the whole file once
            d = _transcribe(audio_path, key, language)
            return d.get("words") or [], d.get("segments") or [], d.get("language") or language or "hi"
        words_all, segs_all, lang = [], [], (language or "hi")
        for offset, cpath in parts:
            d = _transcribe(cpath, key, language)
            lang = d.get("language") or lang
            _offset(d.get("words") or [], offset, words_all)
            _offset(d.get("segments") or [], offset, segs_all)
        return words_all, segs_all, lang
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _cleanup(text: str, api_key: str, system: str, model: str = CLEANUP_MODEL) -> str:
    resp = _post_with_retry(
        f"{API_ROOT}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", **_HEADERS_UA},
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise GroqError(f"cleanup failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def _romanize_lines(texts: list[str], api_key: str, system: str) -> list[str]:
    """Romanize many segment texts with as few LLM calls as possible.

    Sends up to _BATCH segments per request (numbered lines in, numbered lines
    out). If a batch's response doesn't line up 1:1, that batch falls back to the
    raw native-script text rather than risk misaligning captions.
    """
    out: list[str] = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i:i + _BATCH]
        numbered = "\n".join(f"{j + 1}| {t}" for j, t in enumerate(chunk))
        cleaned = _cleanup(numbered, api_key, system)
        lines = [_NUM_PREFIX.sub("", ln).strip()
                 for ln in cleaned.splitlines() if ln.strip()]
        out.extend(lines if len(lines) == len(chunk) else chunk)
    return out


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


def transcribe_and_romanize(audio_path: str, api_key: str,
                            language: str = "hi", romanize: bool = True) -> Transcript:
    """Full Groq path: audio -> cleaned, word-timestamped Transcript.

    `language` is an ISO code (falsy = auto-detect). When `romanize` is True the
    transcript is cleaned into natural Latin script (the Hinglish approach, for any
    language); when False the native-script whisper text is kept verbatim.

    Segment.text is the (cleaned) line; Segment.words keep the original word
    timestamps so karaoke timing (distribute_timing) still works after rewording.
    """
    key = (api_key or "").strip()
    if not key:
        raise GroqError("no API key")
    words, raw_segments, detected = _collect(audio_path, key, language)
    # (segment dict, raw text) for non-empty segments only
    kept = [(s, (s.get("text") or "").strip()) for s in raw_segments]
    kept = [(s, t) for s, t in kept if t]
    raw_texts = [t for _, t in kept]
    if romanize:
        system = _cleanup_system(language or detected, batch=True)
        cleaned_texts = _romanize_lines(raw_texts, key, system)
    else:
        cleaned_texts = raw_texts
    segments: list[Segment] = []
    for (s, raw_text), cleaned in zip(kept, cleaned_texts):
        start, end = float(s["start"]), float(s["end"])
        seg_words = _bucket_words(words, start, end)
        segments.append(Segment(text=cleaned or raw_text, start=start, end=end, words=seg_words))
    return Transcript(segments=segments, language=detected)


def key_from_env() -> str:
    return os.environ.get("GROQ_API_KEY", "")
