# Captioner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A free, local Mac tool that auto-captions the owner's Hinglish videos with romanized subtitles synced to speech, with a human edit step before burn-in.

**Architecture:** All-Python. A pure core (data model → timing → ASS/SRT) with TDD, wrapped by two ML adapters (faster-whisper transcription, IndicXlit romanization) and an ffmpeg export layer, driven by a thin local FastAPI web UI. The risky seam — mapping an owner-edited romanized line back onto per-word timestamps — is pure logic built and tested early (Task 2).

**Tech Stack:** Python 3.11+ (venv), faster-whisper (`large-v3-turbo`), ai4bharat-transliteration (IndicXlit indic→en), ffmpeg (ASS burn-in), FastAPI + uvicorn, vanilla JS front-end.

**Spec:** `docs/specs/2026-10-28-captioner-design.md`

## Global Constraints

- **Local & free only.** No network calls at runtime; models downloaded once and cached. No telemetry, no paid APIs.
- **Transcription model:** faster-whisper `large-v3-turbo`, `vad_filter=True`, `word_timestamps=True`, `language="hi"`, `task="transcribe"`, with a Hinglish `initial_prompt`. `large-v3` is a selectable fallback.
- **Runtime target:** a 2-minute clip (the max supported) must transcribe+romanize+burn fast enough to beat reaching for Kalakar. Measure on a real 2-min clip (Task 4, Task 8). If faster-whisper misses the target, the documented lever is swapping to mlx-whisper (requires separate VAD) — a later optimization, not v1 scope.
- **Romanize at segment (sentence) level**, never per-word (per-word loses context: `chala` vs `chal`).
- **IndicXlit loading:** patch `torch.load` to `weights_only=False` before importing/loading the fairseq checkpoint (trusted AI4Bharat model; torch ≥2.6 defaults to True and fails).
- **ffmpeg** at `/opt/homebrew/bin/ffmpeg`; burning subtitles re-encodes video at visually-lossless quality (`-crf 18`, preserve resolution/aspect), audio stream copied.
- **Caption style v1:** one clean readable style with per-word karaoke highlight; style presets are out of scope.

## Review Focus

- **Edited line token count ≠ word-timestamp count (the N:M seam):** owner adds/removes words; timing must still produce a valid per-token span and never crash or drop a word. → Task 2.
- **Video with no speech / music only:** VAD returns no segments; the app shows an empty transcript rather than crashing. → Task 4.
- **Pure-English (or English-heavy) speech:** romanization leaves real English words intact, does not mangle them. → Task 3.
- **Vertical vs landscape video:** burn-in preserves the source resolution/aspect and places captions inside the lower safe area for both. → Task 8.
- **ASS-breaking characters in caption text** (`{`, `}`, `\`, newlines): escaped so they render as text, not as override tags. → Task 6.

---

### Task 1: Project scaffold + data model

**Files:**
- Create: `captioner/__init__.py`, `captioner/models.py`, `requirements.txt`, `.gitignore`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Word(text: str, start: float, end: float)`, `Segment(text: str, start: float, end: float, words: list[Word])`, `Transcript(segments: list[Segment], language: str)` — all frozen dataclasses with `to_dict()`/`from_dict()` for the web layer. `Segment.text` is the romanized line and the owner's edit unit; `Segment.words` carries the original per-word time spans.

- [ ] **Step 1: Write failing tests** in `tests/test_models.py`: `test_transcript_roundtrips_through_dict()` builds a `Transcript` with one `Segment` and two `Word`s, asserts `Transcript.from_dict(t.to_dict()) == t`; `test_segment_dict_keys()` asserts `to_dict()` has keys `text,start,end,words`.
- [ ] **Step 2: Run, verify fail** — `pytest tests/test_models.py -v` → FAIL (module/classes not defined).
- [ ] **Step 3: Implement** the three dataclasses + `to_dict`/`from_dict` in `captioner/models.py`. Create `requirements.txt` (faster-whisper, ai4bharat-transliteration, fastapi, uvicorn, python-multipart, pytest) and `.gitignore` (venv, `__pycache__`, `models/`, `*.mp4`, `*.wav`, output dir).
- [ ] **Step 4: Run, verify pass** — `pytest tests/test_models.py -v` → PASS.
- [ ] **Step 5: Commit** — `git init` if needed, then `feat: project scaffold and transcript data model`.

---

### Task 2: Timing distribution (the N:M seam)

**Files:**
- Create: `captioner/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `distribute_timing(start: float, end: float, tokens: list[str], word_spans: list[tuple[float, float]]) -> list[Word]` — returns one `Word` per token with monotonic, non-overlapping `[start,end]` inside `[start,end]`. When `len(tokens) == len(word_spans)`, maps 1:1. Otherwise distributes `[start,end]` across tokens proportional to `len(token)` (min width guard so empty/short tokens still get a slice).

- [ ] **Step 1: Write failing tests** in `tests/test_timing.py`:
  - `test_one_to_one_uses_word_spans()`: 3 tokens, 3 spans → each `Word` gets the matching span.
  - `test_more_tokens_than_spans_distributes_proportionally()`: 4 tokens, 2 spans → 4 Words, first.start==start, last.end==end, spans monotonic non-overlapping.
  - `test_fewer_tokens_than_spans()`: 1 token, 3 spans → 1 Word spanning `[start,end]`.
  - `test_empty_tokens_returns_empty()`: `[]` tokens → `[]`.
- [ ] **Step 2: Run, verify fail** — `pytest tests/test_timing.py -v` → FAIL.
- [ ] **Step 3: Implement** `distribute_timing` in `captioner/timing.py`. Approach: if counts equal, zip; else cumulative proportional split of `(end-start)` by token character length with a small floor per token.
- [ ] **Step 4: Run, verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat: timing distribution for edited caption lines`.

---

### Task 3: Romanization adapter (IndicXlit)

**Files:**
- Create: `captioner/romanize.py`
- Test: `tests/test_romanize.py`

**Interfaces:**
- Consumes: `Transcript`, `Segment` (Task 1).
- Produces: `romanize_transcript(t: Transcript) -> Transcript` — returns a new `Transcript` where each `Segment.text` is the casual-roman transliteration of its Devanagari/English content (English words preserved); `start`, `end`, `words` unchanged. Lazily constructs one module-level `XlitEngine("hi", beam_width=10, rescore=False, src_script_type="indic")`.

- [ ] **Step 1: Write failing tests** in `tests/test_romanize.py` (these load the real model; mark `@pytest.mark.slow`):
  - `test_devanagari_line_romanizes()`: segment text `"मैं ऑफिस जा रहा हूँ"` → result contains `"office"` and starts with `"main"`.
  - `test_pure_english_preserved()`: segment text `"happy diwali let's go"` → result lowercased-equals the input (English left intact, Review Focus).
  - `test_timings_unchanged()`: `words`/`start`/`end` identical before and after.
- [ ] **Step 2: Run, verify fail** → FAIL (function not defined).
- [ ] **Step 3: Implement** `romanize_transcript`. At module import, patch `torch.load` to force `weights_only=False`, then import `XlitEngine`. Romanize each `Segment.text` with `translit_sentence(text, lang_code="hi")`.
- [ ] **Step 4: Run, verify pass** — `pytest tests/test_romanize.py -v -m slow` → PASS.
- [ ] **Step 5: Commit** — `feat: IndicXlit romanization adapter`.

---

### Task 4: Transcription adapter (faster-whisper)

**Files:**
- Create: `captioner/transcribe.py`
- Test: `tests/test_transcribe.py` (integration, uses a real fixture clip)

**Interfaces:**
- Consumes: `models.py`.
- Produces: `transcribe(audio_path: str, model_name: str = "large-v3-turbo") -> Transcript` — raw (pre-romanize) segments with Devanagari/English text and per-word spans. Empty audio/no speech → `Transcript(segments=[], language=...)`.

- [ ] **Step 1: Write failing tests** in `tests/test_transcribe.py` (`@pytest.mark.slow`; fixture: the owner's 20s clip's extracted wav, copied into `tests/fixtures/`):
  - `test_transcribes_known_clip()`: result's joined text contains `"happy"` and `"Namaste"` (case-insensitive); at least one segment has non-empty `words` with increasing `start`.
  - `test_silence_returns_empty_transcript()`: a 2s silent wav (generate with ffmpeg in the test) → `segments == []`, no exception (Review Focus: no-speech).
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `transcribe` using `WhisperModel(model_name, device="auto", compute_type="int8")`, `transcribe(..., vad_filter=True, word_timestamps=True, language="hi", task="transcribe", initial_prompt=<Hinglish prompt biasing against aur→and over-translation>)`. Map segments/words into `Transcript`.
- [ ] **Step 4: Run, verify pass** — `pytest tests/test_transcribe.py -v -m slow` → PASS.
- [ ] **Step 5: Measure runtime** — time `transcribe` on a real ~2-min clip; record seconds in a comment. If far over the §Global runtime target, note it in the commit body as the trigger for the mlx-whisper lever (do not switch now).
- [ ] **Step 6: Commit** — `feat: faster-whisper transcription adapter`.

---

### Task 5: Pipeline (video → romanized transcript)

**Files:**
- Create: `captioner/pipeline.py`
- Modify: `captioner/export.py` is created in Task 8; here add only `extract_audio` (create `export.py` with just that function, Task 8 extends it).
- Test: `tests/test_pipeline.py` (integration)

**Interfaces:**
- Consumes: `transcribe` (Task 4), `romanize_transcript` (Task 3).
- Produces: `extract_audio(video_path: str, out_wav: str) -> None` (ffmpeg 16k mono wav); `process_video(video_path: str, model_name: str = "large-v3-turbo") -> Transcript` (extract audio to a temp wav → transcribe → romanize).

- [ ] **Step 1: Write failing test** in `tests/test_pipeline.py` (`@pytest.mark.slow`): `test_process_video_end_to_end()` on the fixture clip → returns a `Transcript` whose first segment `text` is romanized Latin (no Devanagari codepoints) and contains `"happy"`; `words` present with timings. This exercises the full transcribe→romanize→timing-ready seam on a real clip (spec §5 validation).
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `extract_audio` (subprocess ffmpeg `-ar 16000 -ac 1 -vn`) and `process_video` (tempfile wav, call Task 4 then Task 3).
- [ ] **Step 4: Run, verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat: video-to-romanized-transcript pipeline`.

---

### Task 6: ASS subtitle generation (karaoke)

**Files:**
- Create: `captioner/ass.py`
- Test: `tests/test_ass.py`

**Interfaces:**
- Consumes: `Transcript`, `Segment` (Task 1), `distribute_timing` (Task 2).
- Produces: `AssStyle` dataclass (font name, size, primary color, highlight color, outline, vertical `margin_v`, `alignment=2` bottom-center) with a sensible default; `build_ass(t: Transcript, style: AssStyle = AssStyle()) -> str` — a full ASS document. Each segment is one Dialogue line; words carry `\k<centiseconds>` karaoke tags computed from `distribute_timing(seg.start, seg.end, seg.text.split(), [(w.start,w.end) for w in seg.words])`. Caption text is ASS-escaped.

- [ ] **Step 1: Write failing tests** in `tests/test_ass.py`:
  - `test_ass_has_script_info_and_dialogue()`: output contains `[Script Info]`, `[V4+ Styles]`, `[Events]`, and one `Dialogue:` per segment.
  - `test_dialogue_timecodes_match_segment()`: a segment `[1.0,2.5]` → Dialogue start `0:00:01.00` end `0:00:02.50`.
  - `test_karaoke_durations_sum_to_segment()`: sum of `\k` centiseconds in a line ≈ segment duration in cs (±1).
  - `test_escapes_brace_characters()`: segment text `"a {b} c"` → output has no raw `{b}` override; braces escaped (Review Focus).
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `AssStyle` + `build_ass` in `captioner/ass.py`. Provide an ASS timecode formatter and a brace/backslash escaper.
- [ ] **Step 4: Run, verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat: ASS karaoke subtitle generation`.

---

### Task 7: SRT export

**Files:**
- Create: `captioner/srt.py`
- Test: `tests/test_srt.py`

**Interfaces:**
- Consumes: `Transcript` (Task 1).
- Produces: `build_srt(t: Transcript) -> str` — one SRT cue per segment (index, `HH:MM:SS,mmm` timecodes, `Segment.text`).

- [ ] **Step 1: Write failing tests** in `tests/test_srt.py`: `test_srt_cue_format()` (segment `[1.0,2.5]`, text "hello" → `1\n00:00:01,000 --> 00:00:02,500\nhello`); `test_multiple_cues_numbered()` (two segments → indices 1,2).
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `build_srt` in `captioner/srt.py`.
- [ ] **Step 4: Run, verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat: SRT export`.

---

### Task 8: Burn-in export (ffmpeg)

**Files:**
- Modify: `captioner/export.py` (add `burn_captions`)
- Test: `tests/test_export.py` (integration)

**Interfaces:**
- Consumes: `extract_audio` already in `export.py` (Task 5); ASS string (Task 6).
- Produces: `burn_captions(video_path: str, ass_path: str, out_path: str) -> None` — ffmpeg `subtitles` filter burns the ASS onto the video, re-encode `-crf 18 -preset medium`, `-c:a copy`, output preserves source width/height.

- [ ] **Step 1: Write failing tests** in `tests/test_export.py` (`@pytest.mark.slow`): `test_burn_produces_playable_output()` — burn a 2-line ASS onto a short fixture clip → output file exists, non-zero size, and `ffprobe` reports the same width/height as the source (Review Focus: vertical/landscape aspect preserved). Use both a landscape and a synthesized vertical (1080x1920) fixture.
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `burn_captions` (subprocess ffmpeg; escape the ass path for the filter). Default `margin_v` in `AssStyle` keeps text in the lower safe area for vertical.
- [ ] **Step 4: Run, verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat: ffmpeg caption burn-in export`.

---

### Task 9: Local web UI + run script

**Files:**
- Create: `captioner/webapp.py`, `captioner/static/index.html` (inline JS/CSS), `run.sh`
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `process_video` (Task 5), `build_ass`/`AssStyle` (Task 6), `build_srt` (Task 7), `burn_captions` (Task 8).
- Produces: FastAPI app with `GET /` (the page), `POST /process` (multipart video → saves to a work dir → `process_video` → returns `Transcript.to_dict()` + a job id), `POST /export` (job id + edited segments JSON + style + want_srt → `build_ass` → `burn_captions` → returns download URL; also SRT if requested), `GET /download/{job}` (serves the output). Front-end: drag-drop upload, editable list of segment text fields, a Burn button, a video/download result, optional "also save .srt".

- [ ] **Step 1: Write failing tests** in `tests/test_webapp.py` using `fastapi.testclient`: `test_root_serves_page()` (`GET /` → 200, HTML contains the upload control); `test_export_builds_from_edited_segments()` — POST `/export` with a stub job and two hand-written edited segments + style, monkeypatching `burn_captions` to touch an output file → 200 and a download URL; asserts the ASS passed to `burn_captions` reflects the edited text (proves the owner's edits drive the burn — spec §5.4).
- [ ] **Step 2: Run, verify fail** → FAIL.
- [ ] **Step 3: Implement** `webapp.py` + `index.html`. Keep a simple in-memory/work-dir job map keyed by uuid. Write `run.sh`: create venv if missing, `pip install -r requirements.txt`, launch `uvicorn captioner.webapp:app`, open `http://127.0.0.1:8000`.
- [ ] **Step 4: Run, verify pass** — `pytest tests/test_webapp.py -v` → PASS.
- [ ] **Step 5: Manual smoke** — `./run.sh`, drop the 20s fixture clip, edit a word, burn, confirm the exported video plays with highlighted captions. Record result.
- [ ] **Step 6: Commit** — `feat: local web UI and run script`.

---

## Notes for the executor

- Mark model-loading tests `@pytest.mark.slow`; register the marker in `pyproject.toml`/`pytest.ini`. Fast suite (Tasks 1,2,6,7 + webapp) runs without models; `-m slow` runs the real-model/integration tests.
- First run downloads `large-v3-turbo` (~1.6GB) and IndicXlit (~800MB) to cache; that is expected and one-time.
- Do not add network calls, accounts, or paid services. Free and offline is a hard requirement.
