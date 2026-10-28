# Captioner — Design Document

**Date:** 2026-10-28
**Status:** Draft for approval
**Author:** Design captured via brainstorming with the project owner, informed by a working spike.

---

## 1. Intent & Why

The owner shoots short-form videos (Reels/Shorts-style) in **Hinglish** — Hindi spoken
with English words mixed in ("main office ja raha hoon", "ye reel viral ho gaya"). They
want on-screen captions, synced to speech, **romanized** (Latin script, the way Desi
creators actually caption), generated **for free on their own Mac**.

A funded incumbent (**Kalakar**, kalakar.io, 100k+ users) already does this commercially
with per-month pricing and video-length caps. Captioner is **not** a competitor or a
product to sell. It is a **personal tool** so the owner never has to pay a subscription
or hit a length cap to caption their own videos.

**Why now:** the owner is actively making videos and does not want to pay Kalakar's
tiers. They already build Mac/Swift apps (notchfy), so a Mac tool is in reach.

---

## 2. Goals

1. **Input:** owner drops a video file into the app.
2. **Transcribe** the speech on-device with word-level timestamps.
3. **Romanize** Hindi words to casual Latin; keep English words in correct English spelling.
4. **Editable transcript:** owner reviews and fixes the ~20% of auto-errors (a few seconds
   of typing) before rendering. This is a first-class step, not an afterthought.
5. **Captions render** over the video — a clean readable bar with each word highlighting
   as it is spoken (karaoke style).
6. **Export** a burned-in video at full quality, plus optional `.srt` sidecar.
7. **Free & local:** no subscription, no API cost, no per-video cap, works offline.

### Accuracy bar (owner's explicit priority)
Normal **English and Hindi words must be as accurate as possible.** Slang/swear words
mis-hearing is acceptable ("bhosdike" → garbled is fine). The spike confirmed this bar is
met by model choice (see §5).

---

## 3. Non-Goals

- Not a mobile/iPhone app. (Investigated and rejected — see §7.)
- Not a product to sell, no accounts, no payments, no licensing, no multi-user.
- Not dubbing, translation, or voice cloning. Captions only.
- Not animated/branded caption templates (MrBeast-style packs) in v1. Simple style only.
- Not true lip/viseme sync. "Sync" means **word appears when spoken**, driven by
  word-level timestamps — not mouth-shape matching.
- Not App Store distribution. Runs locally for the owner.

## 4. Anti-Goals (would make this a failure even if it technically works)

- Captions that read like a machine: diacritics (`maiṁ`), schwa junk (`bahuta`, `masta`),
  or English words phonetically mangled (`haapi` for "happy"). If output looks like
  rule-based transliteration, the tool has failed.
- A pipeline with no human-fix step, forcing the owner to accept wrong words.
- A tool so slow or fiddly it's easier to just pay Kalakar.
- Quality loss in the exported video.

---

## 5. What Was Decided (and the spike that proved it)

A working spike ran on a real 20s Hinglish clip on the owner's Mac. **Proven
separately:** transcription quality by model (§5.2), romanization quality including
English-word recovery (§5.3), and that word-level timestamps exist. **Not yet run as
one stitched pipeline:** sentence-level romanization aligned back onto per-word
timestamps (§5.4) — the editor data model below is designed so this seam cannot break
the tool, and the builder validates it first (§8).

### 5.1 Platform: **Mac app, all-Python, local web UI**
- The ML stack (PyTorch) is required by the romanizer regardless of app language, so a
  Swift wrapper would add complexity for no size saving. All-Python is the least-effort
  path and the spike already runs in it.
- UI = a **thin local web UI** (browser): drop video → transcript (editable) → pick style
  → export. The browser previews video far better than a native widget.
- Exported video goes to the owner's Mac; they AirDrop to phone to post (AirDrop is
  lossless — confirmed — so no quality hit).

### 5.2 Transcription: **faster-whisper, `large-v3-turbo` model**
- Spike result on the real clip:
  - `medium`: heard "happy" as `हापि` → wrong English. **Rejected.**
  - `large-v3-turbo`: "Namaste, and happy Diwali" — English words correct, 1.6GB,
    ~2× faster than large-v3. **Chosen** (owner decision: meets the accuracy bar at
    workable speed for clips up to 2 min).
  - `large-v3`: also English-correct, 3GB, slower — kept as a higher-accuracy option the
    owner can switch to if a specific clip needs it.
- **VAD filter is required** — without it, music-only sections produced garbage (`।`).
- `word_timestamps=True` gives the per-word timing that drives karaoke highlight.
- **Known quirk:** large models sometimes over-translate a Hindi function word to English
  (`aur` → `and`). Mitigation: a Hinglish `initial_prompt`, plus the editable step catches
  the rest.

### 5.3 Romanization: **AI4Bharat IndicXlit (indic→en direction)**
- Converts Devanagari → casual Latin **and recovers English words to real spelling**
  (office, phone, reel, seriously) — the thing rule-based transliteration (indic-
  transliteration ITRANS/IAST) totally failed at. Rule-based rejected in spike.
- **Romanize at sentence level, then align back to word timings.** Per-word romanization
  is worse (loses context: `chala` instead of `chal`).
- Torch 2.6 loads the fairseq checkpoint with `weights_only=True` by default and fails;
  the loader must force `weights_only=False` (the model is a trusted AI4Bharat download).
- Auto-accuracy ~80%. The remaining errors (`haapi`→happy, `chala`→chal, `ye`→`yeye`,
  `hai`→`haye`) are fixed in the editable step.

### 5.4 Editable transcript step (mandatory) + caption/timing data model
After transcribe+romanize, show the owner the result as a list of **caption segments**
(one Whisper segment ≈ one on-screen line). This is the decided data model (builder's
call to refine, settled here because it de-risks the §5 alignment seam):

- **Edit unit = the segment's romanized text**, a free text field. The owner fixes
  `haapi`→happy, `chala`→chal, `aur` wrongly shown as `and`, or adds/removes a word —
  without being constrained to a 1:1 word grid. This is how the tool clears the accuracy
  bar that automation cannot, and satisfies the anti-goal against being forced to accept
  wrong words.
- **Karaoke timing is derived, not hand-aligned.** Each segment carries its start/end and
  Whisper's per-word timestamps. After the owner edits the roman text, split it into
  tokens and distribute the segment's time span across them (map by index to the original
  word timings when counts match; otherwise distribute proportionally by token length).
  Because timing is *derived* from the segment span, an N:M mismatch between Devanagari
  tokens and roman tokens degrades highlight precision slightly but never drops a word or
  breaks the caption.
- **Fallback** if per-word karaoke proves unreliable in validation: show the whole
  segment at once (no per-word highlight). Captions still work; only the highlight is lost.

### 5.5 Caption style: **Simple (option A)**
- One clean style: readable text bar, current word highlighted as spoken.
- Burned in via **ffmpeg** (already installed) using an **ASS** subtitle track — ASS
  supports the per-word karaoke highlight and styling for free.
- Export full-quality video + optional `.srt` sidecar.
- Style **presets** (fonts/colors/positions, Kalakar-style packs) are explicitly deferred
  to a later version.

---

## 6. Constraints

- **Local & free only.** No paid APIs, no cloud calls, no telemetry.
- **Owner's Mac, Apple Silicon.** Models: `large-v3-turbo` (~1.6GB) + IndicXlit (~800MB)
  + torch (~1.5GB) ≈ 2–2.5GB on disk. Acceptable (one machine, one-time). `large-v3`
  (~3GB) only if the owner opts into it for a clip.
- **Max clip length = 2 minutes** (owner's typical use). This is the runtime target:
  transcribe + romanize + burn a 2-min clip should finish fast enough that the owner
  won't reach for Kalakar instead.
- **CPU inference is too slow at this length.** Builder should use **MLX-Whisper**
  (Apple Silicon GPU/ANE) as the default runtime, or faster-whisper with the best compute
  type, to hit the 2-min target. Validate actual runtime on a real 2-min clip.
- **Python 3.9** is the system Python; the spike used a venv. Builder may choose a newer
  Python via the project venv.
- ffmpeg present at `/opt/homebrew/bin/ffmpeg`; brew available.

---

## 7. Investigated & Rejected

- **iPhone app (on-device):** rejected. IndicXlit is a fairseq/PyTorch model with no iOS
  runtime; converting it to CoreML is high-risk and uncertain. Mac removes the wall.
- **iPhone + server romanize step:** rejected — breaks offline/free intent.
- **Rule-based transliteration (indic-transliteration):** rejected — produced
  `maiM ऑphisa jA rahA hU~`, unusable.
- **Whisper `small`/`medium`:** rejected on the accuracy bar (English words wrong).

---

## 8. Left to the Builder

The design fixes the pipeline and the choices above. These are the builder's calls,
to settle sensibly without changing what gets built:

- **Local UI framework** — e.g. FastAPI + a static page, Gradio, Streamlit, or similar.
  Pick the lightest that gives drag-drop, an editable word list, a video preview, and a
  download. (Owner leaned "thin local web UI"; exact framework is open.)
- **Whisper runtime for speed** — faster-whisper vs MLX-Whisper on Apple Silicon; pick
  what makes large-v3 (or turbo) tolerable, per §6.
- **ASS generation details** — exact font, size, highlight color, position, safe-area
  margins for vertical video. Start with one tasteful default; presets are later.
- **Word-timing → ASS karaoke** mapping per the §5.4 model (distribute segment span over
  edited roman tokens). **Validate this seam first** on a real clip before building the UI
  around it; if per-word highlight is unreliable, take the §5.4 segment-at-once fallback.
- **`initial_prompt` wording** to reduce the `aur→and` over-translation.
- **Project layout, venv, and a one-command run** (e.g. `./run.sh` opens the local UI).
- **Vertical vs landscape** handling and how captions position for each.

---

## 9. Resolved with the Owner

- **Max clip length = 2 minutes.** → runtime target in §6; MLX-Whisper is the default
  runtime.
- **Default model = `large-v3-turbo`** (§5.2); `large-v3` kept as an opt-in for a clip
  that needs it.

No open questions blocking the plan.
