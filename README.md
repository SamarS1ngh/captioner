# Captioner

Put romanized **Hinglish** (and many other languages) karaoke captions on your
videos, locally. Upload a clip, fix the lines, pick a style, burn them in. The
video never leaves your machine — only the audio goes to [Groq](https://console.groq.com)
to transcribe.

It's a small local web app: a Python server you run on your own computer and use
in your browser. Nothing is hosted; nothing is public.

---

## Just use it (no setup) — the app

Download the app for your system from the [Releases](https://github.com/samarsingh-winit/captioner/releases)
page, unzip, and open it. It starts the local server and opens your browser.

| System | Download | First launch |
|--------|----------|--------------|
| **macOS** (Apple Silicon) | `Captioner-macos-arm64.zip` | Right-click the app → **Open** → **Open** (one time — it's unsigned). |
| **Windows** | `Captioner-windows-x64.zip` | Double-click `Captioner.exe`. On the blue "Windows protected your PC" box: **More info → Run anyway**. |
| **Linux** | `Captioner-linux-x64.zip` | `chmod +x Captioner/Captioner` then run it. |

You'll also need a **free Groq API key** (2-minute signup at
[console.groq.com](https://console.groq.com/keys)). Paste it into the app once;
it's stored only in your browser and sent only to Groq.

Those one-time "unknown developer" prompts are expected — the apps aren't paid
code-signed. Keep the window open while you work; closing it quits the server.

---

## Run from source

Needs Python 3.9+ and an internet connection (it fetches a static ffmpeg on
first run).

```bash
# macOS
./run.sh
```

```bash
# any OS, manually
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python scripts/fetch_ffmpeg.py        # downloads ffmpeg + ffprobe into bin/
python launcher.py                    # opens http://127.0.0.1:8000
```

Tests:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

---

## Build the apps yourself

The desktop apps are built by [GitHub Actions](.github/workflows/build-apps.yml)
on macOS, Windows, and Linux runners (PyInstaller can't cross-compile, so each
OS builds its own). Trigger it from the **Actions** tab, or push a `v*` tag to
also attach the zips to a Release.

To build for your current OS locally:

```bash
pip install -r requirements.txt pyinstaller
python scripts/fetch_ffmpeg.py
pyinstaller --noconfirm captioner.spec
# result in dist/
```

---

## How it works

- **Frontend** — one static HTML/CSS/JS page served by the backend.
- **Backend** — FastAPI + uvicorn (`captioner/webapp.py`).
- **ffmpeg** — extracts 16 kHz FLAC audio and burns the ASS captions (libass).
- **Groq** — `whisper-large-v3` transcribes; `gpt-oss-120b` romanizes the text
  naturally (the "-glish" styles) or it's kept in native script.

Long videos are chunked automatically to stay under Groq's request size limit.
