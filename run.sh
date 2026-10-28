#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  /usr/bin/python3 -m venv .venv
fi
.venv/bin/python -m pip install -q 'pip<24.1'
.venv/bin/python -m pip install -q -r requirements.txt fastapi uvicorn python-multipart httpx

BASE="https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release"
mkdir -p vendor/ffmpeg
for tool in ffmpeg ffprobe; do
  if [ ! -x "vendor/ffmpeg/$tool" ]; then
    echo "Downloading $tool..."
    tmp="$(mktemp -d)"
    curl -fL "$BASE/$tool.zip" -o "$tmp/$tool.zip"
    unzip -o -q "$tmp/$tool.zip" -d vendor/ffmpeg
    rm -rf "$tmp"
    chmod +x "vendor/ffmpeg/$tool"
    xattr -dr com.apple.quarantine "vendor/ffmpeg/$tool" 2>/dev/null || true
  fi
done

(sleep 2 && open "http://127.0.0.1:8000") &
exec .venv/bin/python -m uvicorn captioner.webapp:app --host 127.0.0.1 --port 8000
