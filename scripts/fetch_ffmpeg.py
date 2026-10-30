#!/usr/bin/env python3
"""Download a static ffmpeg + ffprobe (with libass) for the current OS/arch.

Writes them into ``bin/`` next to the repo root so the PyInstaller build and the
local launcher can bundle/use them. Idempotent: skips tools already present.

Sources (both static, libass-enabled):
- macOS / Linux: ffmpeg.martin-riedl.de (per-tool zips)
- Windows:       BtbN/FFmpeg-Builds GitHub release (one zip with both exes)
"""
import io
import os
import platform
import stat
import sys
import urllib.request
import zipfile

MARTIN = "https://ffmpeg.martin-riedl.de/redirect/latest"
BTBN_WIN = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
            "ffmpeg-master-latest-win64-gpl.zip")
_UA = {"User-Agent": "Mozilla/5.0 (captioner-build)"}  # default urllib UA is 403'd


def _bin_dir() -> str:
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
    os.makedirs(out, exist_ok=True)
    return out


def _get(url: str) -> bytes:
    print(f"  fetch {url}")
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=300).read()  # noqa: S310 (trusted hosts)


def _make_executable(path: str) -> None:
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _fetch_unix(os_part: str, arch: str, out: str) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        dest = os.path.join(out, tool)
        if os.path.exists(dest):
            print(f"  have {dest}")
            continue
        with zipfile.ZipFile(io.BytesIO(_get(f"{MARTIN}/{os_part}/{arch}/release/{tool}.zip"))) as z:
            member = next(n for n in z.namelist() if n.rstrip("/").endswith(tool))
            with z.open(member) as src, open(dest, "wb") as f:
                f.write(src.read())
        _make_executable(dest)
        print(f"  wrote {dest}")


def _fetch_windows(out: str) -> None:
    if all(os.path.exists(os.path.join(out, t + ".exe")) for t in ("ffmpeg", "ffprobe")):
        print("  have ffmpeg.exe + ffprobe.exe")
        return
    with zipfile.ZipFile(io.BytesIO(_get(BTBN_WIN))) as z:  # one zip holds both exes
        for tool in ("ffmpeg", "ffprobe"):
            member = next(n for n in z.namelist() if n.endswith(f"bin/{tool}.exe"))
            dest = os.path.join(out, tool + ".exe")
            with z.open(member) as src, open(dest, "wb") as f:
                f.write(src.read())
            print(f"  wrote {dest}")


def main() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    out = _bin_dir()
    if system == "Darwin":
        _fetch_unix("macos", "arm64" if machine in ("arm64", "aarch64") else "amd64", out)
    elif system == "Linux":
        _fetch_unix("linux", "arm64" if machine in ("aarch64", "arm64") else "amd64", out)
    elif system == "Windows":
        _fetch_windows(out)
    else:
        sys.exit(f"Unsupported OS for packaging: {system}")


if __name__ == "__main__":
    main()
