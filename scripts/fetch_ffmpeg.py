#!/usr/bin/env python3
"""Download a static ffmpeg + ffprobe (with libass) for the current OS/arch.

Writes them into ``bin/`` next to the repo root so the PyInstaller build and the
local launcher can bundle/use them. Idempotent: skips tools already present.

Source: ffmpeg.martin-riedl.de static builds (same family already used by run.sh
for macOS arm64; their Linux/Windows builds include libass too).
"""
import io
import os
import platform
import stat
import sys
import urllib.request
import zipfile

BASE = "https://ffmpeg.martin-riedl.de/redirect/latest"


def target() -> tuple[str, str, str]:
    """(os_part, arch, exe_suffix) for the machine we're running on."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        return "macos", arch, ""
    if system == "Linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
        return "linux", arch, ""
    if system == "Windows":
        return "windows", "amd64", ".exe"
    sys.exit(f"Unsupported OS for packaging: {system}")


def main() -> None:
    os_part, arch, suffix = target()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
    os.makedirs(out, exist_ok=True)
    for tool in ("ffmpeg", "ffprobe"):
        dest = os.path.join(out, tool + suffix)
        if os.path.exists(dest):
            print(f"  have {dest}")
            continue
        url = f"{BASE}/{os_part}/{arch}/release/{tool}.zip"
        print(f"  fetch {url}")
        data = urllib.request.urlopen(url, timeout=180).read()  # noqa: S310 (trusted host)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(
                n for n in z.namelist()
                if n.rstrip("/").endswith(tool) or n.endswith(tool + suffix)
            )
            with z.open(member) as src, open(dest, "wb") as f:
                f.write(src.read())
        if suffix == "":
            os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
