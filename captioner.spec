# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the Captioner desktop app (Mac / Windows / Linux).

One-dir build off launcher.py, bundling the captioner package, the static UI,
and the ffmpeg/ffprobe binaries fetched into bin/. On macOS it's wrapped as a
.app; on Windows/Linux it's a folder with a console executable (the console
shows the local URL and closing it quits).
"""
import glob
import sys

from PyInstaller.utils.hooks import collect_submodules

is_mac = sys.platform == "darwin"

# ffmpeg/ffprobe (downloaded by scripts/fetch_ffmpeg.py) -> bundled under bin/.
# They're mach-o/ELF/PE executables, so list them as binaries (not datas): that
# preserves the exec bit and lets PyInstaller sign them with the bundle, so we
# never have to chmod inside the signed .app at runtime (which would break the
# signature and get the app killed on macOS).
bin_files = [(f, "bin") for f in glob.glob("bin/*")]
datas = [("captioner/static", "captioner/static")]

# uvicorn resolves its loop/protocol implementations by dynamic import
hiddenimports = collect_submodules("uvicorn") + ["h11"]

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=bin_files,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Captioner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=not is_mac,          # console window on Win/Linux; windowed .app on Mac
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Captioner",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="Captioner.app",
        icon=None,
        bundle_identifier="dev.samarsingh.captioner",
        info_plist={
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
