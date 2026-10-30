"""Desktop launcher for Captioner.

Double-clickable entry point used by the packaged apps (PyInstaller). It points
the pipeline at the bundled ffmpeg/ffprobe, starts the local FastAPI server on a
free port, opens the browser, and keeps running until the window is closed.

Running it directly from a checkout works too (``python launcher.py``); it just
uses whatever ffmpeg CAPTIONER_FFMPEG / vendor/ / PATH resolve to.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
import webbrowser

# A GUI launch (double-click on macOS / pythonw on Windows) hands the process no
# console: sys.stdout / sys.stderr are None, so any print() or logging write would
# raise and kill the server before it binds. Send all output to a log file instead.
LOG_PATH = os.path.join(tempfile.gettempdir(), "captioner.log")


def _ensure_streams() -> None:
    needs_file = sys.stdout is None or sys.stderr is None
    if not needs_file:
        return
    log = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def _base_dir() -> str:
    # PyInstaller unpacks data/binaries next to this at runtime (sys._MEIPASS)
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _wire_bundled_ffmpeg(base: str) -> None:
    """Point the exporter at the ffmpeg/ffprobe shipped inside the app, if present."""
    exe = ".exe" if os.name == "nt" else ""
    frozen = getattr(sys, "frozen", False)
    for env_var, name in (("CAPTIONER_FFMPEG", "ffmpeg"), ("CAPTIONER_FFPROBE", "ffprobe")):
        path = os.path.join(base, "bin", name + exe)
        if os.path.exists(path) and not os.environ.get(env_var):
            # Only chmod when running from a plain checkout; never touch files inside
            # a packaged (signed) bundle — modifying them breaks the signature.
            if not frozen and os.name != "nt":
                try:
                    os.chmod(path, 0o755)
                except OSError:
                    pass
            os.environ[env_var] = path


def _pick_port(preferred: int = 8000) -> int:
    # Walk a fixed ladder before giving up to a random port, so the origin (and
    # therefore the saved API key in localStorage) stays stable across launches.
    candidates = list(range(preferred, preferred + 16)) + [0]
    for port in candidates:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    return preferred


def main() -> None:
    _ensure_streams()
    base = _base_dir()
    _wire_bundled_ffmpeg(base)

    import uvicorn  # imported here so bundling picks it up
    from captioner.webapp import app

    port = _pick_port(int(os.environ.get("CAPTIONER_PORT") or 8000))
    url = f"http://127.0.0.1:{port}"

    def open_browser() -> None:
        time.sleep(1.3)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    print("\n  Captioner is running.")
    print(f"  Open {url} in your browser (it should open automatically).")
    print("  Keep this window open while you work; close it to quit.\n")
    # log_config=None: don't let uvicorn install its own stdout handlers, which can
    # fail under a GUI launch; our guarded streams handle output.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)


def _guarded_main() -> None:
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # last-resort: a GUI launch has nowhere else to show this
        import traceback
        try:
            with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
                f.write("\n--- Captioner crashed ---\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    _guarded_main()
