"""Local web UI: upload -> transcribe -> edit -> burn captions."""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from captioner.ass import AssStyle, build_ass, get_style, preset_names
from captioner.export import burn_captions
from captioner.groq_engine import LANG_MENU
from captioner.models import Transcript
from captioner.pipeline import process_video
from captioner.srt import build_srt

STATIC = Path(__file__).parent / "static"
WORK_ROOT = Path(tempfile.gettempdir()) / "captioner-jobs"

app = FastAPI(title="Captioner")

# job id -> {"dir": Path, "video": Path, "out": Path, "srt": Path | None}
JOBS: dict[str, dict[str, Any]] = {}


class ExportRequest(BaseModel):
    job: str
    segments: list[dict]
    preset: str = "classic"
    style: dict = {}  # optional per-field overrides on top of the preset
    want_srt: bool = False
    language: Optional[str] = "hi"


@app.get("/")
def index() -> FileResponse:
    # no-store: the UI is iterated often; never let the browser serve a stale page
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/styles")
def styles() -> dict:
    # full field set per preset so the browser preview can render it faithfully
    return {
        "presets": [
            {"name": name, **dataclasses.asdict(get_style(name))}
            for name in preset_names()
        ],
        "languages": LANG_MENU,
    }


@app.post("/process")
async def process(
    video: UploadFile = File(...),
    api_key: str = Form(""),
    language: str = Form("hi"),
    romanize: str = Form("true"),
) -> dict:
    lang = (language or "").strip()
    if lang in ("", "auto"):
        lang = None
    rom = str(romanize).strip().lower() not in ("false", "0", "no", "off", "")
    job = uuid.uuid4().hex
    work = WORK_ROOT / job
    work.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "").suffix or ".mp4"
    path = work / f"input{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        transcript = await run_in_threadpool(
            process_video, str(path), api_key, lang, rom
        )
    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Processing failed: {exc}")
    engine = "groq" if (api_key.strip() or os.environ.get("GROQ_API_KEY")) else "local"
    JOBS[job] = {"dir": work, "video": path}
    return {"job": job, "transcript": transcript.to_dict(), "engine": engine}


@app.post("/export")
async def export(req: ExportRequest) -> dict:
    info = JOBS.get(req.job)
    if info is None:
        raise HTTPException(404, "Unknown job")
    allowed = {f.name for f in dataclasses.fields(AssStyle)}
    overrides = {k: v for k, v in req.style.items() if k in allowed}
    style = dataclasses.replace(get_style(req.preset), **overrides)
    transcript = Transcript.from_dict(
        {"segments": req.segments, "language": req.language or "hi"}
    )
    work: Path = info["dir"]
    ass_path = work / "captions.ass"
    out_path = work / "captioned.mp4"
    ass_path.write_text(build_ass(transcript, style), encoding="utf-8")
    try:
        await run_in_threadpool(
            burn_captions, str(info["video"]), str(ass_path), str(out_path)
        )
    except Exception as exc:
        raise HTTPException(500, f"Burn failed: {exc}")
    info["out"] = out_path
    result = {"download": f"/download/{req.job}"}
    if req.want_srt:
        srt_path = work / "captions.srt"
        srt_path.write_text(build_srt(transcript) + "\n", encoding="utf-8")
        info["srt"] = srt_path
        result["srt"] = f"/download/{req.job}?kind=srt"
    return result


@app.get("/download/{job}")
def download(job: str, kind: str = "video") -> FileResponse:
    info = JOBS.get(job)
    key = "srt" if kind == "srt" else "out"
    if info is None or key not in info or not Path(info[key]).exists():
        raise HTTPException(404, "Not found")
    path = Path(info[key])
    media = "application/x-subrip" if key == "srt" else "video/mp4"
    return FileResponse(path, media_type=media, filename=f"captioned{path.suffix}")
