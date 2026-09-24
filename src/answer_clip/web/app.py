"""FastAPI app: list videos, ask, clip (localhost demo)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from answer_clip.clip import ClipError, export_clip
from answer_clip.paths import SEGMENTS_FILENAME, data_dir
from answer_clip.query.ask import AskError, ask


class AskBody(BaseModel):
    video_id: str
    question: str
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    pad_sec: float = Field(default=0.0, ge=0.0)
    use_llm: bool = False


class ClipBody(BaseModel):
    video_id: str
    start: float
    end: float
    pad_sec: float = Field(default=0.0, ge=0.0)


def list_video_ids(data_root: Path) -> list[str]:
    videos = data_root / "videos"
    if not videos.is_dir():
        return []
    ids: list[str] = []
    for child in sorted(videos.iterdir()):
        if child.is_dir() and (child / SEGMENTS_FILENAME).is_file():
            ids.append(child.name)
    return ids


def create_app(*, data_root: Path | None = None) -> FastAPI:
    """Build the localhost demo app.

    ``data_root`` is the data directory itself (same as CLI ``--data-dir``).
    """
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    templates = Jinja2Templates(
        directory=str(Path(__file__).resolve().parent / "templates")
    )

    app = FastAPI(
        title="answer-clip",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.data_root = root

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "video_ids": list_video_ids(root),
                "data_root": str(root),
            },
        )

    @app.get("/api/videos")
    def api_videos() -> dict:
        return {"video_ids": list_video_ids(root), "data_root": str(root)}

    @app.post("/api/ask")
    def api_ask(body: AskBody) -> JSONResponse:
        try:
            result = ask(
                body.video_id,
                body.question,
                data_root=root,
                top_k=body.top_k,
                pad_sec=body.pad_sec,
                use_llm=body.use_llm,
                mode=body.mode,  # type: ignore[arg-type]
            )
        except AskError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/ask/form")
    def api_ask_form(
        video_id: str = Form(...),
        question: str = Form(...),
        mode: str = Form("hybrid"),
        top_k: int = Form(5),
    ) -> JSONResponse:
        return api_ask(
            AskBody(video_id=video_id, question=question, mode=mode, top_k=top_k)
        )

    @app.post("/api/clip")
    def api_clip(body: ClipBody) -> JSONResponse:
        if body.end <= body.start:
            raise HTTPException(status_code=400, detail="end must be greater than start")
        try:
            job = export_clip(
                body.video_id,
                start=body.start,
                end=body.end,
                pad_sec=body.pad_sec,
                data_root=root,
            )
        except ClipError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = job.model_dump(mode="json")
        out = Path(job.output_path)
        payload["download_url"] = f"/files/clips/{body.video_id}/{out.name}"
        return JSONResponse(content=payload)

    @app.get("/files/clips/{video_id}/{filename}")
    def download_clip(video_id: str, filename: str) -> FileResponse:
        if "/" in video_id or ".." in video_id or "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid path")
        path = (root / "videos" / video_id / "clips" / filename).resolve()
        clips_root = (root / "videos" / video_id / "clips").resolve()
        if not str(path).startswith(str(clips_root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="clip not found")
        return FileResponse(path, filename=filename)

    return app
