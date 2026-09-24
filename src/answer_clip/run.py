"""One-shot pipeline: ingest → asr → ask → clip."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from answer_clip.asr.base import AsrBackend, AsrBackendError
from answer_clip.asr.pipeline import AsrError, run_asr
from answer_clip.clip import ClipError, export_clip
from answer_clip.ingest import IngestError, ingest
from answer_clip.models import ExportJob, QueryResult, VideoMeta
from answer_clip.query.ask import AskError, ask


class RunError(RuntimeError):
    """End-to-end pipeline failure."""


def _ensure_faster_whisper_installed() -> None:
    if importlib.util.find_spec("faster_whisper") is None:
        raise RunError(
            'ASR dependency missing: install with pip install "answer-clip[asr]" '
            "(also requires ffmpeg/ffprobe on PATH)."
        )


def run_pipeline(
    video_path: str | Path,
    question: str,
    *,
    data_root: Path | None = None,
    backend: str = "faster-whisper",
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    download_root: Path | None = None,
    backend_impl: AsrBackend | None = None,
    top_k: int = 5,
    pad_sec: float = 0.0,
    use_llm: bool | None = False,
    hit: int = 0,
    out: str | Path | None = None,
    skip_clip: bool = False,
) -> dict[str, Any]:
    """Run ingest → asr → ask → clip and return a summary dict.

    ``use_llm`` defaults to ``False`` for deterministic one-shot runs.
    Pass ``None`` to auto-enable when an API key is present, or ``True``
    to force an attempt (still sets ``llm_skipped`` without a key).

    ``backend_impl`` injects a mock/real ASR backend (tests).
    """
    q = (question or "").strip()
    if not q:
        raise RunError("question must be non-empty")

    try:
        meta: VideoMeta = ingest(video_path, data_root=data_root)
    except IngestError as exc:
        raise RunError(f"ingest failed: {exc}") from exc

    backend_key = backend.strip().lower().replace("_", "-")
    if backend_impl is None and backend_key in {"faster-whisper", "whisper"}:
        _ensure_faster_whisper_installed()

    try:
        asr_info = run_asr(
            video_id=meta.id,
            data_root=data_root,
            backend=backend,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            backend_impl=backend_impl,
        )
    except (AsrError, AsrBackendError) as exc:
        msg = str(exc)
        if "answer-clip[asr]" in msg or "faster-whisper is not installed" in msg:
            raise RunError(
                f"asr failed: {msg}. "
                'Install ASR extras: pip install "answer-clip[asr]"'
            ) from exc
        raise RunError(f"asr failed: {exc}") from exc

    try:
        result: QueryResult = ask(
            meta.id,
            q,
            data_root=data_root,
            top_k=top_k,
            pad_sec=pad_sec,
            use_llm=use_llm,
        )
    except AskError as exc:
        raise RunError(f"ask failed: {exc}") from exc

    clip_job: ExportJob | None = None
    if not skip_clip:
        if not result.hits:
            raise RunError(
                "ask returned no hits; cannot clip. "
                "Try different keywords or check ASR segments."
            )
        if hit < 0 or hit >= len(result.hits):
            raise RunError(
                f"hit index {hit} out of range (hits={len(result.hits)})"
            )
        chosen = result.hits[hit]
        try:
            clip_job = export_clip(
                meta.id,
                start=chosen.start,
                end=chosen.end,
                pad_sec=0.0,
                out=out,
                data_root=data_root,
                hit_index=hit,
            )
        except ClipError as exc:
            raise RunError(f"clip failed: {exc}") from exc

    return {
        "video_id": meta.id,
        "meta": meta.model_dump(mode="json"),
        "asr": {k: v for k, v in asr_info.items() if k != "segments"},
        "ask": result.model_dump(mode="json"),
        "clip": clip_job.model_dump(mode="json") if clip_job else None,
        "hit_index": hit if clip_job else None,
    }
