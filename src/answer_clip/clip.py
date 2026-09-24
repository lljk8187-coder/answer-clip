"""ffmpeg clip export for ingested videos."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from answer_clip.ffprobe import probe
from answer_clip.models import ExportJob, QueryResult, VideoMeta
from answer_clip.paths import (
    CLIPS_DIRNAME,
    META_FILENAME,
    SEGMENTS_FILENAME,
    SRT_FILENAME,
    clips_dir,
    data_dir,
)


class ClipError(ValueError):
    """Invalid args or failed clip export."""


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise ClipError(
            "ffmpeg not found on PATH. Install ffmpeg and retry."
        )
    return path


def find_media(video_id: str, *, data_root: Path | None = None) -> Path:
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    vdir = root / "videos" / video_id
    if not vdir.is_dir():
        raise ClipError(f"video id not found: {video_id} ({vdir})")
    meta_path = vdir / META_FILENAME
    if meta_path.is_file():
        meta = VideoMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        candidate = vdir / meta.filename
        if candidate.is_file():
            return candidate
    skip = {META_FILENAME, SEGMENTS_FILENAME, SRT_FILENAME, CLIPS_DIRNAME}
    media_files = sorted(
        p
        for p in vdir.iterdir()
        if p.is_file() and p.name not in skip and p.suffix.lower() not in {".json", ".srt"}
    )
    if not media_files:
        raise ClipError(f"no media file found under {vdir}")
    return media_files[0]


def resolve_span(
    *,
    start: float | None,
    end: float | None,
    pad_sec: float,
    hit_index: int | None,
    ask_result: Path | None,
) -> tuple[float, float, int | None]:
    """Return padded (start, end, hit_index).

    ``hit_index`` alone is metadata when ``start``/``end`` are already set
    (e.g. ``run`` pipeline). Loading from ``ask_result`` only happens when
    times are missing.
    """
    hit_i = hit_index
    if (start is None or end is None) and hit_index is not None:
        if ask_result is None:
            raise ClipError("--hit requires --ask-result pointing to a QueryResult JSON")
        path = Path(ask_result).expanduser()
        if not path.is_file():
            raise ClipError(f"ask result not found: {path}")
        result = QueryResult.model_validate_json(path.read_text(encoding="utf-8"))
        if hit_index < 0 or hit_index >= len(result.hits):
            raise ClipError(
                f"hit index {hit_index} out of range (hits={len(result.hits)})"
            )
        hit = result.hits[hit_index]
        start = float(hit.start)
        end = float(hit.end)
    if start is None or end is None:
        raise ClipError("provide --start and --end, or --hit with --ask-result")
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        raise ClipError(f"end ({end_f}) must be greater than start ({start_f})")
    pad = max(0.0, float(pad_sec))
    start_f = max(0.0, start_f - pad)
    end_f = end_f + pad
    if end_f <= start_f:
        raise ClipError("padded window is empty")
    return start_f, end_f, hit_i


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def export_clip(
    video_id: str,
    *,
    start: float | None = None,
    end: float | None = None,
    pad_sec: float = 0.0,
    out: str | Path | None = None,
    data_root: Path | None = None,
    hit_index: int | None = None,
    ask_result: str | Path | None = None,
    write_sidecar: bool = True,
) -> ExportJob:
    """Cut ``[start, end]`` (plus pad) from an ingested video via ffmpeg.

    Prefers stream copy (``-c copy``); falls back to libx264/aac re-encode.
    """
    ffmpeg = require_ffmpeg()
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    media = find_media(video_id, data_root=root)
    start_f, end_f, hit_i = resolve_span(
        start=start,
        end=end,
        pad_sec=pad_sec,
        hit_index=hit_index,
        ask_result=Path(ask_result) if ask_result else None,
    )
    duration = end_f - start_f

    if out is not None:
        output = Path(out).expanduser().resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = media.suffix or ".mp4"
        name = f"clip_{start_f:.2f}_{end_f:.2f}_{ts}{suffix}"
        output = clips_dir(video_id, root) / name

    output.parent.mkdir(parents=True, exist_ok=True)

    base = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_f:.3f}",
        "-i",
        str(media),
        "-t",
        f"{duration:.3f}",
    ]
    copy_cmd = base + ["-c", "copy", "-avoid_negative_ts", "make_zero", str(output)]
    completed = _run_ffmpeg(copy_cmd)
    codec_mode: str = "copy"
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        if output.exists():
            output.unlink(missing_ok=True)
        reenc_cmd = base + [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = _run_ffmpeg(reenc_cmd)
        codec_mode = "reencode"
        if completed.returncode != 0 or not output.is_file():
            err = (completed.stderr or completed.stdout or "").strip()[-800:]
            raise ClipError(f"ffmpeg failed to export clip: {err}")

    actual_duration: float | None = None
    try:
        actual_duration = probe(output).duration_sec
    except Exception:  # noqa: BLE001
        actual_duration = duration

    job = ExportJob(
        video_id=video_id,
        source_path=str(media),
        start=start_f,
        end=end_f,
        pad_sec=pad_sec,
        output_path=str(output),
        duration_sec=actual_duration,
        codec_mode=codec_mode,  # type: ignore[arg-type]
        hit_index=hit_i,
        sidecar_path=None,
    )
    if write_sidecar:
        side = Path(str(output) + ".json")
        job.sidecar_path = str(side)
        side.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return job
