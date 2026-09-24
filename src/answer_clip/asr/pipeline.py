"""Resolve media + write segments.json / transcript.srt."""

from __future__ import annotations

import json
from pathlib import Path

from answer_clip.asr.base import AsrBackend, AsrBackendError, get_backend
from answer_clip.asr.normalize import normalize_segments
from answer_clip.asr.srt import segments_to_srt
from answer_clip.models import SubtitleSegment, VideoMeta
from answer_clip.paths import (
    META_FILENAME,
    SEGMENTS_FILENAME,
    SRT_FILENAME,
    data_dir,
    model_cache_dir,
    video_dir,
)


class AsrError(ValueError):
    """Invalid video id / path or failed ASR pipeline."""


def _find_media_in_dir(vdir: Path) -> Path:
    meta_path = vdir / META_FILENAME
    if meta_path.is_file():
        meta = VideoMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        candidate = vdir / meta.filename
        if candidate.is_file():
            return candidate
        # Fall back to stored_relpath basename under this dir.
        stored = Path(meta.stored_relpath).name
        candidate = vdir / stored
        if candidate.is_file():
            return candidate
    media_files = sorted(
        p
        for p in vdir.iterdir()
        if p.is_file() and p.name not in {META_FILENAME, SEGMENTS_FILENAME, SRT_FILENAME}
    )
    if not media_files:
        raise AsrError(f"no media file found under {vdir}")
    return media_files[0]


def resolve_target(
    *,
    video_id: str | None = None,
    path: str | Path | None = None,
    data_root: Path | None = None,
) -> tuple[Path, Path, str | None]:
    """Return ``(media_path, output_dir, video_id_or_none)``."""
    if bool(video_id) == bool(path):
        raise AsrError("provide exactly one of video_id or --path")

    root = Path(data_root).resolve() if data_root is not None else data_dir()

    if video_id:
        vdir = root / "videos" / video_id
        if not vdir.is_dir():
            raise AsrError(f"video id not found under data store: {video_id} ({vdir})")
        media = _find_media_in_dir(vdir)
        return media, vdir, video_id

    media = Path(path).expanduser().resolve()
    if not media.is_file():
        raise AsrError(f"path does not exist or is not a file: {media}")

    # If the file already lives in data/videos/<id>/..., use that folder.
    try:
        videos_root = (root / "videos").resolve()
        rel = media.relative_to(videos_root)
        parts = rel.parts
        if len(parts) >= 2:
            vid = parts[0]
            return media, videos_root / vid, vid
    except ValueError:
        pass

    return media, media.parent, None


def write_outputs(
    output_dir: Path,
    segments: list[SubtitleSegment],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = normalize_segments(segments)
    seg_path = output_dir / SEGMENTS_FILENAME
    srt_file = output_dir / SRT_FILENAME
    payload = [s.model_dump(mode="json") for s in segments]
    seg_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    srt_file.write_text(segments_to_srt(segments), encoding="utf-8")
    return seg_path, srt_file


def run_asr(
    *,
    video_id: str | None = None,
    path: str | Path | None = None,
    data_root: Path | None = None,
    backend: str = "faster-whisper",
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    download_root: Path | None = None,
    backend_impl: AsrBackend | None = None,
) -> dict:
    """Transcribe and write ``segments.json`` + ``transcript.srt``.

    ``backend_impl`` injects a mock/real backend (tests); otherwise resolve by name.
    """
    media, out_dir, vid = resolve_target(
        video_id=video_id, path=path, data_root=data_root
    )
    cache = download_root
    if cache is None and data_root is not None:
        cache = Path(data_root) / "models"
    elif cache is None:
        cache = model_cache_dir(data_root)

    engine = backend_impl or get_backend(
        backend,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        download_root=cache,
        language=language,
    )
    try:
        segments = engine.transcribe(media)
    except AsrBackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AsrBackendError(f"ASR backend failed: {exc}") from exc

    seg_path, srt_file = write_outputs(out_dir, segments)
    return {
        "video_id": vid,
        "media_path": str(media),
        "output_dir": str(out_dir),
        "segments_path": str(seg_path),
        "srt_path": str(srt_file),
        "segment_count": len(segments),
        "backend": getattr(engine, "name", backend),
        "segments": [s.model_dump(mode="json") for s in segments],
    }
