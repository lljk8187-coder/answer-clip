"""Register a local video under ``data/videos/<id>/`` with ffprobe metadata."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from answer_clip.ffprobe import FfprobeError, probe
from answer_clip.models import CopyStrategy, VideoMeta
from answer_clip.paths import META_FILENAME, data_dir


class IngestError(ValueError):
    """Invalid path or failed registration."""


def content_id(path: Path, *, length: int = 12) -> str:
    """Stable id from SHA-256 of file contents (first ``length`` hex chars)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _place_media(src: Path, dest: Path) -> CopyStrategy:
    """Prefer hardlink on the same filesystem; fall back to ``copy2``.

    If ``dest`` already exists and is not the same inode as ``src``, it is
    replaced so re-ingest refreshes the stored media.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if dest.samefile(src):
                return "hardlink"
        except OSError:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def ingest(
    source: str | Path,
    *,
    data_root: Path | None = None,
) -> VideoMeta:
    """Copy/hardlink ``source`` into the data store and write ``meta.json``.

    Layout::

        <data_root>/videos/<id>/<original-filename>
        <data_root>/videos/<id>/meta.json

    ``data_root`` defaults to ``paths.data_dir()`` (``./data`` or
    ``$ANSWER_CLIP_DATA``). Large media under ``data/`` stays gitignored.
    """
    src = Path(source).expanduser()
    if not src.exists():
        raise IngestError(f"path does not exist: {src}")
    if not src.is_file():
        raise IngestError(f"not a file: {src}")

    src = src.resolve()
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    video_id = content_id(src)
    filename = src.name
    vdir = root / "videos" / video_id
    dest = vdir / filename
    meta_path = vdir / META_FILENAME

    try:
        probed = probe(src)
    except FfprobeError:
        raise

    strategy = _place_media(src, dest)
    stored_relpath = f"videos/{video_id}/{filename}"

    meta = VideoMeta(
        id=video_id,
        source_path=str(src),
        filename=filename,
        stored_relpath=stored_relpath,
        copy_strategy=strategy,
        duration_sec=probed.duration_sec,
        width=probed.width,
        height=probed.height,
        video_codec=probed.video_codec,
        audio_codec=probed.audio_codec,
        fps=probed.fps,
        size_bytes=src.stat().st_size,
    )
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return meta
