"""Data and cache path helpers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIRNAME = "data"
VIDEOS_DIRNAME = "videos"
META_FILENAME = "meta.json"


def data_dir(root: Path | None = None) -> Path:
    """Return the project data root (``$ANSWER_CLIP_DATA`` or ``<cwd>/data``)."""
    env = os.environ.get("ANSWER_CLIP_DATA")
    if env:
        return Path(env).expanduser().resolve()
    base = root if root is not None else Path.cwd()
    return (base / DEFAULT_DATA_DIRNAME).resolve()


def videos_dir(root: Path | None = None) -> Path:
    return data_dir(root) / VIDEOS_DIRNAME


def video_dir(video_id: str, root: Path | None = None) -> Path:
    return videos_dir(root) / video_id


def video_meta_path(video_id: str, root: Path | None = None) -> Path:
    return video_dir(video_id, root) / META_FILENAME


def video_media_path(video_id: str, filename: str, root: Path | None = None) -> Path:
    return video_dir(video_id, root) / filename
