"""Shared data types."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class HitSpan(BaseModel):
    """A transcript span that answers a query (placeholder)."""

    start_sec: float
    end_sec: float
    text: str = ""


CopyStrategy = Literal["copy", "hardlink"]


class VideoMeta(BaseModel):
    """Local video registration metadata written to ``meta.json``.

    Fields that ffprobe cannot provide stay ``None`` rather than inventing zeros.
    """

    id: str = Field(description="Stable video id (content hash prefix).")
    source_path: str = Field(description="Absolute path of the original file at ingest.")
    filename: str = Field(description="Original basename.")
    stored_relpath: str = Field(
        description="Path relative to data root, e.g. videos/<id>/<filename>."
    )
    copy_strategy: CopyStrategy = Field(
        description="How the file was placed under data/videos/."
    )
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    fps: float | None = None
    size_bytes: int | None = None
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
