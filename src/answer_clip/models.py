"""Shared data types."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SubtitleSegment(BaseModel):
    """One timed subtitle / ASR segment."""

    start: float = Field(ge=0, description="Segment start in seconds.")
    end: float = Field(ge=0, description="Segment end in seconds.")
    text: str = ""
    confidence: float | None = Field(
        default=None,
        description="Optional backend confidence in [0, 1].",
    )


class HitSpan(BaseModel):
    """A transcript span that answers a query (legacy alias shape)."""

    start_sec: float
    end_sec: float
    text: str = ""


class QueryHit(BaseModel):
    """One ranked answer window from ``ask``."""

    start: float
    end: float
    text: str
    score: float = Field(ge=0, description="Relevance score (higher is better).")
    evidence: str = Field(
        default="",
        description="Matched tokens or short rationale for the hit.",
    )


class QueryResult(BaseModel):
    """Structured output of ``answer-clip ask``."""

    video_id: str
    question: str
    hits: list[QueryHit] = Field(default_factory=list)
    top_k: int = 5
    pad_sec: float = 0.0
    keyword_backend: str = "simple"
    llm_used: bool = False
    llm_skipped: str | None = Field(
        default=None,
        description="Reason LLM rerank was skipped, if any.",
    )


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
