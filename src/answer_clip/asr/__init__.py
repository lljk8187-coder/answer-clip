"""ASR package: backends + write segments.json / transcript.srt."""

from __future__ import annotations

from answer_clip.asr.base import AsrBackend, AsrBackendError
from answer_clip.asr.pipeline import AsrError, run_asr
from answer_clip.asr.srt import segments_to_srt, srt_timestamp
from answer_clip.models import SubtitleSegment

__all__ = [
    "AsrBackend",
    "AsrBackendError",
    "AsrError",
    "SubtitleSegment",
    "run_asr",
    "segments_to_srt",
    "srt_timestamp",
]
