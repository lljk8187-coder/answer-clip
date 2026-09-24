"""OpenAI-compatible / cloud ASR stub (not implemented in MVP)."""

from __future__ import annotations

from pathlib import Path

from answer_clip.asr.base import AsrBackendError
from answer_clip.models import SubtitleSegment


class OpenAICompatibleBackend:
    """Placeholder for a remote Whisper-compatible HTTP API."""

    name = "openai-compatible"

    def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
        raise AsrBackendError(
            "openai-compatible ASR backend is not implemented yet "
            f"(refusing to call cloud for {media_path})"
        )
