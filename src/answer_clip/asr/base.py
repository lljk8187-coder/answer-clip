"""ASR backend protocol and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from answer_clip.models import SubtitleSegment


class AsrBackendError(RuntimeError):
    """Backend missing, misconfigured, or transcription failed."""


@runtime_checkable
class AsrBackend(Protocol):
    """Pluggable speech-to-text backend."""

    name: str

    def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
        """Return ordered subtitle segments for ``media_path``."""
        ...


def get_backend(
    name: str,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    download_root: Path | None = None,
    language: str | None = None,
) -> AsrBackend:
    """Resolve a backend by name.

    Built-ins:
    - ``faster-whisper`` / ``whisper`` — local faster-whisper (MVP)
    - ``openai-compatible`` / ``cloud`` — stub (not implemented)
    """
    key = name.strip().lower().replace("_", "-")
    if key in {"faster-whisper", "whisper", "faster_whisper"}:
        from answer_clip.asr.faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            language=language,
        )
    if key in {"openai-compatible", "openai", "cloud"}:
        from answer_clip.asr.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend()
    raise AsrBackendError(
        f"unknown ASR backend {name!r}; "
        "supported: faster-whisper, openai-compatible (stub)"
    )
