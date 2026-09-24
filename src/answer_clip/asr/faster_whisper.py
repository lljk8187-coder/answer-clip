"""faster-whisper local backend."""

from __future__ import annotations

from pathlib import Path

from answer_clip.asr.base import AsrBackendError
from answer_clip.asr.normalize import normalize_segments
from answer_clip.models import SubtitleSegment


class FasterWhisperBackend:
    """Transcribe with the ``faster-whisper`` package (optional extra)."""

    name = "faster-whisper"

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.language = language
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrBackendError(
                "faster-whisper is not installed. "
                'Install with: pip install "answer-clip[asr]"'
            ) from exc

        kwargs: dict = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if self.download_root is not None:
            self.download_root.mkdir(parents=True, exist_ok=True)
            kwargs["download_root"] = str(self.download_root)

        try:
            self._model = WhisperModel(self.model_size, **kwargs)
        except Exception as exc:  # noqa: BLE001 — surface install/runtime issues
            raise AsrBackendError(
                f"failed to load faster-whisper model {self.model_size!r}: {exc}"
            ) from exc
        return self._model

    def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
        model = self._load()
        path = Path(media_path)
        if not path.is_file():
            raise AsrBackendError(f"media file not found: {path}")

        kwargs: dict = {"vad_filter": True}
        if self.language:
            kwargs["language"] = self.language

        try:
            segments_iter, _info = model.transcribe(str(path), **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AsrBackendError(f"faster-whisper transcription failed: {exc}") from exc

        raw: list[SubtitleSegment] = []
        for seg in segments_iter:
            conf = None
            avg = getattr(seg, "avg_logprob", None)
            if avg is not None:
                # Map logprob (~[-1, 0]) into a soft [0, 1] score.
                conf = max(0.0, min(1.0, 1.0 + float(avg)))
            raw.append(
                SubtitleSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=(seg.text or "").strip(),
                    confidence=conf,
                )
            )
        return normalize_segments(raw)
