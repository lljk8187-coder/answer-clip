"""Embedding backend protocol and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class EmbedBackendError(RuntimeError):
    """Backend missing, misconfigured, or encode failed."""


@runtime_checkable
class EmbedBackend(Protocol):
    """Pluggable text embedding backend."""

    name: str
    model_id: str

    @property
    def dim(self) -> int:
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array shaped ``(len(texts), dim)``."""
        ...


def get_backend(
    name: str,
    *,
    model_id: str = "BAAI/bge-small-zh-v1.5",
    device: str | None = None,
    download_root: Path | None = None,
) -> EmbedBackend:
    key = name.strip().lower().replace("_", "-")
    if key in {"sentence-transformers", "st", "local", "bge"}:
        from answer_clip.index.sentence_transformers import SentenceTransformersBackend

        return SentenceTransformersBackend(
            model_id=model_id,
            device=device,
            download_root=download_root,
        )
    if key in {"openai-compatible", "openai", "cloud"}:
        from answer_clip.index.openai_compatible import OpenAICompatibleEmbedBackend

        return OpenAICompatibleEmbedBackend(model_id=model_id)
    raise EmbedBackendError(
        f"unknown embed backend {name!r}; "
        "supported: sentence-transformers, openai-compatible (stub)"
    )
