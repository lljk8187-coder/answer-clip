"""OpenAI-compatible embeddings stub (not implemented in this milestone)."""

from __future__ import annotations

import numpy as np

from answer_clip.index.base import EmbedBackendError


class OpenAICompatibleEmbedBackend:
    """Placeholder for a remote embeddings HTTP API."""

    name = "openai-compatible"

    def __init__(self, *, model_id: str = "text-embedding-3-small") -> None:
        self.model_id = model_id
        self._dim = 1536

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        raise EmbedBackendError(
            "openai-compatible embeddings backend is not implemented yet "
            f"(model={self.model_id}, n_texts={len(texts)})"
        )
