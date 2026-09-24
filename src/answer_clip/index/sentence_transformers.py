"""Local sentence-transformers embedding backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from answer_clip.index.base import EmbedBackendError


class SentenceTransformersBackend:
    """Encode texts with ``sentence-transformers`` (optional ``[embed]`` extra)."""

    name = "sentence-transformers"

    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        download_root: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.download_root = download_root
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedBackendError(
                "sentence-transformers is not installed. "
                'Install with: pip install "answer-clip[embed]"'
            ) from exc

        kwargs: dict = {}
        if self.device:
            kwargs["device"] = self.device
        if self.download_root is not None:
            self.download_root.mkdir(parents=True, exist_ok=True)
            kwargs["cache_folder"] = str(self.download_root)

        try:
            self._model = SentenceTransformer(self.model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise EmbedBackendError(
                f"failed to load sentence-transformers model {self.model_id!r}: {exc}"
            ) from exc
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            model = self._load()
            self._dim = int(model.get_sentence_embedding_dimension())
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbedBackendError(f"encode failed: {exc}") from exc
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise EmbedBackendError(f"unexpected embedding shape: {arr.shape}")
        self._dim = int(arr.shape[1])
        return arr
