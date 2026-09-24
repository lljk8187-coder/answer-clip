"""Embedding index: segments.json → embeddings.npz."""

from answer_clip.index.base import EmbedBackend, EmbedBackendError, get_backend
from answer_clip.index.pipeline import IndexBuildError, build_index, write_embeddings_npz

__all__ = [
    "EmbedBackend",
    "EmbedBackendError",
    "IndexBuildError",
    "build_index",
    "get_backend",
    "write_embeddings_npz",
]
