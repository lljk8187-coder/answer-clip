"""Build ``embeddings.npz`` from ``segments.json``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from answer_clip.index.base import EmbedBackend, EmbedBackendError, get_backend
from answer_clip.models import SubtitleSegment
from answer_clip.paths import (
    EMBEDDINGS_FILENAME,
    SEGMENTS_FILENAME,
    data_dir,
    model_cache_dir,
)


class IndexBuildError(ValueError):
    """Invalid video id / missing segments / failed index build.

    Named ``IndexError_`` to avoid clashing with the builtin ``IndexError``.
    """


def load_segments(video_id: str, *, data_root: Path | None = None) -> list[SubtitleSegment]:
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    seg_path = root / "videos" / video_id / SEGMENTS_FILENAME
    if not seg_path.is_file():
        raise IndexBuildError(
            f"segments.json not found for video_id={video_id!r} at {seg_path}; "
            "run `answer-clip asr` first"
        )
    raw = json.loads(seg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise IndexBuildError(f"segments.json must be a list: {seg_path}")
    return [SubtitleSegment.model_validate(item) for item in raw]


def write_embeddings_npz(
    path: Path,
    *,
    vectors: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    model_id: str,
    video_id: str,
    backend: str,
) -> Path:
    """Persist vectors + metadata into a compressed ``.npz``."""
    if vectors.dtype != np.float32:
        vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise IndexBuildError(f"vectors must be 2-D, got {vectors.shape}")
    n, dim = vectors.shape
    if starts.shape != (n,) or ends.shape != (n,):
        raise IndexBuildError("starts/ends length must match vectors rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors,
        starts=np.asarray(starts, dtype=np.float64),
        ends=np.asarray(ends, dtype=np.float64),
        model_id=np.asarray(model_id),
        dim=np.int32(dim),
        video_id=np.asarray(video_id),
        backend=np.asarray(backend),
        segment_count=np.int32(n),
    )
    return path


def build_index(
    video_id: str,
    *,
    data_root: Path | None = None,
    backend: str = "sentence-transformers",
    model_id: str = "BAAI/bge-small-zh-v1.5",
    device: str | None = None,
    download_root: Path | None = None,
    backend_impl: EmbedBackend | None = None,
) -> dict:
    """Encode segments and write ``data/videos/<id>/embeddings.npz``."""
    if not video_id:
        raise IndexBuildError("video_id is required")

    root = Path(data_root).resolve() if data_root is not None else data_dir()
    segments = load_segments(video_id, data_root=root)
    texts = [(s.text or "").strip() or " " for s in segments]
    starts = np.asarray([float(s.start) for s in segments], dtype=np.float64)
    ends = np.asarray([float(s.end) for s in segments], dtype=np.float64)

    cache = download_root
    if cache is None:
        cache = model_cache_dir(root)

    engine = backend_impl or get_backend(
        backend,
        model_id=model_id,
        device=device,
        download_root=cache,
    )
    try:
        vectors = engine.encode(texts)
    except EmbedBackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EmbedBackendError(f"embed backend failed: {exc}") from exc

    out = root / "videos" / video_id / EMBEDDINGS_FILENAME
    write_embeddings_npz(
        out,
        vectors=vectors,
        starts=starts,
        ends=ends,
        model_id=getattr(engine, "model_id", model_id),
        video_id=video_id,
        backend=getattr(engine, "name", backend),
    )
    return {
        "video_id": video_id,
        "embeddings_path": str(out),
        "segment_count": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "model_id": getattr(engine, "model_id", model_id),
        "backend": getattr(engine, "name", backend),
    }
