"""Semantic ranking from ``embeddings.npz`` + query encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from answer_clip.index.base import EmbedBackend, EmbedBackendError
from answer_clip.models import QueryHit, SubtitleSegment
from answer_clip.paths import EMBEDDINGS_FILENAME, data_dir


class EmbedIndexError(ValueError):
    """Missing or invalid embeddings.npz."""


def load_embeddings_npz(path: Path) -> dict:
    """Load vectors + metadata; raise ``EmbedIndexError`` if unusable."""
    if not path.is_file():
        raise EmbedIndexError(f"embeddings.npz not found: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise EmbedIndexError(f"failed to load embeddings.npz: {exc}") from exc
    if "vectors" not in data.files:
        raise EmbedIndexError("embeddings.npz missing 'vectors'")
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise EmbedIndexError(f"invalid vectors shape: {vectors.shape}")
    n, dim = vectors.shape
    starts = np.asarray(data["starts"], dtype=np.float64) if "starts" in data.files else None
    ends = np.asarray(data["ends"], dtype=np.float64) if "ends" in data.files else None
    if starts is None or ends is None or starts.shape != (n,) or ends.shape != (n,):
        raise EmbedIndexError("embeddings.npz starts/ends must match vector rows")
    model_id = str(data["model_id"]) if "model_id" in data.files else ""
    return {
        "vectors": vectors,
        "starts": starts,
        "ends": ends,
        "dim": int(dim),
        "model_id": model_id,
        "segment_count": n,
    }


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def cosine_scores(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Return cosine similarities shaped ``(n,)``."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.ndim != 1:
        raise EmbedIndexError(f"query vector must be 1-D, got {q.shape}")
    if vectors.shape[1] != q.shape[0]:
        raise EmbedIndexError(
            f"query dim {q.shape[0]} != index dim {vectors.shape[1]}"
        )
    qn = _l2_normalize(q.reshape(1, -1))[0]
    vn = _l2_normalize(vectors)
    return vn @ qn


def rank_embedded(
    question: str,
    segments: list[SubtitleSegment],
    *,
    data_root: Path | None,
    video_id: str,
    backend: EmbedBackend,
    top_k: int = 20,
    min_cosine: float = 0.0,
) -> tuple[list[QueryHit], dict]:
    """Encode ``question`` and rank segments by cosine vs ``embeddings.npz``.

    Returns ``(hits, meta)`` where meta includes model_id / path.
    """
    path = resolve_embeddings_path(video_id, data_root)
    index = load_embeddings_npz(path)
    vectors = index["vectors"]
    starts = index["starts"]
    ends = index["ends"]

    try:
        q_vecs = backend.encode([(question or "").strip() or " "])
    except EmbedBackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EmbedBackendError(f"query encode failed: {exc}") from exc

    q = np.asarray(q_vecs, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 1:
        raise EmbedBackendError(f"encoder returned bad shape: {q.shape}")
    scores = cosine_scores(q[0], vectors)

    # Map (start,end) → segment text when available.
    text_by_span: dict[tuple[float, float], str] = {}
    for seg in segments:
        text_by_span[(round(float(seg.start), 4), round(float(seg.end), 4))] = seg.text or ""

    hits: list[QueryHit] = []
    order = np.argsort(-scores)
    for idx in order:
        i = int(idx)
        cos = float(scores[i])
        if cos <= min_cosine:
            continue
        start = float(starts[i])
        end = float(ends[i])
        key = (round(start, 4), round(end, 4))
        text = text_by_span.get(key)
        if text is None and 0 <= i < len(segments):
            text = segments[i].text or ""
        elif text is None:
            text = ""
        # Map cosine [-1,1] → [0,1] for QueryHit.score ge=0.
        score = max(0.0, min(1.0, (cos + 1.0) / 2.0))
        hits.append(
            QueryHit(
                start=start,
                end=end,
                text=text,
                score=round(score, 4),
                evidence=f"embed:{cos:.4f}",
            )
        )
        if top_k > 0 and len(hits) >= top_k:
            break

    meta = {
        "embeddings_path": str(path),
        "model_id": index["model_id"],
        "dim": index["dim"],
    }
    return hits, meta


def resolve_embeddings_path(video_id: str, data_root: Path | None = None) -> Path:
    """``data_root`` is the data directory itself (same as ask/index)."""
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    return root / "videos" / video_id / EMBEDDINGS_FILENAME


def try_load_embeddings_path(video_id: str, data_root: Path | None) -> Path | None:
    path = resolve_embeddings_path(video_id, data_root)
    return path if path.is_file() else None
