"""``ask``: keyword / embed / hybrid (RRF) ± optional LLM window rerank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from answer_clip.index.base import EmbedBackend, EmbedBackendError, get_backend
from answer_clip.index.pipeline import build_index
from answer_clip.models import QueryResult, SubtitleSegment
from answer_clip.paths import SEGMENTS_FILENAME, data_dir, model_cache_dir
from answer_clip.query.embed_rank import (
    EmbedIndexError,
    rank_embedded,
    try_load_embeddings_path,
)
from answer_clip.query.llm import default_llm_rerank, llm_config_from_env
from answer_clip.query.merge import merge_hits
from answer_clip.query.rrf import reciprocal_rank_fusion
from answer_clip.query.score import rank_segments

AskMode = Literal["hybrid", "keyword", "embed"]


class AskError(ValueError):
    """Invalid video id / missing segments / bad args."""


def load_segments(video_id: str, *, data_root: Path | None = None) -> list[SubtitleSegment]:
    root = Path(data_root).resolve() if data_root is not None else data_dir()
    vdir = root / "videos" / video_id
    seg_path = vdir / SEGMENTS_FILENAME
    if not seg_path.is_file():
        raise AskError(
            f"segments.json not found for video_id={video_id!r} at {seg_path}; "
            "run `answer-clip asr` first"
        )
    raw = json.loads(seg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AskError(f"segments.json must be a list: {seg_path}")
    return [SubtitleSegment.model_validate(item) for item in raw]


def _resolve_embed_backend(
    *,
    embed_backend: EmbedBackend | None,
    embed_model: str,
    embed_device: str | None,
    download_root: Path | None,
    data_root: Path | None,
) -> EmbedBackend:
    if embed_backend is not None:
        return embed_backend
    cache = download_root
    if cache is None:
        root = Path(data_root).resolve() if data_root is not None else data_dir()
        cache = model_cache_dir(root)
    return get_backend(
        "sentence-transformers",
        model_id=embed_model,
        device=embed_device,
        download_root=cache,
    )


def _semantic_hits(
    question: str,
    segments: list[SubtitleSegment],
    *,
    video_id: str,
    data_root: Path | None,
    pool_k: int,
    embed_backend: EmbedBackend | None,
    embed_model: str,
    embed_device: str | None,
    download_root: Path | None,
    strict: bool,
) -> tuple[list, str | None, bool]:
    """Return ``(hits, embed_skipped, embed_used)``.

    When ``strict`` (embed-only mode), missing index / backend / encode raises
    ``AskError``. Otherwise degrade with ``embed_skipped`` and empty hits.
    """
    path = try_load_embeddings_path(video_id, data_root)
    if path is None:
        msg = (
            f"embeddings.npz not found for video_id={video_id!r}; "
            "run `answer-clip index` first"
        )
        if strict:
            raise AskError(msg)
        return [], msg, False

    try:
        backend = _resolve_embed_backend(
            embed_backend=embed_backend,
            embed_model=embed_model,
            embed_device=embed_device,
            download_root=download_root,
            data_root=data_root,
        )
        hits, _meta = rank_embedded(
            question,
            segments,
            data_root=data_root,
            video_id=video_id,
            backend=backend,
            top_k=pool_k,
        )
        return hits, None, True
    except EmbedIndexError as exc:
        if strict:
            raise AskError(str(exc)) from exc
        return [], str(exc), False
    except EmbedBackendError as exc:
        if strict:
            raise AskError(str(exc)) from exc
        return [], str(exc), False
    except Exception as exc:  # noqa: BLE001
        msg = f"embed retrieval failed: {exc}"
        if strict:
            raise AskError(msg) from exc
        return [], msg, False


def ask(
    video_id: str,
    question: str,
    *,
    data_root: Path | None = None,
    top_k: int = 5,
    pad_sec: float = 0.0,
    use_llm: bool | None = None,
    candidate_k: int = 20,
    rerank_fn=None,
    mode: AskMode = "hybrid",
    build_embed: bool = False,
    embed_backend: EmbedBackend | None = None,
    embed_model: str = "BAAI/bge-small-zh-v1.5",
    embed_device: str | None = None,
    download_root: Path | None = None,
    rrf_k: int = 60,
) -> QueryResult:
    """Retrieve Top-K hits for ``question`` over ``segments.json``.

    ``mode``:
      - ``hybrid`` (default): keyword always; add semantic if index + [embed]
        work; fuse with RRF; never fails solely because embed is missing
      - ``keyword``: keyword only
      - ``embed``: semantic only; missing index / backend → ``AskError``

    ``build_embed``: when True, run ``index`` before retrieval (opt-in;
    default False so ask never silently downloads models).

    ``use_llm``: same as before (optional window rerank after retrieval).
    """
    q = (question or "").strip()
    if not q:
        raise AskError("question must be non-empty")
    if not video_id:
        raise AskError("video_id is required")
    mode_key = (mode or "hybrid").strip().lower()
    if mode_key not in {"hybrid", "keyword", "embed"}:
        raise AskError(f"unknown ask mode {mode!r}; use hybrid|keyword|embed")
    mode_lit: AskMode = mode_key  # type: ignore[assignment]

    if build_embed:
        try:
            build_index(
                video_id,
                data_root=data_root,
                model_id=embed_model,
                device=embed_device,
                download_root=download_root,
                backend_impl=embed_backend,
            )
        except Exception as exc:  # noqa: BLE001
            if mode_lit == "embed":
                raise AskError(f"--build-embed failed: {exc}") from exc
            # hybrid/keyword continue; embed path may still skip

    segments = load_segments(video_id, data_root=data_root)
    pool_k = max(top_k, candidate_k)

    embed_used = False
    embed_skipped: str | None = None
    keyword_hits: list = []
    embed_hits: list = []

    if mode_lit in {"hybrid", "keyword"}:
        keyword_hits = rank_segments(q, segments, top_k=pool_k)

    if mode_lit == "keyword":
        embed_skipped = "mode=keyword"
        hits = keyword_hits
    elif mode_lit == "embed":
        embed_hits, embed_skipped, embed_used = _semantic_hits(
            q,
            segments,
            video_id=video_id,
            data_root=data_root,
            pool_k=pool_k,
            embed_backend=embed_backend,
            embed_model=embed_model,
            embed_device=embed_device,
            download_root=download_root,
            strict=True,
        )
        hits = embed_hits
    else:  # hybrid
        embed_hits, embed_skipped, embed_used = _semantic_hits(
            q,
            segments,
            video_id=video_id,
            data_root=data_root,
            pool_k=pool_k,
            embed_backend=embed_backend,
            embed_model=embed_model,
            embed_device=embed_device,
            download_root=download_root,
            strict=False,
        )
        if embed_used and embed_hits:
            hits = reciprocal_rank_fusion(
                [keyword_hits, embed_hits],
                k=rrf_k,
            )
        else:
            hits = keyword_hits

    llm_used = False
    llm_skipped: str | None = None
    cfg = llm_config_from_env()
    want_llm = use_llm if use_llm is not None else bool(cfg["api_key"])

    if want_llm:
        if rerank_fn is None and not cfg["api_key"]:
            llm_skipped = "no API key (set OPENAI_API_KEY or ANSWER_CLIP_LLM_API_KEY)"
        else:
            fn = rerank_fn or default_llm_rerank
            try:
                hits = fn(q, hits)
                llm_used = True
            except Exception as exc:  # noqa: BLE001 — degrade to retrieval hits
                llm_skipped = f"llm rerank failed: {exc}"
    else:
        if use_llm is False:
            llm_skipped = "disabled via --no-llm"
        else:
            llm_skipped = "no API key (set OPENAI_API_KEY or ANSWER_CLIP_LLM_API_KEY)"

    if pad_sec and pad_sec > 0:
        hits = merge_hits(hits, pad_sec=pad_sec)
        hits.sort(key=lambda h: (-h.score, h.start))

    hits = hits[: max(top_k, 0)] if top_k > 0 else hits
    return QueryResult(
        video_id=video_id,
        question=q,
        hits=hits,
        top_k=top_k,
        pad_sec=pad_sec,
        mode=mode_lit,
        keyword_backend="simple",
        embed_used=embed_used,
        embed_skipped=embed_skipped if not embed_used else None,
        llm_used=llm_used,
        llm_skipped=llm_skipped if not llm_used else None,
    )


def ask_to_json(result: QueryResult) -> str:
    return result.model_dump_json(indent=2) + "\n"
