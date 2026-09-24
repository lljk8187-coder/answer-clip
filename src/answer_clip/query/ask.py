"""``ask``: keyword retrieval ± optional LLM window rerank."""

from __future__ import annotations

import json
from pathlib import Path

from answer_clip.models import QueryResult, SubtitleSegment
from answer_clip.paths import SEGMENTS_FILENAME, data_dir
from answer_clip.query.llm import default_llm_rerank, llm_config_from_env
from answer_clip.query.merge import merge_hits
from answer_clip.query.score import rank_segments


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
) -> QueryResult:
    """Retrieve Top-K hits for ``question`` over ``segments.json``.

    ``use_llm``:
      - False / ``--no-llm``: never call LLM
      - True: attempt LLM; if no key → ``llm_skipped``, still return keyword hits
      - None (default): use LLM only when an API key is present
    """
    q = (question or "").strip()
    if not q:
        raise AskError("question must be non-empty")
    if not video_id:
        raise AskError("video_id is required")

    segments = load_segments(video_id, data_root=data_root)
    # Pull a wider candidate pool before pad-merge / LLM, then trim to top_k.
    pool_k = max(top_k, candidate_k)
    hits = rank_segments(q, segments, top_k=pool_k)

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
            except Exception as exc:  # noqa: BLE001 — degrade to keyword hits
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
        keyword_backend="simple",
        llm_used=llm_used,
        llm_skipped=llm_skipped if not llm_used else None,
    )


def ask_to_json(result: QueryResult) -> str:
    return result.model_dump_json(indent=2) + "\n"
