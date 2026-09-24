"""Keyword scoring over subtitle segments."""

from __future__ import annotations

from answer_clip.models import QueryHit, SubtitleSegment
from answer_clip.query.tokenize import phrase_normalized, tokenize


def score_segment(question: str, segment: SubtitleSegment) -> QueryHit | None:
    """Score one segment; return None if no token overlap and no phrase hit."""
    q_tokens = tokenize(question)
    if not q_tokens and not question.strip():
        return None

    text = segment.text or ""
    hay = text.lower()
    hay_compact = phrase_normalized(text)
    q_compact = phrase_normalized(question)

    matched: list[str] = []
    for tok in q_tokens:
        if tok.isascii():
            # word-ish: require token boundary-ish via substring on lower text
            if tok in hay:
                matched.append(tok)
        else:
            if tok in hay_compact or tok in hay:
                matched.append(tok)

    phrase_bonus = 0.0
    if q_compact and len(q_compact) >= 2 and q_compact in hay_compact:
        phrase_bonus = 0.35

    if not matched and phrase_bonus == 0.0:
        return None

    base = (len(matched) / max(len(q_tokens), 1)) if q_tokens else 0.0
    score = min(1.0, base + phrase_bonus)
    evidence = ",".join(matched[:12]) if matched else "phrase"
    return QueryHit(
        start=float(segment.start),
        end=float(segment.end),
        text=text,
        score=round(score, 4),
        evidence=evidence,
    )


def rank_segments(
    question: str,
    segments: list[SubtitleSegment],
    *,
    top_k: int = 5,
) -> list[QueryHit]:
    hits: list[QueryHit] = []
    for seg in segments:
        hit = score_segment(question, seg)
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: (-h.score, h.start))
    if top_k <= 0:
        return hits
    return hits[:top_k]
