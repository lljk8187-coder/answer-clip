"""Reciprocal Rank Fusion over ranked hit lists."""

from __future__ import annotations

from answer_clip.models import QueryHit


def _hit_key(hit: QueryHit) -> tuple[float, float]:
    return (round(float(hit.start), 4), round(float(hit.end), 4))


def reciprocal_rank_fusion(
    rankings: list[list[QueryHit]],
    *,
    k: int = 60,
) -> list[QueryHit]:
    """Fuse multiple ranked lists with classic RRF: ``1 / (k + rank)``.

    Rank is 1-based within each list. Hits are keyed by ``(start, end)``.
    The fused score is the RRF sum; text/evidence prefer the highest
    single-list score's fields, with evidence tagged ``rrf``.
    """
    if not rankings:
        return []

    scores: dict[tuple[float, float], float] = {}
    best: dict[tuple[float, float], QueryHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = _hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            prev = best.get(key)
            if prev is None or hit.score > prev.score:
                best[key] = hit

    fused: list[QueryHit] = []
    for key, rrf_score in scores.items():
        base = best[key]
        evidence = base.evidence or ""
        if evidence and "rrf" not in evidence:
            evidence = f"{evidence}|rrf"
        elif not evidence:
            evidence = "rrf"
        fused.append(
            QueryHit(
                start=base.start,
                end=base.end,
                text=base.text,
                score=round(float(rrf_score), 6),
                evidence=evidence,
            )
        )
    fused.sort(key=lambda h: (-h.score, h.start))
    return fused
