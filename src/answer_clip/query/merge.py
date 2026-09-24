"""Merge adjacent / overlapping query hits with padding."""

from __future__ import annotations

from answer_clip.models import QueryHit


def merge_hits(hits: list[QueryHit], *, pad_sec: float = 0.0) -> list[QueryHit]:
    """Expand each hit by ``pad_sec`` and merge overlaps (keep max score)."""
    if not hits:
        return []
    pad = max(0.0, float(pad_sec))
    expanded = [
        QueryHit(
            start=max(0.0, h.start - pad),
            end=h.end + pad,
            text=h.text,
            score=h.score,
            evidence=h.evidence,
        )
        for h in hits
    ]
    expanded.sort(key=lambda h: (h.start, h.end))
    merged: list[QueryHit] = [expanded[0]]
    for h in expanded[1:]:
        prev = merged[-1]
        if h.start <= prev.end:
            merged[-1] = QueryHit(
                start=prev.start,
                end=max(prev.end, h.end),
                text=(prev.text + " " + h.text).strip(),
                score=max(prev.score, h.score),
                evidence=",".join(
                    x for x in dict.fromkeys(
                        (prev.evidence + "," + h.evidence).split(",")
                    )
                    if x
                ),
            )
        else:
            merged.append(h)
    return merged
