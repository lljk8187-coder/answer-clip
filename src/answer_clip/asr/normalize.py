"""Segment cleanup: sort, clamp, de-overlap."""

from __future__ import annotations

from answer_clip.models import SubtitleSegment


def normalize_segments(
    segments: list[SubtitleSegment],
    *,
    min_duration: float = 0.01,
) -> list[SubtitleSegment]:
    """Return monotonic, non-overlapping segments.

    Strategy:
    1. Drop empty text.
    2. Ensure ``start <= end`` (swap if inverted); drop if duration < min.
    3. Sort by start, then end.
    4. If a segment overlaps the previous, clamp ``start`` to ``prev.end``;
       drop if that collapses the span below ``min_duration``.
    """
    cleaned: list[SubtitleSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start, end = float(seg.start), float(seg.end)
        if end < start:
            start, end = end, start
        if end - start < min_duration:
            continue
        cleaned.append(
            SubtitleSegment(
                start=start,
                end=end,
                text=text,
                confidence=seg.confidence,
            )
        )

    cleaned.sort(key=lambda s: (s.start, s.end))
    out: list[SubtitleSegment] = []
    for seg in cleaned:
        if not out:
            out.append(seg)
            continue
        prev = out[-1]
        start = seg.start
        if start < prev.end:
            start = prev.end
        if seg.end - start < min_duration:
            # Absorb into previous text if tiny leftover overlap.
            continue
        out.append(
            SubtitleSegment(
                start=start,
                end=seg.end,
                text=seg.text,
                confidence=seg.confidence,
            )
        )
    return out
