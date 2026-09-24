"""SRT serialization helpers."""

from __future__ import annotations

from answer_clip.models import SubtitleSegment


def srt_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` for SRT."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list[SubtitleSegment]) -> str:
    """Render segments as an SRT document (trailing newline)."""
    blocks: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx = len(blocks) + 1
        start = srt_timestamp(seg.start)
        end = srt_timestamp(seg.end)
        blocks.append(f"{idx}\n{start} --> {end}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"
