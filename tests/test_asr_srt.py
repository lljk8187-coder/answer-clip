"""Pure tests for SubtitleSegment + SRT helpers (no Whisper)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from answer_clip.asr.normalize import normalize_segments
from answer_clip.asr.srt import segments_to_srt, srt_timestamp
from answer_clip.models import SubtitleSegment


def test_srt_timestamp_basic() -> None:
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert srt_timestamp(3661.234) == "01:01:01,234"


def test_srt_timestamp_negative_clamped() -> None:
    assert srt_timestamp(-1) == "00:00:00,000"


def test_segments_to_srt_round_trip_shape() -> None:
    segs = [
        SubtitleSegment(start=0.0, end=1.0, text="Hello"),
        SubtitleSegment(start=1.2, end=2.5, text="world"),
    ]
    doc = segments_to_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:01,000\nHello" in doc
    assert "2\n00:00:01,200 --> 00:00:02,500\nworld" in doc
    assert doc.endswith("\n")


def test_segments_to_srt_skips_empty_text() -> None:
    segs = [
        SubtitleSegment(start=0.0, end=1.0, text="  "),
        SubtitleSegment(start=1.0, end=2.0, text="ok"),
    ]
    doc = segments_to_srt(segs)
    assert doc.startswith("1\n")
    assert "ok" in doc
    assert doc.count("-->") == 1


def test_subtitle_segment_schema() -> None:
    seg = SubtitleSegment(start=0.0, end=1.0, text="hi", confidence=0.9)
    assert seg.model_dump()["confidence"] == 0.9
    with pytest.raises(ValidationError):
        SubtitleSegment(start=-1.0, end=1.0, text="x")


def test_normalize_sorts_and_deoverlaps() -> None:
    raw = [
        SubtitleSegment(start=2.0, end=3.0, text="b"),
        SubtitleSegment(start=0.0, end=1.5, text="a"),
        SubtitleSegment(start=1.0, end=2.5, text="overlap"),
    ]
    out = normalize_segments(raw)
    assert [s.text for s in out] == ["a", "overlap", "b"]
    assert out[0].end == 1.5
    assert out[1].start == 1.5
    assert out[2].start == 2.5
    for i in range(1, len(out)):
        assert out[i].start >= out[i - 1].end
        assert out[i].start < out[i].end


def test_normalize_swaps_inverted() -> None:
    out = normalize_segments([SubtitleSegment(start=2.0, end=1.0, text="x")])
    assert len(out) == 1
    assert out[0].start == 1.0
    assert out[0].end == 2.0
