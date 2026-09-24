"""Optional real faster-whisper smoke (skipped unless explicitly enabled)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from answer_clip.asr.pipeline import run_asr
from answer_clip.ingest import ingest

RUN_SLOW = os.environ.get("ANSWER_CLIP_ASR_SLOW", "").strip() in {"1", "true", "yes"}


def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_SLOW, reason="set ANSWER_CLIP_ASR_SLOW=1 to run"),
    pytest.mark.skipif(not _have_faster_whisper(), reason="faster-whisper not installed"),
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required"),
]


def test_faster_whisper_tiny_on_silent(tmp_path: Path) -> None:
    media = tmp_path / "silent.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:a",
        "aac",
        str(media),
    ]
    # Prefer audio-only container; if that fails try video+audio
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(media),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            pytest.skip(f"ffmpeg failed: {completed.stderr[-300:]}")

    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    result = run_asr(
        video_id=meta.id,
        data_root=data_root,
        backend="faster-whisper",
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        download_root=data_root / "models",
    )
    assert Path(result["segments_path"]).is_file()
    assert Path(result["srt_path"]).is_file()
    # Silent/sine may yield zero speech segments; files must still exist.
    assert result["segment_count"] >= 0
