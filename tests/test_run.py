"""run pipeline with mock ASR (no Whisper download)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from answer_clip import __version__
from answer_clip.cli import main
from answer_clip.models import SubtitleSegment
from answer_clip.run import RunError, run_pipeline


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe required",
)


class _FakeAsr:
    name = "fake"

    def __init__(self, text: str = "Today we explain gradient descent clearly.") -> None:
        self.text = text

    def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
        assert media_path.is_file()
        return [
            SubtitleSegment(start=0.0, end=0.8, text="Welcome to class."),
            SubtitleSegment(start=0.8, end=1.6, text=self.text),
        ]


def _silent_mp4(tmp_path: Path, duration: float = 2.0) -> Path:
    out = tmp_path / "silent.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=320x240:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=44100:cl=mono:d={duration}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg fixture failed: {completed.stderr[-300:]}")
    return out


def test_version_is_release() -> None:
    assert __version__ == "0.1.0"


def test_run_pipeline_mock_asr(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"
    summary = run_pipeline(
        media,
        "gradient descent",
        data_root=data_root,
        backend_impl=_FakeAsr(),
        use_llm=False,
        hit=0,
        pad_sec=0.0,
    )
    assert summary["video_id"]
    assert summary["ask"]["hits"]
    assert summary["clip"] is not None
    assert Path(summary["clip"]["output_path"]).is_file()
    assert summary["hit_index"] == 0


def test_run_skip_clip(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"
    summary = run_pipeline(
        media,
        "gradient",
        data_root=data_root,
        backend_impl=_FakeAsr(),
        use_llm=False,
        skip_clip=True,
    )
    assert summary["clip"] is None
    assert summary["ask"]["hits"]


def test_run_no_hits(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"

    class _Emptyish(_FakeAsr):
        def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
            return [SubtitleSegment(start=0.0, end=1.0, text="zzzz unrelated pasta")]

    with pytest.raises(RunError, match="no hits"):
        run_pipeline(
            media,
            "quantum entanglement xyzzy",
            data_root=data_root,
            backend_impl=_Emptyish(),
            use_llm=False,
        )


def test_run_missing_asr_dep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _silent_mp4(tmp_path)
    import answer_clip.run as run_mod

    real_find = importlib.util.find_spec

    def _find(name: str, package: str | None = None):
        if name == "faster_whisper":
            return None
        return real_find(name, package)

    monkeypatch.setattr(run_mod.importlib.util, "find_spec", _find)
    with pytest.raises(RunError, match=r"answer-clip\[asr\]"):
        run_pipeline(
            media,
            "hello",
            data_root=tmp_path / "data",
            use_llm=False,
            skip_clip=True,
        )


def test_cli_run_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
