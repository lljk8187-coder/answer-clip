"""ASR pipeline with mock backend (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answer_clip.asr.base import AsrBackendError, get_backend
from answer_clip.asr.pipeline import AsrError, run_asr, write_outputs
from answer_clip.cli import main
from answer_clip.ingest import ingest
from answer_clip.models import SubtitleSegment
from answer_clip.paths import META_FILENAME, SEGMENTS_FILENAME, SRT_FILENAME


class _FakeBackend:
    name = "fake"

    def __init__(self, segments: list[SubtitleSegment] | None = None) -> None:
        self.segments = segments or [
            SubtitleSegment(start=0.0, end=0.8, text="hello"),
            SubtitleSegment(start=0.9, end=1.5, text="world"),
        ]
        self.calls: list[Path] = []

    def transcribe(self, media_path: Path) -> list[SubtitleSegment]:
        self.calls.append(media_path)
        return list(self.segments)


def _silent_mp4(tmp_path: Path) -> Path:
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    out = tmp_path / "silent.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=1",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
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
        pytest.skip(f"ffmpeg failed: {completed.stderr[-300:]}")
    return out


def test_write_outputs(tmp_path: Path) -> None:
    segs = [SubtitleSegment(start=0.0, end=1.0, text="hi")]
    seg_path, srt_path = write_outputs(tmp_path, segs)
    assert seg_path.name == SEGMENTS_FILENAME
    assert srt_path.name == SRT_FILENAME
    data = json.loads(seg_path.read_text(encoding="utf-8"))
    assert data[0]["text"] == "hi"
    assert "00:00:00,000" in srt_path.read_text(encoding="utf-8")


def test_run_asr_mock_on_ingested(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    fake = _FakeBackend()
    result = run_asr(
        video_id=meta.id,
        data_root=data_root,
        backend_impl=fake,
    )
    assert result["segment_count"] == 2
    assert result["video_id"] == meta.id
    assert Path(result["segments_path"]).is_file()
    assert Path(result["srt_path"]).is_file()
    assert fake.calls and fake.calls[0].name == "silent.mp4"


def test_run_asr_mock_via_path(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    stored = data_root / meta.stored_relpath
    fake = _FakeBackend()
    result = run_asr(path=stored, data_root=data_root, backend_impl=fake)
    assert result["video_id"] == meta.id
    assert (data_root / "videos" / meta.id / SEGMENTS_FILENAME).is_file()


def test_run_asr_missing_id(tmp_path: Path) -> None:
    with pytest.raises(AsrError, match="not found"):
        run_asr(video_id="nope", data_root=tmp_path / "data", backend_impl=_FakeBackend())


def test_run_asr_requires_xor() -> None:
    with pytest.raises(AsrError, match="exactly one"):
        run_asr(backend_impl=_FakeBackend())
    with pytest.raises(AsrError, match="exactly one"):
        run_asr(video_id="a", path="/tmp/x", backend_impl=_FakeBackend())


def test_openai_compatible_stub() -> None:
    backend = get_backend("openai-compatible")
    with pytest.raises(AsrBackendError, match="not implemented"):
        backend.transcribe(Path("/tmp/x.mp4"))


def test_faster_whisper_missing_dep_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    backend = get_backend("faster-whisper")
    with pytest.raises(AsrBackendError, match=r"answer-clip\[asr\]"):
        backend.transcribe(Path("/tmp/x.mp4"))


def test_cli_asr_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["asr", "--help"])
    assert exc.value.code == 0


def test_cli_asr_with_mock_via_missing_backend_dep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI path validation without loading Whisper."""
    code = main(["asr", "missing-id", "--data-dir", str(tmp_path / "data")])
    assert code == 1
    assert "asr error" in capsys.readouterr().err
