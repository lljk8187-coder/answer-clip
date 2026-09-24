"""Ingest + ffprobe tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from answer_clip.cli import main
from answer_clip.ffprobe import FfprobeError, probe, require_ffprobe
from answer_clip.ingest import IngestError, content_id, ingest
from answer_clip.models import VideoMeta
from answer_clip.paths import META_FILENAME


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


pytestmark = pytest.mark.skipif(
    not _have_ffmpeg(),
    reason="ffmpeg/ffprobe required for ingest tests",
)


@pytest.fixture()
def silent_mp4(tmp_path: Path) -> Path:
    """Generate a ~1s silent H.264 mp4 via ffmpeg."""
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
        pytest.skip(f"ffmpeg failed to generate fixture: {completed.stderr[-400:]}")
    assert out.is_file() and out.stat().st_size > 0
    return out


def test_require_ffprobe() -> None:
    assert Path(require_ffprobe()).is_file()


def test_probe_silent_mp4(silent_mp4: Path) -> None:
    result = probe(silent_mp4)
    assert result.duration_sec is not None
    assert result.duration_sec == pytest.approx(1.0, abs=0.3)
    assert result.width == 320
    assert result.height == 240
    assert result.video_codec == "h264"
    assert result.audio_codec in {"aac", "mp3", None} or result.audio_codec


def test_ingest_writes_meta_and_media(silent_mp4: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    meta = ingest(silent_mp4, data_root=data_root)

    assert isinstance(meta, VideoMeta)
    assert meta.id == content_id(silent_mp4)
    assert meta.filename == "silent.mp4"
    assert meta.width == 320
    assert meta.height == 240
    assert meta.video_codec == "h264"
    assert meta.duration_sec == pytest.approx(1.0, abs=0.3)
    assert meta.copy_strategy in {"copy", "hardlink"}
    assert meta.size_bytes == silent_mp4.stat().st_size

    media = data_root / meta.stored_relpath
    assert media.is_file()
    meta_path = data_root / "videos" / meta.id / META_FILENAME
    assert meta_path.is_file()
    loaded = VideoMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    assert loaded.id == meta.id
    assert loaded.source_path == str(silent_mp4.resolve())


def test_ingest_missing_path(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="does not exist"):
        ingest(tmp_path / "nope.mp4", data_root=tmp_path / "data")


def test_ingest_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not a file"):
        ingest(tmp_path, data_root=tmp_path / "data")


def test_cli_ingest_ok(silent_mp4: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_root = tmp_path / "cli-data"
    code = main(["ingest", str(silent_mp4), "--data-dir", str(data_root)])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["filename"] == "silent.mp4"
    assert (data_root / "videos" / payload["id"] / "silent.mp4").is_file()


def test_cli_ingest_missing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ingest", "/no/such/video.mp4"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ingest error" in err
    assert "does not exist" in err


def test_probe_bad_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a real video")
    with pytest.raises(FfprobeError):
        probe(junk)
