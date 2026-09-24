"""ffmpeg clip export tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from answer_clip.cli import main
from answer_clip.clip import ClipError, export_clip
from answer_clip.ffprobe import probe
from answer_clip.ingest import ingest
from answer_clip.models import ExportJob, QueryHit, QueryResult


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe required",
)


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


def test_export_clip_half_second(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path, duration=2.0)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    job = export_clip(
        meta.id,
        start=0.0,
        end=0.5,
        data_root=data_root,
    )
    assert isinstance(job, ExportJob)
    out = Path(job.output_path)
    assert out.is_file() and out.stat().st_size > 0
    assert "clips" in out.parts
    assert job.duration_sec is not None
    # Allow keyframe / container slack
    assert 0.2 <= job.duration_sec <= 1.2
    probed = probe(out)
    assert probed.duration_sec is not None
    assert 0.2 <= probed.duration_sec <= 1.2
    assert job.sidecar_path and Path(job.sidecar_path).is_file()
    side = json.loads(Path(job.sidecar_path).read_text(encoding="utf-8"))
    assert side["video_id"] == meta.id
    assert side["codec_mode"] in {"copy", "reencode"}


def test_export_clip_custom_out(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path, duration=2.0)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    out = tmp_path / "custom" / "piece.mp4"
    job = export_clip(
        meta.id,
        start=0.2,
        end=0.8,
        out=out,
        data_root=data_root,
    )
    assert Path(job.output_path) == out.resolve()
    assert out.is_file()


def test_export_from_ask_hit(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path, duration=2.0)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    ask = QueryResult(
        video_id=meta.id,
        question="demo",
        hits=[QueryHit(start=0.1, end=0.6, text="x", score=1.0, evidence="x")],
    )
    ask_path = tmp_path / "ask.json"
    ask_path.write_text(ask.model_dump_json(indent=2), encoding="utf-8")
    job = export_clip(
        meta.id,
        hit_index=0,
        ask_result=ask_path,
        data_root=data_root,
    )
    assert job.hit_index == 0
    assert Path(job.output_path).is_file()
    assert job.start == pytest.approx(0.1)
    assert job.end == pytest.approx(0.6)


def test_clip_bad_range(tmp_path: Path) -> None:
    media = _silent_mp4(tmp_path)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    with pytest.raises(ClipError, match="greater than start"):
        export_clip(meta.id, start=1.0, end=0.5, data_root=data_root)


def test_cli_clip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    media = _silent_mp4(tmp_path, duration=2.0)
    data_root = tmp_path / "data"
    meta = ingest(media, data_root=data_root)
    code = main(
        [
            "clip",
            meta.id,
            "--start",
            "0",
            "--end",
            "0.5",
            "--data-dir",
            str(data_root),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["output_path"]).is_file()


def test_cli_clip_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["clip", "--help"])
    assert exc.value.code == 0
