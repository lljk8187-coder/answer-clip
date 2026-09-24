"""Thin wrapper around system ``ffprobe``."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfprobeError(RuntimeError):
    """ffprobe missing, failed, or returned unusable output."""


@dataclass(frozen=True)
class ProbeResult:
    duration_sec: float | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    fps: float | None


def require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FfprobeError(
            "ffprobe not found on PATH. Install ffmpeg (which provides ffprobe) "
            "and retry."
        )
    return path


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        try:
            num, den = float(num_s), float(den_s)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(rate)
    except ValueError:
        return None


def probe(path: Path) -> ProbeResult:
    """Run ffprobe and extract common media fields."""
    ffprobe = require_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise FfprobeError(f"failed to run ffprobe: {exc}") from exc

    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip() or "unknown error"
        raise FfprobeError(f"ffprobe failed for {path}: {err}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FfprobeError(f"ffprobe returned invalid JSON for {path}") from exc

    duration_sec: float | None = None
    fmt = payload.get("format") or {}
    if fmt.get("duration") is not None:
        try:
            duration_sec = float(fmt["duration"])
        except (TypeError, ValueError):
            duration_sec = None

    width = height = None
    video_codec = audio_codec = None
    fps: float | None = None

    for stream in payload.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_codec is None:
            video_codec = stream.get("codec_name")
            try:
                width = int(stream["width"]) if stream.get("width") is not None else None
            except (TypeError, ValueError):
                width = None
            try:
                height = (
                    int(stream["height"]) if stream.get("height") is not None else None
                )
            except (TypeError, ValueError):
                height = None
            fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            if duration_sec is None and stream.get("duration") is not None:
                try:
                    duration_sec = float(stream["duration"])
                except (TypeError, ValueError):
                    pass
        elif codec_type == "audio" and audio_codec is None:
            audio_codec = stream.get("codec_name")

    return ProbeResult(
        duration_sec=duration_sec,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        fps=fps,
    )
