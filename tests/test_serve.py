"""Localhost serve: TestClient routes with mock data (no Whisper/GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answer_clip.cli import main
from answer_clip.models import ExportJob


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from answer_clip.web.app import create_app  # noqa: E402


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    video_id = "vid_web"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to the lecture."},
        {"start": 2.0, "end": 5.0, "text": "Today we explain gradient descent."},
        {"start": 5.0, "end": 7.0, "text": "Unrelated pasta tips."},
    ]
    (vdir / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False), encoding="utf-8"
    )
    # Empty sibling without segments should be ignored
    (tmp_path / "videos" / "empty").mkdir()
    return tmp_path


@pytest.fixture()
def client(data_root: Path) -> TestClient:
    return TestClient(create_app(data_root=data_root))


def test_list_videos_and_home(client: TestClient) -> None:
    r = client.get("/api/videos")
    assert r.status_code == 200
    payload = r.json()
    assert payload["video_ids"] == ["vid_web"]

    home = client.get("/")
    assert home.status_code == 200
    assert b"vid_web" in home.content
    assert b"localhost" in home.content.lower()


def test_api_ask(client: TestClient) -> None:
    r = client.post(
        "/api/ask",
        json={
            "video_id": "vid_web",
            "question": "gradient descent",
            "mode": "keyword",
            "top_k": 3,
            "use_llm": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "keyword"
    assert data["hits"]
    assert any("gradient" in h["text"].lower() for h in data["hits"])


def test_api_ask_missing_video(client: TestClient) -> None:
    r = client.post(
        "/api/ask",
        json={"video_id": "nope", "question": "x", "mode": "keyword"},
    )
    assert r.status_code == 400


def test_api_clip_mocked(client: TestClient, data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = data_root / "videos" / "vid_web" / "clips" / "clip_demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"fake-mp4")

    def _fake_export(video_id, **kwargs):
        return ExportJob(
            video_id=video_id,
            source_path=str(data_root / "videos" / video_id / "x.mp4"),
            start=float(kwargs.get("start") or 0),
            end=float(kwargs.get("end") or 1),
            pad_sec=float(kwargs.get("pad_sec") or 0),
            output_path=str(out),
            duration_sec=1.0,
            codec_mode="copy",
        )

    monkeypatch.setattr("answer_clip.web.app.export_clip", _fake_export)
    r = client.post(
        "/api/clip",
        json={"video_id": "vid_web", "start": 2.0, "end": 5.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["output_path"] == str(out)
    assert body["download_url"] == "/files/clips/vid_web/clip_demo.mp4"

    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert dl.content == b"fake-mp4"


def test_cli_serve_missing_web_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("answer_clip.web") or name in {"fastapi", "uvicorn"}:
            raise ImportError("simulated missing web")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Also clear cached modules if any
    import sys

    for key in list(sys.modules):
        if key.startswith("answer_clip.web"):
            sys.modules.pop(key, None)

    code = main(["serve", "--host", "127.0.0.1", "--port", "8765"])
    assert code == 1
    err = capsys.readouterr().err
    assert "answer-clip[web]" in err or "[web]" in err


def test_cli_serve_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
