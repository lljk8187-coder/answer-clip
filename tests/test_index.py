"""index → embeddings.npz (mock encoder; no model download)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from answer_clip.cli import main
from answer_clip.index import EmbedBackendError, IndexBuildError, build_index, get_backend
from answer_clip.index.pipeline import write_embeddings_npz


class _FakeEmbed:
    name = "fake"
    model_id = "fake-dim-8"

    @property
    def dim(self) -> int:
        return 8

    def encode(self, texts: list[str]) -> np.ndarray:
        # Deterministic tiny vectors from text length.
        rows = []
        for i, t in enumerate(texts):
            v = np.zeros(8, dtype=np.float32)
            v[0] = float(len(t))
            v[1] = float(i)
            n = np.linalg.norm(v) or 1.0
            rows.append(v / n)
        return np.stack(rows, axis=0)


@pytest.fixture()
def video_with_segments(tmp_path: Path) -> tuple[str, Path]:
    video_id = "vid_embed"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    segs = [
        {"start": 0.0, "end": 1.0, "text": "gradient descent"},
        {"start": 1.0, "end": 2.5, "text": "梯度下降"},
        {"start": 2.5, "end": 4.0, "text": "unrelated pasta"},
    ]
    (vdir / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False), encoding="utf-8"
    )
    return video_id, tmp_path


def test_write_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    vectors = np.eye(3, 4, dtype=np.float32)
    write_embeddings_npz(
        path,
        vectors=vectors,
        starts=np.array([0.0, 1.0, 2.0]),
        ends=np.array([1.0, 2.0, 3.0]),
        model_id="m",
        video_id="v",
        backend="fake",
    )
    data = np.load(path, allow_pickle=False)
    assert data["vectors"].shape == (3, 4)
    assert int(data["dim"]) == 4
    assert str(data["model_id"]) == "m"
    assert int(data["segment_count"]) == 3


def test_build_index_mock(video_with_segments: tuple[str, Path]) -> None:
    video_id, data_root = video_with_segments
    summary = build_index(
        video_id,
        data_root=data_root,
        backend_impl=_FakeEmbed(),
    )
    assert summary["segment_count"] == 3
    assert summary["dim"] == 8
    path = Path(summary["embeddings_path"])
    assert path.is_file()
    data = np.load(path, allow_pickle=False)
    assert data["vectors"].shape == (3, 8)
    assert str(data["video_id"]) == video_id
    assert str(data["model_id"]) == "fake-dim-8"


def test_build_index_missing_segments(tmp_path: Path) -> None:
    with pytest.raises(IndexBuildError, match="segments.json"):
        build_index("missing", data_root=tmp_path, backend_impl=_FakeEmbed())


def test_missing_embed_dep_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers"):
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    backend = get_backend("sentence-transformers")
    with pytest.raises(EmbedBackendError, match=r"answer-clip\[embed\]"):
        backend.encode(["hello"])


def test_openai_embed_stub() -> None:
    backend = get_backend("openai-compatible")
    with pytest.raises(EmbedBackendError, match="not implemented"):
        backend.encode(["x"])


def test_cli_index_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["index", "--help"])
    assert exc.value.code == 0


def test_cli_index_mock_via_missing(
    video_with_segments: tuple[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    video_id, data_root = video_with_segments
    # CLI always uses real backend; missing segments path already covered.
    # Here: unknown video without segments.
    code = main(["index", "nope", "--data-dir", str(data_root)])
    assert code == 1
    assert "index error" in capsys.readouterr().err
