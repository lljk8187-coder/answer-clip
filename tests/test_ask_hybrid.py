"""Hybrid ask: RRF over keyword + mock embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from answer_clip.cli import main
from answer_clip.index.pipeline import write_embeddings_npz
from answer_clip.models import QueryHit
from answer_clip.query.ask import AskError, ask
from answer_clip.query.rrf import reciprocal_rank_fusion


class _MockEmbed:
    """Maps a few probe phrases to fixed vectors; else zeros."""

    name = "mock"
    model_id = "mock-dim-4"

    def __init__(self, query_vec: np.ndarray):
        self._query_vec = np.asarray(query_vec, dtype=np.float32)

    @property
    def dim(self) -> int:
        return 4

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            low = (t or "").lower()
            if "optimization" in low or "优化" in t:
                rows.append(self._query_vec.copy())
            else:
                rows.append(np.zeros(4, dtype=np.float32))
        return np.stack(rows, axis=0)


@pytest.fixture()
def hybrid_video(tmp_path: Path) -> tuple[str, Path]:
    video_id = "vid_hybrid"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to the lecture."},
        {
            "start": 2.0,
            "end": 5.0,
            "text": "Today we explain gradient descent step by step.",
        },
        {
            "start": 5.0,
            "end": 8.0,
            "text": "Completely unrelated iterative cooking tips for pasta.",
        },
        {"start": 8.0, "end": 11.0, "text": "接下来讲梯度下降的直观含义。"},
    ]
    (vdir / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False), encoding="utf-8"
    )
    # Unit vectors: index 1 = GD English, 2 = pasta, 3 = GD Chinese
    vectors = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    starts = np.array([0.0, 2.0, 5.0, 8.0], dtype=np.float64)
    ends = np.array([2.0, 5.0, 8.0, 11.0], dtype=np.float64)
    write_embeddings_npz(
        vdir / "embeddings.npz",
        vectors=vectors,
        starts=starts,
        ends=ends,
        model_id="mock-dim-4",
        video_id=video_id,
        backend="mock",
    )
    return video_id, tmp_path


def test_rrf_prefers_consensus() -> None:
    # b is rank-1 in both lists; a is rank-2 only in one list.
    a = [
        QueryHit(start=3.0, end=4.0, text="b", score=0.5, evidence="kw"),
        QueryHit(start=1.0, end=2.0, text="a", score=0.4, evidence="kw"),
    ]
    b = [
        QueryHit(start=3.0, end=4.0, text="b", score=0.9, evidence="emb"),
        QueryHit(start=5.0, end=6.0, text="c", score=0.2, evidence="emb"),
    ]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert fused[0].start == 3.0  # consensus rank-1 → highest RRF


def test_hybrid_elevates_semantic_synonym(hybrid_video: tuple[str, Path]) -> None:
    video_id, data_root = hybrid_video
    question = "iterative optimization"  # keyword hits pasta via "iterative"
    backend = _MockEmbed(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    kw = ask(
        video_id,
        question,
        data_root=data_root,
        mode="keyword",
        use_llm=False,
        top_k=3,
        embed_backend=backend,
    )
    assert kw.mode == "keyword"
    assert kw.embed_used is False
    assert kw.hits
    assert "pasta" in kw.hits[0].text.lower() or "iterative" in kw.hits[0].text.lower()

    hyb = ask(
        video_id,
        question,
        data_root=data_root,
        mode="hybrid",
        use_llm=False,
        top_k=3,
        embed_backend=backend,
    )
    assert hyb.mode == "hybrid"
    assert hyb.embed_used is True
    assert hyb.embed_skipped is None
    assert hyb.hits
    top_text = hyb.hits[0].text.lower()
    assert "gradient descent" in top_text or "梯度下降" in hyb.hits[0].text
    # Semantic-relevant hit must outrank pure keyword pasta decoy.
    assert "pasta" not in top_text


def test_hybrid_no_npz_embed_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = "vid_no_npz"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    (vdir / "segments.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "gradient descent"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ask(
        video_id,
        "gradient descent",
        data_root=tmp_path,
        mode="hybrid",
        use_llm=False,
        top_k=2,
    )
    assert result.embed_used is False
    assert result.embed_skipped and "embeddings.npz" in result.embed_skipped
    assert result.hits
    assert "gradient" in result.hits[0].text.lower()


def test_keyword_mode_ignores_npz(hybrid_video: tuple[str, Path]) -> None:
    video_id, data_root = hybrid_video
    backend = _MockEmbed(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    result = ask(
        video_id,
        "iterative optimization",
        data_root=data_root,
        mode="keyword",
        use_llm=False,
        top_k=2,
        embed_backend=backend,
    )
    assert result.mode == "keyword"
    assert result.embed_used is False
    assert result.embed_skipped == "mode=keyword"
    assert "pasta" in result.hits[0].text.lower()


def test_embed_mode_requires_npz(tmp_path: Path) -> None:
    video_id = "vid_embed_only"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    (vdir / "segments.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "hi"}], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(AskError, match="embeddings.npz"):
        ask(
            video_id,
            "hi",
            data_root=tmp_path,
            mode="embed",
            use_llm=False,
            embed_backend=_MockEmbed(np.ones(4, dtype=np.float32)),
        )


def test_embed_mode_with_npz(hybrid_video: tuple[str, Path]) -> None:
    video_id, data_root = hybrid_video
    backend = _MockEmbed(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    result = ask(
        video_id,
        "optimization",
        data_root=data_root,
        mode="embed",
        use_llm=False,
        top_k=2,
        embed_backend=backend,
    )
    assert result.mode == "embed"
    assert result.embed_used is True
    assert "gradient" in result.hits[0].text.lower() or "梯度" in result.hits[0].text


def test_cli_ask_mode_keyword(
    hybrid_video: tuple[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id, data_root = hybrid_video
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWER_CLIP_LLM_API_KEY", raising=False)
    code = main(
        [
            "ask",
            video_id,
            "iterative cooking",
            "--data-dir",
            str(data_root),
            "--mode",
            "keyword",
            "--no-llm",
            "--top-k",
            "2",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "keyword"
    assert payload["embed_used"] is False
