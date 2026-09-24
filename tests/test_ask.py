"""ask: keyword hits + optional LLM rerank (mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answer_clip.cli import main
from answer_clip.models import QueryHit, QueryResult
from answer_clip.query.ask import AskError, ask
from answer_clip.query.merge import merge_hits
from answer_clip.query.score import rank_segments
from answer_clip.query.tokenize import tokenize
from answer_clip.models import SubtitleSegment


FIXTURE = Path(__file__).parent / "fixtures" / "segments_lecture.json"


@pytest.fixture()
def video_data(tmp_path: Path) -> tuple[str, Path]:
    video_id = "vid_lecture"
    vdir = tmp_path / "videos" / video_id
    vdir.mkdir(parents=True)
    (vdir / "segments.json").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return video_id, tmp_path


def test_tokenize_latin_and_cjk() -> None:
    toks = tokenize("Gradient Descent 梯度下降")
    assert "gradient" in toks
    assert "descent" in toks
    assert "梯" in toks
    assert "梯度" in toks
    assert "梯度下" in toks
    assert "度下降" in toks


def test_tokenize_cjk_trigram_short_phrase() -> None:
    """CJK trigrams improve short Chinese query matching (no jieba)."""
    toks = tokenize("梯度下降算法")
    assert "梯度下" in toks
    assert "度下降" in toks
    assert "下降算" in toks
    assert "降算法" in toks
    # unigram / bigram still present
    assert "梯" in toks
    assert "下降" in toks
    # Latin path untouched
    assert tokenize("Gradient Descent") == ["gradient", "descent"]


def test_tokenize_cjk_trigram_scores_hit() -> None:
    from answer_clip.models import SubtitleSegment
    from answer_clip.query.score import rank_segments

    segs = [
        SubtitleSegment(start=0.0, end=1.0, text="今天讲完全无关的烹饪技巧。"),
        SubtitleSegment(start=1.0, end=3.0, text="本节介绍梯度下降算法的步骤。"),
    ]
    # Trigram overlap with 梯度下降算法 even if query is a 3-char slice.
    hits = rank_segments("度下降", segs, top_k=2)
    assert hits
    assert "梯度下降" in hits[0].text


def test_rank_hits_known_keyword() -> None:
    segs = [SubtitleSegment.model_validate(x) for x in json.loads(FIXTURE.read_text())]
    hits = rank_segments("gradient descent", segs, top_k=3)
    assert hits
    assert any("gradient" in h.text.lower() for h in hits)
    assert hits[0].score > 0


def test_rank_hits_chinese() -> None:
    segs = [SubtitleSegment.model_validate(x) for x in json.loads(FIXTURE.read_text())]
    hits = rank_segments("梯度下降", segs, top_k=3)
    assert hits
    assert any("梯度" in h.text for h in hits)


def test_ask_no_llm(video_data: tuple[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    video_id, data_root = video_data
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWER_CLIP_LLM_API_KEY", raising=False)
    result = ask(video_id, "gradient descent", data_root=data_root, use_llm=False, top_k=3)
    assert isinstance(result, QueryResult)
    assert result.llm_used is False
    assert result.llm_skipped is not None
    assert result.hits
    assert result.hits[0].start < result.hits[0].end
    assert "gradient" in result.hits[0].text.lower() or any(
        "gradient" in h.text.lower() for h in result.hits
    )


def test_ask_llm_skipped_without_key(
    video_data: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id, data_root = video_data
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWER_CLIP_LLM_API_KEY", raising=False)
    result = ask(video_id, "backpropagation", data_root=data_root, use_llm=True, top_k=2)
    assert result.llm_used is False
    assert result.llm_skipped and "API key" in result.llm_skipped
    assert result.hits
    assert any("backpropagation" in h.text.lower() for h in result.hits)


def test_ask_mock_llm_rerank(video_data: tuple[str, Path]) -> None:
    video_id, data_root = video_data

    def _rerank(question: str, hits: list[QueryHit]) -> list[QueryHit]:
        assert question
        # Reverse order and bump scores to prove rerank ran.
        out = list(reversed(hits))
        return [
            QueryHit(
                start=h.start,
                end=h.end,
                text=h.text,
                score=1.0 - i * 0.01,
                evidence="mock-llm",
            )
            for i, h in enumerate(out)
        ]

    result = ask(
        video_id,
        "learning",
        data_root=data_root,
        use_llm=True,
        top_k=5,
        rerank_fn=_rerank,
    )
    assert result.llm_used is True
    assert result.llm_skipped is None
    assert result.hits
    assert result.hits[0].evidence == "mock-llm"


def test_merge_pad_sec() -> None:
    hits = [
        QueryHit(start=1.0, end=2.0, text="a", score=0.5, evidence="a"),
        QueryHit(start=2.5, end=3.0, text="b", score=0.8, evidence="b"),
    ]
    merged = merge_hits(hits, pad_sec=0.5)
    assert len(merged) == 1
    assert merged[0].start == pytest.approx(0.5)
    assert merged[0].end == pytest.approx(3.5)
    assert merged[0].score == 0.8


def test_ask_missing_segments(tmp_path: Path) -> None:
    with pytest.raises(AskError, match="segments.json"):
        ask("missing", "q", data_root=tmp_path, use_llm=False)


def test_cli_ask_json(
    video_data: tuple[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id, data_root = video_data
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWER_CLIP_LLM_API_KEY", raising=False)
    out = data_root / "ask-out.json"
    code = main(
        [
            "ask",
            video_id,
            "梯度下降",
            "--data-dir",
            str(data_root),
            "--no-llm",
            "--top-k",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["video_id"] == video_id
    assert payload["hits"]
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["hits"]


def test_cli_ask_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--help"])
    assert exc.value.code == 0
