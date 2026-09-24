"""CLI help mentions Phase 2 retrieval (hybrid / index)."""

from __future__ import annotations

import pytest

from answer_clip.cli import main


def test_root_help_mentions_hybrid(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "hybrid" in out
    assert "index" in out


def test_ask_help_mentions_modes_and_build_embed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "--mode" in out
    assert "hybrid" in out
    assert "embed_skipped" in out or "embed" in out
    assert "--build-embed" in out


def test_index_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["index", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "embeddings.npz" in out or "npz" in out
