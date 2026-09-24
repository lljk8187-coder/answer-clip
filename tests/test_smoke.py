"""Minimal smoke tests for the scaffold / CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from answer_clip import __version__
from answer_clip.cli import build_parser, main


def test_version_constant() -> None:
    assert __version__ == "0.1.0a1"


def test_import_package() -> None:
    import answer_clip.asr  # noqa: F401
    import answer_clip.asr.pipeline  # noqa: F401
    import answer_clip.clip  # noqa: F401
    import answer_clip.ffprobe  # noqa: F401
    import answer_clip.index  # noqa: F401
    import answer_clip.ingest  # noqa: F401
    import answer_clip.models  # noqa: F401
    import answer_clip.paths  # noqa: F401
    import answer_clip.query  # noqa: F401
    import answer_clip.query.ask  # noqa: F401


def test_help_exit_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_ingest_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0


def test_build_parser_help() -> None:
    parser = build_parser()
    text = parser.format_help()
    assert "answer-clip" in text
    assert "ingest" in text


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "answer-clip" in out


def test_cli_help_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "answer_clip", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "answer-clip" in result.stdout
