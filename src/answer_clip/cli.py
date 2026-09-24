"""CLI entry for answer-clip (Phase 1 stub)."""

from __future__ import annotations

import argparse
import sys

from answer_clip import __version__


def _cmd_stub(name: str) -> int:
    print(
        f"answer-clip {name}: not implemented yet (Phase 1 scaffold).",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="answer-clip",
        description=(
            "Search lecture/video transcripts locally and clip matching moments."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, help_text in (
        ("ingest", "Ingest media into the local data store (stub)."),
        ("asr", "Run speech-to-text on ingested media (stub)."),
        ("index", "Build or refresh the transcript index (stub)."),
        ("query", "Ask a question against indexed transcripts (stub)."),
        ("clip", "Cut a media clip for a hit span (stub)."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(_handler=lambda _args, n=name: _cmd_stub(n))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
