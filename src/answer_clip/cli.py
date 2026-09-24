"""CLI entry for answer-clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from answer_clip import __version__
from answer_clip.ffprobe import FfprobeError
from answer_clip.ingest import IngestError, ingest


def _cmd_stub(name: str) -> int:
    print(
        f"answer-clip {name}: not implemented yet.",
        file=sys.stderr,
    )
    return 2


def _cmd_ingest(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    try:
        meta = ingest(args.path, data_root=data_root)
    except IngestError as exc:
        print(f"ingest error: {exc}", file=sys.stderr)
        return 1
    except FfprobeError as exc:
        print(f"ffprobe error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


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

    ingest_p = sub.add_parser(
        "ingest",
        help=(
            "Register a local video under data/videos/<id>/ "
            "(hardlink preferred, else copy; write meta.json via ffprobe)."
        ),
    )
    ingest_p.add_argument("path", help="Path to a local media file.")
    ingest_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    ingest_p.set_defaults(_handler=_cmd_ingest)

    for name, help_text in (
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
