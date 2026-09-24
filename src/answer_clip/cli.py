"""CLI entry for answer-clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from answer_clip import __version__
from answer_clip.asr.base import AsrBackendError
from answer_clip.asr.pipeline import AsrError, run_asr
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


def _cmd_asr(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    download_root = (
        Path(args.download_root).expanduser() if args.download_root else None
    )
    try:
        result = run_asr(
            video_id=args.video_id,
            path=args.path,
            data_root=data_root,
            backend=args.backend,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            download_root=download_root,
        )
    except AsrError as exc:
        print(f"asr error: {exc}", file=sys.stderr)
        return 1
    except AsrBackendError as exc:
        print(f"asr backend error: {exc}", file=sys.stderr)
        return 1
    # Omit full segment dump from stdout summary (still on disk).
    summary = {k: v for k, v in result.items() if k != "segments"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
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

    asr_p = sub.add_parser(
        "asr",
        help=(
            "Transcribe an ingested video (or --path) with faster-whisper; "
            "write segments.json + transcript.srt."
        ),
    )
    asr_p.add_argument(
        "video_id",
        nargs="?",
        default=None,
        help="Ingested video id under data/videos/<id>/.",
    )
    asr_p.add_argument(
        "--path",
        default=None,
        help="Media file path (alternative to video_id).",
    )
    asr_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    asr_p.add_argument(
        "--backend",
        default="faster-whisper",
        help="ASR backend (default: faster-whisper; openai-compatible is stub).",
    )
    asr_p.add_argument(
        "--model",
        default="small",
        help="faster-whisper model size (default: small).",
    )
    asr_p.add_argument(
        "--device",
        default="cpu",
        help="Device for faster-whisper (default: cpu).",
    )
    asr_p.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 compute type (default: int8, CPU-friendly).",
    )
    asr_p.add_argument(
        "--language",
        default=None,
        help="Force language code (default: auto-detect).",
    )
    asr_p.add_argument(
        "--download-root",
        default=None,
        help="Model cache directory (default: <data>/models or $ANSWER_CLIP_MODEL_CACHE).",
    )
    asr_p.set_defaults(_handler=_cmd_asr)

    for name, help_text in (
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
