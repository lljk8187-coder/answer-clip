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
from answer_clip.query.ask import AskError, ask, ask_to_json


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
    summary = {k: v for k, v in result.items() if k != "segments"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    if args.no_llm and args.llm:
        print("ask error: use only one of --llm / --no-llm", file=sys.stderr)
        return 2
    use_llm: bool | None
    if args.no_llm:
        use_llm = False
    elif args.llm:
        use_llm = True
    else:
        use_llm = None
    try:
        result = ask(
            args.video_id,
            args.question,
            data_root=data_root,
            top_k=args.top_k,
            pad_sec=args.pad_sec,
            use_llm=use_llm,
        )
    except AskError as exc:
        print(f"ask error: {exc}", file=sys.stderr)
        return 1
    payload = ask_to_json(result)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload, end="")
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

    ask_p = sub.add_parser(
        "ask",
        help=(
            "Keyword search over segments.json (Top-K hits); "
            "optional LLM window rerank when an API key is set."
        ),
    )
    ask_p.add_argument("video_id", help="Ingested video id under data/videos/<id>/.")
    ask_p.add_argument("question", help="Natural-language question / keywords.")
    ask_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    ask_p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max hits to return (default: 5).",
    )
    ask_p.add_argument(
        "--pad-sec",
        type=float,
        default=0.0,
        help="Pad and merge adjacent hit windows by this many seconds.",
    )
    ask_p.add_argument(
        "--no-llm",
        action="store_true",
        help="Force keyword-only scoring (never call LLM).",
    )
    ask_p.add_argument(
        "--llm",
        action="store_true",
        help="Attempt LLM rerank (skips with llm_skipped if no API key).",
    )
    ask_p.add_argument(
        "--out",
        default=None,
        help="Also write QueryResult JSON to this path.",
    )
    ask_p.set_defaults(_handler=_cmd_ask)

    for name, help_text in (
        ("index", "Build or refresh the transcript index (stub)."),
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
