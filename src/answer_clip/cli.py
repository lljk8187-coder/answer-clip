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
from answer_clip.clip import ClipError, export_clip
from answer_clip.index import EmbedBackendError, IndexBuildError, build_index
from answer_clip.run import RunError, run_pipeline
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



def _cmd_clip(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    try:
        job = export_clip(
            args.video_id,
            start=args.start,
            end=args.end,
            pad_sec=args.pad_sec,
            out=args.out,
            data_root=data_root,
            hit_index=args.hit,
            ask_result=args.ask_result,
        )
    except ClipError as exc:
        print(f"clip error: {exc}", file=sys.stderr)
        return 1
    print(job.model_dump_json(indent=2))
    return 0



def _cmd_run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    download_root = (
        Path(args.download_root).expanduser() if args.download_root else None
    )
    if args.no_llm and args.llm:
        print("run error: use only one of --llm / --no-llm", file=sys.stderr)
        return 2
    if args.no_llm:
        use_llm: bool | None = False
    elif args.llm:
        use_llm = True
    else:
        use_llm = False  # run defaults to keyword-only
    try:
        summary = run_pipeline(
            args.video_path,
            args.question,
            data_root=data_root,
            backend=args.backend,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            download_root=download_root,
            top_k=args.top_k,
            pad_sec=args.pad_sec,
            use_llm=use_llm,
            hit=args.hit,
            out=args.out,
            skip_clip=args.skip_clip,
        )
    except RunError as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0



def _cmd_index(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser() if args.data_dir else None
    download_root = (
        Path(args.download_root).expanduser() if args.download_root else None
    )
    try:
        summary = build_index(
            args.video_id,
            data_root=data_root,
            backend=args.backend,
            model_id=args.model,
            device=args.device,
            download_root=download_root,
        )
    except IndexBuildError as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 1
    except EmbedBackendError as exc:
        print(f"index backend error: {exc}", file=sys.stderr)
        return 1
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

    clip_p = sub.add_parser(
        "clip",
        help=(
            "Export a time range with ffmpeg to data/videos/<id>/clips/ "
            "(stream copy, re-encode fallback)."
        ),
    )
    clip_p.add_argument("video_id", help="Ingested video id under data/videos/<id>/.")
    clip_p.add_argument("--start", type=float, default=None, help="Clip start seconds.")
    clip_p.add_argument("--end", type=float, default=None, help="Clip end seconds.")
    clip_p.add_argument(
        "--pad-sec",
        type=float,
        default=0.0,
        help="Pad start/end by this many seconds before cutting.",
    )
    clip_p.add_argument(
        "--out",
        default=None,
        help="Output media path (default: data/videos/<id>/clips/clip_*.mp4).",
    )
    clip_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    clip_p.add_argument(
        "--hit",
        type=int,
        default=None,
        help="Use start/end from QueryResult hits[index] (needs --ask-result).",
    )
    clip_p.add_argument(
        "--ask-result",
        default=None,
        help="Path to ask QueryResult JSON (for --hit).",
    )
    clip_p.set_defaults(_handler=_cmd_clip)

    run_p = sub.add_parser(
        "run",
        help=(
            "One-shot: ingest → asr → ask → clip top hit "
            "(pass-through model / no-llm / pad-sec / out)."
        ),
    )
    run_p.add_argument("video_path", help="Local media file to ingest.")
    run_p.add_argument("question", help="Question / keywords for ask.")
    run_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    run_p.add_argument(
        "--backend",
        default="faster-whisper",
        help="ASR backend (default: faster-whisper).",
    )
    run_p.add_argument("--model", default="small", help="ASR model size.")
    run_p.add_argument("--device", default="cpu", help="ASR device.")
    run_p.add_argument(
        "--compute-type",
        default="int8",
        help="ASR compute type (default: int8).",
    )
    run_p.add_argument("--language", default=None, help="Force ASR language.")
    run_p.add_argument(
        "--download-root",
        default=None,
        help="ASR model cache directory.",
    )
    run_p.add_argument("--top-k", type=int, default=5, help="Ask top-k.")
    run_p.add_argument(
        "--pad-sec",
        type=float,
        default=0.0,
        help="Pad/merge ask windows before clipping.",
    )
    run_p.add_argument("--no-llm", action="store_true", help="Keyword-only ask.")
    run_p.add_argument("--llm", action="store_true", help="Attempt LLM rerank.")
    run_p.add_argument(
        "--hit",
        type=int,
        default=0,
        help="Which ask hit to clip (default: 0 = top).",
    )
    run_p.add_argument(
        "--out",
        default=None,
        help="Clip output path (default under data/videos/<id>/clips/).",
    )
    run_p.add_argument(
        "--skip-clip",
        action="store_true",
        help="Stop after ask (do not export a clip).",
    )
    run_p.set_defaults(_handler=_cmd_run)


    index_p = sub.add_parser(
        "index",
        help=(
            "Encode segments.json into embeddings.npz "
            "(sentence-transformers; requires [embed] extra)."
        ),
    )
    index_p.add_argument("video_id", help="Ingested video id under data/videos/<id>/.")
    index_p.add_argument(
        "--data-dir",
        default=None,
        help="Data root (default: ./data or $ANSWER_CLIP_DATA).",
    )
    index_p.add_argument(
        "--backend",
        default="sentence-transformers",
        help="Embed backend (default: sentence-transformers; openai-compatible is stub).",
    )
    index_p.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding model id (default: BAAI/bge-small-zh-v1.5).",
    )
    index_p.add_argument(
        "--device",
        default=None,
        help="Optional device for sentence-transformers (e.g. cpu, cuda).",
    )
    index_p.add_argument(
        "--download-root",
        default=None,
        help="Model cache directory (default: <data>/models).",
    )
    index_p.set_defaults(_handler=_cmd_index)

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
