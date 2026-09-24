# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] — 2026-09-24

Phase 1 MVP.

### Added

- Project scaffold (`answer-clip` CLI, MIT license, hatchling packaging).
- `ingest`: register local media under `data/videos/<id>/` with ffprobe `VideoMeta`.
- `asr`: faster-whisper backend (optional `[asr]` extra), `segments.json` + `transcript.srt`,
  pluggable backend protocol with openai-compatible stub.
- `ask`: CJK/Latin keyword retrieval, optional LLM window rerank, `--pad-sec` merge,
  `QueryResult` JSON.
- `clip`: ffmpeg export with stream-copy then re-encode fallback, `ExportJob` sidecar.
- `run`: one-shot ingest → asr → ask → clip.
- pytest suite (slow Whisper gated); optional GitHub Actions CI for `not slow`.

### Notes

- Ends the `0.1.0a1` pre-release line.
