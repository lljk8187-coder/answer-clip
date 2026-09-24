# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

Phase 2 retrieval (still versioned as `0.1.0` until the `0.2.0` cut).

### Added

- **M7 — `index`**: `answer-clip index <video_id>` encodes `segments.json` into
  `embeddings.npz` (vectors + metadata). Optional `[embed]` extra
  (`sentence-transformers`, default `BAAI/bge-small-zh-v1.5`); pluggable
  local ST backend + openai-compatible embeddings stub.
- **M8 — hybrid ask + RRF**: default `--mode hybrid` always runs keyword
  retrieval; when a valid index and `[embed]` are available, adds semantic
  hits and fuses with Reciprocal Rank Fusion before pad / `top_k` / optional
  LLM rerank. Missing index / deps / encode → degrade with `embed_skipped`
  (no failure). `--mode keyword` / `--mode embed`; opt-in `--build-embed`
  (ask never silently downloads models).
- **M9 — CJK trigram**: keyword `tokenize` emits CJK unigram + bigram +
  **trigram** (no jieba / new deps) for better short Chinese recall.

### Docs

- README / CLI help aligned for index, `[embed]`, ask modes, `embed_skipped`,
  and localhost-demo-friendly non-goals (M10).

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
