# answer-clip

Local CLI that turns lecture / course videos into searchable transcripts, then
clips the moments that answer your question.

**Version:** `0.1.0a1` (Phase 1)

## Non-goals (for now)

- Hosted / multi-tenant SaaS
- A web UI or browser extension
- Shipping pretrained models inside the repo
- Real-time streaming ASR
- Automatic upload to YouTube / cloud storage

## Install

Requires Python 3.11+ and system **ffmpeg/ffprobe**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# optional local ASR:
pip install -e ".[asr]"
answer-clip --help
```

## Ingest

```bash
answer-clip ingest /path/to/lecture.mp4
# optional: --data-dir /tmp/ac-data
```

Prefers a **hardlink** into `data/videos/<id>/`; falls back to **copy**. Writes
`meta.json` via `ffprobe`. Override root with `--data-dir` or `$ANSWER_CLIP_DATA`.

## ASR

```bash
answer-clip asr <video_id>
answer-clip asr --path /path/to/media.mp4
# options: --backend faster-whisper --model small --device cpu --compute-type int8
```

Writes `segments.json` (`SubtitleSegment`: start/end/text/confidence?) and
`transcript.srt` next to the media (under the video dir when using `video_id`).
Segments are sorted and de-overlapped (later starts clamped to previous end).

Default backend is **faster-whisper** (install `[asr]`). Model weights download
into `data/models/` (gitignored) or `$ANSWER_CLIP_MODEL_CACHE`. An
`openai-compatible` backend name is reserved as a stub.

## Ask

```bash
answer-clip ask <video_id> "梯度下降是什么" --no-llm
answer-clip ask <video_id> "what is backpropagation" --top-k 3 --pad-sec 1.5
# optional LLM rerank when OPENAI_API_KEY or ANSWER_CLIP_LLM_API_KEY is set:
answer-clip ask <video_id> "..." --llm
```

Default path is keyword scoring (Latin words length≥2; CJK unigrams+bigrams).
With an API key, windows may be reranked via an OpenAI-compatible chat API;
without a key the result sets `llm_skipped` and still returns keyword hits.
JSON goes to stdout (and optional `--out`).

## Layout

```
src/answer_clip/
  cli.py
  ingest.py / ffprobe.py
  asr/          # backends + pipeline
  query/        # ask: keyword ± optional LLM rerank
  models.py     # VideoMeta, SubtitleSegment, QueryResult, …
  paths.py
```

## License

MIT — see [LICENSE](LICENSE).
