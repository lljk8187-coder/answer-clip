# answer-clip

Local CLI that turns lecture / course videos into searchable transcripts, then
clips the moments that answer your question.

**Version:** `0.1.0` (Phase 1)

## Non-goals

- Hosted / multi-tenant SaaS or Web UI
- Embedding vector databases
- Shipping pretrained model weights in the repo
- Real-time / live streaming ASR
- Face / person detection
- Auto-upload to short-video platforms

## Requirements

- Python **3.11+**
- System **ffmpeg** and **ffprobe** on `PATH`
- Optional ASR: `faster-whisper` via the `[asr]` extra

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# local speech-to-text:
pip install -e ".[asr]"
answer-clip --help
```

## Quick start

```bash
# One-shot: ingest → asr → ask → clip top hit
answer-clip run ./lecture.mp4 "梯度下降是什么" --no-llm --model small

# Or step by step:
answer-clip ingest ./lecture.mp4
answer-clip asr <video_id> --model small
answer-clip ask <video_id> "what is backpropagation" --no-llm --top-k 3
answer-clip clip <video_id> --hit 0 --ask-result ask.json
# or: answer-clip clip <video_id> --start 12.0 --end 18.5
```

`run` prints a JSON summary (video id, ask hits, clip `ExportJob`). Clips land
under `data/videos/<id>/clips/` unless you pass `--out`.

## Subcommands

| Command | Purpose |
|---------|---------|
| `ingest <path>` | Register media under `data/videos/<id>/` + `meta.json` (ffprobe) |
| `asr <video_id\|--path>` | Write `segments.json` + `transcript.srt` (faster-whisper) |
| `ask <video_id> "…"` | Keyword Top-K hits; optional LLM window rerank |
| `clip <video_id>` | ffmpeg export (`--start/--end` or `--hit` + `--ask-result`) |
| `run <path> "…"` | ingest → asr → ask → clip |
| `index` | Stub (later phases) |

## Environment variables

| Variable | Meaning |
|----------|---------|
| `ANSWER_CLIP_DATA` | Data root (default `./data`) |
| `ANSWER_CLIP_MODEL_CACHE` | Whisper weight cache (default `<data>/models`) |
| `OPENAI_API_KEY` / `ANSWER_CLIP_LLM_API_KEY` | Enable optional ask LLM rerank |
| `ANSWER_CLIP_LLM_BASE_URL` | OpenAI-compatible base URL |
| `ANSWER_CLIP_LLM_MODEL` | Chat model name (default `gpt-4o-mini`) |
| `ANSWER_CLIP_ASR_SLOW=1` | Opt into slow real-Whisper pytest |

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

## License

MIT — see [LICENSE](LICENSE).
