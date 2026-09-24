# answer-clip

Local CLI that turns lecture / course videos into searchable transcripts, then
clips the moments that answer your question.

**Version:** `0.2.0` (Phase 2: hybrid semantic retrieval + localhost `serve` demo)

## Non-goals

- Hosted / multi-tenant SaaS (a **localhost demo** UI is fine and planned)
- Shipping pretrained model weights in the repo
- Remote / hosted vector databases (local `embeddings.npz` only)
- Real-time / live streaming ASR
- Face / person detection
- Auto-upload to short-video platforms

## Requirements

- Python **3.11+**
- System **ffmpeg** and **ffprobe** on `PATH`
- Optional ASR: `faster-whisper` via the `[asr]` extra
- Optional embeddings: `sentence-transformers` via the `[embed]` extra

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# local speech-to-text / embeddings:
pip install -e ".[asr]"
pip install -e ".[embed]"
pip install -e ".[web]"
answer-clip --help
```

## Quick start

```bash
# One-shot: ingest → asr → ask → clip top hit
answer-clip run ./lecture.mp4 "梯度下降是什么" --no-llm --model small

# Or step by step:
answer-clip ingest ./lecture.mp4
answer-clip asr <video_id> --model small
answer-clip index <video_id>          # needs [embed]; writes embeddings.npz
answer-clip ask <video_id> "what is backpropagation" --no-llm --top-k 3
# default ask mode is hybrid (keyword + embed RRF when index exists)
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
| `index <video_id>` | Encode `segments.json` → `embeddings.npz` (`[embed]` extra) |
| `serve` | Localhost demo UI (ask → hits → clip; `[web]` extra, binds `127.0.0.1`) |
| `ask <video_id> "…"` | Hybrid / keyword / embed retrieval + optional LLM rerank |
| `clip <video_id>` | ffmpeg export (`--start/--end` or `--hit` + `--ask-result`) |
| `run <path> "…"` | ingest → asr → ask → clip |

## Index (`[embed]`)

```bash
pip install -e ".[embed]"
pip install -e ".[web]"
answer-clip index <video_id>
# default model: BAAI/bge-small-zh-v1.5
answer-clip index <video_id> --model BAAI/bge-small-zh-v1.5 --device cpu
```

Writes `data/videos/<id>/embeddings.npz` with `vectors`, `starts` / `ends`,
`model_id`, `dim`, `video_id`, `backend`, `segment_count`.

Ask does **not** download embedding weights by default. Pass
`--build-embed` only when you explicitly want ask to run `index` first.

## Ask modes

Default **`--mode hybrid`**:

1. Always run keyword retrieval (Latin words + CJK uni/bi/**trigram**).
2. If a valid `embeddings.npz` exists and `[embed]` can encode the query,
   add semantic hits and fuse with **RRF**.
3. Optional LLM window rerank still runs afterward (when enabled / keyed).

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Keyword + embed RRF; missing index / `[embed]` / encode failure → keyword only, `embed_skipped` set (does **not** fail) |
| `keyword` | Lexical only (`embed_skipped: mode=keyword`) |
| `embed` | Semantic only; missing `embeddings.npz` (or encode failure) → clear error |

Useful flags:

```bash
answer-clip ask <id> "优化算法" --mode hybrid --no-llm
answer-clip ask <id> "梯度下降" --mode keyword --no-llm
answer-clip ask <id> "backpropagation" --mode embed --no-llm
answer-clip ask <id> "…" --build-embed   # opt-in: index then ask
```

`QueryResult` fields of note: `mode`, `embed_used`, `embed_skipped`,
`llm_used`, `llm_skipped`, `hits`.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `ANSWER_CLIP_DATA` | Data root (default `./data`) |
| `ANSWER_CLIP_MODEL_CACHE` | Model weight cache (default `<data>/models`) |
| `OPENAI_API_KEY` / `ANSWER_CLIP_LLM_API_KEY` | Enable optional ask LLM rerank |
| `ANSWER_CLIP_LLM_BASE_URL` | OpenAI-compatible base URL |
| `ANSWER_CLIP_LLM_MODEL` | Chat model name (default `gpt-4o-mini`) |
| `ANSWER_CLIP_ASR_SLOW=1` | Opt into slow real-Whisper pytest |

## Serve (localhost demo)

```bash
pip install -e ".[web]"
answer-clip serve          # http://127.0.0.1:8765/  (loopback only)
```

No auth. Not for multi-tenant or public internet.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

## License

MIT — see [LICENSE](LICENSE).
