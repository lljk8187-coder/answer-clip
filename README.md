# answer-clip

Local CLI that turns lecture / course videos into searchable transcripts, then
clips the moments that answer your question.

**Version:** `0.1.0a1` (Phase 1 scaffold)

## Non-goals (for now)

- Hosted / multi-tenant SaaS
- A web UI or browser extension
- Shipping pretrained models inside the repo
- Real-time streaming ASR
- Automatic upload to YouTube / cloud storage

## Install (preview)

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
answer-clip --help
```

Subcommands (`ingest`, `asr`, `index`, `query`, `clip`, …) are placeholders in
this release; ASR / ffmpeg / embedding logic lands in later milestones.

## Layout

```
src/answer_clip/
  cli.py      # entry: answer-clip
  ingest.py   # media ingest (stub)
  asr.py      # speech-to-text (stub)
  index.py    # transcript index (stub)
  query.py    # question → hit spans (stub)
  clip.py     # cut media by span (stub)
  models.py   # shared types (stub)
  paths.py    # data/cache path helpers (stub)
```

## License

MIT — see [LICENSE](LICENSE).
