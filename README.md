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

Requires Python 3.11+ and system **ffmpeg/ffprobe** (used by `ingest`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
answer-clip --help
```

## Ingest

Register a local media file under `data/videos/<id>/`:

```bash
answer-clip ingest /path/to/lecture.mp4
# optional: answer-clip ingest ./clip.mp4 --data-dir /tmp/ac-data
```

Behaviour:

1. Compute a content-hash id (SHA-256 prefix).
2. Prefer a **hardlink** into `data/videos/<id>/<filename>`; if that fails
   (cross-device), **copy** the file.
3. Run `ffprobe` and write `meta.json` (`VideoMeta`: duration, width/height,
   codecs, fps, source path, copy strategy, …).

`data/` (including large media) is gitignored. Override the data root with
`--data-dir` or `$ANSWER_CLIP_DATA`.

Subcommands `asr` / `index` / `query` / `clip` are still placeholders.

## Layout

```
src/answer_clip/
  cli.py
  ingest.py   # register local media + meta.json
  ffprobe.py  # system ffprobe wrapper
  asr.py      # stub
  index.py    # stub
  query.py    # stub
  clip.py     # stub
  models.py   # VideoMeta, HitSpan
  paths.py    # data/videos/<id>/ helpers
```

## License

MIT — see [LICENSE](LICENSE).
