"""Optional LLM window rerank (OpenAI-compatible HTTP)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from answer_clip.models import QueryHit

RerankFn = Callable[[str, list[QueryHit]], list[QueryHit]]


def llm_config_from_env() -> dict[str, str | None]:
    api_key = (
        os.environ.get("ANSWER_CLIP_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANSWER_CLIP_LLM_KEY")
    )
    base_url = (
        os.environ.get("ANSWER_CLIP_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    model = (
        os.environ.get("ANSWER_CLIP_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    return {"api_key": api_key, "base_url": base_url, "model": model}


def default_llm_rerank(question: str, hits: list[QueryHit]) -> list[QueryHit]:
    """Ask an OpenAI-compatible chat API to reorder / rescore candidate windows."""
    cfg = llm_config_from_env()
    api_key = cfg["api_key"]
    if not api_key:
        raise RuntimeError("no LLM API key configured")

    catalog = [
        {"index": i, "start": h.start, "end": h.end, "text": h.text, "score": h.score}
        for i, h in enumerate(hits)
    ]
    system = (
        "You rerank transcript windows for a user question. "
        "Reply with ONLY JSON: "
        '{"order":[{"index":0,"score":0.9,"evidence":"..."}, ...]} '
        "covering each candidate index exactly once. "
        "Higher score = more relevant."
    )
    user = json.dumps(
        {"question": question, "candidates": catalog},
        ensure_ascii=False,
    )
    body: dict[str, Any] = {
        "model": cfg["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc

    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    order = parsed.get("order") or parsed.get("hits") or []
    by_index = {i: h for i, h in enumerate(hits)}
    reranked: list[QueryHit] = []
    seen: set[int] = set()
    for item in order:
        idx = int(item["index"])
        if idx not in by_index or idx in seen:
            continue
        seen.add(idx)
        base = by_index[idx]
        score = float(item.get("score", base.score))
        evidence = str(item.get("evidence") or base.evidence)
        reranked.append(
            QueryHit(
                start=base.start,
                end=base.end,
                text=base.text,
                score=score,
                evidence=evidence,
            )
        )
    for i, h in enumerate(hits):
        if i not in seen:
            reranked.append(h)
    reranked.sort(key=lambda h: (-h.score, h.start))
    return reranked
