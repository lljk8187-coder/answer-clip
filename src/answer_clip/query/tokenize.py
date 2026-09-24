"""Lightweight CJK + Latin tokenization for keyword ask.

Strategy (MVP, not perfect NLP):
- Lowercase the text.
- Latin runs: split on non-alphanumeric; keep tokens with length >= 2
  (or length 1 if the whole query is a single letter — rare).
- CJK runs (U+4E00–U+9FFF etc.): emit unigrams, bigrams, and trigrams
  so short Chinese queries still match segment text without a dictionary
  (no jieba / external NLP deps).
- Digits length >= 2 kept as tokens.
"""

from __future__ import annotations

import re

_LATIN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# CJK Unified Ideographs + common extensions used in zh text
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def tokenize(text: str) -> list[str]:
    """Return ordered unique-preserving tokens for scoring."""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for m in _LATIN.finditer(lowered):
        tok = m.group(0)
        if len(tok) >= 2 or len(lowered.strip()) == 1:
            _add(tok)

    for m in _CJK_RUN.finditer(lowered):
        run = m.group(0)
        for ch in run:
            _add(ch)
        for i in range(len(run) - 1):
            _add(run[i : i + 2])
        for i in range(len(run) - 2):
            _add(run[i : i + 3])

    return tokens


def phrase_normalized(text: str) -> str:
    """Compact form for substring / phrase bonus checks."""
    return re.sub(r"\s+", "", text.lower())
