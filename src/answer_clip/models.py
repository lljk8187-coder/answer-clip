"""Shared data types (stub)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitSpan:
    """A transcript span that answers a query (placeholder)."""

    start_sec: float
    end_sec: float
    text: str = ""
