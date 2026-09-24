"""Data and cache path helpers (stub)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")


def data_dir(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / DEFAULT_DATA_DIR
