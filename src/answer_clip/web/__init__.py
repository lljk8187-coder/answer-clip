"""Optional localhost demo UI (``answer-clip serve``; requires ``[web]``)."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from answer_clip.web.app import create_app as _create

    return _create(*args, **kwargs)
