"""Run uvicorn for ``answer-clip serve``."""

from __future__ import annotations

from pathlib import Path


def run_serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_root: Path | None = None,
) -> None:
    import uvicorn

    from answer_clip.web.app import create_app

    app = create_app(data_root=data_root)
    # Bind defaults to loopback; do not advertise public exposure.
    uvicorn.run(app, host=host, port=port, log_level="info")
