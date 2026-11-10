"""Launch the existing FastAPI application on the container network interface."""
from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.app import app, read_settings
from src.utils.phase2_common import load_params


def main() -> None:
    """Reuse configured runtime settings and expose the API inside Compose."""
    import uvicorn

    settings = read_settings()
    if settings.workers != 1:
        raise ValueError("Container launcher supports one worker")
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        app,
        host=load_params()["phase2_api"]["container_host"],
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
