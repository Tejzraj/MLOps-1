"""Load and validate frontend settings without changing shared configuration."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings for the UI and HTTP client."""

    title: str
    log_level: str
    max_answer_chars: int
    backend_url: str
    predict_path: str
    timeout_seconds: float
    mode: str
    mock_label: str
    quality_labels: tuple[str, ...]
    question_bank_path: Path


def load_settings(path: Path | None = None) -> Settings:
    """Read local params.yaml; environment overrides support later deployment."""
    config_path = path or Path(
        os.environ.get("FRONTEND_PARAMS", Path(__file__).with_name("params.yaml"))
    )
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        app, backend = data["app"], data["backend"]
        settings = Settings(
            title=app["title"],
            log_level=app["log_level"],
            max_answer_chars=int(app["max_answer_chars"]),
            backend_url=os.environ.get("BACKEND_URL", backend["url"]).rstrip("/"),
            predict_path=backend["predict_path"],
            timeout_seconds=float(
                os.environ.get("BACKEND_TIMEOUT_SECONDS", backend["timeout_seconds"])
            ),
            mode=os.environ.get("FRONTEND_MODE", backend["mode"]).lower(),
            mock_label=backend["mock_label"],
            quality_labels=tuple(data["quality_labels"]),
            question_bank_path=config_path.parent / data["question_bank_path"],
        )
        url = urlsplit(settings.backend_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("backend URL must be an HTTP(S) URL without credentials")
        if url.query or url.fragment:
            raise ValueError("backend URL must not contain a query or fragment")
        _ = url.port  # Validate malformed ports before making requests.
        if not settings.predict_path.startswith("/") or settings.predict_path.startswith("//"):
            raise ValueError("predict_path must be an absolute endpoint path")
        if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
            raise ValueError("timeout must be a positive finite number")
        if settings.max_answer_chars <= 0:
            raise ValueError("max_answer_chars must be positive")
        if settings.mode not in {"mock", "api"}:
            raise ValueError("FRONTEND_MODE must be mock or api")
        if not settings.quality_labels or settings.mock_label not in settings.quality_labels:
            raise ValueError("mock_label must belong to quality_labels")
        if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("invalid log_level")
        return settings
    except (OSError, KeyError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid frontend configuration: {exc}") from exc
