"""Shared configuration, integrity, and output guards for Phase 2 experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read repository parameters without modifying legacy configuration."""
    return yaml.safe_load((path or ROOT / 'params.yaml').read_text(encoding='utf-8-sig'))


def sha256(path: Path) -> str:
    """Hash a source or artifact without loading its contents as Python objects."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def output_path(root: Path, relative: str) -> Path:
    """Reject absolute paths and escapes from the intended generated directory."""
    part = Path(relative)
    result = (root / part).resolve()
    if part.is_absolute() or not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError('Generated path must stay below its output root')
    return result


def write_json(path: Path, content: Any) -> None:
    """Write inspectable, finite JSON into a generated output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git_identity() -> dict[str, Any]:
    """Record commit identity and dirty file names without capturing secrets."""
    def git(*args: str) -> str:
        return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--short')}
