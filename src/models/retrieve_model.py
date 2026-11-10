"""Download and integrity-check the configured immutable Registry bundle; never load it."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.phase2_common import ROOT, load_params

LOGGER = logging.getLogger(__name__)


def safe_relative(name: str) -> Path:
    """Reject ambiguous, absolute and traversing artifact paths on all platforms."""
    if not name or '\\' in name or PureWindowsPath(name).drive or Path(name).is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('Unsafe artifact entry path')
    return Path(name)


def digest(path: Path) -> str:
    """Hash bytes without deserializing model files."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_bundle(directory: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pin the manifest before trusting its complete, exact file checksum contract."""
    manifest_path = directory / safe_relative(cfg['manifest_filename'])
    if digest(manifest_path) != cfg['expected_manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['bundle_version'] != cfg['expected_version'] or manifest['selected_configuration'] != cfg['expected_configuration']:
        raise ValueError('Model identity mismatch')
    expected = set(manifest['files_sha256']) | {cfg['manifest_filename']}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if len(expected) != cfg['expected_file_count'] or actual != expected:
        raise ValueError('Bundle file allowlist mismatch')
    for name, checksum in manifest['files_sha256'].items():
        path = directory / safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or digest(path) != checksum:
            raise ValueError('Bundle file checksum mismatch')
    return manifest


def retrieve(cfg: dict[str, Any], destination: Path, api: Any = None) -> Path:
    """Download into a fresh staging directory; expose the target only after verification."""
    if not re.search(r':v[0-9]+$', cfg['registry_reference']):
        raise ValueError('An immutable numeric Registry version is required')
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Destination exists; refusing to overwrite')
    if api is None:
        import wandb
        api = wandb.Api(overrides={'base_url': cfg['base_url']})
    artifact = api.artifact(cfg['registry_reference'], type=cfg['artifact_type'])
    if artifact.digest != cfg['expected_artifact_digest']:
        raise ValueError('Registry artifact digest mismatch')
    entries = artifact.manifest.entries
    if len(entries) != cfg['expected_file_count']:
        raise ValueError('Registry file count mismatch')
    for name, entry in entries.items():
        safe_relative(name)
        if getattr(entry, 'ref', None):
            raise ValueError('External artifact references are not permitted')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + '-unverified-', dir=destination.parent))
    # A failed staging directory remains for diagnosis; never marked ready or loaded.
    artifact.download(root=str(staging), skip_cache=True)
    verify_bundle(staging, cfg)
    staging.rename(destination)
    LOGGER.info('Bundle ready: immutable version and every SHA-256 verified; directory=%s', destination)
    return destination


def main() -> None:
    """Read params, allow a destination override, and fail without logging credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=None)
    parser.add_argument('--destination', type=Path, default=None)
    args = parser.parse_args()
    cfg = load_params(args.params)['phase2_retrieval']
    destination = args.destination or Path(cfg['destination'])
    if not destination.is_absolute():
        destination = ROOT / destination
    try:
        retrieve(cfg, destination)
    except Exception as error:
        LOGGER.error('Retrieval failed (%s). Check login, download permissions, immutable version and integrity; no bundle marked ready.', type(error).__name__)
        raise SystemExit(1) from None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
