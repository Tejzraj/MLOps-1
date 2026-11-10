"""Startup-only loading of frozen inference code without importing training modules."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from src.inference.predict import BundlePredictor, local_path


class ModelUnavailableError(RuntimeError):
    """An integrity, dependency or namespace mismatch prevents safe model loading."""


def digest(path: Path) -> str:
    """Hash a file without loading serialized objects."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verified_module(name: str, path: Path, expected: str) -> ModuleType:
    """Load exact checksum-verified compatibility source without creating bytecode."""
    existing = sys.modules.get(name)
    if existing is not None:
        origin = getattr(existing, '__file__', None)
        if origin is None or digest(Path(origin)) != expected:
            raise ModelUnavailableError('Model runtime namespace collision')
        return existing
    parent_name, attribute = name.rsplit('.', 1)
    parent = importlib.import_module(parent_name)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ModelUnavailableError('Compatibility source integrity failure')
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise ModelUnavailableError('Compatibility source cannot be loaded')
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    setattr(parent, attribute, module)
    return module


def load_predictor(bundle: Path, manifest_filename: str, expected_sha256: str, expected_version: str, expected_configuration: str) -> BundlePredictor:
    """Verify the frozen trust anchor and all files before executing code/loading models."""
    bundle = bundle.resolve()
    manifest_path = local_path(bundle, manifest_filename)
    if digest(manifest_path) != expected_sha256:
        raise ModelUnavailableError('Frozen bundle manifest identity mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['bundle_version'] != expected_version or manifest['selected_configuration'] != expected_configuration:
        raise ModelUnavailableError('Frozen model identity mismatch')
    for relative, expected in manifest['files_sha256'].items():
        if digest(local_path(bundle, relative)) != expected:
            raise ModelUnavailableError('Frozen bundle file integrity failure')
    # Reuse the existing wrapper, and verify it is the frozen implementation too.
    wrapper = sys.modules['src.inference.predict']
    if digest(Path(wrapper.__file__)) != manifest['files_sha256']['src/inference/predict.py']:
        raise ModelUnavailableError('Inference wrapper differs from frozen bundle')
    for name in ['src.features.build_features', 'src.models.run_imbalance_experiments']:
        relative = name.replace('.', '/') + '.py'
        verified_module(name, local_path(bundle, relative), manifest['files_sha256'][relative])
    return BundlePredictor(bundle, manifest_filename)
