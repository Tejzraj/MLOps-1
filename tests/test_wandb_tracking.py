"""Verify offline W&B boundaries without credentials or network calls."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import os

import pytest

from src.utils import wandb_tracking
from src.utils.phase2_common import load_params


def settings() -> dict[str, Any]:
    """Read the approved tracking policy from params."""
    return deepcopy(load_params()['phase2_experiments']['wandb'])


def test_online_and_model_uploads_are_rejected(tmp_path: Path) -> None:
    """An environment override cannot authorize online/model registration."""
    policy = settings()
    policy['mode'] = 'online'
    with pytest.raises(ValueError):
        wandb_tracking.OfflineTracker(policy, tmp_path, 'group', 'B0', {}, [])
    policy = settings()
    policy['save_fold_models'] = True
    with pytest.raises(ValueError):
        wandb_tracking.OfflineTracker(policy, tmp_path, 'group', 'B0', {}, [])


def test_offline_context_and_artifact_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only reviewed JSON is staged; credentials never enter run configuration."""
    calls: dict[str, Any] = {}
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setenv('WANDB_API_KEY', 'unit-test-placeholder-never-saved')
    run = SimpleNamespace(summary={}, log=lambda payload: calls.update(metrics=payload), log_artifact=lambda artifact: calls.update(artifact=artifact), finish=lambda **kwargs: calls.update(finish=kwargs))
    def initialize(**kwargs: Any) -> Any:
        calls['init'] = kwargs
        assert os.environ['WANDB_MODE'] == 'offline'
        return run
    class Artifact:
        """An in-memory stand-in to verify filenames without uploading files."""
        def __init__(self, **kwargs: Any) -> None:
            self.files: list[str] = []
        def add_file(self, path: str, name: str) -> None:
            self.files.append(name)
    monkeypatch.setattr(wandb_tracking.wandb, 'init', initialize)
    monkeypatch.setattr(wandb_tracking.wandb, 'Artifact', Artifact)
    file = tmp_path / 'metrics.json'
    file.write_text('{}')
    with wandb_tracking.OfflineTracker(settings(), tmp_path, 'group', 'B0', {'strategy': 'balanced'}, ['metrics.json']) as tracker:
        tracker.log({'macro_f1': 0})
        tracker.log_reviewed([file])
        with pytest.raises(ValueError):
            tracker.log_reviewed([tmp_path / 'model.joblib'])
        with pytest.raises(ValueError):
            tracker.log_reviewed([tmp_path / 'raw_answers.json'])
    assert calls['init']['mode'] == 'offline'
    assert calls['init']['settings'].disable_code is True
    assert calls['init']['settings'].disable_git is True
    assert 'unit-test-placeholder-never-saved' not in repr(calls)
    assert calls['artifact'].files == ['metrics.json']
    assert calls['finish']['exit_code'] == 0
    assert os.environ['WANDB_MODE'] == 'online'


def test_finish_marks_failure_and_restores_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed fold marks its run failed and restores the caller's environment."""
    calls: dict[str, Any] = {}
    monkeypatch.delenv('WANDB_MODE', raising=False)
    run = SimpleNamespace(finish=lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr(wandb_tracking.wandb, 'init', lambda **kwargs: run)
    with pytest.raises(RuntimeError):
        with wandb_tracking.OfflineTracker(settings(), tmp_path, 'group', 'B0', {}, []):
            raise RuntimeError('test fold failure')
    assert calls['exit_code'] == 1
    assert 'WANDB_MODE' not in os.environ
