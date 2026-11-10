"""Offline-only W&B logging of explicitly reviewed, text-free experiment artifacts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb

from src.utils.phase2_common import output_path


class OfflineTracker:
    """Log one configuration locally; never authenticate, sync, or register models."""

    def __init__(self, settings: dict[str, Any], directory: Path, group: str, name: str, config: dict[str, Any], artifact_files: list[str]) -> None:
        """Require offline operation and keep all SDK storage below generated output."""
        if settings['mode'] != 'offline':
            raise ValueError('This framework currently permits offline mode only')
        if settings['save_fold_models'] or settings['register_models'] or settings['upload_raw_text']:
            raise ValueError('Raw text, model persistence, and registry operations are disabled')
        self.settings = settings
        self.directory = directory.resolve()
        self.artifact_files = set(artifact_files)
        self.group = group
        self.name = name
        self.config = config
        self.run: Any = None
        self.previous_environment: dict[str, str | None] = {}

    def __enter__(self) -> OfflineTracker:
        """Start a local run with metadata/code capture and system statistics disabled."""
        environment = {'WANDB_MODE': 'offline', 'WANDB_DISABLE_CODE': 'true', 'WANDB_DISABLE_GIT': 'true'}
        for key, setting in [('WANDB_CACHE_DIR', 'cache_dir'), ('WANDB_CONFIG_DIR', 'config_dir'), ('WANDB_DATA_DIR', 'data_dir')]:
            folder = output_path(self.directory, self.settings[setting])
            folder.mkdir(parents=True, exist_ok=True)
            environment[key] = str(folder)
        for key, value in environment.items():
            self.previous_environment[key] = os.environ.get(key)
            os.environ[key] = value
        run_dir = output_path(self.directory, self.settings['run_dir'])
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.run = wandb.init(project=self.settings['project'], entity=self.settings['entity'], group=self.group, name=self.name, mode='offline', dir=str(run_dir), config=self.config, settings=wandb.Settings(disable_git=True, disable_code=True, console='off', x_disable_stats=True, x_disable_meta=True))
        except BaseException:
            self._restore_environment()
            raise
        return self

    def log(self, metrics: dict[str, Any]) -> None:
        """Log only explicit metrics; no environment or raw DataFrame capture."""
        self.run.log(metrics)

    def log_reviewed(self, files: list[Path]) -> None:
        """Stage allowlisted JSON artifacts in the offline run, excluding binaries."""
        if not self.settings['log_reviewed_artifacts']:
            return
        for file in files:
            if file.name not in self.artifact_files or file.suffix != '.json' or not file.resolve().is_relative_to(self.directory):
                raise ValueError('Artifact is not an allowlisted local JSON report')
        artifact = wandb.Artifact(name=f'{self.group}-{self.name}', type='experiment-report', metadata={'contains_models': False, 'contains_raw_text': False})
        for file in files:
            artifact.add_file(str(file), name=file.name)
        self.run.log_artifact(artifact)

    def _restore_environment(self) -> None:
        """Restore only the SDK settings changed by this context manager."""
        for key, value in self.previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        """Finish the local run and restore caller settings even on failure."""
        try:
            if self.run is not None:
                self.run.finish(exit_code=0 if kind is None else 1)
        finally:
            self._restore_environment()
