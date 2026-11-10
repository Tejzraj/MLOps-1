"""Check pinned Phase 2 dependencies and record a reproducible environment lock."""
from __future__ import annotations

import importlib.metadata
import logging
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.phase2_common import ROOT, load_params, output_path, write_json

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate exact package versions and record all installed distributions."""
    cfg = load_params()['phase2_experiments']['environment']
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    if platform.python_version() != cfg['expected_python']:
        raise RuntimeError('Python version does not match the approved runtime')
    for name, expected in cfg['expected_packages'].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {actual}')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    # Imports exercise sklearn-compat without fitting or touching any saved models.
    import imblearn
    import wandb
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import RandomOverSampler
    if not all([imblearn, wandb, BalancedRandomForestClassifier, RandomOverSampler]):
        raise RuntimeError('Required imports failed')
    report = {'python': platform.python_version(), 'packages': versions, 'pip_check': 'passed', 'imports': 'passed'}
    write_json(output_path(ROOT, cfg['report_path']), report)
    lock = '\n'.join(f'{name}=={version}' for name, version in sorted(versions.items(), key=lambda item: item[0].lower()))
    output_path(ROOT, cfg['lock_path']).write_text(lock + '\n', encoding='utf-8')
    LOGGER.info('Environment compatible: Python %s; %d packages', platform.python_version(), len(versions))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
