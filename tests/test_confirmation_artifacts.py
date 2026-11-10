"""Post-training checks of only the newly generated trusted confirmation pairs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest

from src.models.confirm_validation import verify_pair
from src.utils.phase2_common import ROOT, load_params, sha256


@pytest.fixture(scope='module')
def candidates() -> dict[str, Any]:
    """Verify checksums before loading this session's locally generated artifacts."""
    params = load_params()
    cfg = params['phase2_confirmation']
    directories = list((ROOT / cfg['output_root']).glob(cfg['confirmation_id'] + '-*/decision.json'))
    assert len(directories) == 1, 'Exactly one completed confirmation is required'
    result = {}
    train = pd.read_parquet(cfg['train_path'])
    sample = train.groupby(params['phase2_experiments']['target_column'], group_keys=False).head(cfg['compatibility_sample_per_class'])[params['phase2_experiments']['input_columns']]
    for name in cfg['configurations']:
        folder = directories[0].parent / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        for filename, expected in manifest['artifacts_sha256'].items():
            assert sha256(folder / filename) == expected
        pipeline = joblib.load(folder / cfg['pipeline_filename'])
        model = joblib.load(folder / cfg['model_filename'])
        result[name] = {'manifest': manifest, 'pipeline': pipeline, 'model': model, 'folder': folder, 'sample': sample, 'params': params}
    return result


@pytest.mark.parametrize('name', ['B0', 'F1'])
def test_saved_pair_contract(name: str, candidates: dict[str, Any]) -> None:
    """Actual saved pairs agree on dimensions, classes and deterministic predictions."""
    record = candidates[name]
    names = json.loads((record['folder'] / record['params']['phase2_confirmation']['feature_names_filename']).read_text())
    check = verify_pair(record['pipeline'], record['model'], record['sample'], record['params'], names)
    assert check['feature_count'] == record['manifest']['feature_count']
    assert check['classes'] == sorted(record['params']['data']['target_classes'])


@pytest.mark.parametrize('name', ['B0', 'F1'])
def test_saved_estimator_settings(name: str, candidates: dict[str, Any]) -> None:
    """Saved estimator parameters match the manifest and frozen training budget."""
    record = candidates[name]
    assert record['model'].get_params(deep=False) == record['manifest']['resolved_estimator_parameters']
    assert record['model'].n_estimators == record['params']['phase2_experiments']['profiles']['full']['n_estimators']
    assert record['manifest']['datasets']['train']['rows'] == record['params']['phase2_confirmation']['expected_train_rows']
    assert record['manifest']['test_set_accessed'] is False
