"""Post-evaluation inference-bundle verification without reading the test dataset."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import pytest

from src.inference.predict import BundlePredictor
from src.utils.phase2_common import ROOT, load_params, sha256


@pytest.fixture(scope='module')
def exported() -> dict[str, Any]:
    """Load only this session's verified bundle; use training rows for API checks."""
    params = load_params()
    cfg = params['phase2_final']
    completed = list((ROOT / cfg['output_root']).glob(cfg['final_id'] + '-*/completion.json'))
    assert len(completed) == 1
    folder = completed[0].parent
    bundle = folder / cfg['bundle_directory']
    predictor = BundlePredictor(bundle, cfg['bundle_manifest_filename'])
    train = pd.read_parquet(params['phase2_confirmation']['train_path'])
    sample = train.groupby(params['data']['target_column'], group_keys=False).head(params['phase2_confirmation']['compatibility_sample_per_class'])
    records = sample[cfg['preprocessing']['required_columns']].to_dict(orient='records')
    return {'params': params, 'bundle': bundle, 'predictor': predictor, 'records': records, 'folder': folder}


def test_exported_bytes_are_selected_candidate(exported: dict[str, Any]) -> None:
    """Packaging copies the exact B0 bytes, not Phase 1 originals or a refitted model."""
    params = exported['params']
    candidate = json.loads((ROOT / params['phase2_final']['candidate_directory'] / 'manifest.json').read_text())
    for name, expected in candidate['artifacts_sha256'].items():
        assert sha256(exported['bundle'] / name) == expected
    assert exported['predictor'].model.n_features_in_ == candidate['feature_count']
    assert exported['predictor'].model.classes_.tolist() == sorted(params['data']['target_classes'])


def test_repeat_single_batch_and_label_mapping(exported: dict[str, Any]) -> None:
    """Actual saved-pair inference is deterministic and decodes every prediction correctly."""
    predictor, records = exported['predictor'], exported['records']
    batch = predictor.predict_batch(records)
    assert batch == predictor.predict_batch(records)
    assert batch == [predictor.predict_one(record) for record in records]
    for prediction in batch:
        assert prediction['quality_label'] == exported['params']['data']['target_classes'][prediction['class_id']]


def test_question_text_is_not_predictive(exported: dict[str, Any]) -> None:
    """Changing only question wording leaves the fitted feature/prediction path unchanged."""
    records = exported['records']
    changed = [{**record, 'question': record['question'] + ' ' + record['question']} for record in records]
    assert exported['predictor'].predict_batch(records) == exported['predictor'].predict_batch(changed)


def test_portable_process_uses_only_bundle_source(exported: dict[str, Any], tmp_path: Path) -> None:
    """An isolated subprocess runs without repository imports, imblearn or W&B."""
    cfg = exported['params']['phase2_final']
    program = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from src.inference.predict import BundlePredictor
predictor=BundlePredictor(Path(sys.argv[1]),sys.argv[2])
records=json.loads(sys.stdin.read())
result=predictor.predict_batch(records)
assert 'wandb' not in sys.modules and 'imblearn' not in sys.modules
assert all(Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()) for name,module in sys.modules.items() if name.startswith('src') and getattr(module,'__file__',None))
sys.stdout.write(json.dumps(result))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(exported['bundle']), cfg['bundle_manifest_filename']], cwd=tmp_path, input=json.dumps(exported['records']), text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == exported['predictor'].predict_batch(exported['records'])
