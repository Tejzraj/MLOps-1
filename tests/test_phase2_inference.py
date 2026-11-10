"""Preflight checks for trusted artifact identity and exact raw-input preprocessing."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.preprocess import clean_text
from src.inference import predict
from src.models.finalize_model import verify_selected
from src.utils.phase2_common import load_params, write_json


def real_request() -> tuple[dict[str, str], dict[str, Any]]:
    """Use one existing training example; no test partition is read."""
    params = load_params()
    columns = params['phase2_final']['preprocessing']['required_columns']
    row = pd.read_parquet(params['phase2_confirmation']['train_path']).iloc[0]
    return {column: str(row[column]) for column in columns}, params


def test_selected_artifact_hashes_and_decision() -> None:
    """Both model/pipeline checksums match the saved B0 manifest before loading."""
    params = load_params()
    candidate, selection = verify_selected(params)
    assert candidate['configuration'] == params['phase2_final']['selected_configuration']
    assert selection['selected_configuration'] == candidate['configuration']
    changed = deepcopy(params)
    changed['phase2_final']['selected_configuration'] = params['phase2_confirmation']['candidate']
    with pytest.raises(ValueError, match='selection identity'):
        verify_selected(changed)


def test_exact_training_cleaning() -> None:
    """Case and combining Unicode are preserved; candidate br tags precede spacing."""
    record, params = real_request()
    cfg = params['phase2_final']['preprocessing']
    decorated = {key: ' \n' + value + '\t ' for key, value in record.items()}
    decorated[cfg['candidate_column']] += '<br/> e\u0301'
    cleaned = predict.clean_record(decorated, cfg)
    assert cleaned[cfg['candidate_column']] == clean_text(decorated[cfg['candidate_column']].replace('<br/>', ''))
    assert cleaned['reference_answer'] == clean_text(decorated['reference_answer'])
    assert cleaned['question'] == clean_text(decorated['question'])
    assert cleaned[cfg['candidate_column']].endswith('e\u0301')


@pytest.mark.parametrize('bad', [None, 1, [], True])
def test_non_string_fields_rejected(bad: Any) -> None:
    """No implicit conversion of invalid API fields into strings."""
    record, params = real_request()
    record['candidate_answer'] = bad
    with pytest.raises(ValueError, match='must be a string'):
        predict.clean_record(record, params['phase2_final']['preprocessing'])


def test_missing_extra_and_empty_candidate_rejected() -> None:
    """Validation prevents missing fields, accidental score inputs and empty candidates."""
    record, params = real_request()
    cfg = params['phase2_final']['preprocessing']
    missing = dict(record)
    missing.pop('question')
    with pytest.raises(ValueError):
        predict.clean_record(missing, cfg)
    with pytest.raises(ValueError, match='Unexpected'):
        predict.clean_record({**record, 'score_avg': None}, cfg)
    with pytest.raises(ValueError, match='must not be empty'):
        predict.clean_record({**record, 'candidate_answer': ' \t<br/> '}, cfg)


def test_empty_batch_rejected_before_inference() -> None:
    """An invalid batch does not need or touch model artifacts."""
    predictor = object.__new__(predict.BundlePredictor)
    with pytest.raises(ValueError, match='nonempty list'):
        predictor.predict_batch([])


def test_checksum_failure_precedes_deserialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt bytes cannot reach joblib.load."""
    params = load_params()
    cfg = params['phase2_final']
    candidate, _ = verify_selected(params)
    name = candidate['artifact_files']['model']
    (tmp_path / name).write_bytes(b'not-a-serialized-model')
    write_json(tmp_path / cfg['bundle_manifest_filename'], {'files_sha256': {name: candidate['artifacts_sha256'][name]}})
    monkeypatch.setattr(predict.joblib, 'load', lambda *args, **kwargs: pytest.fail('Deserialization must not occur'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        predict.BundlePredictor(tmp_path, cfg['bundle_manifest_filename'])
