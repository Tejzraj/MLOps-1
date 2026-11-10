"""Preflight tests for the frozen validation protocol and inference contracts."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models import confirm_validation as confirmation
from src.utils.phase2_common import ROOT, load_params


def test_frozen_configuration_and_dependency_identity() -> None:
    """The exact approved CV recipes and implementation are still available."""
    params = load_params()
    configs = confirmation.frozen_settings(params)
    cfg = params['phase2_confirmation']
    assert list(configs) == cfg['configurations']
    for settings in configs.values():
        assert settings['profile_settings'] == params['phase2_experiments']['profiles']['full']


def test_reject_changed_features_or_model_recipe() -> None:
    """Any drift from the selected CV settings stops before fitting."""
    params = load_params()
    modified = deepcopy(params)
    modified['features']['tfidf']['min_df'] += 1
    with pytest.raises(ValueError, match='Feature parameters'):
        confirmation.frozen_settings(modified)
    modified = deepcopy(params)
    modified['phase2_experiments']['model_seed'] += 1
    with pytest.raises(ValueError, match='model_seed'):
        confirmation.frozen_settings(modified)


def test_predeclared_strict_confirmation_rule() -> None:
    """Use recorded CV scores to exercise win, loss and conservative equality branches."""
    params = load_params()
    cfg = params['phase2_confirmation']
    records = json.loads((ROOT / cfg['cv_directory'] / 'cv_review.json').read_text())['configurations']
    scores = {record['configuration']: {'macro_f1': record['aggregate']['macro_f1_mean']} for record in records}
    assert confirmation.decide(scores, params)['selected_configuration'] == cfg['candidate']
    reversed_scores = {cfg['baseline']: scores[cfg['candidate']], cfg['candidate']: scores[cfg['baseline']]}
    assert confirmation.decide(reversed_scores, params)['selected_configuration'] == cfg['baseline']
    tied = {name: scores[cfg['baseline']] for name in cfg['configurations']}
    assert confirmation.decide(tied, params)['selected_configuration'] == cfg['baseline']


def test_partition_loader_never_opens_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only checksum-approved train/validation files are read; groups stay separate."""
    params = load_params()
    cfg = params['phase2_confirmation']
    read = pd.read_parquet
    opened: list[str] = []
    def guarded(path: Any) -> pd.DataFrame:
        opened.append(str(path))
        assert str(path).replace('\\', '/') in [cfg['train_path'], cfg['validation_path']]
        return read(path)
    monkeypatch.setattr(confirmation.pd, 'read_parquet', guarded)
    train, validation = confirmation.load_partitions(params)
    assert len(opened) == 2
    assert len(train) == cfg['expected_train_rows']
    assert len(validation) == cfg['expected_validation_rows']
    group = params['phase2_experiments']['group_column']
    assert not set(train[group]) & set(validation[group])


class ContractPipeline:
    """Minimal inference-only stand-in for testing compatibility failures."""
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
    def transform(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(inputs), len(self.columns)))
    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.columns)


class ContractModel:
    """Deterministic inference stand-in; never fits or assigns dataset labels."""
    def __init__(self, classes: list[int], dimension: int) -> None:
        self.classes_ = np.array(classes)
        self.n_features_in_ = dimension
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.classes_[0])
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.eye(len(self.classes_))[0], (features.shape[0], 1))


def contract_fixture() -> tuple[Any, Any, pd.DataFrame, dict[str, Any]]:
    """Read real training examples for contract tests without fitting a classifier."""
    params = load_params()
    cfg = params['phase2_experiments']
    inputs = pd.read_parquet(params['phase2_confirmation']['train_path'])[cfg['input_columns']].head(params['phase2_confirmation']['compatibility_sample_per_class'])
    pipeline = ContractPipeline(cfg['input_columns'])
    model = ContractModel(sorted(params['data']['target_classes']), len(cfg['input_columns']))
    return pipeline, model, inputs, params


def test_pair_contract_accepts_deterministic_predictions() -> None:
    """A compatible deterministic inference contract passes shape and class guards."""
    pipeline, model, inputs, params = contract_fixture()
    result = confirmation.verify_pair(pipeline, model, inputs, params)
    assert result['single_batch_equal'] and result['deterministic_predictions']


def test_pair_contract_rejects_feature_dimension_and_order() -> None:
    """Dimension and ordered-feature drift must fail before trusting a bundle."""
    pipeline, model, inputs, params = contract_fixture()
    with pytest.raises(ValueError, match='Ordered feature names'):
        confirmation.verify_pair(pipeline, model, inputs, params, list(reversed(pipeline.columns)))
    model.n_features_in_ += 1
    with pytest.raises(ValueError, match='dimension'):
        confirmation.verify_pair(pipeline, model, inputs, params)


def test_pair_contract_rejects_wrong_class_mapping() -> None:
    """A wrong classifier class order cannot silently corrupt label decoding."""
    pipeline, model, inputs, params = contract_fixture()
    model.classes_ = model.classes_[::-1]
    with pytest.raises(ValueError, match='classes'):
        confirmation.verify_pair(pipeline, model, inputs, params)


def test_pair_contract_rejects_batch_dependent_predictions() -> None:
    """Batch-size-dependent output is detected by single-row equivalence checks."""
    pipeline, model, inputs, params = contract_fixture()
    model.predict = lambda features: np.full(features.shape[0], model.classes_[0 if features.shape[0] == 1 else -1])
    with pytest.raises(ValueError, match='deterministic'):
        confirmation.verify_pair(pipeline, model, inputs, params)
