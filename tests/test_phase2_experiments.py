"""Behavioral checks for grouped splits, fresh features, sampling scope, and metrics."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models import run_imbalance_experiments as experiment
from src.utils.phase2_common import load_params, output_path


@pytest.fixture(scope='module')
def params() -> dict[str, Any]:
    """Use the approved real configuration without changing the source file."""
    return load_params()


@pytest.fixture(scope='module')
def frame(params: dict[str, Any]) -> pd.DataFrame:
    """Read only real training rows; never access external validation or test."""
    return pd.read_parquet(params['phase2_experiments']['train_path'])


def test_profiles_preserve_design(params: dict[str, Any]) -> None:
    """Smoke is six one-fold fits; full is six five-fold fits with 300 trees."""
    cfg = params['phase2_experiments']
    assert [c['id'] for c in cfg['configurations']] == ['B0', 'W1', 'W2', 'W3', 'O1', 'F1']
    assert cfg['profiles']['smoke']['fold_indices'] == [0]
    assert cfg['profiles']['smoke']['n_estimators'] < cfg['profiles']['full']['n_estimators']
    assert cfg['profiles']['full']['n_estimators'] == params['model']['random_forest']['n_estimators']
    assert len(cfg['profiles']['full']['fold_indices']) == cfg['cv']['n_splits']
    assert cfg['profiles']['smoke']['eligible_for_selection'] is False


def test_group_folds_are_disjoint_and_repeatable(frame: pd.DataFrame, params: dict[str, Any]) -> None:
    """Every training row is validated once; no question crosses a fold boundary."""
    first = experiment.grouped_folds(frame, params)
    second = experiment.grouped_folds(frame, params)
    held_out = []
    group = params['phase2_experiments']['group_column']
    for (train, val), (again_train, again_val) in zip(first, second):
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(val, again_val)
        assert not set(frame.iloc[train][group]) & set(frame.iloc[val][group])
        held_out.extend(val.tolist())
    assert sorted(held_out) == list(range(len(frame)))


def test_reject_overlapping_groups_before_fitting(frame: pd.DataFrame, params: dict[str, Any]) -> None:
    """Invalid fold boundaries fail before any feature fitting or sampling."""
    cfg = params['phase2_experiments']
    train, _ = experiment.grouped_folds(frame, params)[0]
    with pytest.raises(ValueError, match='leakage'):
        experiment.fit_fold(frame, train, train, params, cfg['configurations'][0], cfg['profiles']['smoke'])


def test_both_vocabularies_exclude_validation_text(frame: pd.DataFrame, params: dict[str, Any]) -> None:
    """A validation-only marker enters neither fitted TF-IDF vocabulary."""
    cfg = params['phase2_experiments']
    train, val = experiment.grouped_folds(frame, params)[0]
    pipeline = experiment.feature_pipeline(params)
    X_train = frame.iloc[train][cfg['input_columns']]
    X_val = frame.iloc[val][cfg['input_columns']].copy()
    marker = 'heldoutuniquemarker'
    for column in cfg['input_columns']:
        X_val[column] = X_val[column] + ' ' + marker
    fitted = pipeline.fit_transform(X_train)
    transformed = pipeline.transform(X_val)
    assert marker not in pipeline.named_transformers_['tfidf'].vocabulary_
    assert marker not in pipeline.named_transformers_['ref_sim_features'].tfidf_vectorizer_.vocabulary_
    assert fitted.shape[1] == transformed.shape[1]
    assert pipeline is not experiment.feature_pipeline(params)


def test_fold_local_oversampling(frame: pd.DataFrame, params: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual sparse oversampling duplicates only train rows; validation is untouched."""
    cfg = params['phase2_experiments']
    train, val = experiment.grouped_folds(frame, params)[0]
    original = frame.copy(deep=True)
    calls: list[list[int]] = []
    original_builder = experiment.feature_pipeline

    def builder(settings: dict[str, Any]) -> Any:
        pipeline = original_builder(settings)
        fit_transform = pipeline.fit_transform
        def record(X: pd.DataFrame, y: Any = None) -> Any:
            calls.append(X.index.tolist())
            return fit_transform(X, y)
        pipeline.fit_transform = record
        return pipeline

    class DummyClassifier:
        """Avoid extra forest fits while testing the actual feature/sampler boundary."""
        n_estimators = cfg['profiles']['smoke']['n_estimators']
        class_weight = None
        def fit(self, X: Any, y: np.ndarray) -> DummyClassifier:
            self.n_features_in_ = X.shape[1]
            self.label = int(y[0])
            return self
        def predict(self, X: Any) -> np.ndarray:
            return np.full(X.shape[0], self.label)

    monkeypatch.setattr(experiment, 'feature_pipeline', builder)
    monkeypatch.setattr(experiment, 'classifier', lambda *args: DummyClassifier())
    result = experiment.fit_fold(frame, train, val, params, cfg['configurations'][4], cfg['profiles']['smoke'])
    audit = result['audit']
    assert calls == [frame.iloc[train].index.tolist()]
    assert set(audit['sampled_source_row_positions']) <= set(train)
    assert not set(audit['sampled_source_row_positions']) & set(val)
    assert len(set(audit['counts_after_external_sampling'].values())) == 1
    assert audit['validation_rows'] == len(val)
    assert len(result['predictions']) == len(val)
    pd.testing.assert_frame_equal(frame, original)


def test_class_weights_are_fold_derived(frame: pd.DataFrame, params: dict[str, Any]) -> None:
    """Custom weights depend on supplied fold labels; BRF avoids double balancing."""
    cfg = params['phase2_experiments']
    train, _ = experiment.grouped_folds(frame, params)[0]
    y = frame.iloc[train][cfg['target_column']].to_numpy()
    model = experiment.classifier(cfg['configurations'][3], y, params, cfg['profiles']['smoke'])
    for label in params['data']['target_classes']:
        expected = (len(y) / (len(params['data']['target_classes']) * np.sum(y == label))) ** cfg['weight_exponent']
        assert model.class_weight[label] == pytest.approx(expected)
    balanced = experiment.classifier(cfg['configurations'][5], y, params, cfg['profiles']['smoke'])
    assert balanced.class_weight is None
    assert balanced.bootstrap is False
    assert balanced.replacement is True


def test_fixed_class_metric_order(frame: pd.DataFrame, params: dict[str, Any]) -> None:
    """Metric reports retain human label mapping and support totals."""
    cfg = params['phase2_experiments']
    y = frame[cfg['target_column']].to_numpy()
    scores = experiment.score_predictions(y, y, params)
    assert list(scores['per_class']) == list(params['data']['target_classes'].values())
    assert sum(c['support'] for c in scores['per_class'].values()) == len(frame)
    assert scores['macro_f1'] == pytest.approx(1)
    assert scores['balanced_accuracy'] == pytest.approx(1)


@pytest.mark.parametrize('escape', ['../escape', '../../outside'])
def test_generated_path_escape_rejected(tmp_path: Path, escape: str) -> None:
    """Generated output cannot reach original workspaces."""
    with pytest.raises(ValueError):
        output_path(tmp_path, escape)
