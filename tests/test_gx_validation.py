"""GX rules tested with real-data copies and intentionally invalid in-memory samples."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from src.data.validate_gx import expectations, load_frames, load_params, profile, validate_frame


@pytest.fixture(scope='module')
def params() -> dict[str, Any]:
    """Read exactly the Phase 2 data-quality configuration."""
    return load_params()


@pytest.fixture(scope='module')
def sample(params: dict[str, Any]) -> pd.DataFrame:
    """Keep human labels unchanged in a small copy of actual training data."""
    return load_frames(params)['train'].head(params['phase2_gx']['unit_test_sample_rows']).copy()


def core(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Run critical GX rules without the separate content-duplicate diagnostic."""
    settings = deepcopy(params)
    settings['phase2_gx']['duplicate_content_diagnostic'] = False
    return validate_frame(frame, settings, 'unit')['critical']['validation']


def failed_types(result: dict[str, Any]) -> set[str]:
    """Return actual failed GX expectation names for explicit assertions."""
    return {item['expectation_config']['type'] for item in result['results'] if not item['success']}


def test_real_sample_passes_without_mutation(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """The string compatibility view preserves source values and pandas dtypes."""
    original = sample.copy(deep=True)
    result = core(sample, params)
    assert result['success']
    assert result['statistics']['evaluated_expectations'] == len(expectations(sample, params))
    pd.testing.assert_frame_equal(sample, original)


def test_missing_required_column_fails(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """Presence fails cleanly without running GX metrics on nonexistent columns."""
    missing = sample.drop(columns=['reference_answer'])
    result = core(missing, params)
    assert 'expect_table_columns_to_match_set' in failed_types(result)


@pytest.mark.parametrize('column', ['candidate_answer', 'reference_answer'])
@pytest.mark.parametrize('invalid', ['', ' \n\t '])
def test_blank_answers_fail(sample: pd.DataFrame, params: dict[str, Any], column: str, invalid: str) -> None:
    """Both empty and whitespace-only answers fail, without source-file changes."""
    corrupted = sample.copy()
    corrupted.loc[corrupted.index[0], column] = invalid
    assert 'expect_column_values_to_match_regex' in failed_types(core(corrupted, params))


@pytest.mark.parametrize('column', ['question_id', 'question', 'reference_answer', 'candidate_answer', 'quality_label', 'score_grader1', 'score_grader2', 'score_avg'])
def test_null_fields_fail(sample: pd.DataFrame, params: dict[str, Any], column: str) -> None:
    """Nulls in essential and existing score fields are never silently accepted."""
    corrupted = sample.copy()
    corrupted[column] = corrupted[column].astype(object)
    corrupted.loc[corrupted.index[0], column] = None
    assert 'expect_column_values_to_not_be_null' in failed_types(core(corrupted, params))


def test_encoded_label_domain_and_dtype(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """Quality labels are integer IDs; neither unknown IDs nor display text is valid."""
    corrupted = sample.copy()
    corrupted.loc[corrupted.index[0], params['data']['target_column']] = max(params['data']['target_classes']) + 1
    assert 'expect_column_values_to_be_in_set' in failed_types(core(corrupted, params))
    corrupted = sample.copy()
    column = params['data']['target_column']
    corrupted[column] = corrupted[column].astype(object)
    corrupted.loc[corrupted.index[0], column] = next(iter(params['data']['target_classes'].values()))
    assert 'expect_column_values_to_be_in_type_list' in failed_types(core(corrupted, params))


def test_text_type_validation_does_not_coerce_integers(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """The pandas-3 compatibility view preserves bad numeric values so GX rejects them."""
    corrupted = sample.copy()
    corrupted['candidate_answer'] = corrupted['candidate_answer'].astype(object)
    corrupted.loc[corrupted.index[0], 'candidate_answer'] = len(params['data']['target_classes'])
    assert 'expect_column_values_to_be_in_type_list' in failed_types(core(corrupted, params))


@pytest.mark.parametrize('column', ['score_grader1', 'score_grader2', 'score_avg'])
@pytest.mark.parametrize('boundary', ['below', 'above'])
def test_score_ranges_fail(sample: pd.DataFrame, params: dict[str, Any], column: str, boundary: str) -> None:
    """Out-of-range scores exist only in these intentionally invalid test copies."""
    limits = params['phase2_gx']['scores'][column]
    value = limits['minimum'] - 1 if boundary == 'below' else limits['maximum'] + 1
    corrupted = sample.copy()
    corrupted.loc[corrupted.index[0], column] = value
    assert 'expect_column_values_to_be_between' in failed_types(core(corrupted, params))


def test_score_checks_are_conditional(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """A schema without optional numeric scores does not invent numeric columns/checks."""
    frame = sample.drop(columns=list(params['phase2_gx']['scores']))
    result = core(frame, params)
    assert result['success']
    assert not any(item['expectation_config']['type'] == 'expect_column_values_to_be_between' for item in result['results'])


def test_optional_native_id_uniqueness(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """A native-ID-like test column is checked if present; repeated question IDs are valid."""
    column = params['phase2_gx']['optional_row_id_columns'][0]
    frame = sample.copy()
    frame[column] = frame.index.to_numpy()
    assert core(frame, params)['success']
    frame.loc[frame.index[1], column] = frame.loc[frame.index[0], column]
    assert 'expect_column_values_to_be_unique' in failed_types(core(frame, params))
    assert not profile(sample, params)['native_row_id_columns']
    assert not any(rule.expectation_type == 'expect_column_values_to_be_unique' for rule in expectations(sample, params))


def test_duplicate_content_is_visible_diagnostic(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """Report strict duplicate-content failures without deleting real repeated observations."""
    frame = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    result = validate_frame(frame, params, 'duplicates')
    assert result['critical']['validation']['success']
    assert not result['duplicate_content_diagnostic']['validation']['success']
    serialized = str(result['duplicate_content_diagnostic']['validation'])
    assert 'partial_unexpected_list' not in serialized
    assert sample.iloc[0]['candidate_answer'] not in serialized
