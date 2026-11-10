"""Separate, real-split question leakage tests reusing the existing Phase 1 assertion."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.preprocess import assert_disjoint_splits
from src.data.validate_gx import group_audit, load_frames, load_params


@pytest.fixture(scope='module')
def frames() -> dict[str, pd.DataFrame]:
    """Read all three immutable processed splits for group checks, not model evaluation."""
    return load_frames(load_params())


@pytest.mark.parametrize('left,right', [('train','validation'), ('train','test'), ('validation','test')])
def test_pairwise_question_groups(frames: dict[str, pd.DataFrame], left: str, right: str) -> None:
    """Each real pair has no shared question IDs."""
    group = load_params()['data']['group_column']
    assert set(frames[left][group]).isdisjoint(frames[right][group])


def test_existing_assertion_and_summary(frames: dict[str, pd.DataFrame]) -> None:
    """Keep group leakage as an automated assertion separate from GX suites."""
    params = load_params()
    assert_disjoint_splits(frames['train'], frames['validation'], frames['test'], params['data']['group_column'])
    audit = group_audit(frames, params)
    assert audit['success']
    assert not any(audit['pairwise_overlap_counts'].values())


def test_deliberate_question_leakage_is_detected(frames: dict[str, pd.DataFrame]) -> None:
    """An in-memory contamination fixture fails without modifying any partition."""
    changed = dict(frames)
    changed['validation'] = pd.concat([frames['validation'], frames['train'].iloc[[0]]], ignore_index=True)
    group = load_params()['data']['group_column']
    with pytest.raises(AssertionError):
        assert_disjoint_splits(changed['train'], changed['validation'], changed['test'], group)
    assert not group_audit(changed, load_params())['success']
