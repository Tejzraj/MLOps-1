"""GX validation of immutable processed Mohler splits; no model operations."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid
import warnings
import xml.etree.ElementTree as ET

os.environ['GX_ANALYTICS_ENABLED'] = 'False'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.data_context.types.base import ProgressBarsConfig, DataContextConfig, InMemoryStoreBackendDefaults
import pandas as pd

from src.data.preprocess import assert_disjoint_splits
from src.utils.phase2_common import ROOT, git_identity, load_params, output_path, sha256, write_json

LOGGER = logging.getLogger(__name__)


def load_frames(params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Hash-check and read source files without writing to their directories."""
    frames = {}
    for name, settings in params['phase2_gx']['inputs'].items():
        path = Path(settings['path'])
        if sha256(path) != settings['sha256']:
            raise ValueError(f'{name} input checksum mismatch')
        frames[name] = pd.read_parquet(path)
    return frames


def profile(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Describe real schema/values using counts only, never answer-text examples."""
    cfg = params['phase2_gx']
    target = params['data']['target_column']
    return {'rows': len(frame), 'columns': frame.columns.tolist(), 'dtypes': {column: str(dtype) for column, dtype in frame.dtypes.items()}, 'null_counts': {column: int(value) for column, value in frame.isna().sum().items()}, 'blank_counts': {column: int(frame[column].fillna('').astype(str).str.strip().eq('').sum()) for column in cfg['nonblank_columns'] if column in frame}, 'label_counts': {str(label): int(count) for label, count in frame[target].value_counts(dropna=False).sort_index().items()} if target in frame else {}, 'score_ranges': {column: {'min': float(frame[column].min()), 'max': float(frame[column].max())} for column in cfg['scores'] if column in frame}, 'full_row_duplicates_beyond_first': int(frame.duplicated().sum()), 'full_row_duplicate_members': int(frame.duplicated(keep=False).sum()), 'native_row_id_columns': [column for column in cfg['optional_row_id_columns'] if column in frame], 'row_id_note': 'No native answer identifier: uniqueness expectations apply only if one exists. question_id is a repeating group key; split-local index is not a source answer ID.'}


def expectations(frame: pd.DataFrame, params: dict[str, Any]) -> list[Any]:
    """Build critical rules from observed schema and configured human-label encoding."""
    cfg = params['phase2_gx']
    rules = [gxe.ExpectTableRowCountToBeBetween(min_value=cfg['minimum_rows']), gxe.ExpectTableColumnsToMatchSet(column_set=cfg['required_columns'], exact_match=False)]
    for column, types in cfg['required_types'].items():
        if column in frame:
            rules.append(gxe.ExpectColumnValuesToBeInTypeList(column=column, type_list=types))
    for column in cfg['non_null_columns']:
        if column in frame:
            rules.append(gxe.ExpectColumnValuesToNotBeNull(column=column))
    for column in cfg['nonblank_columns']:
        if column in frame:
            rules.append(gxe.ExpectColumnValuesToMatchRegex(column=column, regex=cfg['nonblank_regex']))
    target = params['data']['target_column']
    if target in frame:
        rules.append(gxe.ExpectColumnValuesToBeInSet(column=target, value_set=sorted(params['data']['target_classes'])))
    for column, settings in cfg['scores'].items():
        if column in frame:
            rules.extend([gxe.ExpectColumnValuesToBeInTypeList(column=column, type_list=settings['types']), gxe.ExpectColumnValuesToNotBeNull(column=column), gxe.ExpectColumnValuesToBeBetween(column=column, min_value=settings['minimum'], max_value=settings['maximum'])])
    for column in cfg['optional_row_id_columns']:
        if column in frame:
            rules.extend([gxe.ExpectColumnValuesToNotBeNull(column=column), gxe.ExpectColumnValuesToBeUnique(column=column)])
    return rules


def sanitized(value: Any) -> Any:
    """Strip raw unexpected values/rows and traceback text from saved GX results."""
    blocked = {'partial_unexpected_list', 'unexpected_list', 'unexpected_rows', 'partial_unexpected_counts', 'unexpected_index_list', 'partial_unexpected_index_list', 'unexpected_index_query', 'exception_traceback', 'exception_message'}
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items() if key not in blocked}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    return value


def validate_frame(frame: pd.DataFrame, params: dict[str, Any], name: str) -> dict[str, Any]:
    """Run separate critical and diagnostic GX suites on a dataframe, in memory."""
    cfg = params['phase2_gx']
    context = gx.get_context(mode='ephemeral', project_config=DataContextConfig(store_backend_defaults=InMemoryStoreBackendDefaults(init_temp_docs_sites=False), data_docs_sites={}, analytics_enabled=False))
    context.enable_analytics(False)
    context.variables.progress_bars = ProgressBarsConfig(globally=cfg['progress_bars'], metric_calculations=cfg['progress_bars'])
    # GX 1.24.1 does not recognize pandas 3's str ExtensionDtype in its
    # aggregate type metric. An object-backed COPY enables strict per-value str
    # checks without coercing values; numeric mistakes remain numeric and fail.
    if cfg['text_validation_view'] != 'object_preserve_values':
        raise ValueError('Unsupported text validation view')
    validation_view = frame.copy(deep=True)
    for column in cfg['nonblank_columns']:
        if column in validation_view:
            validation_view[column] = validation_view[column].astype(object)
    source = context.data_sources.add_pandas(name='mohler')
    asset = source.add_dataframe_asset(name=name)
    batch = asset.add_batch_definition_whole_dataframe(name='whole')
    suites = {'critical': expectations(frame, params)}
    if cfg['duplicate_content_diagnostic']:
        suites['duplicate_content_diagnostic'] = [gxe.ExpectCompoundColumnsToBeUnique(column_list=frame.columns.tolist(), ignore_row_if='never')]
    output = {}
    for severity, rules in suites.items():
        suite = context.suites.add(gx.ExpectationSuite(name=f'{name}_{severity}'))
        for rule in rules:
            suite.add_expectation(rule)
        definition = context.validation_definitions.add(gx.ValidationDefinition(name=f'{name}_{severity}', data=batch, suite=suite))
        result = definition.run(batch_parameters={'dataframe': validation_view}, result_format=cfg['result_format'])
        output[severity] = {'validation': sanitized(result.to_json_dict()), 'suite': suite.to_json_dict()}
    return output


def group_audit(frames: dict[str, pd.DataFrame], params: dict[str, Any]) -> dict[str, Any]:
    """Keep existing group-disjointness assertion separate from GX expectations."""
    column = params['data']['group_column']
    overlaps = {f'{left}__{right}': len(set(frames[left][column]) & set(frames[right][column])) for left, right in itertools.combinations(frames, 2)}
    try:
        assert_disjoint_splits(frames['train'], frames['validation'], frames['test'], column)
    except AssertionError:
        return {'success': False, 'pairwise_overlap_counts': overlaps}
    return {'success': True, 'pairwise_overlap_counts': overlaps}


def run(params: dict[str, Any]) -> Path:
    """Run tests and separate train/validation/test validations; save unique reports."""
    cfg = params['phase2_gx']
    if importlib.metadata.version('great-expectations') != cfg['expected_gx_version']:
        raise RuntimeError('GX version does not match pinned configuration')
    directory = output_path(ROOT, cfg['output_root']) / ('gx-validation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'pytest', '-o', 'addopts=', '-o', 'filterwarnings=default', '-p', 'no:cacheprovider', '--basetemp='+str(directory/'pytest-temp'), '--junitxml='+str(directory/'pytest-results.xml'), *cfg['tests']]
    tests = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (directory/'pytest.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
    if tests.returncode:
        LOGGER.error('Focused tests failed; validation stopped. Details: %s', directory/'pytest.log')
        raise RuntimeError('GX preflight tests failed')
    frames = load_frames(params)
    results = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for name, frame in frames.items():
            descriptions = profile(frame, params)
            report = validate_frame(frame, params, name)
            results[name] = {'profile': descriptions, 'suites': report}
            write_json(directory/f'{name}_profile.json', descriptions)
            for severity, values in report.items():
                write_json(directory/f'{name}_{severity}_results.json', values['validation'])
                write_json(directory/f'{name}_{severity}_suite.json', values['suite'])
            LOGGER.info('%s critical=%s diagnostic=%s', name, report['critical']['validation']['statistics'], report['duplicate_content_diagnostic']['validation']['statistics'] if 'duplicate_content_diagnostic' in report else 'disabled')
    grouping = group_audit(frames, params)
    for settings in cfg['inputs'].values():
        if sha256(Path(settings['path'])) != settings['sha256']:
            raise RuntimeError('Read-only source file changed')
    test_suite = ET.parse(directory/'pytest-results.xml').getroot().find('testsuite').attrib
    summary = {'gx_version': importlib.metadata.version('great-expectations'), 'source': git_identity(), 'input_checksums': {name: settings['sha256'] for name, settings in cfg['inputs'].items()}, 'splits': {name: {'profile': value['profile'], 'critical': value['suites']['critical']['validation']['statistics'], 'critical_success': value['suites']['critical']['validation']['success'], 'duplicate_content_diagnostic': value['suites']['duplicate_content_diagnostic']['validation']['statistics'] if 'duplicate_content_diagnostic' in value['suites'] else None} for name, value in results.items()}, 'question_groups': grouping, 'tests': test_suite, 'test_command': command, 'warnings': sorted(set(f'{warning.category.__name__}: {warning.message}' for warning in captured)), 'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'gx_context': 'ephemeral; analytics disabled; no Data Docs or remote actions', 'row_id_policy': 'Native IDs checked only if present. No generated ID is presented as a source identifier.', 'duplicate_policy': 'Strict duplicate-content GX diagnostic; failures reported as warnings, not hidden or used to deduplicate data.'}
    write_json(directory/'summary.json', summary)
    (directory/'requirements-data-validation.lock.txt').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(summary['dependencies'].items(),key=lambda item:item[0].lower()))+'\n')
    lines = ['# GX processed-split validation', '', f"GX {summary['gx_version']}; tests: {test_suite['tests']}, failures: {test_suite['failures']}, errors: {test_suite['errors']}.", '', '| Split | Rows | Critical evaluated | Passed | Failed | Duplicate diagnostic failed | Duplicate rows beyond first |','|---|---:|---:|---:|---:|---:|---:|']
    for name, values in summary['splits'].items():
        stats = values['critical']
        diagnostic = values['duplicate_content_diagnostic']
        lines.append(f"| {name} | {values['profile']['rows']} | {stats['evaluated_expectations']} | {stats['successful_expectations']} | {stats['unsuccessful_expectations']} | {diagnostic['unsuccessful_expectations'] if diagnostic else 0} | {values['profile']['full_row_duplicates_beyond_first']} |")
    lines += ['', f"Separate question-group check: {grouping}.", '', 'Content duplicates require review; they are not proof of incorrect human annotations. question_id repeats by design. There is no native answer ID in these files, so source-ID uniqueness is explicitly not asserted. No rows, labels or scores were changed.', '', 'Machine-readable GX results preserve per-expectation success/counts and errors while excluding raw unexpected answers/rows. All results are local; no training, evaluation, W&B, DVC operation or external validation service was invoked.']
    (directory/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    LOGGER.info('GX validation complete; output=%s', directory)
    if not grouping['success'] or any(not value['critical_success'] for value in summary['splits'].values()):
        raise RuntimeError('Critical validation failure; inspect saved results')
    return directory


def main() -> None:
    """Execute params-driven validation from python src/data/validate_gx.py."""
    run(load_params())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
