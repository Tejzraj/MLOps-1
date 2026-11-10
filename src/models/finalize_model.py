"""Evaluate only the selected validated B0 once and package its unchanged fitted pair."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any
import uuid
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import yaml

from src.inference.predict import BundlePredictor, clean_record
from src.models.run_imbalance_experiments import score_predictions
from src.utils.phase2_common import ROOT, git_identity, load_params, output_path, sha256, write_json
from src.utils.wandb_tracking import OfflineTracker

LOGGER = logging.getLogger(__name__)


def verify_selected(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the selected B0 candidate to the saved decision and checksums before load."""
    cfg = params['phase2_final']
    candidate = output_path(ROOT, cfg['candidate_directory'])
    manifest_path = candidate / 'manifest.json'
    decision_path = output_path(ROOT, cfg['selection_record'])
    if sha256(manifest_path) != cfg['candidate_manifest_sha256']:
        raise ValueError('Selected candidate manifest checksum changed')
    if sha256(decision_path) != cfg['selection_record_sha256']:
        raise ValueError('Selection record checksum changed')
    manifest = json.loads(manifest_path.read_text())
    decision = json.loads(decision_path.read_text())
    if manifest['configuration'] != cfg['selected_configuration'] or decision['selected_configuration'] != cfg['selected_configuration']:
        raise ValueError('Model selection identity mismatch')
    for filename, expected in manifest['artifacts_sha256'].items():
        if sha256(candidate / filename) != expected:
            raise ValueError('Selected B0 artifact checksum mismatch')
    return manifest, decision


def make_bundle(directory: Path, params: dict[str, Any], candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Copy verified bytes and exact custom-class source into a portable inference tree."""
    cfg = params['phase2_final']
    bundle = output_path(directory, cfg['bundle_directory'])
    bundle.mkdir(exist_ok=False)
    source = output_path(ROOT, cfg['candidate_directory'])
    for filename, expected in candidate['artifacts_sha256'].items():
        shutil.copy2(source / filename, bundle / filename)
        if sha256(bundle / filename) != expected:
            raise ValueError('Copy changed selected artifact bytes')
    shutil.copy2(source / 'manifest.json', bundle / cfg['candidate_manifest_filename'])
    for relative in cfg['source_files']:
        target = output_path(bundle, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    # Preserve the canonical class import path with the exact class body, excluding
    # training/tracking functions and their imblearn/W&B imports from the bundle.
    relative = cfg['reference_class_source']
    original = (ROOT / relative).read_text(encoding='utf-8-sig')
    tree = ast.parse(original)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cfg['reference_class_name'])
    body = ast.get_source_segment(original, definition)
    header = '"""Exact compatibility class from the training source; no experiment runner."""\nfrom __future__ import annotations\nfrom typing import Any\nimport pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom src.features.build_features import ReferenceSimilarityFeatures\n\n'
    output_path(bundle, relative).write_text(header + body + '\n', encoding='utf-8')
    bundle_params = {'preprocessing': cfg['preprocessing'], 'label_mapping': params['data']['target_classes'], 'question_is_predictive': False}
    (bundle / cfg['bundle_params_filename']).write_text(yaml.safe_dump(bundle_params, sort_keys=False), encoding='utf-8')
    dependencies = {package: importlib.metadata.version(package) for package in cfg['runtime_packages']}
    for package, version in dependencies.items():
        if version != candidate['dependencies'][package]:
            raise ValueError('Runtime dependency differs from validated model environment')
    (bundle / cfg['runtime_requirements_filename']).write_text('\n'.join(f'{name}=={version}' for name, version in dependencies.items()) + '\n', encoding='utf-8')
    readme = '# B0 inference bundle\n\nLocally generated trusted Phase 2 candidate; not a submitted Phase 1 binary.\n\nUse Python ' + platform.python_version() + '. Install requirements-inference.txt in an isolated environment. No imbalanced-learn, W&B or training framework is needed at inference.\n\nRun from any directory:\n\n```powershell\npython -B <bundle>/src/inference/predict.py --bundle <bundle> --manifest ' + cfg['bundle_manifest_filename'] + ' --input <request.json>\n```\n\nInput is a JSON object or nonempty list with exactly question, reference_answer and candidate_answer string fields. Output contains class_id and quality_label. Empty cleaned candidate answers are rejected, as training discarded them. Question text is accepted but not used as a feature.\n\nCandidate br tags are removed case-sensitively, then all fields are stripped and repeated whitespace is collapsed. Case is preserved; Unicode normalization is not performed. Use the fitted pair without fitting or resampling.\n\nThe original canonical module names are retained for joblib compatibility. The parameterized reference transformer has its exact original class body, without experiment/tracking code.\n\nChecksums establish integrity, not authenticity. Joblib can execute code: load only this trusted bundle after reviewing its provenance.\n'
    (bundle / 'README.md').write_text(readme, encoding='utf-8')
    manifest = {'bundle_version': cfg['bundle_version'], 'selected_configuration': cfg['selected_configuration'], 'candidate_manifest_file': cfg['candidate_manifest_filename'], 'candidate_manifest_sha256': cfg['candidate_manifest_sha256'], 'selection_record_sha256': cfg['selection_record_sha256'], 'params_file': cfg['bundle_params_filename'], 'runtime_dependencies': dependencies, 'python': platform.python_version(), 'source': git_identity(), 'source_class_body_sha256': sha256(ROOT / relative), 'source_class_extraction': 'Exact AST source segment of ConfiguredReferenceSimilarityFeatures at its canonical import path', 'model_retrained': False, 'question_is_predictive': False, 'test_benchmark_status': 'reused previously inspected benchmark; not independent unseen test', 'files_sha256': {str(path.relative_to(bundle).as_posix()): sha256(path) for path in bundle.rglob('*') if path.is_file()}}
    write_json(bundle / cfg['bundle_manifest_filename'], manifest)
    return bundle, manifest


def run(params: dict[str, Any]) -> Path:
    """Build, verify and evaluate the frozen selected model once without training."""
    cfg = params['phase2_final']
    root = output_path(ROOT, cfg['output_root'])
    if list(root.glob(cfg['final_id'] + '-*/protocol.json')):
        raise RuntimeError('Final evaluation was already attempted; no automatic repeat allowed')
    candidate, selection = verify_selected(params)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    group = cfg['final_id'] + '-' + stamp
    directory = output_path(root, group)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'selected_configuration': cfg['selected_configuration'], 'selection_frozen_before_test': True, 'selection_record_sha256': cfg['selection_record_sha256'], 'candidate_manifest_sha256': cfg['candidate_manifest_sha256'], 'benchmark_status': 'reused_test_benchmark_not_independent_unseen_test', 'test_sha256': cfg['test_sha256'], 'test_metrics_do_not_change_selection': True}
    write_json(directory / 'protocol.json', protocol)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        bundle, bundle_manifest = make_bundle(directory, params, candidate)
        # Both copied artifacts are checked against the saved candidate manifest by
        # BundlePredictor before joblib.load. No original Phase 1 binaries are used.
        predictor = BundlePredictor(bundle, cfg['bundle_manifest_filename'])
        test_path = Path(cfg['test_path'])
        if sha256(test_path) != cfg['test_sha256']:
            raise ValueError('Test partition checksum mismatch')
        frame = pd.read_parquet(test_path)
        if len(frame) != cfg['expected_test_rows']:
            raise ValueError('Test partition row count mismatch')
        columns = cfg['preprocessing']['required_columns']
        target = params['phase2_experiments']['target_column']
        records = frame[columns].to_dict(orient='records')
        # Existing processed benchmark should already have exactly the cleaning
        # used for training. Refuse silent changes before the one prediction pass.
        if any(clean_record(record, cfg['preprocessing']) != record for record in records):
            raise ValueError('Processed test rows disagree with exact inference cleaning')
        predictions = predictor.predict_batch(records)
        prediction_ids = np.array([item['class_id'] for item in predictions])
        metrics = score_predictions(frame[target].to_numpy(), prediction_ids, params)
        cv_path = output_path(ROOT, candidate['linked_cv_directory'])
        cv_metrics = json.loads((cv_path / cfg['selected_configuration'] / 'metrics.json').read_text())
        evaluation = {'benchmark_status': protocol['benchmark_status'], 'selected_configuration': cfg['selected_configuration'], 'selection_changed': False, 'rows': len(frame), 'questions': int(frame[params['phase2_experiments']['group_column']].nunique()), 'test_sha256': cfg['test_sha256'], 'metrics': metrics, 'cv_aggregate': cv_metrics['aggregate'], 'validation_metrics': selection['records'][cfg['selected_configuration']]['metrics'], 'candidate_manifest_sha256': cfg['candidate_manifest_sha256'], 'bundle_directory': str(bundle), 'wall_seconds_before_tracking': time.perf_counter() - started}
        write_json(directory / 'final_evaluation.json', evaluation)
        write_json(directory / 'test_predictions.json', [{'row_position': i, 'human_label': int(actual), 'prediction': int(pred)} for i, (actual, pred) in enumerate(zip(frame[target], prediction_ids))])
        provenance = {'source': git_identity(), 'python': platform.python_version(), 'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'source_sha256': {relative: sha256(ROOT / relative) for relative in ['src/models/finalize_model.py', 'src/inference/predict.py', 'params.yaml']}, 'candidate_manifest_sha256': cfg['candidate_manifest_sha256'], 'selection_record_sha256': cfg['selection_record_sha256'], 'linked_validation_group': selection['experiment_group'], 'linked_cv_directory': candidate['linked_cv_directory'], 'protocol_sha256': sha256(directory / 'protocol.json')}
        write_json(directory / 'provenance.json', provenance)
        write_json(directory / 'config.json', cfg)
        card = '# Model card: Phase 2 B0 v' + cfg['bundle_version'] + '\n\nSelected after the fixed validation confirmation retained B0 over Balanced RF. This is the unchanged newly trained validation B0 pair, not the submitted Phase 1 binaries. No retraining or test-driven selection occurred.\n\nIntended use: experimental short-answer quality classification given a candidate and reference answer. The required question field is not predictive. Outputs are categories (Poor, Needs Improvement, Good, Excellent), not exact grader scores. Labels derive only from Mohler human scores.\n\nTraining: 1,561 rows and 56 question groups. Validation: 229 rows and eight groups. Features: candidate TF-IDF, five length statistics and one candidate/reference cosine similarity, fitted only on training data; 2,824 total.\n\nFinal test results below are a REUSED PREVIOUSLY INSPECTED BENCHMARK, not an independent unseen test. The partition informed earlier analysis, and the results are descriptive only.\n\n```json\n' + json.dumps({'test_metrics': metrics, 'cv_aggregate': cv_metrics['aggregate'], 'validation_metrics': evaluation['validation_metrics']}, indent=2) + '\n```\n\nLimitations: narrow introductory data-structures domain; small minority supports; class imbalance and weak minority recall; no established domain generalization or calibrated probabilities. The model does not assess factual correctness robustly or generate personalized feedback. Do not treat its category as an authoritative human grade.\n\nRaw-input preprocessing removes candidate br tags, strips/collapses whitespace, preserves case, and does not normalize Unicode. Empty cleaned candidates are rejected, matching training exclusion. The paired fitted pipeline must accompany the model.\n\nSource commit and uncommitted source hashes, package versions and artifact identities are recorded in the candidate and bundle manifests. Joblib can execute code: only load trusted files after integrity verification.\n'
        (bundle / cfg['model_card_filename']).write_text(card, encoding='utf-8')
        (directory / cfg['model_card_filename']).write_text(card, encoding='utf-8')
        bundle_manifest['files_sha256'][cfg['model_card_filename']] = sha256(bundle / cfg['model_card_filename'])
        write_json(bundle / cfg['bundle_manifest_filename'], bundle_manifest)
        tracking = deepcopy(params['phase2_experiments']['wandb'])
        with OfflineTracker(tracking, directory, group, cfg['selected_configuration'] + '-reused-test', {'phase': 'final_reused_test_benchmark', 'selected_configuration': cfg['selected_configuration'], 'linked_cv_group': Path(candidate['linked_cv_directory']).name, 'linked_validation_group': selection['experiment_group'], 'candidate_manifest_sha256': cfg['candidate_manifest_sha256'], 'test_sha256': cfg['test_sha256'], 'selection_changed': False}, cfg['reviewed_artifact_files']) as tracker:
            tracker.log({f'reused_test/{key}': value for key, value in metrics.items() if key != 'per_class'})
            for label, values in metrics['per_class'].items():
                tracker.log({f'reused_test/{label}/{key}': value for key, value in values.items()})
            write_json(directory / 'warnings.json', [{'category': item.category.__name__, 'message': str(item.message)} for item in captured])
            tracker.log_reviewed([directory / filename for filename in cfg['reviewed_artifact_files']])
            run_directory = tracker.run.dir
        if sha256(test_path) != cfg['test_sha256']:
            raise RuntimeError('Read-only test dataset changed')
        verify_selected(params)
        write_json(directory / 'completion.json', {'complete': True, 'wall_seconds': time.perf_counter() - started, 'offline_wandb_run_directory': run_directory, 'test_prediction_passes': 1, 'selected_configuration': cfg['selected_configuration'], 'bundle_manifest_sha256': sha256(bundle / cfg['bundle_manifest_filename']), 'warnings': [{'category': item.category.__name__, 'message': str(item.message)} for item in captured]})
    LOGGER.info('Reused test metrics=%s; bundle=%s', metrics, bundle)
    return directory


def main() -> None:
    """Require explicit final evaluation of the fixed selected model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-once', action='store_true', required=True)
    parser.parse_args()
    run(load_params())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
