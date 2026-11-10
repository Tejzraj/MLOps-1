"""Single predeclared validation confirmation of frozen B0 and F1 configurations."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import uuid
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd

from src.models.run_imbalance_experiments import classifier, class_counts, feature_pipeline, score_predictions
from src.utils.phase2_common import ROOT, git_identity, load_params, output_path, sha256, write_json
from src.utils.wandb_tracking import OfflineTracker

LOGGER = logging.getLogger(__name__)


def frozen_settings(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fail on config, feature implementation, dependency or label-map drift from CV."""
    cfg = params['phase2_confirmation']
    experiment = params['phase2_experiments']
    cv = output_path(ROOT, cfg['cv_directory'])
    configs = {}
    for name in cfg['configurations']:
        path = cv / name / 'config.json'
        if sha256(path) != cfg['frozen_config_sha256'][name]:
            raise ValueError('Frozen CV configuration checksum changed')
        frozen = json.loads(path.read_text())
        if frozen['features'] != params['features']:
            raise ValueError('Feature parameters drifted from frozen CV settings')
        for key in ['reference_tfidf', 'estimator', 'model_seed', 'sampler_seed', 'n_jobs', 'metrics']:
            if frozen[key] != experiment[key]:
                raise ValueError(f'{key} drifted from frozen CV settings')
        current = next(item for item in experiment['configurations'] if item['id'] == name)
        if frozen['configuration'] != current or frozen['profile_settings'] != experiment['profiles']['full']:
            raise ValueError('Classifier or tree-count settings drifted from full CV')
        if frozen['label_mapping'] != {str(key): value for key, value in params['data']['target_classes'].items()}:
            raise ValueError('Label mapping differs from frozen CV')
        provenance = json.loads((cv / name / 'provenance.json').read_text())
        for relative, expected in provenance['source_hashes'].items():
            if relative != 'params.yaml' and sha256(ROOT / relative) != expected:
                raise ValueError(f'Frozen CV implementation changed: {relative}')
        for package, version in provenance['dependencies'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f'Dependency changed since CV: {package}')
        configs[name] = frozen
    return configs


def load_partitions(params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the approved train and validation Parquet, checking identity/groups."""
    cfg = params['phase2_confirmation']
    frames = []
    for partition in ['train', 'validation']:
        path = Path(cfg[f'{partition}_path'])
        if sha256(path) != cfg[f'{partition}_sha256']:
            raise ValueError(f'{partition} dataset checksum mismatch')
        frame = pd.read_parquet(path)
        if len(frame) != cfg[f'expected_{partition}_rows']:
            raise ValueError(f'{partition} row count mismatch')
        frames.append(frame)
    experiment = params['phase2_experiments']
    required = experiment['input_columns'] + [experiment['group_column'], experiment['target_column']]
    for frame in frames:
        if frame[required].isna().any().any():
            raise ValueError('Null fields in confirmation data')
        if set(frame[experiment['target_column']]) != set(params['data']['target_classes']):
            raise ValueError('Unexpected or missing class labels')
    if set(frames[0][experiment['group_column']]) & set(frames[1][experiment['group_column']]):
        raise ValueError('Question-group leakage between train and validation')
    return frames[0], frames[1]


def verify_pair(pipeline: Any, model: Any, inputs: pd.DataFrame, params: dict[str, Any], expected_names: list[str] | None = None) -> dict[str, Any]:
    """Check dimensions, class mapping, feature order, and repeat/single/batch behavior."""
    classes = sorted(params['data']['target_classes'])
    if model.classes_.tolist() != classes:
        raise ValueError('Classifier classes do not match params label mapping')
    names = pipeline.get_feature_names_out().tolist()
    transformed = pipeline.transform(inputs)
    if transformed.shape[1] != model.n_features_in_ or len(names) != model.n_features_in_:
        raise ValueError('Feature/model dimension incompatibility')
    if expected_names is not None and names != expected_names:
        raise ValueError('Ordered feature names changed after serialization')
    predictions = model.predict(transformed)
    repeated = model.predict(pipeline.transform(inputs))
    singles = np.array([model.predict(pipeline.transform(inputs.iloc[[i]]))[0] for i in range(len(inputs))])
    if not np.array_equal(predictions, repeated) or not np.array_equal(predictions, singles):
        raise ValueError('Predictions are not deterministic across calls/batching')
    probabilities = model.predict_proba(transformed)
    repeated_probabilities = model.predict_proba(pipeline.transform(inputs))
    if not np.allclose(probabilities, repeated_probabilities, rtol=0, atol=params['phase2_confirmation']['probability_atol']):
        raise ValueError('Probabilities are not deterministic')
    return {'feature_count': int(model.n_features_in_), 'classes': classes, 'deterministic_predictions': True, 'single_batch_equal': True, 'feature_order_verified': True, 'checked_training_rows': len(inputs)}


def decide(metrics: dict[str, dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared strict-greater rule, conservatively retaining ties."""
    cfg = params['phase2_confirmation']
    if cfg['decision_rule'] != 'confirm_candidate_only_if_validation_macro_f1_strictly_exceeds_baseline':
        raise ValueError('Unsupported confirmation decision rule')
    baseline, candidate = cfg['baseline'], cfg['candidate']
    delta = metrics[candidate]['macro_f1'] - metrics[baseline]['macro_f1']
    return {'selected_configuration': candidate if delta > 0 else baseline, 'candidate_confirmed': bool(delta > 0), 'validation_macro_f1_delta_candidate_minus_baseline': float(delta), 'decision_rule': cfg['decision_rule'], 'cv_validation_inconsistency': bool(delta <= 0)}


def run(params: dict[str, Any]) -> Path:
    """Fit each frozen recipe once, save trusted local pairs, and compare validation."""
    frozen = frozen_settings(params)
    cfg = params['phase2_confirmation']
    experiment = params['phase2_experiments']
    root = output_path(ROOT, cfg['output_root'])
    # Refuse a second look, including a previous incomplete training attempt.
    if list(root.glob(cfg['confirmation_id'] + '-*/protocol.json')):
        raise RuntimeError('Confirmation was already attempted; no automatic rerun is allowed')
    train, validation = load_partitions(params)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    group = cfg['confirmation_id'] + '-' + stamp
    directory = output_path(root, group)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'phase': 'single_validation_confirmation', 'decision_rule': cfg['decision_rule'], 'configurations': cfg['configurations'], 'frozen_cv_directory': cfg['cv_directory'], 'frozen_config_sha256': cfg['frozen_config_sha256'], 'train_sha256': cfg['train_sha256'], 'validation_sha256': cfg['validation_sha256'], 'test_set_accessed': False, 'written_before_model_fitting': True}
    write_json(directory / 'protocol.json', protocol)
    target = experiment['target_column']
    group_column = experiment['group_column']
    X_train = train[experiment['input_columns']]
    X_validation = validation[experiment['input_columns']]
    y_train = train[target].to_numpy()
    y_validation = validation[target].to_numpy()
    compatibility = train.groupby(target, group_keys=False).head(cfg['compatibility_sample_per_class'])[experiment['input_columns']]
    identities = git_identity()
    dependencies = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    metrics_by_name = {}
    records = {}
    started = time.perf_counter()
    for name in cfg['configurations']:
        settings = frozen[name]
        folder = directory / name
        folder.mkdir(exist_ok=False)
        fresh_params = deepcopy(params)
        fresh_params['features'] = deepcopy(settings['features'])
        pipeline = feature_pipeline(fresh_params)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fit_started = time.perf_counter()
            training_features = pipeline.fit_transform(X_train)
            model = classifier(settings['configuration'], y_train, fresh_params, settings['profile_settings'])
            model.fit(training_features, y_train)
            fit_seconds = time.perf_counter() - fit_started
            internal_sampling = []
            if settings['configuration']['sampling'] == 'internal':
                for sampler in model.samplers_:
                    indices = sampler.sample_indices_
                    if np.any(indices < 0) or np.any(indices >= len(train)):
                        raise ValueError('Internal sampling escaped training partition')
                    internal_sampling.append(class_counts(y_train[indices], sorted(params['data']['target_classes'])))
            check = verify_pair(pipeline, model, compatibility, params)
            # Only now transform untouched external validation rows, without any fit.
            prediction = model.predict(pipeline.transform(X_validation))
            metrics = score_predictions(y_validation, prediction, fresh_params)
            metrics_by_name[name] = metrics
            names = pipeline.get_feature_names_out().tolist()
            model_path = output_path(folder, cfg['model_filename'])
            pipeline_path = output_path(folder, cfg['pipeline_filename'])
            joblib.dump(model, model_path, compress=cfg['serialization_compress'])
            joblib.dump(pipeline, pipeline_path, compress=cfg['serialization_compress'])
            names_path = output_path(folder, cfg['feature_names_filename'])
            write_json(names_path, names)
            checksums = {file.name: sha256(file) for file in [model_path, pipeline_path, names_path]}
            # These files were generated by this process; no external untrusted pickle is loaded.
            reloaded_pipeline = joblib.load(pipeline_path)
            reloaded_model = joblib.load(model_path)
            roundtrip = verify_pair(reloaded_pipeline, reloaded_model, compatibility, params, names)
            original_predictions = model.predict(pipeline.transform(compatibility))
            restored_predictions = reloaded_model.predict(reloaded_pipeline.transform(compatibility))
            if not np.array_equal(original_predictions, restored_predictions):
                raise ValueError('Serialization changed predictions')
            if not np.allclose(model.predict_proba(pipeline.transform(compatibility)), reloaded_model.predict_proba(reloaded_pipeline.transform(compatibility)), rtol=0, atol=cfg['probability_atol']):
                raise ValueError('Serialization changed probabilities')
            manifest = {'identity': 'New Phase 2 validation candidate; not a submitted Phase 1 binary', 'configuration': name, 'source': identities, 'source_sha256': {relative: sha256(ROOT / relative) for relative in ['src/models/confirm_validation.py', 'src/models/run_imbalance_experiments.py', 'src/features/build_features.py', 'src/utils/wandb_tracking.py', 'src/utils/phase2_common.py', 'params.yaml']}, 'linked_cv_directory': cfg['cv_directory'], 'linked_cv_configuration_sha256': cfg['frozen_config_sha256'][name], 'protocol_sha256': sha256(directory / 'protocol.json'), 'resolved_parameters': settings, 'resolved_estimator_parameters': model.get_params(deep=False), 'dependencies': dependencies, 'datasets': {'train': {'sha256': cfg['train_sha256'], 'rows': len(train), 'questions': int(train[group_column].nunique())}, 'validation': {'sha256': cfg['validation_sha256'], 'rows': len(validation), 'questions': int(validation[group_column].nunique())}}, 'question_group_overlap': 0, 'feature_count': int(model.n_features_in_), 'feature_blocks': {key: int(span.stop - span.start) for key, span in pipeline.output_indices_.items()}, 'required_input_columns': experiment['input_columns'], 'label_mapping': params['data']['target_classes'], 'artifacts_sha256': checksums, 'artifact_files': {'model': cfg['model_filename'], 'feature_pipeline': cfg['pipeline_filename'], 'ordered_feature_names': cfg['feature_names_filename']}, 'compatibility': check, 'serialization_roundtrip': roundtrip, 'fit_seconds': fit_seconds, 'test_set_accessed': False, 'preprocessing': {'training_input': 'Existing cleaned training partition', 'new_inputs': 'Remove candidate br tags, strip/collapse whitespace; preserve case; no Unicode normalization', 'feature_order': ['candidate TF-IDF', 'five length statistics', 'candidate-reference cosine similarity'], 'inference': 'Use paired fitted pipeline.transform then model.predict; never refit or resample'}, 'limitations': ['Not a submitted Phase 1 binary; source commit has uncommitted Phase 2 code recorded by hashes.', 'Question text is unused; reference answer is required.', 'Human labels come from dataset scores; output is a quality category, not an exact human score.', 'Validation is a single confirmation with only eight question groups; minority supports are small.', 'Test scores were historically inspected but this confirmation does not access the test partition.', 'Balanced sampling changes class priors; probabilities are not established as calibrated.', 'Custom transformer module paths and compatible dependencies must accompany the pair.', 'Joblib files execute Python when loaded; trust and verify SHA256 before loading.']}
            write_json(folder / 'config.json', settings)
            write_json(folder / 'metrics.json', {'validation': metrics, 'fit_seconds': fit_seconds, 'internal_tree_sample_counts': internal_sampling})
            write_json(folder / 'predictions.json', [{'row_position': int(i), 'human_label': int(actual), 'prediction': int(pred)} for i, (actual, pred) in enumerate(zip(y_validation, prediction))])
            write_json(folder / 'manifest.json', manifest)
            # Metadata-only offline logging; model files and vocabulary are not staged.
            tracking = deepcopy(experiment['wandb'])
            with OfflineTracker(tracking, folder, group, name + '-validation', {'phase': 'validation_confirmation', 'linked_cv_group': Path(cfg['cv_directory']).name, 'decision_rule': cfg['decision_rule'], 'frozen_settings': settings}, cfg['reviewed_artifact_files']) as tracker:
                tracker.log({f'validation/{key}': value for key, value in metrics.items() if key != 'per_class'})
                for label, values in metrics['per_class'].items():
                    tracker.log({f'validation/{label}/{metric}': value for metric, value in values.items()})
                tracker.log({'validation/fit_seconds': fit_seconds, 'validation/feature_count': int(model.n_features_in_)})
                write_json(folder / 'warnings.json', [{'category': item.category.__name__, 'message': str(item.message)} for item in captured])
                tracker.log_reviewed([folder / file for file in cfg['reviewed_artifact_files']])
                run_location = tracker.run.dir
        records[name] = {'metrics': metrics, 'artifact_directory': str(folder), 'feature_count': int(model.n_features_in_), 'fit_seconds': fit_seconds, 'offline_wandb_run_dir': run_location, 'warnings': [{'category': item.category.__name__, 'message': str(item.message)} for item in captured]}
        LOGGER.info('%s validation macro_f1=%.6f accuracy=%.6f balanced_accuracy=%.6f', name, metrics['macro_f1'], metrics['accuracy'], metrics['balanced_accuracy'])
    for partition in ['train', 'validation']:
        if sha256(Path(cfg[f'{partition}_path'])) != cfg[f'{partition}_sha256']:
            raise RuntimeError('Read-only input dataset changed')
    decision = decide(metrics_by_name, params)
    decision.update({'experiment_group': group, 'records': records, 'protocol_sha256': sha256(directory / 'protocol.json'), 'wall_seconds': time.perf_counter() - started, 'test_set_accessed': False, 'fresh_pipeline_per_configuration': True})
    write_json(directory / 'decision.json', decision)
    LOGGER.info('Decision: %s; confirmed=%s; output=%s', decision['selected_configuration'], decision['candidate_confirmed'], directory)
    return directory


def main() -> None:
    """Require an explicit command to begin the single confirmation attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm', action='store_true', required=True)
    parser.parse_args()
    run(load_params())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
