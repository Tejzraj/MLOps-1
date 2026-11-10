"""Run reproducible Phase 2 grouped-CV imbalance experiments without touching Phase 1."""
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from src.features.build_features import ReferenceSimilarityFeatures, build_feature_pipeline
from src.utils.phase2_common import ROOT, git_identity, load_params, output_path, sha256, write_json
from src.utils.wandb_tracking import OfflineTracker

LOGGER = logging.getLogger(__name__)


class ConfiguredReferenceSimilarityFeatures(ReferenceSimilarityFeatures):
    """Retain legacy transformation semantics with vectorizer settings from params."""

    def __init__(self, tfidf_parameters: dict[str, Any], include_reference_similarity: bool = True) -> None:
        """Keep constructor parameters clone-compatible."""
        super().__init__(include_reference_similarity=include_reference_similarity)
        self.tfidf_parameters = tfidf_parameters

    def fit(self, X: pd.DataFrame, y: Any = None) -> ConfiguredReferenceSimilarityFeatures:
        """Fit the reference vocabulary only on original fold-training texts."""
        if self.include_reference_similarity:
            settings = dict(self.tfidf_parameters)
            settings['ngram_range'] = tuple(settings['ngram_range'])
            self.tfidf_vectorizer_ = TfidfVectorizer(**settings)
            texts = X['candidate_answer'].fillna('').astype(str).tolist() + X['reference_answer'].fillna('').astype(str).tolist()
            self.tfidf_vectorizer_.fit(texts)
        return self


def feature_pipeline(params: dict[str, Any]) -> Any:
    """Build a new, unfitted feature pipeline without changing legacy source."""
    cfg = params['phase2_experiments']
    pipeline = build_feature_pipeline(params)
    pipeline.set_params(ref_sim_features=ConfiguredReferenceSimilarityFeatures(cfg['reference_tfidf'], params['features']['engineered_features']['include_reference_similarity']))
    return pipeline


def class_counts(y: np.ndarray, classes: list[int]) -> dict[str, int]:
    """Report original or sampled counts with stable class order."""
    return {str(label): int(np.sum(y == label)) for label in classes}


def grouped_folds(frame: pd.DataFrame, params: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shared grouped folds and reject overlap or missing classes."""
    cfg = params['phase2_experiments']
    groups = frame[cfg['group_column']]
    target = frame[cfg['target_column']]
    splitter = StratifiedGroupKFold(**cfg['cv'])
    folds = list(splitter.split(frame, target, groups))
    classes = set(params['data']['target_classes'])
    for train, validation in folds:
        if set(groups.iloc[train]) & set(groups.iloc[validation]):
            raise ValueError('Question-group leakage in CV split')
        if set(target.iloc[train]) != classes or set(target.iloc[validation]) != classes:
            raise ValueError('All four classes must occur on each side of every fold')
    return folds


def classifier(config: dict[str, Any], y: np.ndarray, params: dict[str, Any], profile: dict[str, Any]) -> Any:
    """Construct the configured estimator using only fold-training labels."""
    cfg = params['phase2_experiments']
    settings = dict(cfg['estimator'])
    settings.update(n_estimators=profile['n_estimators'], random_state=cfg['model_seed'], n_jobs=cfg['n_jobs'])
    weight = config['class_weight']
    if weight == 'sqrt_inverse_frequency':
        classes = sorted(params['data']['target_classes'])
        weight = {label: float((len(y) / (len(classes) * np.sum(y == label))) ** cfg['weight_exponent']) for label in classes}
    settings['class_weight'] = weight
    if config['estimator'] == 'balanced_random_forest':
        settings.update(sampling_strategy=config['sampling_strategy'], replacement=config['replacement'], bootstrap=config['bootstrap'])
        return BalancedRandomForestClassifier(**settings)
    if config['estimator'] != 'random_forest':
        raise ValueError('Unknown classifier')
    return RandomForestClassifier(**settings)


def score_predictions(y: np.ndarray, prediction: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Compute fixed-label macro-F1 and supporting metrics on untouched rows."""
    cfg = params['phase2_experiments']
    mapping = params['data']['target_classes']
    labels = sorted(mapping)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=cfg['metrics']['zero_division'])
    return {'accuracy': float(accuracy_score(y, prediction)), 'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', zero_division=cfg['metrics']['zero_division'])), 'balanced_accuracy': float(balanced_accuracy_score(y, prediction)), 'per_class': {mapping[label]: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(n)} for label, p, r, f, n in zip(labels, precision, recall, f1, support)}, 'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist()}


def fit_fold(frame: pd.DataFrame, train: np.ndarray, validation: np.ndarray, params: dict[str, Any], config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Fit fresh features before fold-local sampling and audit sampled row origin."""
    cfg = params['phase2_experiments']
    classes = sorted(params['data']['target_classes'])
    if set(frame.iloc[train][cfg['group_column']]) & set(frame.iloc[validation][cfg['group_column']]):
        raise ValueError('Question-group leakage before fitting')
    X_train = frame.iloc[train][cfg['input_columns']]
    X_validation = frame.iloc[validation][cfg['input_columns']]
    y_train = frame.iloc[train][cfg['target_column']].to_numpy()
    y_validation = frame.iloc[validation][cfg['target_column']].to_numpy()
    pipeline = feature_pipeline(params)
    started = time.perf_counter()
    # Neither features nor samplers ever receive fold-validation rows during fitting.
    train_features = pipeline.fit_transform(X_train)
    validation_features = pipeline.transform(X_validation)
    fit_X, fit_y = train_features, y_train
    sampled_source_rows = train
    if config['sampling'] == 'random_over':
        sampler = RandomOverSampler(sampling_strategy=config['sampling_strategy'], shrinkage=config['shrinkage'], random_state=cfg['sampler_seed'])
        fit_X, fit_y = sampler.fit_resample(train_features, y_train)
        sampled_source_rows = train[sampler.sample_indices_]
    elif config['sampling'] not in ('none', 'internal'):
        raise ValueError('Unknown sampling strategy')
    model = classifier(config, y_train, params, profile)
    model.fit(fit_X, fit_y)
    internal_counts = []
    if config['sampling'] == 'internal':
        for sampler in model.samplers_:
            indices = sampler.sample_indices_
            if np.any(indices < 0) or np.any(indices >= len(train)):
                raise ValueError('Internal sampler indices escape fold-training data')
            if set(train[indices]) & set(validation):
                raise ValueError('Internal sampler touched fold-validation rows')
            internal_counts.append(class_counts(y_train[indices], classes))
    if set(sampled_source_rows) & set(validation) or not set(sampled_source_rows) <= set(train):
        raise ValueError('Resampling used rows outside fold training')
    if model.n_features_in_ != validation_features.shape[1]:
        raise ValueError('Training and prediction dimensions differ')
    prediction = model.predict(validation_features)
    metrics = score_predictions(y_validation, prediction, params)
    return {'metrics': metrics, 'feature_dimension': int(train_features.shape[1]), 'fit_seconds': time.perf_counter() - started, 'n_estimators': model.n_estimators, 'class_weight': model.class_weight, 'audit': {'question_group_overlap': 0, 'feature_fit_row_positions': train.tolist(), 'validation_row_positions': validation.tolist(), 'sampled_source_row_positions': sampled_source_rows.tolist(), 'sampling_scope': 'fold_training_only', 'train_rows_before': len(train), 'train_rows_after_external_sampling': len(fit_y), 'validation_rows': len(validation), 'validation_unresampled': True, 'counts_before': class_counts(y_train, classes), 'counts_after_external_sampling': class_counts(fit_y, classes), 'internal_tree_sample_counts': internal_counts}, 'predictions': [{'row_position': int(row), 'human_label': int(actual), 'prediction': int(pred)} for row, actual, pred in zip(validation, y_validation, prediction)]}


def flatten_metrics(result: dict[str, Any], fold: int) -> dict[str, Any]:
    """Create explicit W&B scalars and a confusion-matrix payload."""
    metrics = result['metrics']
    payload = {'fold': fold, 'fold/macro_f1': metrics['macro_f1'], 'fold/accuracy': metrics['accuracy'], 'fold/balanced_accuracy': metrics['balanced_accuracy'], 'fold/fit_seconds': result['fit_seconds'], 'fold/feature_dimension': result['feature_dimension'], 'fold/confusion_matrix': metrics['confusion_matrix']}
    for label, values in metrics['per_class'].items():
        payload.update({f'fold/{label}/{metric}': value for metric, value in values.items()})
    return payload


def run(profile_name: str, params: dict[str, Any]) -> Path:
    """Run the selected profile; never read held-out validation or test partitions."""
    cfg = params['phase2_experiments']
    profile = cfg['profiles'][profile_name]
    if cfg['wandb']['mode'] != 'offline':
        raise ValueError('Only offline experiments are authorized')
    if profile_name == 'full' and profile['n_estimators'] != params['model']['random_forest']['n_estimators']:
        raise ValueError('Full profile must retain the configured baseline tree count')
    source = Path(cfg['train_path'])
    before = sha256(source)
    if before != cfg['train_sha256']:
        raise ValueError('Training dataset checksum differs from approved input')
    frame = pd.read_parquet(source)
    required = cfg['input_columns'] + [cfg['group_column'], cfg['target_column']]
    if frame[required].isna().any().any():
        raise ValueError('Null input fields are not accepted')
    folds = grouped_folds(frame, params)
    selected_folds = [(i, folds[i]) for i in profile['fold_indices']]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    group = f"{cfg['wandb']['group_prefix']}-{profile_name}-{stamp}"
    root = output_path(ROOT, cfg['output_root'])
    directory = output_path(root, group)
    directory.mkdir(parents=True, exist_ok=False)
    assignment = np.full(len(frame), -1, dtype=int)
    for i, (_, validation) in enumerate(folds):
        assignment[validation] = i
    assignments = [{'row_position': int(i), 'question_id': str(group_id), 'validation_fold': int(assignment[i])} for i, group_id in enumerate(frame[cfg['group_column']])]
    write_json(directory / 'fold_assignments.json', assignments)
    provenance = {'source': git_identity(), 'train_sha256': before, 'source_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT / 'src/utils/wandb_tracking.py', ROOT / 'src/utils/phase2_common.py', ROOT / 'src/features/build_features.py', ROOT / 'params.yaml']}, 'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'profile': profile_name, 'eligible_for_selection': profile['eligible_for_selection'], 'test_set_read': False, 'external_validation_set_read': False, 'binary_models_saved': False}
    summary = []
    baseline_means: dict[str, float] = {}
    for config in cfg['configurations']:
        name = config['id']
        folder = directory / name
        folder.mkdir(exist_ok=False)
        # Only explicit parameters enter W&B; never dump os.environ or raw rows.
        resolved = {'experiment_id': cfg['experiment_id'], 'profile': profile_name, 'configuration': config, 'profile_settings': profile, 'cv': cfg['cv'], 'model_seed': cfg['model_seed'], 'sampler_seed': cfg['sampler_seed'], 'weight_exponent': cfg['weight_exponent'], 'n_jobs': cfg['n_jobs'], 'estimator': cfg['estimator'], 'metrics': cfg['metrics'], 'features': params['features'], 'reference_tfidf': cfg['reference_tfidf'], 'label_mapping': params['data']['target_classes'], 'train_sha256': before, 'group_column': cfg['group_column'], 'selection_rule': cfg['selection']}
        results = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with OfflineTracker(cfg['wandb'], folder, group, name, resolved, cfg['artifact_files']) as tracker:
                for fold, (train, validation) in selected_folds:
                    result = fit_fold(frame, train, validation, params, config, profile)
                    result['fold'] = fold
                    results.append(result)
                    tracker.log(flatten_metrics(result, fold))
                    LOGGER.info('%s fold=%d macro_f1=%.6f balanced_accuracy=%.6f', name, fold, result['metrics']['macro_f1'], result['metrics']['balanced_accuracy'])
                aggregates = {}
                for metric in ['macro_f1', 'accuracy', 'balanced_accuracy']:
                    values = [r['metrics'][metric] for r in results]
                    aggregates[f'{metric}_mean'] = float(np.mean(values))
                    aggregates[f'{metric}_std'] = float(np.std(values, ddof=cfg['metrics']['std_ddof']))
                fold_scores = {str(r['fold']): r['metrics']['macro_f1'] for r in results}
                if name == cfg['configurations'][0]['id']:
                    baseline_means = fold_scores
                paired = {fold: score - baseline_means[fold] for fold, score in fold_scores.items()}
                pooled_rows = [prediction for result in results for prediction in result['predictions']]
                pooled = score_predictions(np.array([r['human_label'] for r in pooled_rows]), np.array([r['prediction'] for r in pooled_rows]), params)
                tracker.log({f'cv/{key}': value for key, value in aggregates.items()})
                tracker.run.summary.update({'eligible_for_selection': profile['eligible_for_selection'], 'folds_completed': len(results), 'question_group_overlap': 0})
                write_json(folder / 'config.json', resolved)
                write_json(folder / 'provenance.json', provenance)
                write_json(folder / 'fold_assignments.json', assignments)
                write_json(folder / 'metrics.json', {'folds': [{k: v for k, v in r.items() if k != 'predictions'} for r in results], 'aggregate': aggregates, 'pooled_oof': pooled, 'paired_macro_f1_delta_vs_B0': paired})
                write_json(folder / 'predictions.json', pooled_rows)
                warning_records = [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in captured]
                write_json(folder / 'warnings.json', warning_records)
                tracker.log_reviewed([folder / filename for filename in cfg['artifact_files']])
                run_location = tracker.run.dir
                run_id = tracker.run.id
        summary.append({'configuration': name, **aggregates, 'folds_completed': len(results), 'n_estimators': profile['n_estimators'], 'wandb_run_id': run_id, 'wandb_run_dir': run_location, 'warnings': [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in captured]})
    if sha256(source) != before:
        raise RuntimeError('Read-only source changed during the experiment')
    write_json(directory / 'summary.json', {'experiment_group': group, 'profile': profile_name, 'eligible_for_selection': profile['eligible_for_selection'], 'configurations': summary, 'question_group_overlap': 0, 'test_set_read': False, 'external_validation_set_read': False, 'binary_models_saved': False, 'provenance': provenance})
    LOGGER.info('Completed %s; %d fits; output=%s', profile_name, len(summary) * len(selected_folds), directory)
    return directory


def main() -> None:
    """Run a params-driven profile selected explicitly on the command line."""
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(params['phase2_experiments']['profiles']), required=True)
    arguments = parser.parse_args()
    run(arguments.profile, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
