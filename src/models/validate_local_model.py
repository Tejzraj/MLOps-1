"""Validate the separately versioned local Phase 2 candidate without fitting."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.build_features import TextLengthFeatures, ReferenceSimilarityFeatures

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Hash bytes without deserialization."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main() -> None:
    """Inspect types first, or validate when explicitly requested."""
    params = yaml.safe_load((ROOT / 'params.yaml').read_text())
    cfg = params['phase2_validation']
    source = Path(cfg['source_root'])
    out = ROOT / cfg['report_dir']
    out.mkdir(parents=True, exist_ok=True)
    paths = [source / cfg[k] for k in ('feature_path', 'model_path', 'train_path', 'validation_path', 'test_path')]
    inventory = []
    for entry in sorted((source / cfg['mlflow_models_path']).iterdir()):
        meta_path = entry / 'meta.yaml'
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        artifact = entry / 'artifacts' / 'model.skops'
        if meta.get('source_run_id') == cfg['run_id'] and artifact.exists():
            types = sio.get_untrusted_types(file=artifact)
            inventory.append({'id': entry.name, 'name': meta['name'], 'path': str(artifact), 'sha256': digest(artifact), 'untrusted_types': types})
    (out / 'skops_type_inspection.json').write_text(json.dumps(inventory, indent=2))
    for item in inventory:
        LOGGER.info('INSPECTED %s %s types=%s', item['id'], item['name'], item['untrusted_types'])
        unexpected = set(item['untrusted_types']) - set(cfg['allowed_skops_types'])
        if unexpected:
            raise RuntimeError(f'Unexpected skops types: {unexpected}')
    if '--validate' not in sys.argv:
        return
    before = {str(p): digest(p) for p in paths}
    pipeline = joblib.load(paths[0])
    model = joblib.load(paths[1])
    train, val, test = [pd.read_parquet(p) for p in paths[2:]]
    group = params['data']['group_column']
    assert not set(train[group]) & set(test[group])
    assert not set(val[group]) & set(test[group])
    assert not set(train[group]) & set(val[group])
    target = params['data']['target_column']
    classes = sorted(params['data']['target_classes'])
    assert list(model.classes_) == classes
    sample = test.groupby(target, group_keys=False).head(cfg['sample_per_class'])
    inputs = sample.drop(columns=[target])
    features = pipeline.transform(inputs)
    names = pipeline.get_feature_names_out().tolist()
    assert features.shape[1] == model.n_features_in_ == len(names)
    minimal = inputs[['candidate_answer', 'reference_answer']]
    minimal_features = pipeline.transform(minimal)
    def equal_features(left: object, right: object) -> bool:
        """Compare dense or sparse features without changing either."""
        if sparse.issparse(left) or sparse.issparse(right):
            delta = sparse.csr_matrix(left) - sparse.csr_matrix(right)
            return not delta.nnz or bool(np.max(np.abs(delta.data)) <= cfg['probability_atol'])
        return bool(np.allclose(left, right, rtol=0, atol=cfg['probability_atol']))
    assert equal_features(features, minimal_features)
    vectorizer = pipeline.named_transformers_['tfidf']
    expected = params['features']['tfidf']
    actual = vectorizer.get_params()
    for key, value in expected.items():
        assert actual[key] == (tuple(value) if key == 'ngram_range' else value), key
    assert [name for name, _, _ in pipeline.transformers_] == ['tfidf', 'length_features', 'ref_sim_features', 'remainder']
    assert pipeline.named_transformers_['length_features'].include_length == params['features']['engineered_features']['include_length']
    assert pipeline.named_transformers_['ref_sim_features'].include_reference_similarity == params['features']['engineered_features']['include_reference_similarity']
    assert model.n_estimators == params['model']['random_forest']['n_estimators']
    assert model.max_depth == params['model']['random_forest']['max_depth']
    assert model.random_state == params['model']['random_forest']['random_state']
    feature_blocks = {name: len(pipeline.get_feature_names_out()[span]) for name, span in pipeline.output_indices_.items()}
    batch = model.predict(features)
    singles = np.array([model.predict(pipeline.transform(inputs.iloc[[i]]))[0] for i in range(len(inputs))])
    assert np.array_equal(batch, singles)
    probabilities = model.predict_proba(features)
    comparisons = []
    for item in inventory:
        recovered = sio.load(item['path'], trusted=item['untrusted_types'])
        result = {k: v for k, v in item.items() if k != 'path'}
        if item['name'] == 'feature_pipeline':
            result['feature_names_equal'] = recovered.get_feature_names_out().tolist() == names
            result['transforms_equal'] = equal_features(features, recovered.transform(inputs))
            result['candidate_vocabulary_equal'] = recovered.named_transformers_['tfidf'].vocabulary_ == pipeline.named_transformers_['tfidf'].vocabulary_
            result['reference_vocabulary_equal'] = recovered.named_transformers_['ref_sim_features'].tfidf_vectorizer_.vocabulary_ == pipeline.named_transformers_['ref_sim_features'].tfidf_vectorizer_.vocabulary_
        else:
            result['predictions_equal'] = bool(np.array_equal(batch, recovered.predict(features)))
            result['probabilities_equal'] = bool(np.allclose(probabilities, recovered.predict_proba(features), rtol=0, atol=cfg['probability_atol']))
            result['classes_equal'] = bool(np.array_equal(model.classes_, recovered.classes_))
            result['n_features_in'] = int(recovered.n_features_in_)
        comparisons.append(result)
    all_predictions = model.predict(pipeline.transform(test.drop(columns=[target])))
    recalls = recall_score(test[target], all_predictions, labels=classes, average=None, zero_division=0)
    metrics = {'rows': len(test), 'accuracy': accuracy_score(test[target], all_predictions), 'macro_f1': f1_score(test[target], all_predictions, labels=classes, average='macro', zero_division=0), 'per_class': {params['data']['target_classes'][c]: {'recall': float(r), 'support': int((test[target] == c).sum())} for c, r in zip(classes, recalls)}}
    after = {str(p): digest(p) for p in paths}
    assert before == after
    report = {'identity': 'Phase 2 local candidate; NOT the submitted Phase 1 binary', 'provenance': {'recorded_source_commit': '041da0b19096aa511f778223e9af7af5af1bedb2', 'run_id': cfg['run_id'], 'qualification': 'Local metadata and timestamps support origin; no signed binding of joblib bytes to run.'}, 'source_sha256': before, 'python': platform.python_version(), 'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'configuration_checks_passed': True, 'feature_blocks': feature_blocks, 'feature_dimension': int(features.shape[1]), 'model_n_features_in': int(model.n_features_in_), 'fit_dataframe_columns': pipeline.feature_names_in_.tolist(), 'required_transform_columns': minimal.columns.tolist(), 'feature_names': names, 'transformer_order': [name for name, _, _ in pipeline.transformers_], 'classes': model.classes_.tolist(), 'label_mapping': params['data']['target_classes'], 'sample_rows': len(sample), 'single_batch_equal': bool(np.array_equal(batch, singles)), 'sample_predictions': [{'row_index': str(index), 'question_id': str(sample.loc[index, group]), 'human_label': int(sample.loc[index, target]), 'prediction': int(pred)} for index, pred in zip(sample.index, batch)], 'held_out_evaluation': metrics, 'group_disjointness_verified': True, 'mlflow_comparisons': comparisons, 'source_files_unchanged': before == after, 'limitations': ['Evaluation reuses existing Phase 1 test set; not a fresh independent benchmark.', 'Cross-format agreement is measured on representative rows, not proof of equality for all inputs.', 'Question text is ignored. New-domain generalization is not established.', 'Cleaning is outside fitted artifact: remove candidate br tags; strip/collapse whitespace; no actual Unicode normalization.', 'Human score thresholds label categories; predictions are not numeric grader scores.', 'No training or refitting performed.']}
    (out / 'local_model_validation.json').write_text(json.dumps(report, indent=2))
    markdown = '# Phase 2 local model validation\n\nThis validates the local candidate, not the submitted Phase 1 model binaries.\n\n```json\n' + json.dumps({k: v for k, v in report.items() if k != 'feature_names'}, indent=2) + '\n```\n\nComplete ordered feature names are in local_model_validation.json.\n\nBundle recommendation: retain the original fitted pair and import-compatible custom transformer code; include preprocessing, label mapping, params, dependency lock, schema, source checksums, run metadata, ordered feature names, and measured limitations in a versioned manifest. Record source commit as metadata-reported, not a proof of clean training source. Keep Phase 1 DVC metadata unchanged.\n'
    (out / 'local_model_validation.md').write_text(markdown)
    LOGGER.info('RESULT dimensions=%s metrics=%s comparisons=%s', features.shape, metrics, comparisons)


if __name__ == '__main__':
    main()
