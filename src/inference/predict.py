"""Raw-input inference for the selected Phase 2 B0 bundle; never fits models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    """Verify file identity before deserializing trusted local model artifacts."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def local_path(root: Path, relative: str) -> Path:
    """Reject escaping manifest paths before reading files."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle path escapes its root')
    return path


def clean_record(record: Any, config: dict[str, Any]) -> dict[str, str]:
    """Mirror candidate br removal and strip/collapse cleaning used in training."""
    if not isinstance(record, dict):
        raise ValueError('Each input must be an object')
    required = config['required_columns']
    if any(column not in record for column in required):
        raise ValueError('Input requires question, reference_answer and candidate_answer')
    if config['strict_fields'] and set(record) != set(required):
        raise ValueError('Unexpected input fields')
    result = {}
    for column in required:
        value = record[column]
        if not isinstance(value, str):
            raise ValueError(f'{column} must be a string')
        if column == config['candidate_column']:
            value = re.sub(config['candidate_html_pattern'], '', value)
        if config['strip']:
            value = value.strip()
        value = re.sub(config['whitespace_pattern'], ' ', value)
        if column in config['reject_empty_columns'] and not value:
            raise ValueError(f'{column} must not be empty after cleaning')
        result[column] = value
    return result


class BundlePredictor:
    """Load an explicitly trusted bundle after checksum and dependency verification."""

    def __init__(self, directory: Path, manifest_filename: str = 'bundle_manifest.json') -> None:
        """Verify every packaged file and the saved candidate hashes before loading."""
        self.directory = directory.resolve()
        self.manifest = json.loads(local_path(self.directory, manifest_filename).read_text())
        for filename, expected in self.manifest['files_sha256'].items():
            if file_sha256(local_path(self.directory, filename)) != expected:
                raise ValueError(f'Bundle checksum mismatch: {filename}')
        candidate = json.loads(local_path(self.directory, self.manifest['candidate_manifest_file']).read_text())
        for kind in ['model', 'feature_pipeline']:
            filename = candidate['artifact_files'][kind]
            if file_sha256(local_path(self.directory, filename)) != candidate['artifacts_sha256'][filename]:
                raise ValueError(f'Saved candidate checksum mismatch: {kind}')
        if candidate['configuration'] != self.manifest['selected_configuration']:
            raise ValueError('Selected model identity mismatch')
        for package, version in self.manifest['runtime_dependencies'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f'Runtime dependency mismatch: {package}')
        self.params = yaml.safe_load(local_path(self.directory, self.manifest['params_file']).read_text())
        self.preprocessing = self.params['preprocessing']
        self.labels = {int(key): value for key, value in self.params['label_mapping'].items()}
        # Callers must trust the provenance; SHA-256 is integrity, not authentication.
        self.pipeline = joblib.load(local_path(self.directory, candidate['artifact_files']['feature_pipeline']))
        self.model = joblib.load(local_path(self.directory, candidate['artifact_files']['model']))
        if self.model.classes_.tolist() != sorted(self.labels):
            raise ValueError('Classifier classes do not match the label mapping')
        if self.model.n_features_in_ != candidate['feature_count']:
            raise ValueError('Classifier feature dimension differs from candidate manifest')
        names = json.loads(local_path(self.directory, candidate['artifact_files']['ordered_feature_names']).read_text())
        if self.pipeline.get_feature_names_out().tolist() != names or len(names) != self.model.n_features_in_:
            raise ValueError('Ordered feature contract differs from the saved candidate')

    def predict_batch(self, records: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Validate a batch, clean raw text and decode model class IDs into labels."""
        if not isinstance(records, list) or not records:
            raise ValueError('Input batch must be a nonempty list')
        cleaned = [clean_record(record, self.preprocessing) for record in records]
        frame = pd.DataFrame(cleaned)[self.preprocessing['predictive_columns']]
        features = self.pipeline.transform(frame)
        if features.shape[1] != self.model.n_features_in_:
            raise ValueError('Feature/model dimension mismatch')
        predictions = self.model.predict(features)
        return [{'class_id': int(value), 'quality_label': self.labels[int(value)]} for value in predictions]

    def predict_one(self, record: dict[str, str]) -> dict[str, Any]:
        """Predict a single answer with the same path as batch inference."""
        return self.predict_batch([record])[0]


def main() -> None:
    """Read a JSON object/list from a file and write only JSON predictions to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--manifest', default='bundle_manifest.json')
    args = parser.parse_args()
    # The directory is explicit: this also supports running the file inside a bundle.
    sys.path.insert(0, str(args.bundle.resolve()))
    predictor = BundlePredictor(args.bundle, args.manifest)
    records = json.loads(args.input.read_text(encoding='utf-8-sig'))
    result = predictor.predict_batch(records) if isinstance(records, list) else predictor.predict_one(records)
    sys.stdout.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
