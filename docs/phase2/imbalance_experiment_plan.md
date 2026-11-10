# Phase 2 class-imbalance framework and smoke result

The smoke run validates the implementation only. It does not select a model and is not a replacement for submitted Phase 1 results. The full experiment has not run.

## Implementation

- `src/models/run_imbalance_experiments.py` runs the six configurations from `phase2_experiments` in `params.yaml`. Every configuration receives identical saved grouped folds. The runner reads only the checksum-verified original training Parquet, in read-only mode; it never opens external validation/test partitions or serialized baseline models.
- B0: RF balanced; W1: RF unweighted; W2: RF balanced_subsample; W3: fold-derived square-root inverse-frequency weights; O1: plain RandomOverSampler then unweighted RF; F1: balanced RF with internal replacement sampling, bootstrap=False and no extra class weights.
- Fresh TF-IDF features are fitted on original fold-training rows before any sampling. The new parameterized reference-similarity transformer preserves legacy transformation behavior without editing Phase 1 feature source.
- The primary metric is mean grouped-CV macro-F1. Per-fold class precision/recall/F1/support, balanced accuracy, confusion matrices, pooled out-of-fold metrics and paired baseline deltas are recorded separately. Standard deviation uses the configured ddof; smoke standard deviations are zero because there is one fold, not evidence of stability.
- No fold model binaries are persisted. Selected-candidate final fitting/export remains a separate future step after review; enabling model persistence or registry flags currently fails closed.
- No DVC stage was changed. This standalone Phase 2 framework does not invoke DVC or MLflow.

## Actual checks

- Focused Pytest: 12 tests; 0 failures; 0 errors; 0 skipped. JUnit details: `artifacts/phase2/pytest-results.xml`.
- Python 3.11.9; pip dependency check and imbalanced-learn/W&B imports passed.
- Smoke: 6 configurations, 1 grouped fold per configuration, 20 trees each; full profile retains 300 trees and 5 folds per configuration.
- Fold training: 1249 rows / 45 questions; fold validation: 312 rows / 11 questions. Question-group overlap: 0.
- Preservation check: 1374 source/model/MLflow/validation/environment-metadata files hashed before and after; 0 changed. The backup was not accessed.
- Existing Phase 1 test results have previously been inspected. The smoke runner does not read them. Future final test reporting must disclose reuse.

## Smoke metrics

| Configuration | Accuracy | Macro-F1 | Balanced accuracy | Good recall |
|---|---:|---:|---:|---:|
| B0 | 0.516026 | 0.352171 | 0.355602 | 0.180000 |
| W1 | 0.608974 | 0.304742 | 0.303394 | 0.040000 |
| W2 | 0.615385 | 0.294378 | 0.292079 | 0.040000 |
| W3 | 0.557692 | 0.294346 | 0.292149 | 0.060000 |
| O1 | 0.592949 | 0.393027 | 0.378031 | 0.280000 |
| F1 | 0.458333 | 0.381908 | 0.460847 | 0.200000 |

Do not rank or promote configurations using this one-fold, low-tree smoke result. Full per-class metrics and confusion matrices are in each configuration metrics.json.

## Sampling audit

Training class counts before sampling: `{'0': 47, '1': 190, '2': 227, '3': 785}`.
- O1 expands only the 1,249 training rows to 3,140 rows, 785 per class. Sample indices are mapped back to original fold-training positions and checked against validation positions.
- F1 checks each tree sampler against original fold-training positions. Each tree samples 47 rows per class in this smoke fold. Validation remains 312 original rows for every configuration.
- Feature-fit positions, sampled-source positions, validation positions and all per-tree class counts are saved in ignored metrics files; no student-answer text is written.

## Offline W&B locations

Shared group: `mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2`.

All locations below are relative to the Phase 2 worktree. Each run stages only explicitly allowlisted JSON configuration, metrics, fold assignments, predictions without text, warnings and provenance. System statistics, metadata collection, Git capture and code capture are disabled. No login, sync, upload or registry API is called.

- B0: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/B0/wandb/wandb/offline-run-20261109_145102-htm10ft9`
- W1: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/W1/wandb/wandb/offline-run-20261109_145108-5hlr6sda`
- W2: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/W2/wandb/wandb/offline-run-20261109_145108-2yk9768v`
- W3: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/W3/wandb/wandb/offline-run-20261109_145109-0m6tv6jn`
- O1: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/O1/wandb/wandb/offline-run-20261109_145110-a2vwvfo7`
- F1: `artifacts/phase2/mohler-rf-imbalance-v1-smoke-20261109T092102Z-895e8ec2/F1/wandb/wandb/offline-run-20261109_145111-1ekozgd1`

Credentials are never read into configuration or logged. A later online implementation may use WANDB_API_KEY or the supported login mechanism; it requires separate review. The current adapter rejects online mode. All SDK cache/config/data directories stay under ignored experiment outputs.

## Commands actually used

```powershell
python -m venv .venv-experiments
.\.venv-experiments\Scripts\python.exe -m pip install --no-cache-dir -r requirements-phase2.txt
.\.venv-experiments\Scripts\python.exe -B src/models/check_phase2_environment.py
.\.venv-experiments\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider --basetemp=artifacts/phase2/pytest-temp --junitxml=artifacts/phase2/pytest-results.xml tests/test_phase2_experiments.py tests/test_wandb_tracking.py
.\.venv-experiments\Scripts\python.exe -B src/models/run_imbalance_experiments.py --profile smoke
```

The checker writes the resolved lock to `artifacts/phase2/requirements-phase2.lock.txt` and compatibility inventory to `artifacts/phase2/environment.json`. All generated test/run outputs and environments are ignored. Documentation and dependency/configuration source files are reviewable repository changes.

The future full command is the same runner with `--profile full`. It was NOT executed. Validation confirmation, test evaluation, candidate bundle persistence, synchronization and registration are not implemented as automatic follow-on actions.

## Errors, warnings and limitations

- No installation, compatibility, test or training errors occurred; no Python estimator warnings were captured.
- W&B emitted a cached-environment advisory between sequential runs: changes to W&B environment variables may be ignored after the session starts. Explicit per-run offline mode and run directories were honored. SDK cache/config/data paths remain within the first configuration output directory, itself ignored. Future refinement can share stable group-level SDK directories or use separate worker processes.
- W&B prints suggested sync commands automatically. None were executed.
- The standalone tests disable the repository default coverage addopts on the command line because pytest-cov is not installed in this minimal environment. Existing tests and repository settings were not edited.
- Test folds contain few Poor observations; one-fold results are noisy. Oversampling duplicates human-labeled rows and adds no new evidence. Balanced RF trades majority information for minority representation.
- Question text is unused; domain generalization remains unestablished. Resampling changes learned class priors, so probabilities are not established as calibrated.
- The experiment runner intentionally accepts existing cleaned training Parquet. Inference export will need the external cleaning step (candidate br-tag removal; whitespace strip/collapse; no Unicode normalization) with the fitted transformations and classifier.
- No candidate model or inference bundle was created, and no Phase 1 binary was overwritten.
