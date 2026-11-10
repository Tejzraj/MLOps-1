# Single validation confirmation protocol

The rule is fixed before fitting: confirm F1 only when its validation macro-F1 is strictly greater than B0; otherwise retain B0, including exact ties. No thresholds or recipes will be changed after viewing validation results.

The runner reads only the existing training and validation partitions, verifies their hashes and question-group separation, and fits a separate fresh feature pipeline and classifier per configuration. It checks the configuration hashes, dependencies and feature implementation against the frozen full CV run. The test partition is never opened.

Run preflight tests before fitting. `tests/test_confirmation_artifacts.py` is intentionally a separate post-training suite: it requires the freshly generated trusted pairs and checks checksums before deserialization. It reads only training rows for compatibility checks and never refits models.

All generated models, paired pipelines, manifests, feature ordering, metrics and offline W&B records live in a unique ignored confirmation directory. Model binaries and feature vocabulary are not staged as W&B artifacts. Existing Phase 1 artifacts, CV reports and validation reports remain untouched.

Commands and actual results will be recorded after the single confirmation run. No synchronization, registration, deployment, final test evaluation, commit or push is part of this step.


# Single validation confirmation: B0 versus F1

**Final decision: retain B0.** F1 confirmation: False.

The predeclared rule was `confirm_candidate_only_if_validation_macro_f1_strictly_exceeds_baseline`. F1 minus B0 validation macro-F1 is **-0.064975**. This conflicts with the CV ranking; the rule was not changed and no tuning followed.

## Main scores

| Configuration | CV macro-F1 mean ? SD | Validation macro-F1 | Validation accuracy | Validation balanced accuracy |
|---|---:|---:|---:|---:|
| B0 | 0.369026 ? 0.043366 | 0.377762 | 0.576419 | 0.388196 |
| F1 | 0.399188 ? 0.027305 | 0.312787 | 0.497817 | 0.328722 |

## Validation per-class metrics

| Configuration | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| B0 | Poor | 0.200000 | 0.125000 | 0.153846 | 8 |
| B0 | Needs Improvement | 0.344262 | 0.552632 | 0.424242 | 38 |
| B0 | Good | 0.272727 | 0.181818 | 0.218182 | 33 |
| B0 | Excellent | 0.737589 | 0.693333 | 0.714777 | 150 |
| F1 | Poor | 0.045455 | 0.125000 | 0.066667 | 8 |
| F1 | Needs Improvement | 0.245902 | 0.394737 | 0.303030 | 38 |
| F1 | Good | 0.171429 | 0.181818 | 0.176471 | 33 |
| F1 | Excellent | 0.828829 | 0.613333 | 0.704981 | 150 |

## Recall comparison

| Class | B0 CV mean recall | F1 CV mean recall | B0 validation recall | F1 validation recall |
|---|---:|---:|---:|---:|
| Poor | 0.289841 | 0.466984 | 0.125000 | 0.125000 |
| Good | 0.133031 | 0.222536 | 0.181818 | 0.181818 |

F1 improved both recalls in CV relative to B0, but neither improved relative to B0 on validation. Each model correctly classified one of eight Poor answers and six of 33 Good answers. The validation partition contains only eight question groups; the result does not establish general superiority across domains.

## Confusion matrices

Rows are human labels, columns predictions. Order: Poor, Needs Improvement, Good, Excellent.

B0

```text
   1    0    3    4
   1   21    2   14
   1    7    6   19
   2   33   11  104
```

F1

```text
   1    5    2    0
  10   15    7    6
   4   10    6   13
   7   31   20   92
```

## Saved pairs and manifests

Each directory below contains `model.joblib`, its own `feature_pipeline.joblib`, `feature_names.json`, and `manifest.json`. Both pairs were fitted fresh on 1,561 training rows and transformed 229 untouched validation rows. No test partition was accessed. Both produce 2,824 features (2,818 TF-IDF + five length features + one cosine similarity), and classes match params.yaml.

- B0: `artifacts/phase2/mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8/B0`; fit time 1.587s.
  Offline W&B: `artifacts/phase2/mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8/B0/wandb/wandb/offline-run-20261109_151303-5t2sq8pe`.
- F1: `artifacts/phase2/mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8/F1`; fit time 1.384s.
  Offline W&B: `artifacts/phase2/mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8/F1/wandb/wandb/offline-run-20261109_151310-5mc35ss9`.

Each manifest records dataset SHA-256 identities, resolved frozen and estimator parameters, feature counts and ordering-file identity, artifact checksums, dependency versions, source Git commit and dirty state plus code hashes, compatibility checks, preprocessing, and limitations. Binaries and vocabulary files are local only and were not staged as W&B artifacts. Only files freshly created in this run were deserialized, after local generation; post-save tests verified their hashes first.

## Artifact SHA-256

| Configuration | File | SHA-256 |
|---|---|---|
| B0 | model.joblib | `1da106f72ac27cae9c22eba2274fbe44b84276c69cbc659c0cd8c26ccd9d5e23` |
| B0 | feature_pipeline.joblib | `1d13844d2714b514550b2f7f20138d02c292abb0b064b4e08b43724420237e1e` |
| B0 | feature_names.json | `7f49cfb7ba1074e08d9867f7961523b17fd387aa77c8c7a676d4b8ff4b6c6fae` |
| F1 | model.joblib | `620320e7776c32197570db1c02b7ab959458993a6676ab63ff8ddfbd159a1133` |
| F1 | feature_pipeline.joblib | `20b60aaf28d91723f1afb2e7f2b51640ff7e4a91e56bcd603115f0521432971b` |
| F1 | feature_names.json | `7f49cfb7ba1074e08d9867f7961523b17fd387aa77c8c7a676d4b8ff4b6c6fae` |

## Tests and preservation

- Before fitting: 20 tests passed, zero failures/errors.
- After saving: 4 artifact tests passed, zero failures/errors. Actual pairs passed class mapping, feature/model dimensions, ordered features, repeat and single/batch prediction checks, and serialization round trips.
- Preserved files: 1577 checked; zero changes to protected Phase 1 artifacts/source, DVC metadata, MLflow records, previous validation reports, smoke reports, and full CV outputs.
- Independent fresh feature pipelines used only training rows. Balanced RF internal sample indices were checked against training positions; no validation observations were sampled.
- The runner refuses a second confirmation attempt once protocol.json exists, including an incomplete attempt.

## Commands used

```powershell
.\.venv-experiments\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider tests/test_phase2_experiments.py tests/test_wandb_tracking.py tests/test_phase2_confirmation.py
.\.venv-experiments\Scripts\python.exe -B src/models/confirm_validation.py --confirm
.\.venv-experiments\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider tests/test_confirmation_artifacts.py
```

The actual test commands also set --basetemp and --junitxml under the ignored preflight/confirmation directories, keeping generated outputs out of tracked paths. Preflight command and logs are recorded in the preflight directory; execution.json records the confirmation command and duration.

## Warnings and limitations

No training, serialization, compatibility or test errors occurred; Python warnings captured: 0. W&B SDK advisory occurrences: 1.
- W&B warned that environment changes may be ignored after its session starts. Explicit offline mode and run directories were honored; SDK storage remained under ignored Phase 2 outputs. No synchronization was performed.
- CV and this one validation partition disagree. Retaining B0 follows the predeclared conservative rule; this is not proof that F1 is universally worse. Do not rerun seed searches or adjust recipes using these validation scores.
- Historical test metrics have already been seen, but this step did not open or evaluate the test partition.
- The saved B0 pair is a new Phase 2 candidate with documented lineage, not the submitted Phase 1 binary. No existing baseline model was replaced.
- Preprocessing is external to the fitted feature pipeline: candidate br-tag removal and whitespace strip/collapse must be included when packaging raw-input inference. Question text is unused. Probabilities are not established as calibrated.
- Compatible custom transformer import paths and dependencies are required. Joblib loading can execute Python: validate provenance/checksums and load only trusted artifacts.
- No test evaluation, synchronization, registration, deployment, commit or push followed this confirmation.
