# Final reused-test evaluation and B0 inference bundle

B0 remains selected under the predeclared validation rule. Final test metrics will be reported as a reused, previously inspected benchmark, not as an independent unseen test, and will not change selection.

Only the newly trained validation B0 model/pipeline pair may be used. Candidate and selection manifests plus model/pipeline checksums are verified before deserialization. No training or parameter tuning is included.

The raw-input wrapper accepts question, reference_answer and candidate_answer strings. It mirrors candidate br-tag removal and whitespace strip/collapse, preserves case, and does not normalize Unicode. Empty cleaned candidates are rejected because dataset preparation excluded them. Question text is not a predictive feature.

The portable bundle will include the unchanged paired artifact bytes, class mapping, preprocessing configuration, pinned inference requirements, compatibility source modules, provenance manifest, and model card. Generated reports, test outputs, binaries and offline tracking files remain in unique ignored Phase 2 directories. Prior validation and F1 artifacts remain unchanged.

Actual commands, test outcomes, benchmark metrics and limitations will be appended after the single final evaluation and bundle verification.


# Final B0 evaluation and verified inference bundle

**REUSED TEST BENCHMARK: previously inspected; not an independent unseen test.**

B0 remains selected by the validation-confirmation rule. These test results are descriptive and did not change selection. The exact newly trained validation B0 bytes were copied and verified against its saved manifest before deserialization; there was no retraining or tuning.

## Benchmark results

Rows: 483; question groups: 17; test prediction passes: 1.

| Metric | Result |
|---|---:|
| Accuracy | 0.590062 |
| Macro-F1 | 0.392525 |
| Balanced accuracy | 0.388752 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Poor | 0.375000 | 0.300000 | 0.333333 | 10 |
| Needs Improvement | 0.423077 | 0.333333 | 0.372881 | 99 |
| Good | 0.148936 | 0.081395 | 0.105263 | 86 |
| Excellent | 0.691429 | 0.840278 | 0.758621 | 288 |

Confusion matrix: rows human labels, columns predictions; order Poor, Needs Improvement, Good, Excellent.

```text
   3    3    2    2
   2   33   16   48
   1   20    7   58
   2   22   22  242
```

## Descriptive comparison

| Stage | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Grouped CV, fold means | 0.567047 | 0.369026 | 0.367312 |
| Validation | 0.576419 | 0.377762 | 0.388196 |
| Reused test | 0.590062 | 0.392525 | 0.388752 |

The stages contain different question groups and class distributions. Higher test macro-F1 is not evidence of improvement from this step: no model changed, and this benchmark was already inspected. Good recall remains low and no further tuning was performed.

## Bundle and provenance

Bundle v1.0.0: `artifacts/phase2/mohler-b0-inference-v1-20261109T095935Z-d812a332/bundle`.

- Contains unchanged B0 model.joblib, its paired feature_pipeline.joblib, candidate manifest, full ordered features, params.yaml with class mapping and preprocessing, pinned runtime requirements, compatibility source, README, model card and bundle integrity manifest.
- The custom parameterized reference transformer is extracted with its exact class body at its canonical import path. Experiment functions and tracking imports are excluded from that compatibility module. The unchanged legacy feature class source is included, with no training entrypoint invoked.
- Inference requires only the pinned runtime packages. Isolated-process tests verified that bundled source is used without importing imbalanced-learn or W&B. A fresh minimal dependency installation and other Python/OS combinations were not tested.
- Input is one object or a nonempty list of objects with exactly question, reference_answer and candidate_answer string fields. Empty cleaned candidates are rejected because training excluded them.
- Candidate br tags are removed case-sensitively, then fields are stripped and repeated whitespace collapsed. Case is preserved; Unicode normalization is not performed. Only candidate/reference columns enter the fitted transform. **Question text is not predictive.**
- Outputs contain integer class_id and its mapped quality_label. No exact numeric human score is predicted.
- Source commit, dirty source identity, original candidate lineage, dependency versions and artifact checksums are recorded. Checksums establish integrity rather than authenticity; joblib must only load trusted artifacts.

Model SHA-256: `1da106f72ac27cae9c22eba2274fbe44b84276c69cbc659c0cd8c26ccd9d5e23`.
Pipeline SHA-256: `1d13844d2714b514550b2f7f20138d02c292abb0b064b4e08b43724420237e1e`.
Bundle manifest SHA-256: `036f6ac95e47d096d55839f4afe76751af7b7164e8b02b39f1acb8a8a2f85c99`.

## Verification and preservation

- Preflight: 33 passed, zero failures/errors.
- Post-bundle: 4 passed, zero failures/errors. Covers unchanged bytes, class/feature compatibility, deterministic repeat/single/batch predictions, question invariance and isolated portable inference. Only training examples were used for these checks.
- Protected files hashed before/after: 1619; changed: 0. Phase 1 files, frozen original models, DVC metadata, MLflow records, validation B0/F1 pairs and prior reports remain unchanged.
- All generated outputs live in ignored unique directories. No credentials or raw student-answer text are logged or included in source files. No model binaries were staged to W&B.

## Offline tracking

Run: `artifacts/phase2/mohler-b0-inference-v1-20261109T095935Z-d812a332/wandb/wandb/offline-run-20261109_152936-zi48eo8t`.
- Final evaluation links the selected B0 candidate, validation decision and full CV group. Only reviewed evaluation/config/provenance/warning JSON is staged locally. No synchronization, registration or deployment occurred.

## Commands

```powershell
.\.venv-experiments\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider tests/test_phase2_experiments.py tests/test_wandb_tracking.py tests/test_phase2_confirmation.py tests/test_confirmation_artifacts.py tests/test_phase2_inference.py
.\.venv-experiments\Scripts\python.exe -B src/models/finalize_model.py --evaluate-once
.\.venv-experiments\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider tests/test_inference_bundle.py
```

Test commands additionally specified --basetemp and --junitxml inside ignored outputs. Exact preflight command and execution logs are preserved. The finalizer refuses another evaluation attempt once its protocol exists.

Example inference after reviewing/trusting the bundle:

```powershell
python -B <bundle>/src/inference/predict.py --bundle <bundle> --manifest bundle_manifest.json --input <request.json>
```

## Warnings and model limitations

Captured Python warnings: 0; SDK warning lines: 0. No runtime or test errors occurred.
- Narrow Mohler introductory data-structures domain; limited minority supports and weak Good recall. Question text is ignored, probabilities are not calibrated, and category predictions should not replace human grading.
- The test partition is a reused benchmark. External generalization and production performance remain unverified.
- This is a prepared inference bundle, not a deployed service. No FastAPI/Streamlit integration, container deployment or registry action was performed. No commit or push occurred.
