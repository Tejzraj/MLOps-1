# Great Expectations validation for processed Mohler splits

Implemented and run only in the Phase 2 worktree on phase-2-model-api. Existing datasets were read-only; no scores, labels, rows, DVC metadata, models, W&B records, FastAPI files or existing environments were changed. This is data validation, not another model evaluation.

## Environment and commands

Great Expectations **1.24.1**, Python 3.11.9, pandas 3.0.6, NumPy 2.4.6, PyArrow 25.0.1, PyYAML 6.0.3, Pytest 8.2.2. A separate .venv-gx environment was created; pip check passed. scikit-learn 1.9.1 is included only to reuse the existing grouped-split helper/tests, not to train or load a model.

```powershell
# Run from your Phase 2 checkout.
python -m venv .venv-gx
.\.venv-gx\Scripts\python.exe -m pip install --no-cache-dir -r requirements-data-validation.txt
.\.venv-gx\Scripts\python.exe -m pip check
.\.venv-gx\Scripts\python.exe -B src/data/validate_gx.py
```

The runner reads phase2_gx from params.yaml, checks every dataset SHA-256, creates a unique ignored report directory, runs the relevant tests first, then validates each split independently. Tests failing or any core/group validation failing produces a nonzero exit.

GX uses an ephemeral in-memory context with analytics disabled, in-memory stores, progress bars disabled and no Data Docs site. No GX Cloud actions or W&B integration run. Partial unexpected values, raw answer rows, query strings and exception text are excluded from saved machine results.

## Observed processed schema

The schema was inspected from the actual Parquet files before expectations were defined. It was not inferred from the raw-data validator. All three splits have the following columns/types in the pinned pandas environment:

| Column | Observed pandas dtype | Contract |
|---|---|---|
| question_id | str | non-null, nonblank string; repeating group key |
| question | str | non-null, nonblank string |
| reference_answer | str | non-null, nonblank string |
| candidate_answer | str | non-null, nonblank string |
| score_grader1 | float64 | if present: float64, non-null, inclusive [0,5] |
| score_grader2 | float64 | if present: float64, non-null, inclusive [0,5] |
| score_avg | float32 | if present: float32, non-null, inclusive [0,5] |
| quality_label | int64 | non-null integer in configured class IDs |

Encoded mapping: 0=Poor, 1=Needs Improvement, 2=Good, 3=Excellent. Display strings such as Good are not accepted as stored labels. Thresholds/labels were not generated or changed. All score minima/maxima observed are 0.0/5.0; every score-range expectation validates all rows, not only extrema.

GX 1.24.1 does not recognize pandas 3 str ExtensionDtype in its aggregate type-list check. The implementation uses a deep validation-only copy with object-backed text storage so GX performs strict per-value str checks. It does not convert values to strings: an integer in a text field still fails. Original values/dtypes remain unchanged and are reported in profiles. Numeric and label types are never cast. The invalid-type and no-mutation tests cover this adaptation.

## Expectations

| Core check | Count per observed split |
|---|---:|
| Nonempty table and required-column presence | 2 |
| Types: five required columns plus three existing scores | 8 |
| Null checks: five required columns plus three existing scores | 8 |
| Nonblank text, including group ID/question | 4 |
| Encoded quality-label set | 1 |
| Existing numeric score ranges | 3 |
| **Core total** | **26** |

Required columns are question_id, question, reference_answer, candidate_answer, quality_label. Score checks are added only when each score column exists; this conditional path is tested. Missing required columns fail table-column presence without causing metrics on nonexistent columns.

If native answer_id, row_id or id columns exist, additional non-null and unique-value expectations are added. None exists in these files, so native row-ID uniqueness is explicitly inapplicable. No generated locator is misrepresented as source identity. question_id repeats by design, and split-local RangeIndex values are not global answer IDs.

Each split also runs one strict ExpectCompoundColumnsToBeUnique diagnostic over its actual full row. This is separate from the core gate, and its failures are disclosed as warning findings. No existing test/expectation was weakened and no rows were deduplicated to obtain a pass.

## Actual results

| Split | Rows | Core pass / evaluated | Core fail | Diagnostic pass / evaluated | Diagnostic fail | Repeats beyond first | Members of duplicate groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1561 | 26/26 | 0 | 0/1 | 1 | 45 | 63 |
| validation | 229 | 26/26 | 0 | 0/1 | 1 | 8 | 15 |
| test | 483 | 26/26 | 0 | 0/1 | 1 | 15 | 27 |

Total: **78/78 core expectations pass**; **3/3 duplicate diagnostics fail**. That is **81 GX expectations evaluated, 78 passing, 3 failing**. Failures are diagnostic warnings; this is not an all-expectations-pass report.

The duplicate counts use two definitions: pandas duplicated() counts repeats beyond the first row; GX compound uniqueness flags all members of duplicate groups. Hence 45/8/15 repeats correspond to 63/15/27 flagged members. These may be distinct students giving identical answers and scores, or duplicated source observations; the processed files lack answer IDs to decide. Do not silently delete them. Review original source answer identifiers in a separately approved data-versioning task before any remediation.

No nulls, blanks, unknown labels, score-range violations or invalid types were found. No GX execution errors or runtime warnings occurred in the complete validation.

| Split | Poor | Needs Improvement | Good | Excellent |
|---|---:|---:|---:|---:|
| train | 62 | 233 | 277 | 989 |
| validation | 8 | 38 | 33 | 150 |
| test | 10 | 99 | 86 | 288 |

## Group leakage remains a separate check

Existing src/data/preprocess.py assert_disjoint_splits is reused unchanged. New tests/test_gx_group_leakage.py checks the three real pairwise group intersections, the existing assertion/summary, and deliberately contaminated in-memory splits. These are not GX expectations and do not inflate GX expectation counts.

Pairwise overlap counts: `{'train__validation': 0, 'train__test': 0, 'validation__test': 0}`. Overall group audit passed.
Pytest: **43 passed**, zero failures/errors/skips: 25 focused GX tests, 5 separate leakage tests, and 13 unchanged tests from tests/test_split.py. Intentionally invalid test copies cover missing columns, nulls, blanks, non-string text, wrong label representation/IDs, scores outside bounds, optional missing scores, duplicate native IDs and full-row duplicates. These are test fixtures only; no invalid values were added to datasets.

## Outputs and preserved work

Reports: `artifacts/phase2/gx-validation-20261110T052758Z-091e1aa9`.
- Per split: profile JSON, core suite JSON/results JSON, duplicate suite JSON/results JSON.
- Overall: summary.json, summary.md, validation_review.json, JUnit XML, pytest log and complete dependency lock. The readable summary reports the separate group check.
- Preservation records: `artifacts/phase2/gx-preservation-ff81a38d633e`. 1759 protected files hashed before/after; zero changed. This includes the source Parquet files, Phase 1/DVC/model/W&B artifacts, FastAPI implementation and package metadata in .venv-validation, .venv-experiments and .venv-api.
- Only the newly created .venv-gx was installed/changed. All generated reports/caches are ignored below artifacts/phase2, with GX stores in memory. No raw answer table, credentials or failure examples were committed or uploaded.

Files created: requirements-data-validation.txt, src/data/validate_gx.py, tests/test_gx_validation.py, tests/test_gx_group_leakage.py and this document. Existing files modified: .gitignore and params.yaml only. No DVC stage was changed, so no dvc repro/training was run.

## References

- GX dataframe/ValidationDefinition API: https://docs.greatexpectations.io/docs/core/connect_to_data/dataframes/
- Validation result formatting: https://docs.greatexpectations.io/docs/core/run_validations/run_a_validation_definition/
- Pinned GX distribution metadata: https://pypi.org/project/great-expectations/1.24.1/
