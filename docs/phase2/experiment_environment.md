# Phase 2 experiment environment

Python: **3.11.9**. Separate `.venv-experiments`; `.venv-validation` remains unchanged.

Dependency resolution preserved validated package pins through `constraints-phase2-validation.txt`. Direct experiment requirements are in `requirements-phase2.txt`. Compatible pinned Pytest is included. No Phase 1 requirements file was changed.

The environment checker imports imbalanced-learn and W&B, verifies direct package versions, runs pip check and writes all resolved versions into ignored output. Do not install requirements-dev.txt into this environment because it pulls the unrelated Phase 1 dependency set.

## Exact installed versions

| Distribution | Version |
|---|---|
| annotated-types | 0.8.0 |
| anyio | 4.16.0 |
| certifi | 2026.10.23 |
| charset-normalizer | 3.5.2 |
| click | 8.5.1 |
| cloudpickle | 3.1.2 |
| colorama | 0.4.6 |
| cwevals | 0.0.1b1 |
| distro | 1.9.0 |
| googleapis-common-protos | 1.76.0 |
| h11 | 0.16.0 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| idna | 3.20 |
| imbalanced-learn | 0.14.2 |
| iniconfig | 2.3.1 |
| joblib | 1.6.0 |
| narwhals | 2.28.0 |
| numpy | 2.4.6 |
| opentelemetry-api | 1.45.2 |
| opentelemetry-exporter-http-transport | 0.66b2 |
| opentelemetry-exporter-otlp-common | 0.66b2 |
| opentelemetry-exporter-otlp-proto-common | 1.45.2 |
| opentelemetry-exporter-otlp-proto-http | 1.45.2 |
| opentelemetry-proto | 1.45.2 |
| opentelemetry-sdk | 1.45.2 |
| opentelemetry-semantic-conventions | 0.66b2 |
| packaging | 26.3 |
| pandas | 3.0.6 |
| pip | 24.0 |
| platformdirs | 4.14.5 |
| pluggy | 1.7.0 |
| prettytable | 3.18.0 |
| protobuf | 7.36.2 |
| pyarrow | 25.0.1 |
| pydantic | 2.14.3 |
| pydantic_core | 2.50.4 |
| pytest | 8.2.2 |
| python-dateutil | 4.0.0 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| setuptools | 65.5.0 |
| sklearn-compat | 0.1.6 |
| skops | 0.16.0 |
| sniffio | 1.3.1 |
| threadpoolctl | 3.8.0 |
| typing-inspection | 0.4.5 |
| typing_extensions | 4.16.0 |
| tzdata | 2026.6 |
| urllib3 | 2.8.0 |
| wandb | 0.31.0 |
| wcwidth | 0.9.4 |
| xxhash | 4.0.2 |

Installation and import checks passed, and the offline six-configuration smoke additionally exercised sparse sampling, RandomForest, BalancedRandomForest and W&B artifact staging. No cloud synchronization was performed.

Recreate using the same Python and the saved resolved lock when available, or resolve direct requirements under the checked-in constraints and compare the resulting inventory. Runtime/OS differences can still affect serialization and timing.
