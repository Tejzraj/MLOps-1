# Local FastAPI inference backend

The API serves the frozen, validation-selected B0 bundle v1.0.0. No models were trained/refitted, no benchmark evaluation rerun, and no W&B synchronization or Registry publication occurred. The local verification server was stopped after the HTTP checks.

## Files and isolation

- Added `src/api/app.py`, `src/api/model_runtime.py`, `src/api/__init__.py`, `tests/test_api.py`, `requirements-api.txt`, `requirements-api-test.txt`, and this document.
- Updated only `.gitignore` and the new `phase2_api` section in `params.yaml` among existing configuration files. Existing Phase 1 pipeline/MLflow code, inference wrapper and frozen bundle remain unchanged.
- `.venv-api` is a separate ignored Python environment. Runtime requirements pin the frozen bundle dependencies plus FastAPI 0.162.2, Pydantic 2.14.3 and Uvicorn 0.54.0. Test requirements add Pytest 8.2.2 and httpx 0.28.1. `pip check` passed.
- W&B, imbalanced-learn, PyArrow and MLflow are not API dependencies. No exporter or W&B integration is configured; FastAPI resolves its own OpenTelemetry API dependency without installing a telemetry exporter.
- Model loading uses the existing unchanged BundlePredictor. Verified compatibility classes are loaded at canonical import paths, from frozen source bytes, without importing the experiment runner or writing bytecode into the bundle. A conflicting preloaded training module fails closed. Use a dedicated API process.

## Endpoint contract

| Endpoint | Success | Failure |
|---|---|---|
| GET /health | 200: application liveness plus model_ready | model_ready=false when startup load failed |
| GET /model/info | 200: version, supported classes, question-feature flag, public limitations | 503 when model unavailable |
| POST /predict | 200: predicted_quality and model_version only | 422 invalid input; 503 unavailable model; 500 unexpected inference failure |

Startup makes exactly one load attempt per application lifespan. A missing, corrupted, dependency-incompatible or wrong-identity bundle leaves the app alive but not model-ready. No request retries model loading. Fix the installation/configuration and restart. Server logs record error class only; public responses omit private paths and internal exceptions.

All three request fields must be strings and nonempty after whitespace inspection. Unknown fields are forbidden. The API rejects empty/whitespace-only question/reference fields as an HTTP contract, although question text is not predictive. Candidate br-only input is rejected through the frozen cleaner. Validation responses omit submitted input values and Pydantic context.

Exact model cleaning is performed by the existing wrapper: candidate-only case-sensitive br-tag removal, strip and whitespace collapse; preserve case; no Unicode normalization. Only candidate_answer/reference_answer are transformed. The classifier does not generate numeric interview scores, detailed feedback or calibrated confidence.

## Install and start locally

Use Python 3.11.9. The bundle is ignored and is not supplied by Git: your teammate needs an approved verified copy of the complete bundle directory. W&B retrieval is deferred. Do not substitute Phase 1 binaries.

```powershell
# Run from your Phase 2 checkout.
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv-api\Scripts\python.exe -m pip check
# Optional override for a relocated copy of the exact verified bundle:
$env:INTERVIEW_MODEL_BUNDLE = (Resolve-Path 'artifacts/phase2/private-handoff/bundle').Path
# Omit/remove the override to use the default configured in params.yaml.
.\.venv-api\Scripts\python.exe -B src/api/app.py
```

Default URL: `http://127.0.0.1:8000`; docs: `/docs`; OpenAPI: `/openapi.json`. Host, port, log level and one-worker launcher settings come from params.yaml. No reloader is enabled. INTERVIEW_MODEL_BUNDLE may be absolute or relative to the worktree; it cannot change the frozen manifest trust anchor/version/configuration. A relocated bundle must match the same checksums.

If the bundle is elsewhere, pip still needs the frozen inference requirement pins: either retain the default bundle layout for the -r include in requirements-api.txt or install the relocated bundle requirements plus the exact three HTTP pins shown above. This does not authorize modifying the frozen bundle.

ASGI alternative, using equivalent local settings:

```powershell
.\.venv-api\Scripts\python.exe -B -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Use the script launcher when host/port should come directly from params.yaml. Each ASGI worker has its own one-time model load. Keep concurrency modest: the frozen RF already has n_jobs=2. This is local verification, not a deployed/authenticated public service.

## Actual HTTP verification

The following illustrative request is hand-written API input, not a labeled training/test row. The returned category was produced by the frozen model during a real loopback HTTP call.

POST /predict request:

```json
{
  "question": "What is a stack?",
  "reference_answer": "A stack is a last in first out data structure.",
  "candidate_answer": "A stack is a last in first out data structure."
}
```

Actual 200 response:

```json
{
  "predicted_quality": "Excellent",
  "model_version": "1.0.0"
}
```

Actual health response:

```json
{
  "status": "ok",
  "model_ready": true
}
```

Actual whitespace-only candidate response (422):

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "candidate_answer"
      ],
      "msg": "Value error, must not be empty or whitespace-only",
      "type": "value_error"
    }
  ]
}
```

GET /model/info returns the four supported classes (Poor, Needs Improvement, Good, Excellent), model_version=1.0.0, question_used_as_feature=false and reviewed limitations. It does not expose artifact paths, hashes, local identities, Git metadata or dependency inventories. Missing reference_answer also returned 422 in the live check.

## Streamlit integration

Streamlit can make server-side HTTP calls; no CORS configuration is required for that pattern. Its environment is separate from the API environment. Example using requests in the Streamlit client environment:

```python
import requests

payload = {
    "question": "What is a stack?",
    "reference_answer": "A stack is a last in first out data structure.",
    "candidate_answer": "A stack is a last in first out data structure.",
}
response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
response.raise_for_status()
prediction = response.json()
# Display prediction["predicted_quality"] and prediction["model_version"].
# Handle 422 as invalid input and 503 as unavailable model, not as answer quality.
```

Check /health model_ready before accepting predictions. Handle timeouts and 500 errors as operational failures. Do not turn a label into an invented numeric score or detailed feedback. No browser-origin CORS policy or Streamlit UI was added in this task.

## Tests and results

```powershell
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api-test.txt
.\.venv-api\Scripts\python.exe -B -m pytest -o addopts="" -o filterwarnings=default -p no:cacheprovider tests/test_api.py
```

Actual test invocation also set --basetemp and --junitxml below the ignored API verification directory. Existing pytest defaults and tests were not weakened. Run API tests in .venv-api; ML experiment tests retain their separate .venv-experiments dependencies.

- 26 API tests passed; 0 failures; 0 errors.
- 5 real HTTP cases passed. Verification server stopped: True.
- Coverage includes actual model output/class mapping, public metadata, health, deterministic repeated responses, exact equivalent cleaning, question invariance, no fitting, exactly one startup load, missing/corrupt bundles and environment override.
- 1658 protected files were hashed before/after; zero changed. Frozen bundle inventory and every SHA-256 are unchanged, including B0/F1 validation artifacts and earlier W&B/experiment records.
- Full environment lock, JUnit XML, HTTP responses, server log and preservation results: `artifacts/phase2/api-verification-fa95776526df`.
- One test-only warning: Starlette 1.10.0 deprecates httpx-based TestClient in favor of httpx2. Pinned httpx 0.28.1 tests pass. No API inference warning was observed. Do not suppress the warning or change dependency pins silently; migrate the test client in a later reviewed dependency update.

## Limitations and remaining work

- B0 is narrow-domain, ignores question wording, has weak minority-class recall and uncalibrated probabilities. Historical benchmark results were previously inspected; this task did not read the dataset or rerun evaluation.
- No authentication, public hosting, Docker deployment, Registry publication, W&B sync, retraining, model changes, commit or push was performed.
- Other OS/container combinations, load testing, request-size limits and production worker/resource policies remain unverified. The local server binds only loopback by default.

Lifecycle design follows the official FastAPI lifespan pattern: https://fastapi.tiangolo.com/advanced/events/ . Tests use the lifespan-aware TestClient context manager: https://fastapi.tiangolo.com/advanced/testing-events/ .


Private Registry model setup for a fresh checkout: see [model_retrieval.md](model_retrieval.md). Set INTERVIEW_MODEL_BUNDLE to the verified download before starting the existing API.
