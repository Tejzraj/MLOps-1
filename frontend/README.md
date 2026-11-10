# Interview Studio: Fedora integration runbook

Run from the repository root. Use Python 3.11 (`python3.11 --version`), matching
the private bundle's runtime family. The backend and frontend use separate
virtual environments. Create fresh environments; an existing Fedora Python 3.14
frontend environment must not be reused for these Python 3.11 commands. Install Python 3.11 through your Fedora package sources
if it is not available; do not silently substitute Fedora's newer default Python.

## Private bundle prerequisite

Read [the owner handoff](../docs/phase2/teammate_handoff.md). W&B retrieval is
blocked by account access; do not run retrieval or use someone else's credentials.
Receive the private ZIP through the approved private channel, under the ignored
`artifacts/phase2/` directory. Never commit/upload it or extracted files.

From the repository root, verify the ZIP **before extraction**:

```bash
sha256sum artifacts/phase2/b0-inference-v1.0.0-private-handoff.zip
```

The required digest is
`d1c6fa30fb65429f94f75b3f8231eda1424dcce37605ba77b0b0c4751c53840f`.
Stop if it differs. Extract into a fresh directory only:

```bash
test ! -e artifacts/phase2/private-handoff && \
  unzip artifacts/phase2/b0-inference-v1.0.0-private-handoff.zip -d artifacts/phase2/private-handoff
```

Do not continue if this fails. Missing files are a setup blocker, not a successful
prediction. There is no fallback to another model or to simulated grading.

## Backend: first terminal

```bash
python3.11 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements-api-test.txt
export INTERVIEW_MODEL_BUNDLE="$PWD/artifacts/phase2/private-handoff/bundle"
test -f "$INTERVIEW_MODEL_BUNDLE/bundle_manifest.json"
.venv-api/bin/python -B -c 'import os; from pathlib import Path; from src.models.retrieve_model import verify_bundle; from src.utils.phase2_common import load_params; verify_bundle(Path(os.environ["INTERVIEW_MODEL_BUNDLE"]), load_params()["phase2_retrieval"])'
# Continue only after verification exits successfully.
.venv-api/bin/python -B src/api/app.py
```

This uses root `params.yaml` and binds to `127.0.0.1:8000`. It verifies/loads the
frozen bundle once at startup. It does not train or contact W&B.

## Frontend: second terminal

```bash
python3.11 -m venv frontend/.venv-ui311
frontend/.venv-ui311/bin/python -m pip install -r frontend/requirements-dev.txt
export FRONTEND_MODE=api
export BACKEND_URL=http://127.0.0.1:8000
frontend/.venv-ui311/bin/python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501, select Practice, write an answer and evaluate. A real
result shows the category and model version. API failures clear any old result
and display an error. The explicitly selected **Mock · simulated** mode is still
available; its banner and result identify user-selected presentation fixtures.
Mock categories do not grade the submitted text or represent human dataset labels.

## Contract and manual checks

Verified against `PredictionRequest` and `PredictionResponse` in
[src/api/app.py](../src/api/app.py): POST `/predict` accepts **only** `question`,
`reference_answer`, `candidate_answer` as nonblank strings. Do not send
`question_id`. Successful responses contain `predicted_quality` and a nonempty
string `model_version`. Categories are Poor, Needs Improvement, Good, Excellent.
No numeric score, probability or detailed feedback is returned.

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/model/info
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"question":"What is a stack?","reference_answer":"A stack is a last in first out data structure.","candidate_answer":"A stack uses last in first out order."}'
```

Health must report `model_ready: true` before claiming inference works. A 200
health response alone is liveness. Missing/invalid bundles report readiness false
and `/predict` returns 503; invalid request fields return 422. Check backend logs
and the verified bundle path for 503. The frontend uses server-side HTTP, so its
URL must be reachable from Streamlit; browser CORS configuration is unnecessary.
Question text is accepted but is not predictive; generalization beyond Mohler
short answers is unverified and minority-class recall is weak.

## Integration tests

Use a separate combined environment for UI/API integration (Streamlit brings
PyArrow; keep it out of the minimal production API environment):

```bash
python3.11 -m venv frontend/.venv-integration
frontend/.venv-integration/bin/python -m pip install -r frontend/requirements-integration.txt
export INTERVIEW_MODEL_BUNDLE="$PWD/artifacts/phase2/private-handoff/bundle"
frontend/.venv-integration/bin/python -m pytest -c frontend/pytest.ini frontend/tests
```

The integration suite routes the frontend's HTTP boundary through the actual
FastAPI lifecycle and schema. It checks missing-bundle 503 handling, strict 422
validation, and real inference if the private bundle is supplied. Missing bundles
produce an explicit skip for the inference test. No fake predictor supplies a
success. Unit/UI tests use labelled transport fixtures to test presentation.

Run the owner's real-bundle tests separately in the minimal API environment,
after verifying the bundle (they require the private artifacts):

```bash
.venv-api/bin/python -m pytest tests/test_api.py tests/test_model_retrieval.py -q -o addopts=
```

Frontend configuration is in `frontend/params.yaml`. Environment overrides:
`FRONTEND_PARAMS`, `FRONTEND_MODE`, `BACKEND_URL`, `BACKEND_TIMEOUT_SECONDS`.
No ML pipeline stage is changed by this integration.

## Docker Compose on Fedora with SELinux

Requires Docker Engine with the Compose plugin already installed. From the repo
root, set a **relative** verified bundle path beneath `artifacts/phase2`:

```bash
export COMPOSE_MODEL_BUNDLE=artifacts/phase2/private-handoff/bundle
test -f "$COMPOSE_MODEL_BUNDLE/bundle_manifest.json"
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose config --quiet
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose build
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose up -d --wait --wait-timeout 120
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose ps
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose logs --tail=100 backend frontend
```

The frontend defaults to API mode and reaches `http://backend:8000` over Compose
DNS. The backend launches `python src/api/container.py`, uses the configured port
and binds to the container interface. Host ports bind only to loopback. Bundle
contents are mounted read-only, never copied into either image. The bind mount
uses Fedora SELinux `Z` relabelling on that private directory only; no SELinux or
firewall disabling is needed. The container's UID 10001 must have read/traverse
access to the bundle; preserve private permissions, and do not make it world-readable.
A missing source directory is not auto-created. Missing/corrupt bundles make the
backend unhealthy and prevent dependent frontend startup. Inspect readiness with
curl as above. A successful build alone does not verify model loading.

```bash
sudo env COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" docker compose down
```

For a frontend-only mock demonstration without any model or backend:

```bash
sudo docker build -f frontend/Dockerfile -t interview-frontend:local .
sudo docker run --rm -p 127.0.0.1:8501:8501 -e FRONTEND_MODE=mock interview-frontend:local
```

The allowlisted build context excludes artifacts, datasets, ZIPs, credentials,
W&B caches and environments. CI builds both images without private files and
runs integration tests; the real-bundle test is expected to skip there. CI
configuration does not establish a successful remote run before a reviewed push.

## Checks performed in this checkout

The privately supplied handoff ZIP matched the documented SHA-256. It was
extracted into a fresh ignored directory without overwriting files, and
`verify_bundle` verified the pinned manifest and every bundle file before any
Joblib loading. Python 3.11.17 in `.venv-api` matches all pinned API/inference
package versions; `pip check` reports no broken requirements.

With `INTERVIEW_MODEL_BUNDLE` pointing to that verified directory, this command
completed with **42 passed, none skipped**, including real model inference:

```bash
.venv-api/bin/python -B -m pytest -c frontend/pytest.ini \
  frontend/tests/test_api_integration.py tests/test_api.py tests/test_model_retrieval.py
```

The running localhost API returned HTTP 200 for `/health` (model ready),
`/model/info` (version `1.0.0`) and `/predict`. The handoff's example request
returned `predicted_quality: Excellent` and `model_version: 1.0.0`. A separate
headless Streamlit AppTest submitted the displayed reference answer through real
HTTP to the running backend and displayed `Excellent`, version `1.0.0`, with
`simulated=False`; no HTTP transport was mocked. These are illustrative
predictions, not model performance metrics. The test suite emitted a Starlette
HTTPX deprecation warning; tests still passed.

Project syntax, YAML parsing and `git diff --check` passed during integration.
Docker is not installed and Podman has no Compose provider, so image builds,
Compose validation, SELinux mount permissions and container startup remain
unverified locally. CI was updated but has not been triggered. No DVC stage,
training or evaluation logic was changed; `dvc repro` was not run. No commit,
push, upload or deployment was made.

Model quality is separate from integration success. See the
[model-risk report and opt-in diagnostic command](../docs/phase2/model_quality_risk.md)
for reproduced semantic failures, exact inputs and exit-code interpretation.
