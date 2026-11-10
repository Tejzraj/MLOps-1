# Phase 2 teammate handoff: start here

B0 bundle v1.0.0 is frozen and registered privately at `aroramahek21-r-v-c-e-org/wandb-registry-model/interview-answer-quality:v0`. Its source model artifact is `aroramahek21-r-v-c-e/interview-assistant-phase2/interview-answer-quality-b0:v0`.

**Current blocker:** likith1109 has no access to our organization/project/Registry. W&B rejected the read-only Viewer invitation because the current plan does not support Viewer role management. No invitation is pending. We did not upgrade, substitute write-enabled Member access, grant admin rights or change visibility. His own team provides no access to this organization. **Do not run the W&B retrieval script or share owner credentials.** Resolve access later; his own accepted membership/login and successful download will be required before claiming retrieval works.

Use the existing verified private ZIP temporarily. Obtain `b0-inference-v1.0.0-private-handoff.zip` directly from the model owner through an approved encrypted, access-controlled channel. It is not in GitHub, not a release attachment and has not been transferred by the coding agent. The ZIP itself is not encrypted. It includes learned TF-IDF vocabulary and must remain private. Do not upload it or extracted contents to GitHub or Streamlit public hosting.

## Windows PowerShell setup

Open PowerShell at your Phase 2 checkout. Use Python 3.11.9 (the validated runtime). These commands assume a fresh API environment and the privately received ZIP is placed under the ignored artifact tree. If the environment already exists, reuse it instead of recreating it.

```powershell
New-Item -ItemType Directory -Force artifacts/phase2 | Out-Null
# Privately receive the ZIP into artifacts/phase2 before continuing.
$zip = 'artifacts/phase2/b0-inference-v1.0.0-private-handoff.zip'
$expectedZip = 'd1c6fa30fb65429f94f75b3f8231eda1424dcce37605ba77b0b0c4751c53840f'
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedZip) {
    throw 'Private ZIP checksum mismatch; stop and contact the owner'
}
$target = 'artifacts/phase2/private-handoff'
if (Test-Path -LiteralPath $target) { throw 'Choose a fresh extraction directory; do not overwrite' }
Expand-Archive -LiteralPath $zip -DestinationPath $target

py -3.11 -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
if ($LASTEXITCODE -ne 0) { throw 'API installation failed' }
$env:INTERVIEW_MODEL_BUNDLE = (Resolve-Path "$target/bundle").Path
# Verify the pinned manifest and every file without loading a model or contacting W&B.
.\.venv-api\Scripts\python.exe -B -c "import os; from pathlib import Path; from src.models.retrieve_model import verify_bundle; from src.utils.phase2_common import load_params; verify_bundle(Path(os.environ['INTERVIEW_MODEL_BUNDLE']), load_params()['phase2_retrieval'])"
if ($LASTEXITCODE -ne 0) { throw 'Bundle integrity failed; do not start the API' }
.\.venv-api\Scripts\python.exe -B src/api/app.py
```

The offline verification import uses only standard-library code and PyYAML; it does not call the retrieval function, import W&B or require a W&B login. FastAPI independently checks integrity and dependencies before loading the trusted paired artifacts once at startup. No fitting occurs. Keep this terminal open; in a second terminal:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/model/info'
$body = @{
    question = 'What is a stack?'
    reference_answer = 'A stack is a last in first out data structure.'
    candidate_answer = 'A stack is a last in first out data structure.'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' -ContentType 'application/json' -Body $body
```

Expected health shape: `{"status":"ok","model_ready":true}`. Observed prediction for this illustrative request: `{"predicted_quality":"Excellent","model_version":"1.0.0"}`. Missing/blank/non-string fields return 422; unavailable or invalid bundles return model readiness false and prediction 503. Streamlit should send the three strings to POST /predict and display predicted_quality/model_version, without inventing scores or detailed feedback. Question text is accepted but not predictive. Stop the server with Ctrl+C. Set INTERVIEW_MODEL_BUNDLE again in every new server terminal.

## Tests and optional development

Install `requirements-api-test.txt` into .venv-api, then run:

```powershell
.\.venv-api\Scripts\python.exe -B -m pip install -r requirements-api-test.txt
.\.venv-api\Scripts\python.exe -B -m pytest tests/test_api.py tests/test_model_retrieval.py -q -o addopts= -o cache_dir=artifacts/phase2/teammate-test-cache --basetemp=artifacts/phase2/teammate-test-tmp
```

Use INTERVIEW_MODEL_BUNDLE from the first setup block. These API/retrieval tests do not need the Mohler data, original Phase 1 workspace, W&B account or original experiment reports. `-o addopts=` bypasses legacy coverage defaults without changing tests. Experiment and GX dependencies belong in separate environments (`requirements-phase2.txt`, `requirements-data-validation.txt`), not the minimal API environment. Their integration tests additionally require private read-only data and historical artifacts; do not run all suites in a single minimal environment.

Phase 2 experiment/data paths in params.yaml use `../MLOps` relative to the checkout to preserve the existing sibling-source setup without publishing an owner's absolute workstation path. If your clone itself is named MLOps, this resolves to itself. Adjust only the Phase 2 path settings to your authorized read-only data location if needed. The Phase 1 configuration prefix and DVC pipeline are unchanged. API-only setup needs none of those source-data paths. Historical CV/validation/finalization runners enforce historical source hashes; source changes or different bytes must not be used to overwrite or rerun the frozen results. Exact-byte attributes preserve pinned code through checkout.

## Results and publication status

B0 selection is frozen by the predeclared validation rule. See [validation confirmation](validation_confirmation.md), [final evaluation](final_evaluation_and_bundle.md), [GX validation](gx_data_validation.md) and [API contract](fastapi_backend.md). The test benchmark was reused, minority recall is weak, generalization beyond Mohler is unverified and outputs are categorical rather than numeric grader scores.

**Nine historical experiment runs remain offline**: six full grouped-CV runs, two validation runs and one final reused-test evaluation. Only the separate sanitized evidence-publication and release-assembly runs/model artifact were published. No historical native training relationships were invented. Do not sync original offline streams: some contain paths or row-level metadata. [Registry retrieval](model_retrieval.md) documents the future workflow only, after permissions are resolved.

Pre-push verification in the owner's isolated environments: 40 API/retrieval tests, 43 GX/group-leakage tests and 37 ML protocol/artifact tests passed (120 total). Dataset, frozen bundle and private ZIP checksums were verified. No forests were retrained and no experiments synchronized during this review. Frozen binaries, detailed original reports, raw data, learned vocabulary, credentials, environments and generated test outputs are excluded from the contribution.
