> Current teammate setup: Viewer invitation is blocked by the W&B plan; no invite is pending. Use the verified private ZIP and follow [teammate_handoff.md](teammate_handoff.md). Do not run Registry retrieval or synchronize historical runs until separately authorized. Earlier permission proposals below are historical.

# Private B0 Registry retrieval and local API setup

The immutable reference is `aroramahek21-r-v-c-e-org/wandb-registry-model/interview-answer-quality:v0`. Omitting `:v0` does not pin a version. Bundle version is 1.0.0, selected configuration B0. Registry artifact digest is `6ff461f0f861ab0083dfbf1ef036503e`; the independent SHA-256 trust anchor for bundle_manifest.json is `036f6ac95e47d096d55839f4afe76751af7b7164e8b02b39f1acb8a8a2f85c99`.

`src/models/retrieve_model.py` reads `phase2_retrieval` in params.yaml. It uses the read-only W&B Public API (no run, upload, training or historical synchronization), requires a numeric version, checks the W&B artifact digest and safe non-reference entry paths, downloads into a fresh staging directory, pins the manifest SHA-256, checks B0/version identity and all 15 listed file hashes, and verifies exactly 16 files. Only then does it rename the staging directory to the ready destination. Existing targets are never overwritten. On failure it returns exit code 1 with sanitized logging; any partial staging directory remains unverified and must never be served. Integrity checks do not make arbitrary joblib files trustworthy: use only this reviewed immutable artifact and pinned manifest.

The directory may be overridden with `--destination`; references, trust anchors and defaults stay params-driven. The script does not load serialized artifacts. FastAPI separately verifies the pinned manifest and every file at startup before loading, through the existing runtime/wrapper. Set INTERVIEW_MODEL_BUNDLE to the directory containing bundle_manifest.json (not its parent). No FastAPI inference source changes were needed.

## Windows PowerShell commands

Run from the Phase 2 Git checkout containing params.yaml. Python 3.11.9 matches the frozen bundle environment. Use your own W&B account; never put an API key in a command argument, source file, GitHub repository or shared message. Login is interactive and stores credentials through the supported W&B mechanism. The command block assumes a new checkout without these virtual environments; reuse existing matching environments rather than recreating them.

```powershell
# Open PowerShell in your Phase 2 checkout.
py -3.11 -m venv .venv-retrieval
.\.venv-retrieval\Scripts\python.exe -m pip install -r requirements-model-retrieval.txt
.\.venv-retrieval\Scripts\python.exe -m wandb login --host https://api.forge.coreweave.com
if ($LASTEXITCODE -ne 0) { throw 'W&B login failed' }

.\.venv-retrieval\Scripts\python.exe -B src/models/retrieve_model.py --destination artifacts/phase2/retrieved/b0-v1.0.0
if ($LASTEXITCODE -ne 0) { throw 'Retrieval or verification failed; do not start the API' }

py -3.11 -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
if ($LASTEXITCODE -ne 0) { throw 'API dependency installation failed' }
$env:INTERVIEW_MODEL_BUNDLE = (Resolve-Path 'artifacts/phase2/retrieved/b0-v1.0.0').Path
.\.venv-api\Scripts\python.exe -B src/api/app.py
```

No activation or execution-policy change is needed. Keep the API terminal open and use a second PowerShell terminal:

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

Observed local response for that illustrative input: `{"predicted_quality":"Excellent","model_version":"1.0.0"}`; health: `{"status":"ok","model_ready":true}`. This is an inference demonstration, not a labeled evaluation or accuracy claim. Stop the server with Ctrl+C. Question text is unused by the model. No numeric interview score or detailed feedback is generated.

To repeat retrieval, supply a new ignored destination, then update INTERVIEW_MODEL_BUNDLE; do not overwrite a frozen or verified directory. Authentication/403 failures require account access correction, never making the project public or sharing another person's API key.

## Teammate access: pending approval

Read-only cloud membership inspection on 2026-11-10 found:

- `likith1109` is not an organization member of `aroramahek21-r-v-c-e-org` (member query returned null).
- Our team and private project member lists contain only `aroramahek21` (admin).
- The private `wandb-registry-model` member list contains only `aroramahek21` (admin).
- Membership in `likith1109-r-v-c-e` supplies no membership in our organization/team/Registry. No invitation, membership or role was changed. Retrieval was tested using the owner's credentials, not the teammate's credentials; teammate authentication has not been tested.

Minimum Registry download permission is **Viewer**, plus organization membership/Registry entitlement. **Restricted Viewer cannot download files**. Viewer supplies download/use permissions without collection mutation, linking or admin access. Registry permission is separate from the source project's role. See [official Registry role permissions](https://docs.coreweave.com/products/registry/configure_registry).

Proposed steps, requiring owner approval before changes:

1. In organization `aroramahek21-r-v-c-e-org` Settings ? Users, invite `likith1109` with non-admin organization access and Registry access enabled. Have him accept with his own account. Check any seat/plan implications before changing billing or purchasing seats.
2. Open Registry ? model ? gear/settings ? Registry access ? Add access ? select `likith1109`. Set the explicit and effective Registry role to **Viewer**, not Restricted Viewer, Member or Admin. Verify no team inheritance elevates it. This applies to the model Registry, which is organization-private; it is not an invented collection-only ACL.
3. If he also needs the private source project's experiment pages, grant the lowest **read-only/Viewer** project access supported by this tenant in project Settings ? access/members. If access is inherited through the team, use the team's view-only role if available. Do not silently substitute a write-enabled Member role; if the plan does not support read-only access, stop and review that limitation with the owner. Registry download rights do not by themselves prove source-project page access.
4. Keep both project and Registry PRIVATE. Re-read membership/effective roles after acceptance. Have him run the pinned retrieval command with his own login and verify all hashes. Owner-side membership checks cannot prove a teammate credential's actual authorization.

We have not assigned any admin privilege or made any permission change. His lack of access does not affect the already published model, but it currently blocks his download.

## Verification performed

- Focused retrieval guard tests: **14 passed**.
- Fresh live Registry v0 retrieval using the new script: W&B digest matched; manifest and all 15 file SHA-256 checks matched; exactly 16 files present.
- Existing FastAPI tests plus retrieval tests with INTERVIEW_MODEL_BUNDLE pointing at this download: **40 passed**, 3.56 seconds.
- Real localhost FastAPI startup, GET /health, GET /model/info and POST /predict succeeded (HTTP 200). Server stopped after verification.
- Test artifacts: `artifacts/phase2/retrieval-api-tests.xml` and `artifacts/phase2/retrieval-live-http/results.json`. Download: `artifacts/phase2/retrieved/b0-v1.0.0`.
- No package installation was needed: tested with existing .venv-experiments for retrieval and isolated .venv-api for serving. Fresh teammate environment installation commands are documented, not claimed executed here. API requirements now reference tracked requirements-inference.txt with exactly the frozen bundle's pins, rather than requiring an ignored artifact before installation.

To run the focused suite after installing requirements-api-test.txt:

```powershell
$env:INTERVIEW_MODEL_BUNDLE = (Resolve-Path 'artifacts/phase2/retrieved/b0-v1.0.0').Path
.\.venv-api\Scripts\python.exe -B -m pytest tests/test_api.py tests/test_model_retrieval.py -q -o addopts= -o cache_dir=artifacts/phase2/retrieval-api-cache --basetemp=artifacts/phase2/retrieval-api-tmp
```

The existing project pytest defaults request coverage tooling not installed in the minimal API environment; `-o addopts=` avoids requiring that unrelated plugin. API tests reuse the downloaded bundle and no dataset. Production deployment, dependency security audit, teammate access acceptance and teammate-machine execution remain separate tasks. Learned TF-IDF vocabulary and binaries remain private, ignored by Git, and must not be made public. No historical run synchronization, training, commit, push or deployment was performed.
