# Phase 2 continuous delivery to GHCR

## What exists and what this adds

- **CI:** `Frontend CI` tests the frontend/API boundary without private models,
  validates local and GHCR Compose configurations, builds both images and
  smoke-checks frontend health.
  Local real-bundle/Compose verification is separate; public CI cannot establish
  real model readiness without the securely provisioned bundle.
- **Continuous delivery:** `GHCR Delivery` waits for that entire workflow to
  succeed on a same-repository `main` commit, then builds and publishes both
  application images. It checks out the successful run's exact `head_sha`, not
  the moving branch tip. PRs, fork runs, failures and cancelled runs cannot publish.
- **Deployment:** an authorized operator pulls a selected version and starts it
  on an authorized host. No remote host, SSH credentials, automatic rollout,
  production environment or production deployment is configured here.

Image names for this repository are:

```text
ghcr.io/likith1109/mlops/frontend:sha-<full-40-character-commit-SHA>
ghcr.io/likith1109/mlops/backend:sha-<full-40-character-commit-SHA>
```

There is no `latest` tag. Publication summaries record each image digest and
labels record source repository/revision. SHA tags identify source; they are
not registry-enforced immutable tags. Base-image tags and frontend dependency
ranges are mutable, so rebuilding the same source can produce different bytes.
Record the published digests for audit/rollback. Both matrix jobs must succeed
before treating a version as delivered; publication of two images is not atomic.

## Gate, credentials and repository settings

The delivery workflow listens only to completed `Frontend CI` runs for `main`.
It requires conclusion `success`, matching source repository, and a triggering
`push` or `workflow_dispatch` event. It consumes no upstream caches or uploaded
artifacts. The checkout action is pinned, checkout credentials are not persisted,
and registry login occurs after image construction and the private-path audit.
Only publishing jobs receive `contents: read` and `packages: write`; they use
GitHub's short-lived built-in `GITHUB_TOKEN`, with no PAT or owner credentials.
No image/archive is uploaded as a GitHub Actions artifact.

For an optional manual delivery retry, dispatch **Frontend CI** with branch
**main** in the Actions UI. Successful CI then triggers delivery for that exact
commit. Do not manually dispatch a publication job to bypass CI. The delivery
workflow must first exist on the default branch; adding files locally does not
activate it. CI path filters include the delivery workflow and GHCR override,
so changes to either are tested before main-branch publication.

Read-only GitHub inspection on 2026-11-10 found `likith1109/MLOps` public, default
branch `main`, Actions enabled, read-only default workflow token permissions,
no classic main branch protection, and no rulesets. No settings were changed.
Thus CI is enforced as a **delivery gate**, but GitHub currently does not require
it before merges. Configure main branch protection/rulesets to require the
`test-and-build` check and PR review if required by your project policy. New
required workflows must also be added to the delivery gate before claiming they
are prerequisites for publication.

Before the first approved publication, confirm organization/repository policies
permit GHCR writes and the pinned checkout action. First publication needs
permission to create the packages. Existing packages must grant this repository
Actions write access. Verify package visibility/access explicitly; this workflow
does not change visibility. Repository visibility alone is not proof that package
pulls are public. Do not add private model material to a package to resolve access.

## Private model boundary

Delivery builds from an explicit `git archive` list containing only the required
public source/configuration files. Unrelated tracked or untracked files cannot
enter this build context. Both Dockerfiles retain their existing source copies
and `.dockerignore` rules. No private ZIPs, `artifacts/`, model files, learned
vocabulary, datasets, credentials, Git metadata, environments or W&B caches are
included in that archive. No build secret,
model mount or W&B credential is supplied in Actions. A pre-publication filesystem
audit rejects private artifact/credential paths under `/app`; it supplements the
build-context allowlist rather than replacing source review.

Provision the verified handoff bundle separately through the approved private
channel on each authorized host. Never include it in GHCR, CI artifacts, Git,
public hosting or image build context. Reuse the owner's checksum-verification
procedure before starting containers. Keep the quality-risk disclosure visible:
[model quality limitations](model_quality_risk.md) remain unchanged by delivery.

## Authorized Fedora host: pull and manual deployment

These are operator instructions, **not actions executed by this change**. Use a
checkout of the exact delivered commit, containing `compose.yaml` and
`compose.ghcr.yaml`. Docker Compose must support `!reset`; local validation used
v5.7.0. The override removes build definitions, leaving local build configuration
unchanged. Use the same relative dedicated bundle directory as your verified
local setup, beneath `artifacts/phase2`; never substitute a parent directory.

If packages require authentication, authenticate the same account that runs Docker
(e.g. root when using `sudo docker`). Use an approved read-only package credential
with `read:packages`, supplied through `docker login ghcr.io --username YOUR_LOGIN
--password-stdin` from a secure credential source. Do not paste tokens into files,
Git, command arguments or shell history. Configure a Docker credential helper;
without one Docker's login config is not encrypted. Host pull authentication is
separate from the workflow's `GITHUB_TOKEN`, which is not a host credential.
Public packages can be pulled anonymously.

After package authentication as necessary, run from the authorized checkout:

```bash
# Set the full source SHA shown by a successful GHCR Delivery run.
export IMAGE_TAG=sha-REPLACE_WITH_FULL_40_CHARACTER_COMMIT_SHA
export GHCR_REPOSITORY=likith1109/mlops
export COMPOSE_MODEL_BUNDLE=artifacts/phase2/private-handoff-verified-_xaszusi/bundle
export COMPOSE_BACKEND_UID=10001
export COMPOSE_BACKEND_GID=10001

# Reject placeholders or a checkout/version mismatch before pulling.
[[ "$IMAGE_TAG" =~ ^sha-[0-9a-f]{40}$ ]] &&
test "$(git rev-parse HEAD)" = "${IMAGE_TAG#sha-}" &&
test -d "$COMPOSE_MODEL_BUNDLE" &&
test -f "$COMPOSE_MODEL_BUNDLE/bundle_manifest.json"
```

Stop if these checks fail. The example bundle directory above must be replaced
with your actual securely provisioned location if different. After reviewing its
trusted origin, verify bytes without loading Joblib:

```bash
export INTERVIEW_MODEL_BUNDLE="$PWD/$COMPOSE_MODEL_BUNDLE"
.venv-api/bin/python -B -c 'import os; from pathlib import Path; from src.models.retrieve_model import verify_bundle; from src.utils.phase2_common import load_params; verify_bundle(Path(os.environ["INTERVIEW_MODEL_BUNDLE"]), load_params()["phase2_retrieval"])'
```

Use the minimal API environment from [the handoff](teammate_handoff.md). Continue
only after verification succeeds. The dedicated bundle needs read/traverse ACLs
for UID 10001, retaining owner-only permissions otherwise. The mount uses `ro,Z`:
Docker privately relabels **that host bundle directory and its contents only**
for SELinux; this changes host SELinux metadata. Keep SELinux enforcing and do
not relabel the project/home or broaden access globally. Short mount syntax can
create a missing directory, hence the source checks above are mandatory.

```bash
# Includes only non-secret configuration; sudo prompts in the operator's terminal.
delivery_compose() {
  sudo env GHCR_REPOSITORY="$GHCR_REPOSITORY" IMAGE_TAG="$IMAGE_TAG" \
    COMPOSE_MODEL_BUNDLE="$COMPOSE_MODEL_BUNDLE" \
    COMPOSE_BACKEND_UID="$COMPOSE_BACKEND_UID" \
    COMPOSE_BACKEND_GID="$COMPOSE_BACKEND_GID" \
    docker compose -f compose.yaml -f compose.ghcr.yaml "$@"
}

delivery_compose config --quiet
# Both pulls must succeed before rollout; no local-build fallback.
delivery_compose pull backend frontend &&
  delivery_compose up -d --no-build --pull never --wait --wait-timeout 180

delivery_compose ps
delivery_compose images
delivery_compose logs --tail=100 backend frontend
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"question":"What is a stack?","reference_answer":"A stack is a last in first out data structure.","candidate_answer":"A stack uses last in first out order."}'
curl --fail-with-body http://127.0.0.1:8501/_stcore/health
```

Require healthy containers, backend `model_ready: true`, a real prediction with
`model_version: 1.0.0`, and an API-mode submission in Streamlit at localhost:8501.
Streamlit reaches `http://backend:8000`, not host localhost. Host ports remain
loopback-bound. Inspect pulled `RepoDigests` against delivery summaries before
relying on the version. For rollback, select a previously approved SHA for both
images and repeat verification/pull/up; never substitute mock success for failure.

## Verification still required after approval

Local preparation checks passed: actionlint v1.7.12 on both workflows, Bash
syntax for delivery shell steps, seven simulated gate scenarios, construction
of the exact 25-file source-only build context, all three Compose configurations,
`git diff --check`, and the existing frontend/integration suite (38 passed).
These checks did not execute image build/login/push steps or GitHub workflows.

No commit, push, merge, workflow dispatch, package publication or deployment was
performed while preparing these files. An approved merge to `main`, successful
CI, actual package writes, digest inspection, authorized-host pulls and a real
containerized model/UI smoke check are still needed to validate delivery end to
end. No automatic production deployment should be claimed. GitHub service-side
acceptance and permission policies cannot be proved by local workflow linting.

References: [workflow_run security and triggers](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run),
[GHCR with GITHUB_TOKEN](https://docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions),
[registry authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
