> Current teammate setup: Viewer invitation is blocked by the W&B plan; no invite is pending. Use the verified private ZIP and follow [teammate_handoff.md](teammate_handoff.md). Do not run Registry retrieval or synchronize historical runs until separately authorized. Earlier permission proposals below are historical.

# W&B synchronization and Registry readiness workflow

Status: **local integrity passed; external synchronization and registration are pending review**. No authentication, synchronization, artifact publication, registration, deployment, retraining, commit or push was performed by this audit. The B0 bundle remains frozen.

## Actual audit

Read-only scan: `artifacts/phase2/registry-readiness-20261110T045658Z-8b1017f0/audit.json`. Reproduce with `python -B src/models/audit_registry_readiness.py` from the Phase 2 worktree; this creates only a new ignored audit report.

- 9 completed offline runs, three distinct groups, all project `interview-assistant-phase2`; all recorded entity fields are blank.
- 50 staged JSON files: all present with matching SDK digests and sizes. Transaction headers, fragment CRCs and protobuf records parsed successfully; all run exit records are zero.
- Each run records only requirements.txt as an auxiliary uploaded file. No environment, system-statistics or console-output records were found. SDK telemetry/version records are present. No native use_artifact records are present.
- No local synced markers were found; this supports local unsynced status, but remote status has not been queried.
- Bundle: 16 files; every manifest SHA-256 matches. The model and fitted pipeline exactly match the validation-selected B0 candidate. Exact custom-class source and pinned runtime requirements match.
- 84 audited immutable inputs were rehashed after inspection; zero changed. No serialized models were loaded for this audit.

| Phase | Run | ID | Group |
|---|---|---|---|
| CV | B0 | r29el1c7 | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| CV | W1 | ixye7o2d | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| CV | W2 | 0ha6shlk | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| CV | W3 | 66gvgg1v | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| CV | O1 | l753jsmr | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| CV | F1 | 7bcop1v2 | `mohler-rf-imbalance-v1-full-20261109T092712Z-15f5c059` |
| Validation | B0-validation | 5t2sq8pe | `mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8` |
| Validation | F1-validation | 5mc35ss9 | `mohler-rf-b0-f1-validation-v1-20261109T094300Z-6c03b9e8` |
| Reused test | B0-reused-test | zi48eo8t | `mohler-b0-inference-v1-20261109T095935Z-d812a332` |

Smoke runs, preflight output and unrelated run directories are excluded from the synchronization inventory. Preserve each parent experiment directory, especially wandb-data/artifacts/staging. Offline transaction records reference staged payloads by absolute path: copying only the .wandb file or only its run directory is not a complete backup.

## Destination and access requirements

- Choose the intended W&B team entity and organization; neither is established locally. Set non-secret destination fields in phase2_registry only after approval. Confirm the entity belongs to the intended organization and the server/base URL is correct.
- Confirm private project visibility, permission to log artifacts, Registry access, and ability to link model artifacts. Account/organization availability and authorization were not tested.
- Proposed Registry: `Models`; collection: `interview-answer-quality`; artifact type: `model`; source artifact name: `interview-answer-quality-b0`.
- Proposed aliases: `bundle-v1.0.0` and `candidate-b0`. Do not use production aliases. Aliases are mutable; pin returned immutable versions plus digest and the approved bundle-manifest SHA-256 for retrieval.
- Registry collection version numbers and source-project artifact version numbers need not match. Do not assume the first version is v0.

## Upload scope and privacy gate

Approve the following scope before any external operation:

1. Exactly the nine run streams listed above, their recorded requirements.txt files and their 50 explicitly staged JSON payloads. These contain metrics, parameter settings, assignments, row identifiers/labels/predictions, checksums, dependencies and provenance; no raw student-answer table is staged. Class labels and question IDs are still dataset-derived metadata.
2. The deployable artifact contains exactly the bundle_manifest.json plus files_sha256 inventory (listed below). Use explicit add_file calls; never recursively upload the repository or artifacts/phase2.
3. A new dataset-identity manifest may contain only source citation, prepared-split hashes, row/group counts, schema, class mapping and thresholds; never Parquet/CSV contents or raw answer strings.

Known privacy risks requiring explicit acceptance or a separate sanitized-export plan:

- Final staged config.json/evaluation metadata contain absolute local paths with the local account name. SDK telemetry and complete package lists reveal environment details. Do not assume sync removes these.
- The frozen model and feature pipeline contain a learned word/ngram vocabulary; feature_names.json also exposes learned tokens. These are dataset-derived text even though no raw answer table is included. Stripping only feature_names.json does not remove the vocabulary inside the fitted pipeline.
- Source code contains training CLI functions behind main guards, and candidate manifests include dirty file-name status; these are part of the reviewed immutable bundle. No .env, key file, notebook, raw dataset, Git directory, virtual environment, SDK cache, debug log, unrelated report, F1 binary or original Phase 1 binary is included in the model artifact.
- If metadata needs redaction, preserve existing streams and v1.0.0. Create separately reviewed sanitized report artifacts/runs with provenance back to the originals. Do not edit transaction logs or claim that a selective replay kept original run identity unchanged.

The installed W&B 0.31.0 CLI no longer supports the old --include-globs/--exclude-globs sync flags. A directory allowlist alone does not filter artifact payloads already embedded in its transaction log. The audit verifies their actual payload scope.

## Bundle file allowlist

- `MODEL_CARD.md`
- `README.md`
- `bundle_manifest.json`
- `candidate_manifest.json`
- `feature_names.json`
- `feature_pipeline.joblib`
- `model.joblib`
- `params.yaml`
- `requirements-inference.txt`
- `src/__init__.py`
- `src/features/__init__.py`
- `src/features/build_features.py`
- `src/inference/__init__.py`
- `src/inference/predict.py`
- `src/models/__init__.py`
- `src/models/run_imbalance_experiments.py`

Approved integrity anchor for bundle_manifest.json: `036f6ac95e47d096d55839f4afe76751af7b7164e8b02b39f1acb8a8a2f85c99`.

## Secure authentication and synchronization commands ? NOT EXECUTED

Run only after approval of destination and payload scope. Use a W&B-provided interactive login prompt or inject WANDB_API_KEY through a secret manager into the current process. Never put the key in YAML, a script, command-line arguments, Git, a transcript, or an environment dump. Do not inspect or print existing key stores. A supported login may persist credentials outside the repository; process-scoped secret injection avoids that persistence.

```powershell
# Run from your Phase 2 checkout.
$env:WANDB_ENTITY = '<approved-team-entity>'
$env:WANDB_ORGANIZATION = '<approved-organization>'
$env:WANDB_MODE = 'online'
$env:WANDB_DIR = (Join-Path (Get-Location) 'artifacts/phase2/registry-operations/wandb')
$env:WANDB_CACHE_DIR = (Join-Path (Get-Location) 'artifacts/phase2/registry-operations/cache')
$env:WANDB_CONFIG_DIR = (Join-Path (Get-Location) 'artifacts/phase2/registry-operations/config')
$env:WANDB_DATA_DIR = (Join-Path (Get-Location) 'artifacts/phase2/registry-operations/data')
@($env:WANDB_DIR, $env:WANDB_CACHE_DIR, $env:WANDB_CONFIG_DIR, $env:WANDB_DATA_DIR) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
# Skip interactive login when a secret manager already supplies WANDB_API_KEY.
.\.venv-experiments\Scripts\python.exe -B -m wandb login
$audit = Get-Content -Raw 'artifacts/phase2/registry-readiness-20261110T045658Z-8b1017f0/audit.json' | ConvertFrom-Json
$runPaths = @($audit.runs | ForEach-Object { Join-Path (Get-Location) $_.directory })
foreach ($runPath in $runPaths) {
  .\.venv-experiments\Scripts\python.exe -B -m wandb sync --dry-run --entity $env:WANDB_ENTITY --project interview-assistant-phase2 $runPath
  if ($LASTEXITCODE -ne 0) { throw 'Sync preview failed' }
}
# Review exactly nine paths. The next loop UPLOADS; separate authorization required.
foreach ($runPath in $runPaths) {
  .\.venv-experiments\Scripts\python.exe -B -m wandb sync --entity $env:WANDB_ENTITY --project interview-assistant-phase2 $runPath
  if ($LASTEXITCODE -ne 0) { throw 'Stop on first synchronization failure' }
}
```

Do not use bare wandb sync, --sync-all, --id overrides, --include-synced or globbing over the output root. The explicit run paths preserve recorded run IDs/groups. After synchronization, verify IDs, all three group names, configuration count, fold metrics, successful states and report artifact digests in the destination UI/API. Resolve each logged artifact to its actual immutable qualified name. Do not replay already-synced runs blindly. Preserve local staging until remote artifact files are verified.

## Lineage and Registry publication design

Current lineage is metadata/checksum based: prepared training split ? shared grouped CV ? fixed validation decision retaining B0 ? unchanged final evaluated B0 ? verified bundle. Existing runs did not call use_artifact, so syncing alone will not create those native dependency edges retrospectively.

After sync, create a metadata-only dataset-manifest artifact in a dedicated lineage-metadata run. Its JSON should contain train/validation/test identities from the candidate/final manifests and the original Mohler citation, schema, label mapping and human-score thresholds; record that the test benchmark is reused. No raw files or public/private storage references that could download data are attached.

Then a separate registry-assembly run consumes that manifest and the nine immutable experiment-report artifacts, and produces the deployable bundle. This gives a native assembly lineage graph; it does not pretend to be the historical training producer. Include the unchanged validation decision hash and final evaluation hash. The decision report itself was not staged in the validation run, so preserve it as a separately approved decision-evidence artifact, with protocol and decision JSON only. Likewise use a checksum-identified reviewed final-evaluation artifact.

The following is the exact core publication pattern, to be saved/run only after permissions, file review and non-secret configuration are approved. It reads no API key directly; W&B handles authentication. The dataset-manifest and decision-evidence artifacts must already have immutable qualified names recorded in an approved release-inputs file. That file must contain no credentials or raw text.

```python
import json, os
from pathlib import Path
import wandb
import yaml
from src.utils.phase2_common import sha256

root = Path.cwd()
params = yaml.safe_load((root / "params.yaml").read_text())
cfg = params["phase2_registry"]
if not cfg["upload_approved"] or not cfg["entity"] or not cfg["organization"]:
    raise RuntimeError("Destination and upload scope must be approved first")
if os.environ.get("WANDB_MODE") != "online":
    raise RuntimeError("Explicit online operation is required for publication")
audit = json.loads((root / "artifacts/phase2/registry-readiness-20261110T045658Z-8b1017f0/audit.json").read_text())
bundle = root / audit["bundle"]["directory"]
manifest_path = bundle / "bundle_manifest.json"
assert sha256(manifest_path) == audit["bundle"]["manifest_sha256"]
manifest = json.loads(manifest_path.read_text())
allowlist = set(manifest["files_sha256"]) | {"bundle_manifest.json"}
assert {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} == allowlist
for relative, expected in manifest["files_sha256"].items():
    assert sha256(bundle / relative) == expected

# This proposed file is created later from verified remote artifact identities.
release_inputs = json.loads((root / "artifacts/phase2/registry-operations/release-inputs.json").read_text())
# Required keys: dataset_manifest_qualified_name, decision_evidence_qualified_name.
api = wandb.Api()
report_inputs = []
for record in audit["runs"]:
    remote = api.run(f"{cfg['entity']}/{cfg['project']}/{record['run']['id']}")
    assert remote.group == record["run"]["group"]
    matches = [a for a in remote.logged_artifacts() if a.name.rsplit(":", 1)[0] == record["artifacts"][0]["name"]]
    assert len(matches) == 1 and matches[0].digest == record["artifacts"][0]["digest"]
    report_inputs.append(matches[0].qualified_name)

with wandb.init(entity=cfg["entity"], project=cfg["project"],
                job_type="registry-assembly", name="b0-bundle-v1.0.0", mode="online",
                dir=os.environ["WANDB_DIR"],
                settings=wandb.Settings(disable_git=True, disable_code=True,
                                        console="off", x_disable_stats=True, x_disable_meta=True)) as run:
    for name in report_inputs + [release_inputs["dataset_manifest_qualified_name"],
                                 release_inputs["decision_evidence_qualified_name"]]:
        run.use_artifact(name)
    artifact = wandb.Artifact(cfg["model_artifact_name"], type=cfg["artifact_type"], metadata={
        "bundle_version": "1.0.0", "selected_configuration": "B0",
        "bundle_manifest_sha256": audit["bundle"]["manifest_sha256"],
        "selection_record_sha256": manifest["selection_record_sha256"],
        "reused_test_benchmark": True, "retrained_for_registry": False,
        "source_class_hash_note": "source_class_body_sha256 in frozen v1.0.0 is the original full source-file hash"})
    for relative in sorted(allowlist):
        artifact.add_file(str(bundle / relative), name=relative)
    logged = run.log_artifact(artifact, aliases=cfg["proposed_aliases"]).wait()
    linked = run.link_artifact(logged,
        target_path=f"wandb-registry-{cfg['registry_name']}/{cfg['collection_name']}",
        aliases=cfg["proposed_aliases"])
    linked.wait()
    # Save returned immutable source and Registry qualified names/digests to an
    # ignored release record; confirm Registry collection version in the UI/API.
    release_record = {"source_qualified_name": logged.qualified_name,
                      "source_digest": logged.digest,
                      "registry_qualified_name": linked.qualified_name,
                      "registry_digest": linked.digest,
                      "bundle_manifest_sha256": audit["bundle"]["manifest_sha256"]}
    (root / "artifacts/phase2/registry-operations/release.json").write_text(json.dumps(release_record, indent=2))
```

Before publication, inspect the existing collection and aliases in the intended destination. Stop if bundle-v1.0.0 already identifies different content; do not overwrite it. If the identical artifact is already present, retrieve and verify that immutable version instead of blindly republishing.

Do not regenerate the bundle or its model card during publication. The frozen bundle has one naming defect: source_class_body_sha256 records the original full source-file SHA, not a class-body-only hash. The audit verified the exact extracted class body; document this externally in Registry metadata rather than silently changing v1.0.0.

### Exact metadata-only lineage preparation ? NOT EXECUTED

Run this after the nine approved syncs and payload/destination review, before the registry-assembly snippet above. It creates two small, deliberately sanitized JSON artifacts and writes the release-inputs file used by that snippet. No model or data table is uploaded here. It does not read dataset rows.

```python
import json, os
from pathlib import Path
import wandb
import yaml
from src.utils.phase2_common import sha256

root = Path.cwd()
params = yaml.safe_load((root / "params.yaml").read_text())
cfg = params["phase2_registry"]
if not cfg["upload_approved"] or not cfg["entity"] or not cfg["organization"]:
    raise RuntimeError("Destination and metadata uploads must be approved first")
if os.environ.get("WANDB_MODE") != "online":
    raise RuntimeError("Explicit online operation is required")
work = root / "artifacts/phase2/registry-operations"
work.mkdir(parents=True, exist_ok=True)
candidate_path = root / params["phase2_final"]["candidate_directory"] / "manifest.json"
decision_path = root / params["phase2_final"]["selection_record"]
assert sha256(candidate_path) == params["phase2_final"]["candidate_manifest_sha256"]
assert sha256(decision_path) == params["phase2_final"]["selection_record_sha256"]
candidate = json.loads(candidate_path.read_text())
decision = json.loads(decision_path.read_text())
final = json.loads((root / cfg["final_directory"] / "final_evaluation.json").read_text())
identities = {
    "dataset": "Mohler ASAG / nkazi/MohlerASAG, cleaned, open_ended",
    "content_included": False,
    "splits": {**candidate["datasets"], "test": {
        "sha256": final["test_sha256"], "rows": final["rows"], "questions": final["questions"]}},
    "label_mapping": params["data"]["target_classes"],
    "human_score_thresholds": params["data"]["label_thresholds"],
    "group_column": params["data"]["group_column"],
    "human_labels_only": True,
    "test_benchmark_status": final["benchmark_status"]}
# Include only reviewed decision fields; exclude local directories/run-file paths.
evidence = {key: decision[key] for key in ["selected_configuration", "candidate_confirmed",
    "decision_rule", "validation_macro_f1_delta_candidate_minus_baseline",
    "cv_validation_inconsistency", "protocol_sha256"]}
evidence["original_decision_sha256"] = sha256(decision_path)
evidence["validation_metrics"] = {name: item["metrics"] for name, item in decision["records"].items()}
evidence["final_evaluation_sha256"] = sha256(root / cfg["final_directory"] / "final_evaluation.json")
identity_path = work / "dataset-identities.json"
decision_evidence_path = work / "decision-evidence.json"
identity_path.write_text(json.dumps(identities, indent=2))
decision_evidence_path.write_text(json.dumps(evidence, indent=2))
release_inputs = {}
with wandb.init(entity=cfg["entity"], project=cfg["project"], mode="online",
                job_type="lineage-metadata", name="b0-release-lineage-metadata",
                dir=os.environ["WANDB_DIR"],
                settings=wandb.Settings(disable_git=True, disable_code=True, console="off",
                                        x_disable_stats=True, x_disable_meta=True)) as run:
    for name, kind, file, key in [
        ("mohler-asag-split-identities", "dataset-manifest", identity_path, "dataset_manifest_qualified_name"),
        ("b0-validation-decision", "model-decision", decision_evidence_path, "decision_evidence_qualified_name")]:
        artifact = wandb.Artifact(name, type=kind, metadata={"contains_raw_rows": False})
        artifact.add_file(str(file), name=file.name)
        logged = run.log_artifact(artifact).wait()
        release_inputs[key] = logged.qualified_name
        release_inputs[key + "_digest"] = logged.digest
(work / "release-inputs.json").write_text(json.dumps(release_inputs, indent=2))
```

Neither this metadata import nor registry assembly reruns training, validation or test evaluation. Native edges are added only for these new import/assembly runs, not falsely attributed to historical experiments.

## Retrieval and verification ? NOT EXECUTED

Resolve the linked collection version through the Registry UI/API and record it; use the Registry organization namespace rather than assuming it equals the source team entity. Example command with an explicitly verified immutable version:

```powershell
$env:WB_REGISTRY_VERSION = '<verified-vN>'
$registryPath = "$($env:WANDB_ORGANIZATION)/wandb-registry-Models/interview-answer-quality:$($env:WB_REGISTRY_VERSION)"
.\.venv-experiments\Scripts\python.exe -B -m wandb artifact get --type model --root artifacts/phase2/registry-operations/retrieved $registryPath
```

Before any joblib loading, verify the downloaded bundle_manifest.json against the previously approved SHA-256 anchor above; then verify every files_sha256 entry and the model/pipeline identities in candidate_manifest.json. Reject missing, escaping, unexpected or changed files. W&B artifact digests and local SHA-256 serve different purposes; record both. Pin vN and digest rather than latest or only a mutable alias.

For a trusted retrieved bundle, use its README CLI with the retrieved directory as --bundle. A separate approval is needed to create a fresh minimal runtime/install dependencies; do not reuse the training stack accidentally. Use Python 3.11.9 and requirements-inference.txt first. Run representative inference checks without rerunning the benchmark or changing selection.

## FastAPI readiness and remaining risks

- Bundle is suitable as an immutable inference artifact: paired fitted state, verified class order, exact cleaning, deterministic batch/single inference, minimal pinned dependencies, model card and source are present. Prior tests included an isolated process and excluded imbalanced-learn/W&B imports.
- Not yet a FastAPI application: HTTP schema/status codes, lifespan loading, health/readiness endpoints, concurrency/resource limits, error handling, logging/privacy and container runtime need implementation and testing. No fresh minimal dependency install, Linux/container or API load test has run.
- The compatibility modules use the generic src namespace. Start the service with the bundle import root before importing other repository src modules, or isolate inference in its own process. Do not mix repository training modules with the bundle namespace.
- Load once per worker after identity checks. Keep downloaded files read-only. n_jobs=2 is frozen; size worker counts appropriately without changing artifacts. Do not refit or perform sampling during serving.
- The model does not use question text; input requires a reference answer. It outputs a category rather than a numeric grader score. It is narrow-domain and probabilities are uncalibrated.
- Reused test benchmark: accuracy 0.590062, macro-F1 0.392525, balanced accuracy 0.388752; Good recall 0.081395 with support 86. Do not describe these as independent unseen-test results or hide the failed F1 validation confirmation.
- Source Git commit is the Phase 1 base and the Phase 2 implementation is uncommitted. Manifests include dirty state/code hashes, but the commit alone cannot reproduce the current implementation. A reviewed source snapshot/version is a later release requirement; no commit was made here.

## Sources used for API/CLI verification

- Installed W&B 0.31.0 `python -B -m wandb sync --help`, artifact get --help and Run.link_artifact docstring/signature are the primary version-specific references.
- Official SDK CLI source: https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py
- Official Registry linking guide: https://github.com/wandb/docs/blob/main/models/registry/link_version.mdx
- Official transaction format reader reference: https://raw.githubusercontent.com/wandb/wandb/v0.29.0/wandb/sdk/internal/datastore.py (used to confirm the unchanged CRC-fragment format; no old SDK installed).
