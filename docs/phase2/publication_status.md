> Current teammate setup: Viewer invitation is blocked by the W&B plan; no invite is pending. Use the verified private ZIP and follow [teammate_handoff.md](teammate_handoff.md). Do not run Registry retrieval or synchronize historical runs until separately authorized. Earlier permission proposals below are historical.

# Final private publication review

Status: READY FOR APPROVAL. No upload or synchronization performed.

Frozen B0 v1.0.0: all 15 manifest entries match; bundle manifest SHA-256 036f6ac95e47d096d55839f4afe76751af7b7164e8b02b39f1acb8a8a2f85c99. Exactly 16 bundle files; no extras. No model loading or retraining performed.

Private project: aroramahek21-r-v-c-e/interview-assistant-phase2. Private Registry: aroramahek21-r-v-c-e-org/wandb-registry-model/interview-answer-quality. Live verification: entity writable/private-only, user team/org/Registry admin with Registry entitlement, collection empty. Admin roles support write/link operations; actual link execution is deferred until approval. Registry PRIVATE is organization-private, not an exclusive owner-only ACL. Teammate account/access deferred.

Exact model artifact allowlist (destination-relative paths):

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

Separate evidence artifact: `release_provenance.json` from `artifacts/phase2/publication-preparation-20261110T063131Z`. Local `upload_allowlist.json` is an execution control file, not uploaded. Class mapping is embedded in params.yaml and candidate_manifest.json; no invented separate mapping file.

Privacy: no raw datasets, row-level predictions, original offline streams, cache directories, credentials, ZIP handoff, or unrelated experiment files. Text files and decompressed joblib byte streams were scanned without deserialization: no readable absolute local paths or credential markers found. This is a targeted review, not proof against arbitrary encoded secrets. TF-IDF vocabulary/feature names and learned statistics remain sensitive, integral model content; private upload only.

Publication actions after explicit approval:

1. Recheck private access, hashes and exact allowlist; authenticate from supported local login without printing secrets. Disable automatic code/git/console/system capture; supply only sanitized explicit config and metadata.
2. Create a private evidence-publication run; add_file only release_provenance.json into immutable interview-answer-quality-evidence (type evaluation), wait for finalized version/digest.
3. Create a distinct release-assembly run, job_type release-assembly, group b0-v1.0.0-publication. use_artifact the exact evidence version. Log the 16 allowlisted unchanged files using add_file, into interview-answer-quality-b0 (type model), aliases bundle-v1.0.0 and candidate-b0. No directory recursion and no latest/production promotion.
4. Link the exact finalized model version to aroramahek21-r-v-c-e-org/wandb-registry-model/interview-answer-quality. Record immutable source/Registry version IDs and digests. This native graph represents packaging from reviewed evidence, not historical training. Historical six CV, two validation and final evaluation runs remain ID/checksum references until separately sanitized synchronization. Dataset lineage is checksum-only: do not log raw data or fabricate consumed dataset artifacts.
5. Download the immutable registered version into a fresh ignored directory and verify every checksum and manifest identity before trusted inference loading; run existing bundle compatibility/inference tests. Do not alter the frozen source bundle.
6. Do not automatically synchronize the nine offline runs: their original staged files include paths and row-level metadata. Review a separate sanitized synchronization plan first, preserving originals.

Limitations: reused test benchmark; weak minority recall (especially Good); question unused; dirty Phase 2 source recorded by hashes rather than a Phase 2 commit; joblib requires trusted source and pinned dependencies. Frozen source_class_body_sha256 is a full source-file hash despite its name; preserve it and document externally. Existing old handoff provenance status is historical and is not uploaded; the newly prepared evidence supersedes it for publication.

Review evidence: `artifacts/phase2/publication-preparation-20261110T063131Z`. No cloud mutation in this preparation step.

## Publication completed

Registry: `aroramahek21-r-v-c-e-org/wandb-registry-model/interview-answer-quality:v0`. Source: `aroramahek21-r-v-c-e/interview-assistant-phase2/interview-answer-quality-b0:v0`. Evidence: `aroramahek21-r-v-c-e/interview-assistant-phase2/interview-answer-quality-evidence:v0`. Digest: `6ff461f0f861ab0083dfbf1ef036503e`. All 16 downloaded files match the approved SHA-256 checksums. Eight isolated inference tests passed; no datasets accessed. Frozen files unchanged. Remote native evidence-input/model-output lineage verified. Cloud run files contain only config, summary and W&B-generated artifact manifests; all reviewed for absolute local paths and credential markers. No nine-run historical synchronization, retraining, deployment, commit or push.

Warnings: W&B applies latest automatically; use immutable v0. Initial pytest startup needed coverage addopts override. Initial cloud run-file assertion omitted service manifests; expanded read-only inspection verified those files. No unexpected user payload was found. Private Registry is organization-private; vocabulary remains sensitive.

Receipt and JUnit: `artifacts/phase2/publication-execution-20261110T063407Z`.
