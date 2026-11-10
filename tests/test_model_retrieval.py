"""Focused integrity and overwrite guards for Registry-only model setup."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
from src.models.retrieve_model import retrieve, safe_relative, verify_bundle


def fixture_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create tiny non-serialized test files; no real models or data involved."""
    root=tmp_path/'fixture';root.mkdir()
    (root/'model.joblib').write_bytes(b'opaque-test-bytes')
    manifest={'bundle_version':'test','selected_configuration':'B0','files_sha256':{'model.joblib':hashlib.sha256((root/'model.joblib').read_bytes()).hexdigest()}}
    (root/'bundle_manifest.json').write_text(json.dumps(manifest))
    cfg={'registry_reference':'org/wandb-registry-model/collection:v0','artifact_type':'model','expected_artifact_digest':'digest','expected_file_count':2,'manifest_filename':'bundle_manifest.json','expected_manifest_sha256':hashlib.sha256((root/'bundle_manifest.json').read_bytes()).hexdigest(),'expected_version':'test','expected_configuration':'B0'}
    return root,cfg


@pytest.mark.parametrize('name',['../escape','/absolute','C:/private','a/../b','a\\b','a//b','./a'])
def test_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError): safe_relative(name)


def test_pinned_manifest_and_corruption(tmp_path: Path) -> None:
    root,cfg=fixture_bundle(tmp_path)
    assert verify_bundle(root,cfg)['selected_configuration']=='B0'
    (root/'model.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'): verify_bundle(root,cfg)
    (root/'bundle_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest checksum'): verify_bundle(root,cfg)


def test_unexpected_file(tmp_path: Path) -> None:
    root,cfg=fixture_bundle(tmp_path)
    (root/'private.txt').write_text('must not be accepted')
    with pytest.raises(ValueError,match='allowlist'): verify_bundle(root,cfg)


def test_existing_target_not_overwritten(tmp_path: Path) -> None:
    root,cfg=fixture_bundle(tmp_path)
    with pytest.raises(FileExistsError): retrieve(cfg,root,api=object())


def test_mutable_alias_rejected_before_api(tmp_path: Path) -> None:
    _,cfg=fixture_bundle(tmp_path);cfg['registry_reference']='org/project/model:latest'
    with pytest.raises(ValueError,match='immutable'): retrieve(cfg,tmp_path/'target',api=object())


def test_download_verified_before_target_ready(tmp_path: Path) -> None:
    import shutil
    source,cfg=fixture_bundle(tmp_path)
    class Artifact:
        digest='digest'
        manifest=SimpleNamespace(entries={name:SimpleNamespace(ref=None) for name in ['bundle_manifest.json','model.joblib']})
        def download(self, root: str, skip_cache: bool) -> None:
            assert skip_cache
            assert not (tmp_path/'target').exists()
            shutil.copytree(source,root,dirs_exist_ok=True)
    api=SimpleNamespace(artifact=lambda *args,**kwargs:Artifact())
    target=retrieve(cfg,tmp_path/'target',api)
    verify_bundle(target,cfg)


def test_failed_verification_never_marks_target_ready(tmp_path: Path) -> None:
    _,cfg=fixture_bundle(tmp_path)
    class Artifact:
        digest='digest'
        manifest=SimpleNamespace(entries={name:SimpleNamespace(ref=None) for name in ['bundle_manifest.json','model.joblib']})
        def download(self, root: str, skip_cache: bool) -> None:
            (Path(root)/'bundle_manifest.json').write_text('{}')
    with pytest.raises(ValueError): retrieve(cfg,tmp_path/'target',SimpleNamespace(artifact=lambda *args,**kwargs:Artifact()))
    assert not (tmp_path/'target').exists()


def test_digest_or_reference_rejected_before_download(tmp_path: Path) -> None:
    _,cfg=fixture_bundle(tmp_path)
    artifact=SimpleNamespace(digest='wrong')
    api=SimpleNamespace(artifact=lambda *args,**kwargs:artifact)
    with pytest.raises(ValueError,match='digest'): retrieve(cfg,tmp_path/'target',api)
    artifact.digest='digest'
    artifact.manifest=SimpleNamespace(entries={name:SimpleNamespace(ref='https://external') for name in ['bundle_manifest.json','model.joblib']})
    with pytest.raises(ValueError,match='External'): retrieve(cfg,tmp_path/'target',api)
