"""FastAPI tests against the real frozen B0 pair; no training or test-data access."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

from fastapi.testclient import TestClient
import pytest

from src.api import app as api
from src.utils.phase2_common import load_params


@pytest.fixture(scope='module')
def request_data() -> dict[str, str]:
    """Use the documented illustrative input, not a fabricated labeled dataset row."""
    return dict(load_params()['phase2_api']['example_request'])


@pytest.fixture(scope='module')
def client() -> Iterator[TestClient]:
    """Exercise the real startup lifecycle and frozen model in the API environment."""
    with TestClient(api.create_app()) as connection:
        assert connection.app.state.predictor is not None
        yield connection


def test_health_and_public_metadata(client: TestClient) -> None:
    """Health and model metadata contain only public, stable fields."""
    params = load_params()
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json() == {'status': 'ok', 'model_ready': True}
    info = client.get('/model/info')
    assert info.status_code == 200
    expected = params['data']['target_classes']
    assert info.json()['supported_classes'] == [expected[key] for key in sorted(expected)]
    assert info.json()['model_version'] == params['phase2_api']['expected_model_version']
    assert info.json()['question_used_as_feature'] is False
    assert info.json()['limitations'] == params['phase2_api']['public_limitations']
    assert set(info.json()) == {'model_version', 'supported_classes', 'question_used_as_feature', 'limitations'}
    for private in ['C:\\', 'C:/', 'artifacts/', 'sha256', 'source_sha256']:
        assert private not in info.text


def test_real_prediction_and_class_mapping(client: TestClient, request_data: dict[str, str]) -> None:
    """HTTP output exactly matches the existing wrapper and configured class map."""
    response = client.post('/predict', json=request_data)
    assert response.status_code == 200
    expected = client.app.state.predictor.predict_one(request_data)
    assert response.json() == {'predicted_quality': expected['quality_label'], 'model_version': api.read_settings().expected_version}
    assert expected['quality_label'] == load_params()['data']['target_classes'][expected['class_id']]
    assert set(response.json()) == {'predicted_quality', 'model_version'}


def test_determinism_cleaning_and_question_invariance(client: TestClient, request_data: dict[str, str]) -> None:
    """Repeat requests and exactly equivalent cleaned answers return identical output."""
    expected = client.post('/predict', json=request_data).json()
    for _ in range(len(load_params()['data']['target_classes'])):
        assert client.post('/predict', json=request_data).json() == expected
    equivalent = {key: '\t ' + value + '\n ' for key, value in request_data.items()}
    equivalent['candidate_answer'] += '<br/>'
    assert client.post('/predict', json=equivalent).json() == expected
    different_question = {**request_data, 'question': request_data['question'] + request_data['question']}
    assert client.post('/predict', json=different_question).json() == expected


@pytest.mark.parametrize('field', ['question', 'reference_answer', 'candidate_answer'])
def test_missing_field(client: TestClient, request_data: dict[str, str], field: str) -> None:
    """Each request field is mandatory."""
    payload = dict(request_data)
    payload.pop(field)
    assert client.post('/predict', json=payload).status_code == 422


@pytest.mark.parametrize('field', ['question', 'reference_answer', 'candidate_answer'])
@pytest.mark.parametrize('invalid', ['', ' \t\n ', None, 12])
def test_invalid_field(client: TestClient, request_data: dict[str, str], field: str, invalid: Any) -> None:
    """Empty, whitespace-only and non-string input is rejected by Pydantic."""
    assert client.post('/predict', json={**request_data, field: invalid}).status_code == 422


def test_preprocessing_empty_candidate(client: TestClient, request_data: dict[str, str]) -> None:
    """A br-only candidate is rejected using the frozen cleaning function."""
    response = client.post('/predict', json={**request_data, 'candidate_answer': ' <br/> '})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'invalid_answer'


def test_extra_fields_and_malformed_json(client: TestClient, request_data: dict[str, str]) -> None:
    """Unexpected score fields and malformed JSON cannot reach the classifier."""
    response = client.post('/predict', json={**request_data, 'private': 'never-echo-this-private-value'})
    assert response.status_code == 422
    assert 'never-echo-this-private-value' not in response.text
    assert client.post('/predict', content='{invalid', headers={'Content-Type': 'application/json'}).status_code == 422


def test_no_fit_during_requests(client: TestClient, request_data: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Serving never invokes fitting on either fitted artifact."""
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Inference must not train/refit')
    monkeypatch.setattr(client.app.state.predictor.model, 'fit', forbidden)
    monkeypatch.setattr(client.app.state.predictor.pipeline, 'fit', forbidden)
    monkeypatch.setattr(client.app.state.predictor.pipeline, 'fit_transform', forbidden)
    assert client.post('/predict', json=request_data).status_code == 200


def test_load_once_per_startup(request_data: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated health/info/predict calls do not reload the model."""
    original = api.load_predictor
    calls = []
    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, 'load_predictor', tracked)
    with TestClient(api.create_app()) as connection:
        assert connection.get('/health').status_code == 200
        assert connection.get('/model/info').status_code == 200
        assert connection.post('/predict', json=request_data).status_code == 200
        assert connection.post('/predict', json=request_data).status_code == 200
        assert len(calls) == 1
    assert connection.app.state.predictor is None


def test_missing_bundle_returns_sanitized_503(tmp_path: Path, request_data: dict[str, str]) -> None:
    """Startup failure keeps liveness available and rejects model endpoints clearly."""
    settings = replace(api.read_settings(), bundle=tmp_path / 'private-missing-bundle')
    with TestClient(api.create_app(settings)) as connection:
        assert connection.get('/health').json() == {'status': 'ok', 'model_ready': False}
        for response in [connection.get('/model/info'), connection.post('/predict', json=request_data)]:
            assert response.status_code == 503
            assert response.json()['detail']['code'] == 'model_unavailable'
            assert 'private-missing-bundle' not in response.text
            assert str(tmp_path) not in response.text


def test_corrupt_model_rejected_without_loading(tmp_path: Path, request_data: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Changed model bytes are rejected before any joblib deserialization."""
    from src.inference import predict as wrapper
    settings = api.read_settings()
    copied = tmp_path / 'bundle'
    shutil.copytree(settings.bundle, copied)
    manifest = json.loads((copied / 'candidate_manifest.json').read_text())
    path = copied / manifest['artifact_files']['model']
    with path.open('ab') as handle:
        handle.write(b'corrupt')
    monkeypatch.setattr(wrapper.joblib, 'load', lambda *args, **kwargs: pytest.fail('Corrupt model must not load'))
    with TestClient(api.create_app(replace(settings, bundle=copied))) as connection:
        assert connection.get('/health').json()['model_ready'] is False
        assert connection.post('/predict', json=request_data).status_code == 503


def test_bundle_path_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented environment variable controls location without changing weights."""
    key = load_params()['phase2_api']['bundle_environment_variable']
    monkeypatch.setenv(key, str(tmp_path))
    assert api.read_settings().bundle == tmp_path.resolve()


def test_runtime_excludes_training_tracking_and_parquet() -> None:
    """The isolated API environment does not depend on W&B, imblearn or PyArrow."""
    assert importlib.util.find_spec('wandb') is None
    assert importlib.util.find_spec('imblearn') is None
    assert importlib.util.find_spec('pyarrow') is None
