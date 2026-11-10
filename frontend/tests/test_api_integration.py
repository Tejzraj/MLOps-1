"""Integrate the frontend HTTP client with real FastAPI routes and lifecycle.

No replacement predictor or synthetic inference result is used. Real inference
is exercised only when the privately supplied, verified bundle is available.
"""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from typing import Any

import requests

import pytest

pytest.importorskip("fastapi", reason="Install frontend/requirements-integration.txt for API integration")
from fastapi.testclient import TestClient

from frontend.api_client import PredictionError, predict
from frontend.config import Settings, load_settings
from frontend.question_bank import load_questions
from src.api.app import create_app, read_settings


@pytest.fixture
def frontend_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Use configured frontend settings without deployment environment overrides."""
    for key in ("FRONTEND_PARAMS", "FRONTEND_MODE", "BACKEND_URL", "BACKEND_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    return load_settings()


def test_missing_bundle_and_strict_contract(tmp_path: Path, frontend_settings: Settings) -> None:
    """The real app rejects old payloads and the client exposes missing-model 503s."""
    settings = replace(read_settings(), bundle=tmp_path)
    question = load_questions(frontend_settings.question_bank_path)[0]
    payload = dict(question=question.question, reference_answer=question.reference_answer,
                   candidate_answer=question.reference_answer)
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json() == {'status': 'ok', 'model_ready': False}
        assert client.get('/model/info').status_code == 503
        assert client.post('/predict', json=payload).status_code == 503
        assert client.post('/predict', json={**payload, 'question_id': question.question_id}).status_code == 422
        for field in payload:
            assert client.post('/predict', json={k: v for k, v in payload.items() if k != field}).status_code == 422
            for invalid in ('', ' ', None, 1):
                assert client.post('/predict', json={**payload, field: invalid}).status_code == 422

        def send(url: str, **kwargs: Any) -> requests.Response:
            """Route the requests boundary through the real ASGI application."""
            assert url == frontend_settings.backend_url + frontend_settings.predict_path
            assert set(kwargs['json']) == set(payload)
            assert kwargs['allow_redirects'] is False
            response = client.post('/predict', json=kwargs['json'])
            # Preserve requests' HTTPError semantics consumed by the frontend.
            result = requests.Response()
            result.status_code = response.status_code
            result._content = response.content
            return result

        with patch('frontend.api_client.requests.post', side_effect=send), \
             patch('frontend.api_client.mock_prediction', side_effect=AssertionError('No mock fallback')):
            with pytest.raises(PredictionError, match='HTTP 503.*Model bundle unavailable'):
                predict(question, question.reference_answer, frontend_settings)


def test_verified_bundle_frontend_prediction(frontend_settings: Settings) -> None:
    """Compare real frontend output to the actual frozen predictor, if supplied."""
    settings = read_settings()
    if not (settings.bundle / settings.manifest_filename).is_file():
        pytest.skip('Private model bundle missing; successful inference cannot be verified')
    question = load_questions(frontend_settings.question_bank_path)[0]
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json()['model_ready'] is True

        def send(url: str, **kwargs: Any) -> requests.Response:
            """Use real FastAPI serialization for the frontend's request."""
            response = client.post('/predict', json=kwargs['json'])
            result = requests.Response()
            result.status_code = response.status_code
            result._content = response.content
            return result

        with patch('frontend.api_client.requests.post', side_effect=send):
            result = predict(question, question.reference_answer, frontend_settings)
        expected = client.app.state.predictor.predict_one(dict(
            question=question.question, reference_answer=question.reference_answer,
            candidate_answer=question.reference_answer))
        assert result.quality_label == expected['quality_label']
        assert result.model_version == settings.expected_version
        assert result.simulated is False
