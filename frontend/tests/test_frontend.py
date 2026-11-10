"""Smoke coverage for the Streamlit workflow and its HTTP boundary."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

from frontend.api_client import PredictionError, mock_prediction, predict
from frontend.config import Settings, load_settings
from frontend.question_bank import load_questions

FRONTEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolate tests from a developer's deployment environment overrides."""
    for name in ("FRONTEND_PARAMS", "FRONTEND_MODE", "BACKEND_URL", "BACKEND_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    return load_settings(FRONTEND / "params.yaml")


@pytest.fixture
def app(settings: Settings) -> AppTest:
    """Start the real entrypoint using Streamlit's headless test harness."""
    return AppTest.from_file(FRONTEND / "app.py").run()


def practice(app: AppTest) -> AppTest:
    """Navigate into the answer workflow."""
    return app.radio(key="navigation").set_value("Practice").run()


def test_bank_and_home(app: AppTest, settings: Settings) -> None:
    """The homepage and curated reference bank load without a backend."""
    questions = load_questions(settings.question_bank_path)
    assert questions
    assert len({question.question_id for question in questions}) == len(questions)
    assert not app.exception
    assert app.title[1].value == settings.title
    assert any("MOCK MODE" in item.value for item in app.warning)


@pytest.mark.parametrize("label", ["Poor", "Needs Improvement", "Good", "Excellent"])
def test_mock_workflow(app: AppTest, label: str) -> None:
    """Each explicitly selected category displays beside the answer comparison."""
    practice(app)
    app.text_area[0].set_value("A stack uses last in, first out.")
    app.selectbox(key="demo_label").set_value(label)
    with patch("frontend.api_client.requests.post") as post:
        app.button(key="evaluate").click().run()
    post.assert_not_called()
    assert not app.exception
    assert label in [item.value for item in app.subheader]
    assert any("SIMULATED RESPONSE" in item.value for item in app.warning)
    assert not any("Experimental quality prediction" in item.value for item in app.warning)
    assert any("Reference answer" in item.value for item in app.markdown)
    app.text_area[0].set_value("Edited answer").run()
    assert "Evaluation result" not in [item.value for item in app.subheader]


def test_blank_answer(app: AppTest) -> None:
    """Whitespace is rejected before any prediction call."""
    practice(app)
    app.text_area[0].set_value("   ")
    with patch("frontend.api_client.requests.post") as post:
        app.button(key="evaluate").click().run()
    post.assert_not_called()
    assert any("Please write" in item.value for item in app.warning)
    assert "Evaluation result" not in [item.value for item in app.subheader]


def test_switch_question_clears_result(app: AppTest) -> None:
    """An answer's result cannot be shown for another practice prompt."""
    practice(app)
    app.text_area[0].set_value("An array supports indexed access.")
    app.button(key="evaluate").click().run()
    app.selectbox(key="question_id").set_value("practice-stacks").run()
    assert not app.exception
    assert "Evaluation result" not in [item.value for item in app.subheader]
    assert app.text_area[0].value == ""


def test_api_failure_visible_without_mock(app: AppTest) -> None:
    """A failed live request never creates a simulated result or retains an old result."""
    practice(app)
    app.text_area[0].set_value("My answer")
    app.button(key="evaluate").click().run()
    app.radio(key="mode").set_value("api").run()
    with patch("frontend.api_client.requests.post", side_effect=requests.ConnectionError):
        app.button(key="evaluate").click().run()
    assert not app.exception
    assert any("Cannot reach" in item.value for item in app.error)
    assert "Evaluation result" not in [item.value for item in app.subheader]
    assert not any("SIMULATED RESPONSE" in item.value for item in app.warning)


def test_live_result_and_retry_failure(app: AppTest) -> None:
    """Live success is attributed to FastAPI; a failed retry removes the old category."""
    practice(app)
    app.radio(key="mode").set_value("api").run()
    app.text_area[0].set_value("My explanation")
    response = Mock(status_code=200)
    response.json.return_value = {"predicted_quality": "Excellent", "model_version": "test-contract"}
    with patch("frontend.api_client.requests.post", return_value=response):
        app.button(key="evaluate").click().run()
    assert "Excellent" in [item.value for item in app.subheader]
    assert any("Model version: test-contract" in item.value for item in app.caption)
    assert any(
        item.value == "Experimental quality prediction, not a factual-correctness check. "
        "Results may be incorrect and require human review."
        for item in app.warning
    )
    with patch("frontend.api_client.requests.post", side_effect=requests.Timeout):
        app.button(key="evaluate").click().run()
    assert app.error
    assert "Evaluation result" not in [item.value for item in app.subheader]


def test_help_navigation(app: AppTest) -> None:
    """Instructions are reachable from the sidebar."""
    app.radio(key="navigation").set_value("How it works").run()
    assert not app.exception
    assert any("How it works" == item.value for item in app.title)


def test_predict_contract(settings: Settings) -> None:
    """Send the documented payload, configured endpoint and finite timeout."""
    question = load_questions(settings.question_bank_path)[0]
    response = Mock(status_code=200)
    response.json.return_value = {"predicted_quality": "Good", "model_version": "test-contract"}
    custom = replace(settings, backend_url="http://example.test/api", timeout_seconds=3.5)
    with patch("frontend.api_client.requests.post", return_value=response) as post:
        result = predict(question, "  My answer  ", custom)
    assert result.quality_label == "Good"
    assert result.simulated is False
    assert result.model_version == "test-contract"
    post.assert_called_once_with(
        "http://example.test/api/predict",
        json={
            "question": question.question,
            "reference_answer": question.reference_answer,
            "candidate_answer": "My answer",
        },
        timeout=3.5,
        allow_redirects=False,
    )


@pytest.mark.parametrize(
    "failure, message",
    [
        (requests.Timeout(), "timed out"),
        (requests.ConnectionError(), "Cannot reach"),
        (requests.RequestException(), "request failed"),
    ],
)
def test_transport_failures(settings: Settings, failure: Exception, message: str) -> None:
    """Transport errors become actionable application errors."""
    question = load_questions(settings.question_bank_path)[0]
    with patch("frontend.api_client.requests.post", side_effect=failure):
        with pytest.raises(PredictionError, match=message):
            predict(question, "answer", settings)


@pytest.mark.parametrize("status", [422, 500, 503])
def test_http_errors(settings: Settings, status: int) -> None:
    """Backend validation and server failures preserve the status for troubleshooting."""
    question = load_questions(settings.question_bank_path)[0]
    response = Mock(status_code=status)
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    with patch("frontend.api_client.requests.post", return_value=response):
        with pytest.raises(PredictionError, match=f"HTTP {status}"):
            predict(question, "answer", settings)


@pytest.mark.parametrize("body", [{}, [], {"quality_label": "Good"}, {"predicted_quality": 2},
     {"predicted_quality": "Unknown"}, {"predicted_quality": "Good"},
     *[{"predicted_quality": "Good", "model_version": value} for value in (None, 1, "", " ")]])
def test_bad_schema(settings: Settings, body: object) -> None:
    """Reject undocumented responses rather than guessing labels or scores."""
    response = Mock(status_code=200)
    response.json.return_value = body
    with patch("frontend.api_client.requests.post", return_value=response):
        with pytest.raises(PredictionError, match="Unexpected API response"):
            predict(load_questions(settings.question_bank_path)[0], "answer", settings)


def test_invalid_json(settings: Settings) -> None:
    """Non-JSON success responses produce a clear integration error."""
    response = Mock(status_code=200)
    response.json.side_effect = requests.exceptions.JSONDecodeError("bad JSON", "<html>", 0)
    with patch("frontend.api_client.requests.post", return_value=response):
        with pytest.raises(PredictionError, match="invalid JSON"):
            predict(load_questions(settings.question_bank_path)[0], "answer", settings)


def test_no_redirect_following(settings: Settings) -> None:
    """A redirect does not silently resend candidate text to another endpoint."""
    response = Mock(status_code=307)
    with patch("frontend.api_client.requests.post", return_value=response):
        with pytest.raises(PredictionError, match="redirected"):
            predict(load_questions(settings.question_bank_path)[0], "answer", settings)


def test_mock_is_only_a_fixture(settings: Settings) -> None:
    """Mock categories depend only on the explicit selection."""
    assert mock_prediction("Poor", settings).simulated
    with pytest.raises(ValueError, match="Unknown"):
        mock_prediction("unknown", settings)


@pytest.mark.parametrize(
    "name,value",
    [
        ("FRONTEND_MODE", "automatic"),
        ("BACKEND_URL", "file:///tmp/api"),
        ("BACKEND_TIMEOUT_SECONDS", "0"),
        ("BACKEND_TIMEOUT_SECONDS", "nan"),
    ],
)
def test_invalid_configuration(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Fail clearly on unusable deployment settings."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="Invalid frontend configuration"):
        load_settings(FRONTEND / "params.yaml")


def test_configuration_error_ui(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration failure stops cleanly with instructions instead of a traceback."""
    monkeypatch.setenv("FRONTEND_PARAMS", str(FRONTEND / "missing.yaml"))
    app = AppTest.from_file(FRONTEND / "app.py").run()
    assert app.error
    assert not app.exception


def test_environment_overrides(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployment can configure mode, backend and timeout without editing tracked files."""
    monkeypatch.setenv("FRONTEND_MODE", "api")
    monkeypatch.setenv("BACKEND_URL", "http://localhost:9000/")
    monkeypatch.setenv("BACKEND_TIMEOUT_SECONDS", "8")
    loaded = load_settings(FRONTEND / "params.yaml")
    assert (loaded.mode, loaded.backend_url, loaded.timeout_seconds) == (
        "api",
        "http://localhost:9000",
        8,
    )
