"""HTTP boundary and explicit demo responses; contains no model or scoring logic."""

import logging
from dataclasses import dataclass

import requests

from frontend.config import Settings
from frontend.question_bank import Question

LOGGER = logging.getLogger(__name__)


class PredictionError(Exception):
    """A user-visible failure to obtain a valid backend prediction."""


@dataclass(frozen=True)
class EvaluationResult:
    """A backend category or an explicitly simulated presentation example."""

    quality_label: str
    simulated: bool
    model_version: str | None = None


def mock_prediction(label: str, settings: Settings) -> EvaluationResult:
    """Return the selected UI fixture; never examine or grade candidate text."""
    if label not in settings.quality_labels:
        raise ValueError("Unknown simulated category")
    return EvaluationResult(quality_label=label, simulated=True)


def predict(question: Question, candidate_answer: str, settings: Settings) -> EvaluationResult:
    """POST the strict FastAPI /predict contract; reject failures without mock fallback."""
    if not candidate_answer.strip():
        raise PredictionError("Write an answer before requesting evaluation.")
    payload = {
        "question": question.question,
        "reference_answer": question.reference_answer,
        "candidate_answer": candidate_answer.strip(),
    }
    try:
        response = requests.post(
            settings.backend_url + settings.predict_path,
            json=payload,
            timeout=settings.timeout_seconds,
            allow_redirects=False,
        )
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise PredictionError("The backend redirected the request. Check the configured URL.")
    except requests.Timeout as exc:
        LOGGER.warning("Prediction request timed out")
        raise PredictionError("The backend timed out. Try again or check the API service.") from exc
    except requests.ConnectionError as exc:
        LOGGER.warning("Prediction backend unavailable")
        raise PredictionError(
            "Cannot reach the backend. Check its URL and that FastAPI is running."
        ) from exc
    except requests.HTTPError as exc:
        LOGGER.warning("Prediction HTTP failure (status %s)", exc.response.status_code)
        detail = (
            "Model bundle unavailable or invalid. Check the private bundle mount and server setup."
            if exc.response.status_code == 503
            else "Check the API contract and server logs."
        )
        raise PredictionError(f"The backend returned HTTP {exc.response.status_code}. {detail}") from exc
    except requests.RequestException as exc:
        LOGGER.warning("Prediction request failed (%s)", type(exc).__name__)
        raise PredictionError("The API request failed. Check the backend configuration.") from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise PredictionError(
            "The backend returned invalid JSON. Expected predicted_quality and model_version."
        ) from exc
    if not isinstance(body, dict) or body.get("predicted_quality") not in settings.quality_labels:
        raise PredictionError(
            "Unexpected API response: predicted_quality must be one of the four categories."
        )
    version = body.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise PredictionError("Unexpected API response: model_version must be a non-empty string.")
    return EvaluationResult(
        quality_label=body["predicted_quality"], simulated=False, model_version=version
    )
