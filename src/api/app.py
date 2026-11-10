"""Local FastAPI adapter for the immutable Phase 2 B0 inference bundle."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from src.api.model_runtime import load_predictor
from src.inference.predict import clean_record
from src.utils.phase2_common import ROOT, load_params

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class APISettings:
    """Non-secret application settings loaded from params with one path override."""
    bundle: Path
    manifest_filename: str
    expected_manifest_sha256: str
    expected_version: str
    expected_configuration: str
    limitations: tuple[str, ...]
    host: str
    port: int
    workers: int
    log_level: str


def read_settings() -> APISettings:
    """Resolve INTERVIEW_MODEL_BUNDLE relative to the worktree when not absolute."""
    cfg = load_params()['phase2_api']
    bundle = Path(os.environ.get(cfg['bundle_environment_variable'], cfg['bundle_path']))
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    return APISettings(bundle.resolve(), cfg['manifest_filename'], cfg['expected_manifest_sha256'], cfg['expected_model_version'], cfg['expected_configuration'], tuple(cfg['public_limitations']), cfg['host'], cfg['port'], cfg['workers'], cfg['log_level'])


class PredictionRequest(BaseModel):
    """Strict three-string HTTP contract; wrapper owns actual training-time cleaning."""
    model_config = ConfigDict(extra='forbid', strict=True)
    question: StrictStr
    reference_answer: StrictStr
    candidate_answer: StrictStr

    @field_validator('question', 'reference_answer', 'candidate_answer')
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Reject empty/whitespace inputs without altering accepted text."""
        if not value.strip():
            raise ValueError('must not be empty or whitespace-only')
        return value


class PredictionResponse(BaseModel):
    """Only information actually produced by the classifier is returned."""
    predicted_quality: str
    model_version: str


class HealthResponse(BaseModel):
    """Application liveness and startup model readiness are separate signals."""
    status: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    """Public model information deliberately excludes local/provenance metadata."""
    model_version: str
    supported_classes: list[str]
    question_used_as_feature: bool
    limitations: list[str]


def unavailable() -> HTTPException:
    """Return a clear operational failure without paths, exceptions or secrets."""
    return HTTPException(status_code=503, detail={'code': 'model_unavailable', 'message': 'Model bundle unavailable or invalid; check server setup.'})


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Construct an app that attempts model loading exactly once per lifespan."""
    settings = settings or read_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Verify and load once, entering degraded mode on any startup failure."""
        application.state.predictor = None
        try:
            application.state.predictor = load_predictor(settings.bundle, settings.manifest_filename, settings.expected_manifest_sha256, settings.expected_version, settings.expected_configuration)
            LOGGER.info('Model ready; version=%s', settings.expected_version)
        except Exception as error:
            LOGGER.error('Model startup rejected (%s); model endpoints unavailable', type(error).__name__)
        try:
            yield
        finally:
            application.state.predictor = None

    application = FastAPI(title='Interview Answer Quality API', version=settings.expected_version, lifespan=lifespan)
    application.state.predictor = None

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        """Exclude submitted answer values/context from validation responses."""
        errors = [{'loc': item['loc'], 'msg': item['msg'], 'type': item['type']} for item in error.errors()]
        return JSONResponse(status_code=422, content={'detail': errors})

    @application.get('/health', response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report application liveness without claiming unavailable models are ready."""
        return HealthResponse(status='ok', model_ready=application.state.predictor is not None)

    @application.get('/model/info', response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        """Return only reviewed, public model identity/classes/limitations."""
        predictor = application.state.predictor
        if predictor is None:
            raise unavailable()
        return ModelInfoResponse(model_version=settings.expected_version, supported_classes=[predictor.labels[key] for key in sorted(predictor.labels)], question_used_as_feature=False, limitations=list(settings.limitations))

    @application.post('/predict', response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Use the frozen wrapper directly, without fitting, feedback or score invention."""
        predictor = application.state.predictor
        if predictor is None:
            raise unavailable()
        record = payload.model_dump()
        try:
            # Detect br-only candidate input using the SAME frozen cleaning function.
            clean_record(record, predictor.preprocessing)
        except ValueError:
            raise HTTPException(status_code=422, detail={'code': 'invalid_answer', 'message': 'Answer is empty or invalid after preprocessing.'}) from None
        try:
            result = predictor.predict_one(record)
            if result['quality_label'] not in predictor.labels.values():
                raise RuntimeError('Unsupported classifier output')
        except Exception as error:
            LOGGER.error('Prediction failed (%s)', type(error).__name__)
            raise HTTPException(status_code=500, detail={'code': 'prediction_failed', 'message': 'Prediction could not be completed.'}) from None
        return PredictionResponse(predicted_quality=result['quality_label'], model_version=settings.expected_version)

    return application


app = create_app()


def main() -> None:
    """Run a local server using params.yaml host/port settings; no reloader or deploy."""
    import uvicorn
    settings = read_settings()
    if settings.workers != 1:
        raise ValueError('Local launcher supports one worker; use reviewed ASGI worker settings separately')
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level, access_log=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
