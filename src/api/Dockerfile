FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements-api.txt requirements-inference.txt /app/
RUN python -m pip install -r requirements-api.txt \
    && useradd --create-home --uid 10001 app
COPY --chown=app:app params.yaml /app/
COPY --chown=app:app src/__init__.py /app/src/
COPY --chown=app:app src/api/ /app/src/api/
COPY --chown=app:app src/inference/__init__.py src/inference/predict.py /app/src/inference/
COPY --chown=app:app src/utils/__init__.py src/utils/phase2_common.py /app/src/utils/
# Compatibility implementations are loaded only from the verified private bundle.
COPY --chown=app:app src/features/__init__.py /app/src/features/
COPY --chown=app:app src/models/__init__.py /app/src/models/
USER app
EXPOSE 8000
# Bind to the container interface; host/port/worker settings otherwise come from params.yaml.
CMD ["python", "src/api/container.py"]
