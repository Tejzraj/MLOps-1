FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    FRONTEND_MODE=mock \
    BACKEND_URL=http://backend:8000

WORKDIR /app

# Install only the frontend runtime dependencies, never the root ML requirements.
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN python -m pip install --no-cache-dir -r frontend/requirements.txt \
    && useradd --create-home --uid 10001 app

# Explicit copies keep tests, secrets and ML files out of the runtime image.
COPY --chown=app:app frontend/__init__.py frontend/app.py frontend/api_client.py \
    frontend/config.py frontend/question_bank.py /app/frontend/
COPY --chown=app:app frontend/params.yaml frontend/questions.yaml /app/frontend/

USER app
EXPOSE 8501

# Liveness of Streamlit only; this does not assert backend/model readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3); assert r.status == 200"

CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
