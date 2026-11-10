# Interview Studio frontend

Python 3.11+ and Streamlit. All new code and configuration live in `frontend/`;
the application does not load a model or modify any ML pipeline artifacts.

## Fedora: install and run

Run these commands in Konsole from the repository root. The virtual environment
keeps frontend dependencies separate from the Phase 1 environment.

```bash
cd /home/likithshashidhar/MLOps
git branch --show-current
# Expected: phase-2-frontend-devops. Do not work on main.
python3 --version
python3 -m venv frontend/.venv
source frontend/.venv/bin/activate
python -m pip install -r frontend/requirements-dev.txt
python -m pytest -c frontend/pytest.ini frontend/tests
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

Open **http://localhost:8501**. Stop with **Ctrl+C**. If Python or pip is missing,
install it with `sudo dnf install python3 python3-pip` first. Always launch from
the repository root using `python -m streamlit` so the frontend package is importable.
For runtime only, install `frontend/requirements.txt` instead of the dev file.

## Demo walkthrough

1. Read Home, then choose **Practice** in the sidebar.
2. Keep **Mock · simulated** selected. A persistent banner identifies mock mode.
3. Choose a topic and type an answer. Blank/whitespace-only submissions are rejected.
4. Choose a **Simulated result category**, then click **Evaluate Answer**.
5. Review the result and side-by-side reference comparison. Try each category:
   Poor, Needs Improvement, Good, Excellent.
6. Edit your answer or change questions: an old result is cleared. Each question
   keeps its own draft within the current session.
7. Select **API · FastAPI** and evaluate to see real connection errors if the
   backend is unavailable. The app never falls back to a simulated result.

Mock responses are user-selected UI fixtures: they do not depend on answer
content, represent human labels, or measure performance. No numeric scores,
confidence values, agreement values or metrics are generated. Curated questions
and educational references are not Mohler records or training data.

## Configuration and backend integration

`frontend/params.yaml` contains frontend settings and the question bank path.
Paths inside it resolve relative to that file. Shared root configuration is not
used or modified. Optional environment overrides:

| Variable | Purpose |
| --- | --- |
| `FRONTEND_PARAMS` | Alternative frontend params.yaml location |
| `FRONTEND_MODE` | Startup mode: `mock` or `api` (sidebar can explicitly change it) |
| `BACKEND_URL` | Backend base URL, including an optional API prefix |
| `BACKEND_TIMEOUT_SECONDS` | Positive, finite request timeout in seconds |

Once your teammate starts FastAPI, stop Streamlit and run:

```bash
source frontend/.venv/bin/activate
export FRONTEND_MODE=api
export BACKEND_URL=http://localhost:8000
export BACKEND_TIMEOUT_SECONDS=15
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

The URL is reached by the Streamlit server, not by the browser. Requests are
server-side, so browser CORS is not needed for this client. The timeout bounds
connection/read waits; a timed-out request is not retried automatically.

**Proposed contract — confirm with the backend teammate before integration.**
The backend is not yet available, so a live integration has not been verified.
`api_client.py` is the single place to adapt serialization if her schema differs.
The configured endpoint defaults to `POST /predict`:

```json
{
  "question_id": "practice-stacks",
  "question": "What is the difference between a stack and a queue? Give a use case for each.",
  "reference_answer": "The full curated reference answer for the selected question",
  "candidate_answer": "The candidate's submitted answer"
}
```

Expected successful JSON response:

```json
{"quality_label": "Good"}
```

`quality_label` must be exactly Poor, Needs Improvement, Good or Excellent.
Extra response fields are ignored. Integer class IDs and numeric scores are not
converted into categories by the frontend. Errors include timeout, unreachable
backend, non-2xx status, redirect, invalid JSON and invalid response schema.
Backend response bodies and candidate answers are not written to application logs.
Answers remain in the Streamlit session; the frontend does not persist them to disk.

## Modules and validation

- `app.py`: navigation, presentation, drafts, and result lifecycle.
- `config.py`: validated frontend configuration and deployment overrides.
- `question_bank.py` / `questions.yaml`: curated question content and loader.
- `api_client.py`: HTTP transport, response validation, explicit mock fixtures.
- `tests/test_frontend.py`: UI smoke tests with Streamlit AppTest plus isolated
  client tests; no running backend or model is required.

```bash
python -m pytest -c frontend/pytest.ini frontend/tests
# Existing Phase 1 tests: use the repository's ML environment, not the frontend venv.
.venv/bin/python -m pytest -o addopts='' tests
```

Frontend tests use their own pytest configuration so Phase 1 coverage settings
and ML dependencies are not required. Docker, CI/CD, deployment, authentication,
history and model evaluation are outside this frontend-only change.
