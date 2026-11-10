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
and ML dependencies are not required.

## Docker demonstration on Fedora

The image uses Python 3.11 slim, frontend runtime requirements only, an unprivileged
user, and port 8501. The build context permits only the required frontend files.
There are no dataset/model mounts, build tools, root ML dependencies or API keys.
Dependencies retain the version ranges already in `frontend/requirements.txt`;
they are not a frozen lockfile. The Python image tag also receives upstream updates.

Docker is not installed on the development laptop. **Image builds, Compose
execution and container health checks are unverified locally.** The CI workflow
is prepared to run these checks, but no remote CI success is claimed until an
approved push produces a successful Actions run. Deployment is not implemented.

Install Docker Engine and the Compose plugin on a maintained Fedora version:

```bash
sudo dnf install dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
sudo docker run --rm hello-world
```

These use the current DNF5 instructions from the
[official Fedora installation guide](https://docs.docker.com/engine/install/fedora/).
If conflicting Docker packages are present, follow that guide before installation.
The commands below use `sudo` so changing Docker group membership is unnecessary.
Installation commands are instructions for you; they have not been executed here.

Build and launch the frontend-only demo from the repository root:

```bash
cd /home/likithshashidhar/MLOps
sudo docker compose -f compose.yaml config --quiet
sudo docker compose -f compose.yaml build --pull frontend
sudo docker compose -f compose.yaml up -d --wait --wait-timeout 120 frontend
sudo docker compose -f compose.yaml ps
curl --fail http://127.0.0.1:8501/_stcore/health
sudo docker compose -f compose.yaml logs --tail=100 frontend
```

Open **http://localhost:8501**. Compose defaults to mock mode. The HTTP health
endpoint reports Streamlit liveness, not successful grading or backend readiness.
The image checks this endpoint with Python's standard library, without curl.
See [Streamlit's Docker guide](https://docs.streamlit.io/deploy/tutorials/docker).
The host port is bound to loopback for the local demo. No Fedora firewall or
SELinux changes are needed; there are no bind mounts requiring relabelling.

Stop the demo:

```bash
sudo docker compose -f compose.yaml down
```

Equivalent standalone image commands (stop Compose first to free port 8501):

```bash
sudo docker build --pull -f frontend/Dockerfile -t interview-frontend:local .
sudo docker run --rm --name interview-frontend -p 127.0.0.1:8501:8501 interview-frontend:local
# In another terminal:
sudo docker inspect --format '{{.State.Health.Status}}' interview-frontend
sudo docker stop interview-frontend
```

If startup fails, inspect `docker compose logs` and check for another process on
port 8501. A healthy container can still show configuration or API errors in the
UI; smoke tests check the application behavior separately.

## Future FastAPI integration

Repository inspection found no FastAPI entrypoint or backend Dockerfile. The
default `compose.yaml` therefore runs only the frontend. Its `BACKEND_URL` is
`http://backend:8000`, using the future Compose service name. `localhost` inside
the frontend container would refer to the frontend itself.

`frontend/compose.backend.example.yaml` is an **opt-in integration template**.
It requires `BACKEND_IMAGE`, a real image supplied/built by the backend teammate.
It uses that image's existing entrypoint; it does not guess a Python module,
model location, W&B credentials or dependency file. Before using it, confirm:

- The image starts FastAPI on `0.0.0.0:8000` and has all required model assets.
- Its `/predict` request/response match the contract above, or adapt `api_client.py`.
- Any backend environment, secrets or mounts are configured with the teammate.
- Its readiness endpoint is known before adding a backend healthcheck. The
  template currently waits for container startup only; early requests may fail
  visibly while the model loads. There is no silent mock fallback.

After that image is available locally, enter its actual tag at the prompt:

```bash
read -r -p 'Verified backend image tag: ' BACKEND_IMAGE
sudo env BACKEND_IMAGE="$BACKEND_IMAGE" docker compose -f compose.yaml -f frontend/compose.backend.example.yaml config --quiet
sudo env BACKEND_IMAGE="$BACKEND_IMAGE" docker compose -f compose.yaml -f frontend/compose.backend.example.yaml up -d --build --wait --wait-timeout 120
sudo env BACKEND_IMAGE="$BACKEND_IMAGE" docker compose -f compose.yaml -f frontend/compose.backend.example.yaml logs --tail=100
# Stop both services:
sudo env BACKEND_IMAGE="$BACKEND_IMAGE" docker compose -f compose.yaml -f frontend/compose.backend.example.yaml down
```

The template switches the frontend to API mode, connects both services on the
default Compose network, and keeps port 8000 internal. It is not a working
backend implementation and has not been integration-tested.

For the default frontend-only stack, optional overrides are
`COMPOSE_FRONTEND_MODE`, `COMPOSE_BACKEND_URL` and `COMPOSE_BACKEND_TIMEOUT_SECONDS`.
Pass them explicitly through `sudo env`, for example:

```bash
sudo env COMPOSE_FRONTEND_MODE=api COMPOSE_BACKEND_URL=http://backend:8000 COMPOSE_BACKEND_TIMEOUT_SECONDS=30 docker compose -f compose.yaml up -d frontend
```

That command requires a reachable backend; it does not create one. Separate
Compose override names prevent a host development setting such as
`BACKEND_URL=http://localhost:8000` from leaking into container configuration.

## GitHub Actions CI

`.github/workflows/frontend-ci.yml` runs on relevant pushes to
`phase-2-frontend-devops` or `main`, pull requests, and manual dispatch. It installs
only frontend test dependencies on Python 3.11, runs the smoke suite, validates
both Compose files, builds the frontend image, and waits for its health check.
The dummy backend image tag used for config validation is never built or pulled.
The workflow grants only `contents: read`, disables persisted checkout credentials,
and has no registry login, custom secrets, image publishing or deployment steps.
See GitHub's [token permissions guidance](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).

Push only after review/approval, then inspect the **Frontend CI** run under the
repository's Actions tab. A workflow file alone does not establish a passing build
or continuous deployment. Authentication, history and model evaluation remain
outside this work.
