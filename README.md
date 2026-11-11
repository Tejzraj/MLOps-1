## Phase 2 ML/backend contribution

Start with **[the teammate setup and Streamlit/API handoff](docs/phase2/teammate_handoff.md)**. It covers the verified private B0 v1.0.0 ZIP, checksum verification, isolated API installation and INTERVIEW_MODEL_BUNDLE. Model binaries and datasets are not in Git.

B0 is privately registered in W&B, but teammate Registry Viewer access is blocked by the current plan; use the private ZIP temporarily and do not run cloud retrieval yet. Nine historical experiment runs remain offline. Phase 1 source, DVC pipeline and MLflow records are preserved. This contribution adds experiments, Pytest, GX validation and local FastAPI inference; Docker, CI/CD, monitoring and deployment are outside this contribution.

---

<div align="center">

<img src="docs/assets/banner.svg" alt="AI-Based Interview Preparation Assistant MLOps Banner" width="100%" />

# AI-Based Interview Preparation Assistant
### Production-Grade MLOps Pipeline & Continuous Delivery Platform

<p align="center">
  <strong>Data</strong> ➔ 
  <strong>Version</strong> ➔ 
  <strong>EDA</strong> ➔ 
  <strong>Features</strong> ➔ 
  <strong>Train</strong> ➔ 
  <strong>Track</strong> ➔ 
  <strong>Register</strong> ➔ 
  <strong>Serve</strong> ➔ 
  <strong>Monitor</strong> ➔ 
  <strong>Retrain</strong>
</p>

<!-- Technology Badges -->
[![Python 3.11+](https://img.shields.io/badge/Python-3.11-FF6B00?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-v3.67_Data_Ops-FF8C00?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking_&_Registry-FFA726?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-FF6B00?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving_(Phase_2)-1A1A1A?style=for-the-badge&logo=fastapi&logoColor=FF8C00)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containers_(Phase_2)-1A1A1A?style=for-the-badge&logo=docker&logoColor=FF8C00)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Actions_(Phase_2)-1A1A1A?style=for-the-badge&logo=github-actions&logoColor=FF8C00)](https://github.com/features/actions)
[![Evidently](https://img.shields.io/badge/Evidently-Drift_Ops_(Phase_3)-1A1A1A?style=for-the-badge&logo=datadog&logoColor=FF8C00)](https://www.evidentlyai.com/)

<!-- Project & Academic Badges -->
[![Course](https://img.shields.io/badge/Course-AI252TA%3A%20MLOps-FF6B00?style=flat-square&logo=bookstack&logoColor=white)](https://github.com/Tejzraj/MLOps)
[![Semester](https://img.shields.io/badge/Semester-V-FF8C00?style=flat-square)](https://github.com/Tejzraj/MLOps)
[![Status](https://img.shields.io/badge/Project_Stage-Phase_1%20(Development_%26_Reproducibility)-FFA726?style=flat-square)](docs/PROJECT_STATUS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-0D1117?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![SDG 4](https://img.shields.io/badge/UN%20SDG-4%20Quality%20Education-FF6B00?style=flat-square)](https://sdgs.un.org/goals/goal4)
[![SDG 8](https://img.shields.io/badge/UN%20SDG-8%20Decent%20Work%20%26%20Economic%20Growth-FF8C00?style=flat-square)](https://sdgs.un.org/goals/goal8)

</div>

---

## Executive Summary

The **AI-Based Interview Preparation Assistant** is an end-to-end Machine Learning Operations (MLOps) platform designed to evaluate and coach candidates on technical and behavioral interview responses. The system analyzes interview question-answer pairs, assesses answer depth and correctness, and classifies response quality into a 4-tier standardized grading rubric to provide actionable, automated feedback.

Engineered for academic course **AI252TA (Machine Learning Operations - Semester V)**, this repository implements industry-standard MLOps practices: **versioned data pipelines with DVC**, **reproducible experiment tracking with MLflow**, **automated quality validation**, **containerized inference serving**, and **drift-triggered continuous retraining**.

> [!NOTE]
> **Status:** Phase 1 (Development & Reproducibility) is finished through test evaluation; see [`reports/phase1/phase1_report.md`](reports/phase1/phase1_report.md). Phase 2 ML experiments, data validation and local FastAPI serving are implemented on phase-2-model-api; no model is deployed. See the Phase 2 handoff below. Faculty review of the dataset change is pending.

---

## Quick Start (Phase 1)

**Prerequisites:** Python 3.11, Git, `make`.

```bash
# 1. Clone and enter the repository
git clone https://github.com/Tejzraj/MLOps.git
cd MLOps

# 2. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Allow MLflow file store (required for local tracking)
export MLFLOW_ALLOW_FILE_STORE=true

# 4. Download the Mohler ASAG dataset (open_ended split) from Hugging Face
python src/data/download_mohler.py

# 5. Run the full DVC pipeline (data → split → features → train → evaluate)
dvc repro

# 6. Run tests
python -m pytest -q
```

**Dataset:** Mohler ASAG (Short Answer Grading) — `nkazi/MohlerASAG` on Hugging Face, subset `cleaned`, split `open_ended`, 2,273 rows, 81 questions, CC BY 4.0 license (Mohler, Bunescu & Mihalcea 2011). The `data/external/mohler/open_ended.parquet` is DVC-tracked; teammates must run the download command above because the DVC remote (`~/dvc_storage`) is local-only and `dvc pull` will not work for them.

**Files not in Git (enforced by `.gitignore`):** `data/`, `models/*.joblib`, `mlruns/`, `.env`. Commit only code, config, and `dvc.lock`/`*.dvc` pointer files.

> [!NOTE]
> **Phase 1 is complete through evaluation.** The quickstart runs the full pipeline through test evaluation. See [`reports/phase1/phase1_report.md`](reports/phase1/phase1_report.md) for results, limitations, and reproducibility details. Phase 2 (serving, CI/CD) is planned next.

---

## Real-Time Milestone Dashboard

```text
========================================================================================
                                 PROJECT PROGRESSION
========================================================================================
Phase 0: Foundation & Environment    [████████████████████] 100% (Completed)
Phase 1: Development & Reproduce     [████████████████░░]  80% (In Progress)
Phase 2: Serving & CI/CD Pipeline    [░░░░░░░░░░░░░░░░░░░░]   0% (Planned)
Phase 3: Observability & Governance  [░░░░░░░░░░░░░░░░░░░░]   0% (Planned)
========================================================================================
CURRENT ACTIVE PHASE : PHASE 1 — DEVELOPMENT & REPRODUCIBILITY
NEXT MILESTONE       : PHASE 1 REPORT & PHASE 2 PLANNING
========================================================================================
```

---

## Phase 1 Lifecycle & Dataset Execution Status

```text
Phase 1 Execution Flow:
Dataset Specification ──► Dataset Acquisition ──► EDA ──► Feature Engineering ──► Baseline Model ──► MLflow Tracking
    [COMPLETED]             [COMPLETED]         [COMPLETED]    [COMPLETED]            [COMPLETED]           [COMPLETED]
```

| Lifecycle Component | Status | Artifact / Documentation Reference |
| :--- | :---: | :--- |
| **Dataset Specification** | `[x] COMPLETED` | [`docs/dataset/DATASET_SPECIFICATION.md`](docs/dataset/DATASET_SPECIFICATION.md) |
| **Labeling Guidelines & Rubric** | `[x] COMPLETED` | [`docs/dataset/LABELING_GUIDELINES.md`](docs/dataset/LABELING_GUIDELINES.md) |
| **Data Collection Protocol** | `[x] COMPLETED` | [`docs/dataset/DATA_COLLECTION.md`](docs/dataset/DATA_COLLECTION.md) |
| **Faculty Approval Dossier** | `[x] COMPLETED` | [`docs/dataset/DATASET_APPROVAL.md`](docs/dataset/DATASET_APPROVAL.md) |
| **Faculty Committee Review** | `[ ] PENDING` | pending |
| **Dataset Acquisition & DVC Tracking** | `[x] COMPLETED` | `data/raw/interview_data.parquet` (DVC-tracked via `dvc.lock`) |
| **Exploratory Data Analysis (EDA)** | `[x] COMPLETED` | [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) |
| **Feature Engineering Pipeline** | `[x] COMPLETED` | [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb), `src/features/build_features.py` |
| **Baseline Model Training (Logistic Regression)** | `[x] COMPLETED` | [`notebooks/04_baseline_models.ipynb`](notebooks/04_baseline_models.ipynb), `src/models/train_baseline.py` |
| **Comparison Model Training (Random Forest)** | `[x] COMPLETED` | `src/models/train_compare.py`, `models/rf_model.joblib` |
| **MLflow Experiment Tracking** | `[x] COMPLETED` | `mlruns/`, experiment: `interview-answer-quality-baseline` |
| **Full Pipeline Reproducibility (`dvc repro`)** | `[x] COMPLETED` | Verified: two consecutive runs skip all stages |
| **Phase 1 Synthesis Report** | `[x] COMPLETED` | [`reports/phase1/phase1_report.md`](reports/phase1/phase1_report.md) |

---

## Machine Learning Problem Formulation

**Dataset Used:** Mohler Short-Answer Grading (ASAG) corpus — `nkazi/MohlerASAG` on Hugging Face, subset `cleaned`, split `open_ended`, 2,273 candidate answers across 81 questions from one introductory data-structures course (University of North Texas), CC BY 4.0 license. Replaces the earlier synthetic interview data specification.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       Interview Question        │   +   │        Candidate Answer         │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    ML Feature Extraction      │
                      │  (TF-IDF / Length / Reference  │
                      │   Similarity / 2,824 features) │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Answer Quality Classification │
                      └───────────────┬───────────────┘
                                      │
          ┌──────────────┬────────────┴────────────┬──────────────┐
          ▼              ▼                         ▼              ▼
       Class 0        Class 1                    Class 2        Class 3
      [ Poor ]  [ Needs Improvement ]           [ Good ]     [ Excellent ]
```

### Problem Definition
- **Input:** Interview question (`question`), reference answer (`reference_answer`), candidate answer (`candidate_answer`).
- **Target Output:** 4-class ordinal quality classification (`quality_label` $\in \{0, 1, 2, 3\}$) derived from `score_avg` (mean of two human graders, 0–5) using thresholds: Poor $\le$ 2.0, Needs Improvement $\le$ 3.5, Good $\le$ 4.5, Excellent $\le$ 5.0.
- **Evaluation Criteria (from Mohler dataset human scores):**
  - **Class 0 (Poor):** score_avg in [0.0, 2.0)
  - **Class 1 (Needs Improvement):** score_avg in [2.0, 3.5)
  - **Class 2 (Good):** score_avg in [3.5, 4.5)
  - **Class 3 (Excellent):** score_avg in [4.5, 5.0]

---

## End-to-End MLOps Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["Data Engineering & Versioning (Phase 1)"]
        Raw["Raw QA Dataset (data/raw)"] --> DVC["DVC Data Tracking"]
        DVC --> Preproc["Preprocessing & Splitting (src/data)"]
        Preproc --> Splits["Train / Val / Test Splits (data/processed)"]
    end

    subgraph Experimentation["Modeling & Tracking (Phase 1)"]
        Splits --> Features["Feature Engineering (src/features)"]
        Features --> Train["Baseline & Comparison Training (src/models)"]
        Train <--> MLflow["MLflow Experiment Tracking (Params, Metrics, Artifacts)"]
        Train --> Eval["Model Evaluation & Error Profiling (src/evaluation)"]
    end

    subgraph CI_CD_Deploy["Productionization & Deployment (Phase 2 - Planned)"]
        Eval --> Registry["MLflow Model Registry"]
        Registry --> API["FastAPI REST Inference Service"]
        API --> Docker["Docker Containerization"]
        Docker --> Actions["GitHub Actions CI/CD"]
        Actions --> Prod["Production API Endpoint"]
    end

    subgraph Monitoring_Loop["Observability & Continuous ML (Phase 3 - Planned)"]
        Prod --> Logs["Prediction & Telemetry Logging"]
        Logs --> Evidently["Evidently AI Drift Analysis"]
        Evidently --> Grafana["Prometheus & Grafana Dashboards"]
        Grafana --> DriftCheck{"Data/Concept Drift Detected?"}
        DriftCheck -- "Yes: Automated Trigger" --> Train
        DriftCheck -- "No: Stable" --> MonitorState["Continuous Monitoring"]
    end

    classDef orange fill:#FF6B00,stroke:#0D1117,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#161B22,stroke:#FF8C00,stroke-width:1.5px,color:#FFFFFF;
    classDef planned fill:#1F242C,stroke:#484F58,stroke-width:1px,stroke-dasharray: 4 4,color:#8B949E;

    class Raw,Splits,Features,Train,Eval orange;
    class DVC,Preproc,MLflow dark;
    class Registry,API,Docker,Actions,Prod,Logs,Evidently,Grafana,DriftCheck,MonitorState planned;
```

---

## Technology Stack

| Domain | Tool / Technology | Status | Role in Pipeline |
| :--- | :--- | :---: | :--- |
| **Language** | **Python 3.11+** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Core runtime environment across entire pipeline |
| **Data Ops** | **Pandas, NumPy** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Numerical transformations, dataframes, and sanitization |
| **Data Versioning** | **DVC (Data Version Control)** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Git-independent dataset versioning and pipeline orchestration |
| **Machine Learning** | **Scikit-learn** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Baseline classifiers, TF-IDF vectorization, evaluation metrics |
| **Prototyping** | **Jupyter Lab / Notebooks** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Exploratory data analysis, error inspection, visualization |
| **Experimentation** | **MLflow** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Experiment tracking, hyperparameter logging, model registry |
| **Testing** | **PyTest & Coverage** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Unit, integration, and model invariance testing (25 tests passing) |
| **Serving API** | **FastAPI & Uvicorn** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | High-performance REST service for real-time inference |
| **Containerization** | **Docker** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | Multi-stage, reproducible microservice container builds |
| **CI/CD** | **GitHub Actions** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | Automated linting, test execution, container push |
| **Observability** | **Evidently AI** | ![Planned P3](https://img.shields.io/badge/Phase_3-1A1A1A?style=flat-square) | Text drift detection, data quality, and model degradation |
| **Metrics & UI** | **Prometheus & Grafana** | ![Planned P3](https://img.shields.io/badge/Phase_3-1A1A1A?style=flat-square) | System metric scraping and real-time operations dashboard |

---

## Detailed Multi-Phase Roadmap

### Legend
- `[x]` Completed
- `[~]` In Progress
- `[ ]` Not Started / Planned

```
Phase 0 ───► Phase 1 ───► Phase 2 ───► Phase 3 ───► Phase 4 ───► Phase 5
Foundation   Dev & Repro  Production   Observability Continuous ML Governance
 [100%]         [0%]         [0%]         [0%]         [0%]         [0%]
```

<details open>
<summary><strong>Phase 0 — Project Foundation & Architecture (COMPLETED)</strong></summary>

- [x] Repository initialized with clean remote tracking on `main`
- [x] Full MLOps directory hierarchy initialized (`src/`, `data/`, `notebooks/`, `docs/`, `tests/`)
- [x] Central parameter management configured (`params.yaml`, `configs/config.yaml`)
- [x] Dependency management configured (`requirements.txt`, `requirements-dev.txt`, `pyproject.toml`)
- [x] DVC pipeline blueprint initialized (`dvc.yaml`)
- [x] Developer automation scripts added (`Makefile`)
- [x] Project governance and licensing created (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`)
- [x] Custom SVG asset and architecture documentation authored
</details>

<details open>
<summary><strong>Phase 1 — Development & Reproducibility (COMPLETED)</strong></summary>

- [x] Dataset specification, labeling guidelines, and data collection protocol authored (`docs/dataset/`)
- [x] Faculty review dossier prepared (`docs/dataset/DATASET_APPROVAL.md`)
- [ ] Faculty committee review and formal approval (pending)
- [x] DVC initialized with pipeline blueprint (`dvc.yaml`, `dvc.lock`)
- [x] Dataset acquired: Mohler ASAG (`nkazi/MohlerASAG`), `open_ended` split, 2,273 answers, 81 questions, CC BY 4.0
- [x] Dataset versioned with DVC (`data/raw/interview_data.parquet` tracked in `dvc.lock`)
- [x] Data quality checks and schema verification implemented (`src/data/load_data.py`, `src/data/preprocess.py`)
- [x] Exploratory Data Analysis completed (`notebooks/01_data_exploration.ipynb`, executed)
- [x] Data preprocessing and cleaning pipeline implemented (`src/data/preprocess.py`)
- [x] Grouped (GroupShuffleSplit) train / validation / test partitioning executed (grouped by `question_id`, 56/8/17 questions)
- [x] Feature engineering pipeline constructed: TF-IDF (up to 5,000 terms, 1-2-grams; 2,818 kept) + 5 length features + 1 reference similarity = 2,824 features (`src/features/build_features.py`)
- [x] Baseline Logistic Regression model trained and logged (`src/models/train_baseline.py`, `models/baseline_model.joblib`)
- [x] Comparison Random Forest model trained and evaluated (`src/models/train_compare.py`, `models/rf_model.joblib`)
- [x] MLflow experiment tracking integrated (hyperparameters, metrics, confusion matrices in `mlruns/`)
- [x] Model selection rule implemented and applied (CV f1_macro tie → validation f1_macro tie-break → Random Forest selected)
- [x] Test evaluation completed (Logistic Regression macro-F1 0.3669, Random Forest macro-F1 0.3925, majority baseline 0.1868)
- [x] Full pipeline reproducibility verified using `dvc repro` (two consecutive runs skip all stages)
- [x] Phase 1 synthesis report authored in `reports/phase1/phase1_report.md`
</details>

<details>
<summary><strong>Phase 2 — Productionization & Deployment (CURRENT)</strong></summary>

- [ ] PyTest test suite implemented (unit tests, integration tests, schema tests)
- [ ] Model behavioral and linguistic invariance tests added
- [ ] Model risk, edge-case, and latency profiling conducted
- [ ] MLflow Model Registry integration with stage transitions (`Staging`, `Production`)
- [ ] FastAPI REST inference service built with Pydantic payload validation
- [ ] Multi-stage Docker container built and optimized for inference
- [ ] GitHub Actions CI pipeline configured (lint, test, build)
- [ ] Continuous Deployment workflow configured to cloud environment
</details>

<details>
<summary><strong>Phase 3 — Observability & Continuous Monitoring (CURRENT)</strong></summary>

- [ ] Structured request/response logging framework deployed
- [ ] Evidently AI integrated for data drift and vocabulary shift detection
- [ ] Prometheus metrics exporter configured for inference latency and class counts
- [ ] Grafana / Power BI operational dashboard configured
- [ ] Simulated production data generation script created
- [ ] Data and concept drift experiment simulated and documented
</details>

<details>
<summary><strong>Phase 4 — Continuous ML & Automated Retraining (PLANNED)</strong></summary>

- [ ] Automated drift-triggered retraining orchestrator designed
- [ ] Shadow model deployment and automated candidate comparison evaluation
- [ ] Automated rollback mechanism on performance degradation
- [ ] Continuous model artifact promotion into Model Registry
</details>

<details>
<summary><strong>Phase 5 — Governance, Compliance & Auditability (PLANNED)</strong></summary>

- [ ] Mitchell et al. Model Card authored (`docs/governance/model_card.md`)
- [ ] Dataset provenance and ethical assessment document completed
- [ ] Fairness and dialectal bias evaluation report published
- [ ] Comprehensive academic viva demonstration and reproducible run documentation
</details>

---

## Repository Structure

```text
MLOps/
│
├── README.md                          # Project documentation and architectural overview
├── LICENSE                            # Open-source MIT License
├── CONTRIBUTING.md                    # Developer guidelines and Git workflows
├── CODE_OF_CONDUCT.md                 # Contributor Covenant Code of Conduct
├── CHANGELOG.md                       # Release notes and milestone log
├── .gitignore                         # Comprehensive Python, DVC, and artifact ignores
├── .gitattributes                     # LF normalization and binary file tracking
├── requirements.txt                   # Production ML & MLOps dependencies
├── requirements-dev.txt               # Development, linting, and testing dependencies
├── pyproject.toml                     # Modern build system, ruff, black, and pytest configuration
├── Makefile                           # Developer automation commands
├── dvc.yaml                           # DVC reproducible pipeline stages
├── params.yaml                        # Centralized pipeline hyperparameters
│
├── configs/
│   └── config.yaml                    # Global system and logging configurations
│
├── data/
│   ├── raw/                           # Raw immutable QA datasets (DVC-tracked)
│   ├── interim/                       # Cleaned intermediate QA data
│   ├── processed/                     # Final train/val/test splits
│   └── README.md                      # Dataset specification and governance
│
├── notebooks/                         # Interactive Jupyter scaffolds for Phase 1
│   ├── 01_data_exploration.ipynb      # EDA and quality distribution analysis
│   ├── 02_data_preprocessing.ipynb    # Text sanitization and grouped splitting
│   ├── 03_feature_engineering.ipynb   # TF-IDF and syntactic feature extraction
│   └── 04_baseline_models.ipynb       # Baseline models & MLflow experiment runs
│
├── src/                               # Modular source package
│   ├── __init__.py
│   ├── data/                          # Ingestion, validation, and cleaning logic
│   │   └── __init__.py
│   ├── features/                      # Feature extractors and text transformers
│   │   └── __init__.py
│   ├── models/                        # Baseline modeling and training scripts
│   │   └── __init__.py
│   ├── evaluation/                    # Metric computation and evaluation plots
│   │   └── __init__.py
│   └── utils/                         # Shared utilities, YAML loaders, logging
│       └── __init__.py
│
├── models/                            # Serialized model weights (.joblib, DVC-tracked)
│   └── .gitkeep
│
├── reports/                           # Generated evaluation outputs and documentation
│   ├── figures/                       # Confusion matrices and distribution plots
│   │   └── .gitkeep
│   └── phase1/
│       └── README.md                  # Phase 1 evaluation report placeholder
│
├── tests/                             # PyTest test suite
│   └── README.md                      # Testing architecture and execution guide
│
├── mlruns/                            # Local MLflow tracking experiments (ignored by Git)
│   └── .gitkeep
│
├── docs/                              # Technical documentation hub
│   ├── PROJECT_STATUS.md              # Live milestone status dashboard
│   ├── assets/
│   │   └── banner.svg                 # Neon orange futuristic control center banner
│   ├── architecture/README.md         # System design and component interactions
│   ├── dataset/README.md              # Dataset approval and annotation rubrics
│   ├── experiments/README.md          # Experiment tracking and MLflow protocol
│   ├── mlops/README.md                # CI/CD, Docker, and reproducibility specifications
│   └── governance/README.md           # Model cards, ethics, and bias analysis
│
└── .github/                           # GitHub repository configuration
    ├── workflows/                     # GitHub Actions CI/CD workflows
    │   └── .gitkeep
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md              # Structured bug report template
    │   └── feature_request.md         # Structured feature proposal template
    └── PULL_REQUEST_TEMPLATE.md       # Pull request checklist
```

---

## Academic Alignment & Sustainable Development Goals

### Course Context
- **Course Title:** Machine Learning Operations
- **Course Code:** `AI252TA`
- **Academic Semester:** Semester V

### Demonstrable MLOps Core Competencies
1. **Data Lifecycle Management:** Decoupled storage, versioning, and provenance via DVC.
2. **Reproducible Experimentation:** Controlled hyperparameter tracking and model metrics via MLflow.
3. **Engineering Rigor:** Automated linting (`ruff`), formatting (`black`), testing (`pytest`), and environment isolation.
4. **Production Architecture:** Microservice inference (`FastAPI`) inside standardized container images (`Docker`).
5. **Observability & Continuous Learning:** Automated drift surveillance (`Evidently`) and metric dashboards (`Prometheus`/`Grafana`).
6. **Governance & Responsible AI:** Detailed Model Cards, candidate data privacy enforcement, and evaluation fairness.

### United Nations Sustainable Development Goals
- **SDG 4 (Quality Education):** Provides students and job candidates with objective, automated, and accessible assessment of interview communication quality, closing access gaps to elite interview coaching.
- **SDG 8 (Decent Work & Economic Growth):** Equips job seekers with personalized feedback to identify weak knowledge areas, improving employability and economic mobility.

---

## Planned Viva & Live Demonstration Milestones

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT DEMONSTRATION MATRIX                           │
├─────────┬──────────────────────────────────────────┬────────────────────────────┤
│ Phase   │ Demonstration Workflow                   │ Expected Artifacts         │
├─────────┼──────────────────────────────────────────┼────────────────────────────┤
│ Phase 1 │ Raw Data ➔ DVC Repro ➔ EDA ➔ MLflow Runs │ DVC DAG, MLflow UI Metrics, Phase 1 Report │
│ Phase 2 │ Model Registry ➔ FastAPI ➔ Docker ➔ CI/CD│ OpenAPI Docs, Docker Image │
│ Phase 3 │ Prod Simulation ➔ Drift Detection ➔ Train│ Evidently Drift Reports    │
└─────────┴──────────────────────────────────────────┴────────────────────────────┘
```

> [!IMPORTANT]
> The demonstration workflows above represent formal academic deliverables. Phase 1 is complete through test evaluation with artifacts in `reports/phase1/phase1_report.md`. Phase 2 is planned next.

---

## Data Privacy & Security Governance

1. **Zero Personally Identifiable Information (PII):** Candidate answers and question datasets must be fully scrubbed of names, personal identifiers, contact details, and proprietary organizational data.
2. **Secret Hygiene:** No API keys, database credentials, or environment secrets may ever be committed to Git. Local secrets are stored in `.env` (strictly enforced by `.gitignore`).
3. **Model & Data Integrity:** Large dataset files and model binaries are managed exclusively via DVC and MLflow artifacts, ensuring the Git repository remains lightweight and auditable.
4. **Fairness & Dialectal Neutrality:** Language models will be assessed for stylistic bias to ensure non-native phrasing is not unfairly penalized when technical content is correct.

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Git & Git LFS
- Make (optional, for developer shortcut commands)

### 2. Clone Repository
```bash
git clone https://github.com/Tejzraj/MLOps.git
cd MLOps
```

### 3. Environment Initialization
```bash
# Create virtual environment
make setup-env
source .venv/bin/activate

# Install core and development dependencies
make install-dev
```

### 4. Verify Development Tooling
```bash
# Check formatting and linting
make lint

# Run initial test suite verification
make test
```

### 5. Launch Local Services (Scaffolded)
```bash
# Launch MLflow tracking dashboard
make mlflow-ui

# Inspect DVC pipeline status
make dvc-status
```

---

## Maintainer & Contributions

- **Repository Maintainer:** Tejzraj ([@Tejzraj](https://github.com/Tejzraj))
- **Course:** AI252TA – Machine Learning Operations, Semester V
- **Contributions:** Please read [CONTRIBUTING.md](CONTRIBUTING.md) and adhere to the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
