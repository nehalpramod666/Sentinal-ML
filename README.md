# SentinelML

**AI-Powered Network Intrusion Risk Scoring Platform**

SentinelML is a production-inspired intrusion detection platform that combines
Ant Colony Optimization (ACO) feature selection, a Gaussian Naive Bayes
classifier, and fuzzy inference to produce graded intrusion risk scores
(Low / Medium / High / Critical) instead of a flat binary alert. It ships
with MLflow experiment tracking, a FastAPI inference service, Docker
packaging, GitHub Actions CI, and an MCP server for natural-language queries
about the deployed model.

> Status: In progress — Day 1 of the 4-week solo build plan.

## Why this exists

Real-world network traffic datasets (this project uses **CICIDS2017**) have
dozens of noisy, redundant, or weakly-informative features. Training on all
of them slows training/inference, hurts interpretability, and raises false
positive rates. SentinelML uses ACO — a swarm-intelligence, wrapper-based
feature selection method — to find a compact, high-signal feature subset
before classification, then converts model output into human-readable risk
levels via fuzzy logic.

## Architecture
CICIDS Dataset
-> Data Cleaning & Encoding
-> Feature Engineering Pipeline
-> Mutual Information Calculation
-> ACO Feature Selection
-> Selected Feature Subset (CSV)
-> Gaussian Naive Bayes
-> Prediction Probabilities
-> Fuzzy Risk Scoring Engine
-> Low / Medium / High / Critical
-> FastAPI Inference API
-> MLflow (tracking + model registry) <-> MCP Server (AI assistant)
-> Docker Deployment


## Repository layout

sentinelml/
├── api/ FastAPI inference service (endpoints, schemas)
├── ml/ Mutual Information, ACO, Gaussian Naive Bayes
├── fuzzy/ Membership functions, fuzzy inference rules
├── mcp/ MCP server + tools for natural-language model queries
├── src/ Shared utilities (data engineering, config, logging)
├── data/
│ ├── raw/ Original CICIDS2017 files (gitignored)
│ └── processed/ Cleaned train/test splits (gitignored)
├── models/ Saved model artifacts (model.pkl, gitignored)
├── notebooks/ Exploratory analysis
├── reports/plots/ EDA and evaluation visualizations
├── scripts/ One-off / CLI utility scripts
├── tests/ pytest test suite
├── docs/ Architecture notes, dataset.md, ACO design doc
└── .github/workflows/ CI pipeline


## Tech stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Python |
| Machine Learning | scikit-learn, NumPy, Pandas, SciPy, custom ACO |
| Fuzzy Logic | scikit-fuzzy |
| MLOps | MLflow |
| DevOps | Docker, GitHub Actions |
| AI | MCP Python SDK, Claude-compatible client |
| Dataset | CICIDS2017 |

## Getting started

```bash
git clone https://github.com/<you>/sentinelml.git
cd sentinelml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pytest
```

## Development workflow

This project uses a lightweight, industry-style Git workflow:

- `main` — always deployable, never pushed to directly
- `develop` — integration branch, everything merges here first
- `feature/*`, `bugfix/*`, `docs/*` — one branch per task

```bash
git checkout develop
git pull
git checkout -b feature/<task-name>
# ... work in small commits using Conventional Commits (feat:, fix:, docs:) ...
git push origin feature/<task-name>
# open a PR into develop, self-review the diff, merge, delete branch
```

`develop` merges into `main` and gets tagged `v1.0.0` at release.

## Roadmap

See [`docs/plan.md`](docs/plan.md) for the full 4-week execution plan.

## License

MIT — see [LICENSE](LICENSE).
