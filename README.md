# SentinelML

**AI-Powered Network Intrusion Risk Scoring Platform**

SentinelML combines Ant Colony Optimization (ACO) feature selection, a
Gaussian Naive Bayes classifier, and fuzzy inference to produce graded
intrusion risk scores (Low / Medium / High / Critical) instead of a flat
binary alert, trained and evaluated on the CICIDS2017 network intrusion
dataset (2.83M flow records, 15 traffic classes).

> **Status: Week 2 of 4 complete** (Days 1-13 of the solo execution plan).
> Data pipeline, ACO feature selection, and fuzzy risk scoring are fully
> implemented end-to-end. MLflow, FastAPI, Docker, CI/CD, and the MCP
> assistant (Weeks 3-4) are in progress.

## What's actually working right now

- **Data pipeline**: 2.83M raw CICIDS2017 rows cleaned, deduplicated, and
  split into a stratified 80/20 train/test set (`ml/preprocessing.py`)
- **EDA**: class distribution, feature correlation, and BENIGN-vs-attack
  distribution analysis (`ml/eda.py`, `reports/plots/`)
- **Mutual Information ranking** of all 77 features against the target
  (`ml/mutual_information.py`, `reports/feature_scores.csv`)
- **ACO feature selection**: a from-scratch Ant Colony Optimization
  implementation (`ml/aco.py`) that selects 35 of 77 features, tuned via a
  7-configuration hyperparameter sweep (`ml/aco_tuning.py`,
  `reports/aco_tuning_experiments.csv`)
- **Trained GaussianNB model** on the ACO-selected features
  (`ml/train_final_model.py`, `models/model.pkl`)
- **Fuzzy risk inference engine**: a Mamdani fuzzy system
  (`fuzzy/engine.py`) combining P(attack), prediction confidence, and
  traffic density into a 0-100 risk score and Low/Medium/High/Critical
  label, with full model integration (`fuzzy/integration.py`) and
  visualization (`fuzzy/visualize.py`)

## Honest findings so far

This project has been built with a "report what's actually true" discipline
rather than only reporting flattering numbers. Three findings worth knowing
before using or extending this code:

1. **GaussianNB struggles badly with this dataset's class imbalance.**
   Both the full-feature baseline (0.16 accuracy) and the ACO-selected
   model (0.18 accuracy) score *worse* than a naive all-BENIGN baseline
   (~0.80 accuracy) would on raw accuracy, because BENIGN traffic — 80% of
   the dataset — is broad and heterogeneous, breaking GaussianNB's
   per-feature Gaussian/independence assumptions. See `docs/dataset.md`,
   "Baseline model results (Day 4)."
2. **ACO feature selection delivers real efficiency gains** (3.1x faster
   training, 2.2x faster inference, 45% of the original features) **and
   modest quality gains** (accuracy, precision, F1 all improved slightly),
   but does not fix GaussianNB's core BENIGN/attack confusion — that's a
   model limitation, not a feature-count problem. See "ACO-selected model
   vs. full-feature baseline (Day 10)."
3. **GaussianNB's predicted probabilities are nearly binary/saturated**
   (a known Naive Bayes pathology from multiplying per-feature likelihoods
   across many features), which limits how much genuine nuance the fuzzy
   risk-scoring layer can extract from them. The fuzzy engine and rule
   base are verified correct; the bottleneck is upstream probability
   calibration. See "Fuzzy visualization and distribution analysis
   (Day 13)."

These aren't project failures — they're the actual empirical case for why
this system is architected the way it is (feature selection + fuzzy
reasoning rather than a single classifier's raw output), and they're
documented in full, with numbers, in `docs/dataset.md` and
`docs/aco_design.md`.

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
-> FastAPI Inference API [Week 3]
-> MLflow (tracking + registry) [Week 3] <-> MCP Server [Week 4]
-> Docker Deployment [Week 3]


## Repository layout

sentinelml/
├── api/ FastAPI inference service [Week 3, not yet built]
├── ml/ Preprocessing, EDA, MI, ACO, model training
├── fuzzy/ Fuzzy engine, model integration, visualization
├── mcp/ MCP server [Week 4, not yet built]
├── src/ Shared utilities
├── data/ raw/ (gitignored) and processed/ (gitignored) CSVs
├── models/ Saved model artifacts (model.pkl, metadata)
├── reports/ Metrics, feature rankings, ACO/fuzzy results
├── reports/plots/ EDA, ACO, and fuzzy visualizations
├── scripts/ Dataset inspection utility
├── tests/ pytest suite
├── docs/ dataset.md (full findings log), aco_design.md
└── .github/workflows/ CI pipeline


## Tech stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Python (Week 3) |
| Machine Learning | scikit-learn, NumPy, Pandas, SciPy, custom ACO |
| Fuzzy Logic | scikit-fuzzy |
| MLOps | MLflow (Week 3) |
| DevOps | Docker, GitHub Actions |
| AI | MCP Python SDK, Claude-compatible client (Week 4) |
| Dataset | CICIDS2017 |

## Getting started

```bash
git clone https://github.com/nehalpramod666/Sentinal-ML.git
cd Sentinal-ML
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

Download CICIDS2017's `MachineLearningCSV.zip` from
[unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
and place the 8 CSVs in `data/raw/`. Then:

```bash
python -m ml.preprocessing        # ~1 min
python -m ml.mutual_information   # ~15-20 min (full dataset)
python -m ml.aco_tuning           # ~10-15 min (7-config sweep + final run)
python -m ml.train_final_model    # ~10 sec
python -m fuzzy.integration       # ~1 min, sample risk scoring on real test data
python -m fuzzy.visualize         # ~1-2 min, membership + distribution plots
pytest                            # smoke tests
```

## Development workflow

- `main` — always deployable, never pushed to directly
- `develop` — integration branch, everything merges here first
- `feature/*`, `fix/*`, `docs/*` — one branch per task

```bash
git checkout develop && git pull
git checkout -b feature/<task-name>
# ... commit, push, open PR into develop, self-review, merge ...
```

## Roadmap

See [`docs/plan.md`](docs/plan.md) for the full 4-week plan and current
checklist status.

## License

MIT — see [LICENSE](LICENSE).
