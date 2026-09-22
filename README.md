# SentinelML

**AI-Powered Network Intrusion Risk Scoring Platform**

SentinelML combines Ant Colony Optimization (ACO) feature selection, a
Gaussian Naive Bayes classifier, and fuzzy inference to produce graded
intrusion risk scores (Low / Medium / High / Critical) instead of a flat
binary alert — trained and evaluated end-to-end on the CICIDS2017 network
intrusion dataset (2.83M flow records, 15 traffic classes).

> **Status: Complete (v1.0.0).** Full pipeline from raw data through a
> Dockerized, MLflow-tracked API with an MCP-based natural-language
> assistant. Built solo over 4 weeks — see [`docs/plan.md`](docs/plan.md)
> for the day-by-day execution log.

<!-- SCREENSHOT: Swagger UI at /docs showing the three main endpoints -->
<!-- SCREENSHOT: MLflow UI showing the tracked run with metrics -->

## Why this exists

Real-world network traffic datasets have dozens of noisy, redundant, or
weakly-informative features. Training on all of them slows
training/inference and hurts interpretability. SentinelML uses ACO — a
swarm-intelligence, wrapper-based feature selection method — to find a
compact, high-signal feature subset before classification, then converts
raw model output into human-readable risk levels via fuzzy logic, because
a single classifier's hard decision is often not trustworthy enough to act
on directly (see "Honest findings," below).

## Quick start

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
and place the 8 CSVs in `data/raw/`.

```bash
python -m ml.preprocessing          # ~1 min
python -m ml.mutual_information     # ~15-20 min (full dataset, one-time)
python -m ml.aco_tuning             # ~10-15 min (7-config sweep + final run)
python -m ml.train_final_model      # ~10 sec
python -m ml.mlflow_tracking        # logs + registers the model
```

Run the API:

```bash
uvicorn api.main:app --reload
# or:
docker-compose up --build
```

Try a prediction against a real test row and compare it to the API's
output directly:

```bash
python -m scripts.test_api
```

Sample output:

--- Comparison ---
predicted_label direct='FTP-Patator' api='FTP-Patator' [OK]
prob_attack direct=1.0 api=1.0 [OK]
confidence direct=0.9922 api=0.9922 [OK]
traffic_density direct=0.0101 api=0.0101 [OK]
risk_score direct=71.06 api=71.06 [OK]
risk_level direct='High' api='High' [OK]

PASS: API matches direct call exactly


Interactive API docs: open `http://127.0.0.1:8000/docs` (Swagger UI) once
the server is running.

## Usage examples

**Score traffic via the API:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Destination Port": 443.0, "...": "... 35 features total, see GET /model-info"}}'
```

**Query the model in natural language** (via Claude Desktop, connected to
`mcp_server/server.py` — see [setup notes](#mcp-assistant-setup) below):
> "What is the latest model's accuracy and how many features does it use?"

<!-- SCREENSHOT: Claude Desktop conversation showing a real natural-language query and response -->

**Explore experiments:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## MCP assistant setup

1. Install [Claude Desktop](https://claude.ai/download)
2. In Settings → Developer → Edit Config, add:
```json
   {
     "mcpServers": {
       "sentinelml": {
         "command": "/absolute/path/to/.venv/Scripts/python.exe",
         "args": ["/absolute/path/to/mcp_server/server.py"]
       }
     }
   }
```
3. Restart Claude Desktop completely (quit from the system tray, not just
   close the window)
4. Ask a question referencing the "sentinelml" connector

**Windows MSIX note**: if the server doesn't appear as connected, the
config file the in-app "Edit Config" button opens may not be the one the
app actually reads — see `docs/dataset.md`, Day 23, for the exact
workaround found for this.

<!-- SCREENSHOT: MCP Inspector tool list (mcp dev mcp_server/server.py) -->

## Honest findings

This project reports what's actually true, not just flattering numbers —
three things worth knowing before using or extending this code:

1. **GaussianNB struggles with this dataset's class imbalance.** Both the
   full-feature baseline (0.16 accuracy) and the ACO-selected model (0.18
   accuracy) score below a naive all-BENIGN baseline (~0.80 accuracy)
   would, because BENIGN traffic — 80% of the dataset — is broad and
   heterogeneous, breaking GaussianNB's per-feature independence
   assumptions.
2. **ACO delivers real efficiency gains** (3.1x faster training, 2.2x
   faster inference, 45% of the original features) **and modest quality
   gains**, but doesn't fix GaussianNB's core BENIGN/attack confusion —
   that's a model limitation, not a feature-count problem.
3. **GaussianNB's predicted probabilities are nearly binary/saturated** (a
   known Naive Bayes pathology), limiting how much nuance the fuzzy
   risk-scoring layer can extract. The fuzzy engine itself is verified
   correct; the bottleneck is upstream probability calibration.

Full detail, with numbers, in [`docs/dataset.md`](docs/dataset.md).

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system
design, component diagram, and a request-sequence diagram.

CICIDS Dataset -> Preprocessing -> Mutual Information -> ACO Feature Selection
-> Gaussian Naive Bayes -> Fuzzy Risk Scoring -> FastAPI
-> MLflow (tracking + registry) <-> MCP Server (AI assistant)
-> Docker Deployment


## Repository layout

sentinelml/
├── api/ FastAPI inference service (main.py, schemas.py)
├── ml/ Preprocessing, EDA, MI, ACO, model training, MLflow
├── fuzzy/ Fuzzy engine, model integration, visualization
├── mcp_server/ MCP server exposing model metadata as natural-
language-queryable tools
├── data/ raw/ (gitignored) and processed/ (gitignored) CSVs
├── models/ Saved model artifacts (model.pkl, metadata)
├── reports/ Metrics, feature rankings, ACO/fuzzy/stress-test results
├── reports/plots/ EDA, ACO, and fuzzy visualizations
├── scripts/ Dataset inspection, API testing, stress testing
├── tests/ pytest suite (unit + end-to-end integration)
├── docs/ architecture.md, api.md, dataset.md (full findings
log), aco_design.md, plan.md
└── .github/workflows/ CI pipeline (lint, test, Docker build)


## Tech stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Python |
| Machine Learning | scikit-learn, NumPy, Pandas, SciPy, custom ACO |
| Fuzzy Logic | scikit-fuzzy |
| MLOps | MLflow (SQLite backend) |
| DevOps | Docker, GitHub Actions |
| AI | MCP Python SDK (FastMCP), Claude Desktop |
| Dataset | CICIDS2017 |

## Testing

```bash
pytest                                 # unit + integration tests
python -m scripts.test_api             # API correctness vs. direct scoring
python -m scripts.stress_test          # load testing, see docs/dataset.md Day 24
```

## Development workflow

- `main` — deployable, tagged at release
- `develop` — integration branch, everything merges here first
- `feature/*`, `fix/*`, `docs/*` — one branch per task

```bash
git checkout develop && git pull
git checkout -b feature/<task-name>
# commit, push, open PR into develop, self-review, merge
```

## Documentation index

- [`docs/architecture.md`](docs/architecture.md) — system design, sequence diagram
- [`docs/api.md`](docs/api.md) — API reference
- [`docs/dataset.md`](docs/dataset.md) — full findings log with real numbers, every bug found and fixed
- [`docs/aco_design.md`](docs/aco_design.md) — ACO's mathematical design
- [`docs/plan.md`](docs/plan.md) — 4-week execution plan and status

## License

MIT — see [LICENSE](LICENSE).

Screenshots — I've marked 4 spots with <!-- SCREENSHOT: ... --> HTML comments (invisible when rendered, so they won't clutter the page until you replace them). Capture these and swap the comments for real Markdown image syntax:

Swagger UI (/docs) — shows the endpoints professionally
MLflow UI showing your tracked run
A real Claude Desktop conversation using the MCP tools (you already have this from Day 23!)
MCP Inspector's tool list

For each, save the image into reports/plots/ or a new docs/images/ folder, then replace e.g. <!-- SCREENSHOT: Swagger UI... --> with ![Swagger UI](docs/images/swagger-ui.png).
