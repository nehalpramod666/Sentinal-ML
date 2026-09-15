"""
Day 20 — FastAPI inference service.

Wraps the trained model and fuzzy engine behind /health, /model-info,
and /predict. Includes input validation, structured logging,
request timing, global error handling, request metrics, and
model self-testing.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient

from api.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from fuzzy.engine import compute_risk
from fuzzy.integration import compute_traffic_density

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "sentinalml-gaussiannb-aco"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sentinelml.api")

MODELS_DIR = Path("models")


class Metrics:
    """In-memory request metrics.

    Simple counters, not Prometheus-format — appropriately scoped for this
    project (no external metrics scraper consumes this); a JSON /metrics
    endpoint is sufficient for visibility into request volume, error rate,
    and latency during development/demo.
    """

    def __init__(self):
        self.start_time = datetime.now(UTC)
        self.request_counts: dict[str, int] = defaultdict(int)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.predict_latencies_ms: list[float] = []
        self.risk_level_counts: dict[str, int] = defaultdict(int)

    def record_request(self, path: str):
        self.request_counts[path] += 1

    def record_error(self, path: str):
        self.error_counts[path] += 1

    def record_predict(self, elapsed_ms: float, risk_level: str):
        self.predict_latencies_ms.append(elapsed_ms)
        self.risk_level_counts[risk_level] += 1

    def summary(self) -> dict:
        uptime_seconds = (datetime.now(UTC) - self.start_time).total_seconds()

        latencies = self.predict_latencies_ms
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        return {
            "uptime_seconds": round(uptime_seconds, 1),
            "request_counts": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "predict_count": len(latencies),
            "predict_avg_latency_ms": (round(avg_latency, 2) if avg_latency is not None else None),
            "predict_max_latency_ms": (round(max(latencies), 2) if latencies else None),
            "risk_level_counts": dict(self.risk_level_counts),
        }


metrics = Metrics()


app = FastAPI(
    title="SentinalML API",
    description=(
        "AI-powered network intrusion risk scoring - "
        "ACO-selected GaussianNB + fuzzy risk inference"
    ),
    version="0.1.0",
)

_model = None
_metadata = None
_model_source = None
_model_version = None


def load_model_from_registry():
    """Query MLflow's model registry for the latest registered version and
    load it. Returns (model, version_string) on success."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No registered versions found for '{REGISTERED_MODEL_NAME}'")

    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, latest.version


@app.on_event("startup")
def load_model():
    global _model, _metadata, _model_source, _model_version

    with open(MODELS_DIR / "model_metadata.json") as f:
        _metadata = json.load(f)
    try:
        _model, _model_version = load_model_from_registry()
        _model_source = "mlflow_registry"
        print(f"Model loaded from MLflow registry: {REGISTERED_MODEL_NAME} v{_model_version}")
    except Exception as e:
        logger.warning(
            f"Could not load model from MLflow registry ({e}); " f"falling back to model/model.pkl"
        )

        _model = joblib.load(MODELS_DIR / "model.pkl")
        _model_source = "local_file"
        _model_version = "unknown"
        print("Model loaded from local file: models/model.pkl (fallback)")


@app.middleware("http")
async def track_requests(request, call_next):
    response = await call_next(request)

    metrics.record_request(request.url.path)

    if response.status_code >= 400:
        metrics.record_error(request.url.path)

    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Catch unexpected exceptions, log details server-side, and return
    a generic error to the client without exposing implementation details.
    """
    logger.error(
        f"Unhandled exception on {request.url.path}: {exc}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    if _model is None or _metadata is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
        )

    try:
        # Self-test: run one real prediction through the model to confirm
        # it is functional, not merely loaded into memory.
        selected_features = _metadata["selected_features"]

        test_row = pd.Series({f: 0.0 for f in selected_features}).to_frame().T

        _model.predict_proba(test_row)

        return HealthResponse(
            status="ok",
            model_loaded=True,
        )

    except Exception as e:
        logger.error(
            f"Health check self-test failed: {e}",
            exc_info=True,
        )

        return HealthResponse(
            status="unhealthy",
            model_loaded=True,
        )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        n_features=len(_metadata["selected_features"]),
        selected_features=_metadata["selected_features"],
        label_mapping={int(k): v for k, v in _metadata["label_mapping"].items()},
        density_feature=_metadata["density_feature"],
        model_source=_model_source,
        model_version=str(_model_version),
    )


@app.get("/metrics")
def get_metrics():
    return metrics.summary()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    if _model is None or _metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    selected_features = _metadata["selected_features"]

    missing = [f for f in selected_features if f not in request.features]

    if missing:
        logger.warning(f"Predict request rejected: missing features {missing}")

        raise HTTPException(
            status_code=422,
            detail=(
                f"Missing required features: {missing}. " "See GET /model-info for the full list."
            ),
        )

    row = pd.Series({f: request.features[f] for f in selected_features})

    X_row = row.to_frame().T

    proba = _model.predict_proba(X_row)[0]

    sorted_proba = np.sort(proba)[::-1]
    top1 = sorted_proba[0]
    top2 = sorted_proba[1] if len(sorted_proba) > 1 else 0.0

    label_mapping = {int(k): v for k, v in _metadata["label_mapping"].items()}

    benign_index = _metadata["benign_index"]

    predicted_class_idx = int(np.argmax(proba))
    predicted_label = label_mapping[predicted_class_idx]

    prob_attack = 1.0 - proba[benign_index]
    confidence = float(top1 - top2)

    density_raw = request.features.get(
        _metadata["density_feature"],
        0.0,
    )

    density = compute_traffic_density(
        density_raw,
        _metadata["density_low"],
        _metadata["density_high"],
    )

    risk_score, risk_level = compute_risk(
        prob_attack,
        confidence,
        density,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Predict: label={predicted_label} "
        f"risk={risk_level} "
        f"score={risk_score:.2f} "
        f"elapsed_ms={elapsed_ms:.1f}"
    )

    metrics.record_predict(
        elapsed_ms,
        risk_level,
    )

    return PredictResponse(
        predicted_label=predicted_label,
        prob_attack=round(prob_attack, 4),
        confidence=round(confidence, 4),
        traffic_density=round(density, 4),
        risk_score=round(float(risk_score), 2),
        risk_level=risk_level,
    )
