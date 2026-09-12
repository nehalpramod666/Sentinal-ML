"""
Day 15 — FastAPI inference service.

Wraps the trained model (Day 12) and fuzzy engine (Day 11-13) behind three
endpoints: /health, /model-info, /predict. Reuses fuzzy.integration's
scoring logic directly rather than duplicating it, so the API and the
standalone scripts always agree on how a prediction becomes a risk score.

Run from the project root:
    uvicorn api.main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from fuzzy.engine import compute_risk
from fuzzy.integration import compute_traffic_density

MODELS_DIR = Path("models")

app = FastAPI(
    title="SentinalML API",
    description="AI-powered network intrusion risk scoring - ACO-selected GaussianNB + fuzzy risk inference",
    version="0.1.0",
)

_model = None
_metadata = None


@app.on_event("startup")
def load_model():
    global _model, _metadata
    _model = joblib.load(MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        _metadata = json.load(f)
    print(f"Model loaded: {len(_metadata['selected_features'])} features")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        n_features=len(_metadata["selected_features"]),
        selected_features=_metadata["selected_features"],
        label_mapping={int(k): v for k, v in _metadata["label_mapping"].items()},
        density_feature=_metadata["density_feature"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    selected_features = _metadata["selected_features"]
    missing = [f for f in selected_features if f not in request.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. See GET /model-info for the full list.",
        )

    row = pd.Series({f: request.features[f] for f in selected_features})
    X_row = row.to_frame().T

    proba = _model.predict_proba(X_row)[0]
    sorted_proba = np.sort(proba)[::-1]
    top1, top2 = sorted_proba[0], sorted_proba[1] if len(sorted_proba) > 1 else 0.0

    label_mapping = {int(k): v for k, v in _metadata["label_mapping"].items()}
    benign_index = _metadata["benign_index"]

    predicted_class_idx = int(np.argmax(proba))
    predicted_label = label_mapping[predicted_class_idx]

    prob_attack = 1.0 - proba[benign_index]
    confidence = float(top1 - top2)

    density_raw = request.features.get(_metadata["density_feature"], 0.0)
    density = compute_traffic_density(
        density_raw, _metadata["density_low"], _metadata["density_high"]
    )

    risk_score, risk_level = compute_risk(prob_attack, confidence, density)

    return PredictResponse(
        predicted_label=predicted_label,
        prob_attack=round(prob_attack, 4),
        confidence=round(confidence, 4),
        traffic_density=round(density, 4),
        risk_score=round(float(risk_score), 2),
        risk_level=risk_level,
    )
