# API Reference

Interactive Swagger UI is available at `/docs` when the API is running
(`uvicorn api.main:app --reload` or `docker-compose up`). This document is
a static reference for browsing without running the server.

## Base URL

- Local: `http://127.0.0.1:8000` (or `:8080` if port 8000 is blocked —
  see `docs/dataset.md`, Day 19, for a Windows-specific note on this)
- Docker: `http://127.0.0.1:8000` (docker-compose maps this automatically)

## `GET /health`

Runs a real self-test prediction (Day 20) rather than only checking the
model object is loaded.

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## `GET /model-info`

Returns which model is currently serving predictions, including whether
it was loaded from the MLflow registry or a local file fallback (Day 21).

**Response 200:**
```json
{
  "n_features": 35,
  "selected_features": ["Destination Port", "Flow Bytes/s", "..."],
  "label_mapping": {"0": "BENIGN", "1": "Bot", "...": "..."},
  "density_feature": "Flow Bytes/s",
  "model_source": "mlflow_registry",
  "model_version": "3"
}
```

## `GET /metrics`

In-memory operational metrics (Day 20) — request counts, error counts,
prediction latency, risk level distribution since server start.

**Response 200:**
```json
{
  "uptime_seconds": 34.6,
  "request_counts": {"/health": 1, "/predict": 1},
  "error_counts": {},
  "predict_count": 1,
  "predict_avg_latency_ms": 15.61,
  "predict_max_latency_ms": 15.61,
  "risk_level_counts": {"High": 1}
}
```

## `POST /predict`

**Request body:**
```json
{
  "features": {
    "Destination Port": 443.0,
    "Flow Duration": 120000.0,
    "...": "... all 35 features listed in GET /model-info ..."
  }
}
```

All 35 ACO-selected features (see `GET /model-info` for the exact list)
must be present. Values with absolute magnitude over 1e15 are rejected
(Day 19 hardening).

**Response 200:**
```json
{
  "predicted_label": "FTP-Patator",
  "prob_attack": 1.0,
  "confidence": 0.9922,
  "traffic_density": 0.0101,
  "risk_score": 71.06,
  "risk_level": "High"
}
```

**Response 422** (missing or invalid features):
```json
{
  "detail": "Missing required features: [...]. See GET /model-info for the full list."
}
```

## Error handling

Any unhandled server-side error returns a generic `500` with
`{"detail": "Internal server error. Check server logs for details."}` —
full details (including stack trace) are logged server-side only, never
exposed to the client (Day 19).
