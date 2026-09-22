"""
Day 25 — End-to-end integration test.

Exercises the full chain from already-processed data through to the API
and MCP layers, verifying each stage's output is consistent with what the
next stage expects. Does not re-run expensive upstream steps (raw CSV
preprocessing ~1 min, full-dataset MI ~15-20 min) — instead validates that
everything downstream of those artifacts (feature selection, model,
fuzzy engine, API, MCP) is internally consistent, which is where this
project's actual bugs have occurred (stale files, relative paths, typos
across module boundaries).

Run from the project root:
    pytest tests/test_integration.py -v
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

REQUIRED_FILES = [
    PROCESSED_DIR / "train.csv",
    PROCESSED_DIR / "test.csv",
    PROCESSED_DIR / "label_mapping.json",
    REPORTS_DIR / "selected_features.csv",
    REPORTS_DIR / "aco_model_results.json",
    MODELS_DIR / "model.pkl",
    MODELS_DIR / "model_metadata.json",
]


def _missing_files() -> list[str]:
    return [str(f) for f in REQUIRED_FILES if not f.exists()]


pytestmark = pytest.mark.skipif(
    bool(_missing_files()),
    reason=f"Pipeline artifacts not present (run Days 3-12 scripts first): {_missing_files()}",
)


@pytest.fixture(scope="module")
def model_and_metadata():
    model = joblib.load(MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


@pytest.fixture(scope="module")
def selected_features():
    return pd.read_csv(REPORTS_DIR / "selected_features.csv")["feature"].tolist()


@pytest.fixture(scope="module")
def sample_test_row():
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return test_df.iloc[0]


class TestFeatureConsistency:
    """Selected features must match exactly what the model was trained on
    and what its metadata claims — this is exactly the class of bug Day 22
    surfaced (a stale file silently describing a different configuration
    than the actual deployed model)."""

    def test_selected_features_match_model_metadata(self, selected_features, model_and_metadata):
        _, metadata = model_and_metadata
        assert selected_features == metadata["selected_features"], (
            "selected_features.csv does not match model_metadata.json — "
            "these must be regenerated together, not independently"
        )

    def test_model_expects_same_feature_count(self, selected_features, model_and_metadata):
        model, _ = model_and_metadata
        # GaussianNB stores per-class feature means with shape (n_classes, n_features)
        assert model.theta_.shape[1] == len(selected_features), (
            f"Model was trained on {model.theta_.shape[1]} features, "
            f"but selected_features.csv lists {len(selected_features)}"
        )


class TestModelToFuzzyPipeline:
    """The core prediction -> risk-scoring chain, using fuzzy.integration's
    real score_row function against a real test row."""

    def test_score_row_produces_valid_output(self, model_and_metadata, sample_test_row):
        from fuzzy.integration import score_row

        model, metadata = model_and_metadata
        result = score_row(model, metadata, sample_test_row)

        assert result["risk_level"] in {"Low", "Medium", "High", "Critical"}
        assert 0 <= result["risk_score"] <= 100
        assert 0 <= result["prob_attack"] <= 1
        assert result["predicted_label"] in metadata["label_mapping"].values()


class TestAPIIntegration:
    """FastAPI's TestClient exercises the real app object in-process — no
    live uvicorn server needed, and it uses the exact same code path a real
    HTTP request would."""

    @pytest.fixture(scope="class")
    def client(self):
        from api.main import app

        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_model_info_endpoint(self, client, selected_features):
        response = client.get("/model-info")
        assert response.status_code == 200
        body = response.json()
        assert body["n_features"] == len(selected_features)
        assert set(body["selected_features"]) == set(selected_features)

    def test_predict_endpoint_matches_direct_scoring(
        self, client, model_and_metadata, sample_test_row, selected_features
    ):
        from fuzzy.integration import score_row

        model, metadata = model_and_metadata
        direct_result = score_row(model, metadata, sample_test_row)

        payload = {f: float(sample_test_row[f]) for f in selected_features}
        response = client.post("/predict", json={"features": payload})

        assert response.status_code == 200
        api_result = response.json()
        assert api_result["predicted_label"] == direct_result["predicted_label"]
        assert api_result["risk_level"] == direct_result["risk_level"]
        assert abs(api_result["risk_score"] - direct_result["risk_score"]) < 0.01

    def test_predict_rejects_missing_features(self, client):
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 422


class TestMCPToolsIntegration:
    """Calls the actual MCP tool functions in-process (via FastMCP's .fn
    attribute, which exposes the undecorated function), exercising the
    same file-path resolution logic that broke in Day 23 when launched
    from a different working directory than the project root."""

    def _get_tool_fn(self, tool_name: str):
        import mcp_server.server as server_module

        tool = getattr(server_module, tool_name)
        # FastMCP's @mcp.tool() decorator wraps the function; the
        # original is accessible via .fn if present, otherwise the
        # decorator returned the plain function unchanged.
        return getattr(tool, "fn", tool)

    def test_get_model_metrics_returns_real_data(self):
        fn = self._get_tool_fn("get_model_metrics")
        result = fn()
        assert result["n_features"] > 0
        assert 0 <= result["accuracy"] <= 1

    def test_get_selected_features_matches_csv(self, selected_features):
        fn = self._get_tool_fn("get_selected_features")
        result = fn()
        assert result["selected_features"] == selected_features

    def test_get_dataset_info_returns_row_counts(self):
        fn = self._get_tool_fn("get_dataset_info")
        result = fn()
        assert result["n_classes"] == 15
        assert "n_train_rows" in result


class TestEndToEndConsistency:
    """The single highest-value test: a prediction made by directly loading
    the model must exactly match a prediction made through the full API
    stack, for the exact same input — this is what proves nothing has
    silently drifted between layers."""

    def test_full_chain_agrees_on_same_input(
        self, model_and_metadata, sample_test_row, selected_features
    ):
        from fastapi.testclient import TestClient

        from api.main import app
        from fuzzy.integration import score_row

        model, metadata = model_and_metadata
        direct = score_row(model, metadata, sample_test_row)

        with TestClient(app) as client:
            payload = {f: float(sample_test_row[f]) for f in selected_features}
            api_response = client.post("/predict", json={"features": payload}).json()

        assert direct["predicted_label"] == api_response["predicted_label"]
        assert direct["risk_level"] == api_response["risk_level"]
