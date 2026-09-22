"""
Day 22 — MCP server exposing SentinelML's model metadata and metrics.

Five tools, each reading from files already produced by earlier days
(reports/, models/, MLflow registry) — no new computation, just a
natural-language-queryable interface over existing project artifacts.

Run standalone for testing:
    mcp dev mcp/server.py
Connected to an AI assistant (Day 23):
    mcp/server.py is launched by the client (e.g. Claude Desktop) via stdio.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mcp.server.fastmcp import FastMCP

# Anchor all paths to this file's location, not the process's current
# working directory. Critical when launched by an external client like
# Claude Desktop, which starts the server from its own working directory,
# not the project root — relative paths silently pointed at the wrong
# location and caused every tool to fail with file-not-found errors when
# first tested from Claude Desktop (worked fine under `mcp dev`, where the
# terminal's cwd happened to already be the project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


REGISTERED_MODEL_NAME = "sentinalml-gaussiannb-aco"

mcp = FastMCP("SentinelML")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@mcp.tool()
def get_model_metrics() -> dict:
    """Get the final ACO-selected model's performance metrics (accuracy,
    precision, recall, F1, ROC AUC), plus a comparison against the
    full-feature baseline from Day 4."""
    aco_results = _load_json(REPORTS_DIR / "aco_model_results.json")
    baseline_results = _load_json(REPORTS_DIR / "baseline_results.json")

    return {
        "model": aco_results["model"],
        "feature_selection": aco_results["feature_selection"],
        "n_features": aco_results["n_features"],
        "accuracy": aco_results["accuracy"],
        "precision_macro": aco_results["precision_macro"],
        "recall_macro": aco_results["recall_macro"],
        "f1_macro": aco_results["f1_macro"],
        "f1_weighted": aco_results["f1_weighted"],
        "roc_auc_macro_ovr": aco_results["roc_auc_macro_ovr"],
        "train_time_seconds": aco_results["train_time_seconds"],
        "inference_time_seconds": aco_results["inference_time_seconds"],
        "comparison_vs_baseline": {
            "baseline_n_features": baseline_results["n_features"],
            "baseline_accuracy": baseline_results["accuracy"],
            "baseline_f1_macro": baseline_results["f1_macro"],
            "accuracy_delta": round(aco_results["accuracy"] - baseline_results["accuracy"], 4),
            "f1_macro_delta": round(aco_results["f1_macro"] - baseline_results["f1_macro"], 4),
        },
    }


@mcp.tool()
def get_selected_features() -> dict:
    """Get the list of features ACO selected (Day 8-9), plus the total
    feature count for context."""
    selected = pd.read_csv(REPORTS_DIR / "selected_features.csv")["feature"].tolist()
    aco_summary = _load_json(REPORTS_DIR / "aco_summary.json")

    return {
        "n_selected": len(selected),
        "n_total": aco_summary["n_features_total"],
        "selected_features": selected,
    }


@mcp.tool()
def get_latest_experiment() -> dict:
    """Get details of the most recent MLflow experiment run: parameters,
    metrics, tags, and status."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("sentinalml-intrusion-detection")
    if experiment is None:
        return {"error": "Experiment 'sentinelml-intrusion-detection' not found"}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {"error": "No runs found for this experiment"}

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
        "status": run.info.status,
        "start_time": run.info.start_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }


@mcp.tool()
def get_deployment_status() -> dict:
    """Get the current deployment status: whether a trained model file
    exists locally, and the latest version registered in the MLflow model
    registry."""
    model_file_exists = (MODELS_DIR / "model.pkl").exists()
    metadata_file_exists = (MODELS_DIR / "model_meatadata.json").exists()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        latest_version = max((int(v.version) for v in versions), default=None)
        registry_available = True
    except Exception:
        latest_version = None
        registry_available = False
    return {
        "local_model_file_exists": model_file_exists,
        "local_metadata_file_exists": metadata_file_exists,
        "mlflow_registry_availabe": registry_available,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "latest_registered_version": latest_version,
    }


@mcp.tool()
@mcp.tool()
def get_dataset_info() -> dict:
    """Get information about the training dataset: row counts, class
    distribution, and label mapping."""
    metadata = _load_json(MODELS_DIR / "model_metadata.json")
    label_mapping = metadata["label_mapping"]

    info = {
        "dataset": "CICIDS2017",
        "n_classes": len(label_mapping),
        "label_mapping": label_mapping,
        "density_feature": metadata["density_feature"],
    }

    # Reuse row counts already computed and saved during Day 10's model
    # evaluation, rather than re-reading multi-million-row CSVs here —
    # keeps this tool fast enough for interactive natural-language queries
    # (an earlier version read train.csv/test.csv directly and timed out).
    aco_results_path = REPORTS_DIR / "aco_model_results.json"
    if aco_results_path.exists():
        aco_results = _load_json(aco_results_path)
        info["n_train_rows"] = aco_results["n_train_rows"]
        info["n_test_rows"] = aco_results["n_test_rows"]
        info["n_total_rows"] = aco_results["n_train_rows"] + aco_results["n_test_rows"]
    else:
        info["note"] = "aco_model_results.json not found; row counts unavailable"

    return info


if __name__ == "__main__":
    mcp.run()
