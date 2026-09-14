"""
Day 17 — MLflow experiment tracking and model registry.

Logs the final model's hyperparameters (from ACO tuning), metrics (from the
Day 10 baseline-vs-ACO comparison and Day 15's model), and key artifacts
(plots, feature rankings) to MLflow, then registers the trained model in
MLflow's model registry so it can be loaded by name/version rather than a
raw file path — this is what Day 21's "auto-load latest/best model" will
query against.

Uses a local file-based tracking store (./mlruns) — no external MLflow
server needed for this project's scope.

Run from the project root:
    python -m ml.mlflow_tracking
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

EXPERIMENT_NAME = "sentinalml-intrusion-detection"
REGISTERED_MODEL_NAME = "sentinalml-gaussiannb-aco"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading model, metadata, and results...")
    model = joblib.load(MODELS_DIR / "model.pkl")
    aco_summary = load_json(REPORTS_DIR / "aco_summary.json")
    aco_results = load_json(REPORTS_DIR / "aco_model_results.json")
    baseline_results = load_json(REPORTS_DIR / "baseline_results.json")

    with mlflow.start_run(run_name="gaussiannb-aco-selected") as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params(
            {
                "model_type": "GaussianNB",
                "feature_selection": "ACO (tuned)",
                "n_features_total": aco_summary["n_features_total"],
                "n_features_selected": aco_summary["n_features_selected"],
                **{f"aco_{k}": v for k, v in aco_summary["config"].items()},
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": aco_results["accuracy"],
                "precision_macro": aco_results["precision_macro"],
                "recall_macro": aco_results["recall_macro"],
                "f1_macro": aco_results["f1_macro"],
                "f1_weighted": aco_results["f1_weighted"],
                "roc_auc_macro_ovr": aco_results["roc_auc_macro_ovr"],
                "train_time_seconds": aco_results["train_time_seconds"],
                "inference_time_seconds": aco_results["inference_time_seconds"],
                "baseline_accuracy": baseline_results["accuracy"],
                "baseline_f1_accuracy": baseline_results["f1_macro"],
                "accuracy_delta_vs_baseline": aco_results["accuracy"]
                - baseline_results["accuracy"],
                "f1_macro_delta_vs_baseline": aco_results["f1_macro"]
                - baseline_results["f1_macro"],
            }
        )

        mlflow.set_tags(
            {
                "dataset": "CICIDS2017",
                "feature_selection_method": "ACO",
            }
        )

        artifact_file = [
            REPORTS_DIR / "selected_features.csv",
            REPORTS_DIR / "feature_scores.csv",
            REPORTS_DIR / "aco_tuning_experiments.csv",
            REPORTS_DIR / "aco_model_results.json",
            REPORTS_DIR / "plots" / "correlation_heatmap.png",
            REPORTS_DIR / "plots" / "fuzzy_membership_functions.png",
            REPORTS_DIR / "plots" / "risk_distribution.png",
        ]
        for f in artifact_file:
            if f.exists():
                mlflow.log_artifact(str(f))
                print(f"Logged artifact: {f}")
            else:
                print(f"WARNING: artifact not found, skipping: {f}")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=None,
        )
        print(f"Model registered as: {REGISTERED_MODEL_NAME}")
        print("Run complete. View with: mlflow ui --backend-store-uri ./ mlruns")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
