"""
Day 4 — Gaussian Naive Bayes baseline (all features, no feature selection).

Trains on data/processed/train.csv, evaluates on data/processed/test.csv,
and saves accuracy/precision/recall/F1 (macro and weighted) to
reports/baseline_results.json. This is the "no ACO" comparison point for
Day 10, where ACO-selected features get evaluated the same way.

Run from the project root:
    python -m ml.baseline
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    with open(PROCESSED_DIR / "label_mapping.json") as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    X_train = train_df.drop(columns=["Label_encoded"])
    y_train = train_df["Label_encoded"]
    X_test = test_df.drop(columns=["Label_encoded"])
    y_test = test_df["Label_encoded"]

    return X_train, y_train, X_test, y_test, label_mapping


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    X_train, y_train, X_test, y_test, label_mapping = load_data()
    print(f"Train: {len(X_train):,} rows, {X_train.shape[1]} features")
    print(f"Test: {len(X_test):,} rows")

    print("\nTraining Gaussian Naive Bayes (all features)...")
    start = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training took {train_time:.2f}s")

    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start
    print(f"Inference on {len(X_test):,} rows took {inference_time:.2f}s")

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(
        f"Macro    — Precision: {precision_macro:.4f}  Recall: {recall_macro:.4f}  F1: {f1_macro:.4f}"
    )
    print(
        f"Weighted — Precision: {precision_weighted:.4f}  Recall: {recall_weighted:.4f}  F1: {f1_weighted:.4f}"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[label_mapping[i] for i in sorted(label_mapping)],
        zero_division=0,
    )
    print("\nPer-class report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    results = {
        "model": "GaussianNB",
        "feature_selection": "none (all features)",
        "n_features": X_train.shape[1],
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "train_time_seconds": round(train_time, 3),
        "inference_time_seconds": round(inference_time, 3),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "label_mapping": label_mapping,
    }

    results_path = REPORTS_DIR / "baseline_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
