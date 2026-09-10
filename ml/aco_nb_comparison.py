"""
Day 10 — ACO-selected feature Naive Bayes vs. full-feature baseline.

Trains GaussianNB on the 35 features ACO selected (Day 8-9), evaluated on
the SAME held-out test.csv Day 4's baseline used (not ACO's internal
subsampled fitness split — this is the final, real comparison). Saves
results and prints a side-by-side table against reports/baseline_results.json.

Run from the project root:
    python -m ml.aco_nb_comparison
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
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    with open(PROCESSED_DIR / "label_mapping.json") as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    selected = pd.read_csv(REPORTS_DIR / "selected_features.csv")["feature"].tolist()

    X_train = train_df[selected]
    y_train = train_df["Label_encoded"]
    X_test = test_df[selected]
    y_test = test_df["Label_encoded"]

    return X_train, y_train, X_test, y_test, label_mapping, selected


def compute_roc_auc(model, X_test, y_test, n_classes):
    # Macro-averaged one vs rest ROC AUC - the standard multiclass approach.
    y_proba = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    present_classes = sorted(y_test.unique())
    try:
        auc = roc_auc_score(
            y_test_bin[:, present_classes],
            y_proba[:, present_classes],
            average="macro",
            multi_class="ovr",
        )
    except ValueError as e:
        print(f"WARNING: ROC AUC computation issue ({e}); falling back to weighted average")
        auc = roc_auc_score(
            y_test_bin[:, present_classes],
            y_proba[:, present_classes],
            average="weighted",
            multi_class="ovr",
        )
    return auc


def main():
    print("Loading data (ACO-selected features)...")
    X_train, y_train, X_test, y_test, label_mapping, selected = load_data()
    print(f"Train: {len(X_train):,} rows, {X_train.shape[1]} features (ACO-selected)")
    print(f"Test: {len(X_test):,} rows")
    print(f"Selected features: {selected}\n")

    print("Training Gaussinan Naive Bayes (ACO-selected features)...")
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

    print("\nComputing ROC AUC (macro, one-vs-rest)...")
    roc_auc = compute_roc_auc(model, X_test, y_test, n_classes=len(label_mapping))

    print(f"\nAccuracy: {accuracy:.4f}")
    print(
        f"Macro   - Precision: {precision_macro:.4f} Recall: {recall_macro:.4f} F1: {f1_macro:.4f}"
    )
    print(
        f"Weighted   - Precision: {precision_weighted:.4f} Recall: {recall_weighted:.4f} F1: {f1_weighted:.4f}"
    )
    print(f"ROC AUC (macro, OvR): {roc_auc:.4f}")

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
        "feature_selection": "ACO (Day 8-9, tuned)",
        "n_features": X_train.shape[1],
        "selected_features": selected,
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
        "roc_auc_macro_ovr": round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
        "label_mapping": label_mapping,
    }

    results_path = REPORTS_DIR / "aco_model_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {results_path}")

    baseline_path = REPORTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: Full-feature baseline vs. ACO-selected")
        print("=" * 70)
        metrics = [
            ("n_features", "Features used"),
            ("accuracy", "Accuracy"),
            ("precision_macro", "Precision (macro)"),
            ("recall_macro", "Recall (macro)"),
            ("f1_macro", "F1 (macro)"),
            ("f1_weighted", "F1 (weighted)"),
            ("train_time_seconds", "Train time (s)"),
            ("inference_time_seconds", "Inference time (s)"),
        ]
        print(f"{'Metric':<22} {'Baseline (77 feat)':>20} {'ACO (35 feat)':>16} {'Delta':>10}")
        for key, label in metrics:
            base_val = baseline.get(key, float("nan"))
            aco_val = results.get(key, float("nan"))
            delta = aco_val - base_val
            print(
                f"{label:<22} {base_val:>20} {aco_val:>16} {delta:>+10.4f}"
                if isinstance(base_val, float)
                else f"{label:<22} {base_val:>20} {aco_val:>16}"
            )
        print(f"{'ROC AUC (macro)':<22} {'N/A (not computed)':>20} {roc_auc:>16.4f}")


if __name__ == "__main__":
    main()
