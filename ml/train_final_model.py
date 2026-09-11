"""
Day 12 (part 1) — Train and save the final model.

Trains GaussianNB on ACO's selected features (reports/selected_features.csv)
using the full training set, then saves the model plus everything needed
for inference: feature order, label mapping, the BENIGN class index, and
traffic-density normalization bounds (1st/99th percentile of Flow Bytes/s,
computed from train data only — using test data here would be leakage).

Run from the project root:
    python -m ml.train_final_model
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

DENSITY_FEATURE = "Flow Bytes/s"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    with open(PROCESSED_DIR / "label_mapping.json") as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}

    selected_features = pd.read_csv(REPORTS_DIR / "selected_features.csv")["feature"].tolist()
    print(f"Training on {len(selected_features)} ACO-selected features, {len(train_df):,} rows")

    X_train = train_df[selected_features]
    y_train = train_df["Label_encoded"]

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Model trained.")

    benign_index = next(idx for idx, label in label_mapping.items() if label == "BENIGN")

    density_low, density_high = train_df[DENSITY_FEATURE].quantile([0.01, 0.99])

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

    metadata = {
        "selected_features": selected_features,
        "label_mapping": label_mapping,
        "benign_index": benign_index,
        "density_feature": DENSITY_FEATURE,
        "density_low": float(density_low),
        "density_high": float(density_high),
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
