"""
Day 12 (part 2) — Fuzzy risk integration.

Loads the saved model (Day 12 part 1) and Day 11's fuzzy engine, runs
predictions on real test data, and computes graded risk levels end-to-end:

    raw traffic row
      -> GaussianNB prediction (predict_proba)
      -> probability = P(attack) = 1 - P(BENIGN)
      -> confidence = margin between top-1 and top-2 class probabilities
      -> traffic_density = normalized Flow Bytes/s (train-set percentile bounds)
      -> fuzzy engine -> risk_score (0-100), risk_level (Low/Medium/High/Critical)

Run from the project root:
    python -m fuzzy.integration
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fuzzy.engine import compute_risk

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

N_SAMPLE_ROWS = 100


def load_model_and_metadata():
    model = joblib.load(MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


def compute_traffic_density(raw_value: float, density_low: float, density_high: float) -> float:
    """Min-max normalize a raw Flow Bytes/s value to [0, 1] using train-set
    percentile bounds (Day 12 part 1), clipping outliers to the boundary
    rather than letting them produce out-of-range fuzzy inputs."""
    if density_high == density_low:
        return 0.5
    normalized = (raw_value - density_low) / (density_high - density_low)
    return float(np.clip(normalized, 0, 1))


def score_row(model, metadata, row: pd.Series) -> dict:
    """Run one traffic row through the full model -> fuzzy pipeline."""
    selected_features = metadata["selected_features"]
    label_mapping = {int(k): v for k, v in metadata["label_mapping"].items()}
    benign_index = metadata["benign_index"]

    X_row = row[selected_features].to_frame().T
    proba = model.predict_proba(X_row)[0]

    sorted_proba = np.sort(proba)[::-1]
    top1, top2 = sorted_proba[0], sorted_proba[1] if len(sorted_proba) > 1 else 0.0

    predicted_class_idx = int(np.argmax(proba))
    predicted_label = label_mapping[predicted_class_idx]

    prob_attack = 1.0 - proba[benign_index]
    confidence = float(top1 - top2)
    density = compute_traffic_density(
        row[metadata["density_feature"]], metadata["density_low"], metadata["density_high"]
    )

    risk_score, risk_level = compute_risk(prob_attack, confidence, density)

    return {
        "predicted_label": predicted_label,
        "true_label": label_mapping.get(int(row["Label_encoded"]), "unknown"),
        "prob_attack": round(prob_attack, 4),
        "confidence": round(confidence, 4),
        "traffic_density": round(density, 4),
        "risk_score": round(float(risk_score), 2),
        "risk_level": risk_level,
    }


def main():
    print("Loading model and metadata...")
    model, metadata = load_model_and_metadata()

    print("Loading test data...")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    label_mapping = {int(k): v for k, v in metadata["label_mapping"].items()}
    test_df["_label_name"] = test_df["Label_encoded"].map(label_mapping)

    sample_rows = []
    per_class_quota = max(1, N_SAMPLE_ROWS // test_df["_label_name"].nunique())
    for _label_name, group in test_df.groupby("_label_name"):
        sample_rows.append(group.sample(n=min(per_class_quota, len(group)), random_state=42))
    sample_df = pd.concat(sample_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    sample_df = sample_df.head(N_SAMPLE_ROWS)

    print(f"\nScoring {len(sample_df)} sample rows through the full pipeline...\n")

    results = []
    for _, row in sample_df.iterrows():
        result = score_row(model, metadata, row)
        results.append(result)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    results_df["true_is_attack"] = results_df["true_label"] != "BENIGN"
    print("\nRisk level distribution by true traffic type:")
    print(results_df.groupby(["true_is_attack", "risk_level"]).size().unstack(fill_value=0))

    out_path = REPORTS_DIR / "fuzzy_integration_sample.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
