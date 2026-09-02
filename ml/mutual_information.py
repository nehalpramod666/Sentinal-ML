"""
Day 6 — Mutual Information feature ranking.

Computes MI between each feature and the target label, ranks features
descending, and saves the ranking to reports/feature_scores.csv. This
ranking becomes the heuristic (eta) term ACO's ants use alongside
pheromone trails during feature selection (Day 7-9).

Run from the project root:
    python -m ml.mutual_information
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    X = train_df.drop(columns=["Label_encoded"])
    y = train_df["Label_encoded"]
    print(f"Computing MI for {X.shape[1]} features against {len(X):,} rows...")

    # discrete_features=False: all our features are continuous flow statistics,
    # not categorical/discrete counts, so let sklearn treat them as continuous.
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)

    scores_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
    scores_df = scores_df.sort_values("mi_score", ascending=False).reset_index(drop=True)
    scores_df["rank"] = scores_df.index + 1

    out_path = REPORTS_DIR / "feature_scores.csv"
    scores_df.to_csv(out_path, index=False)

    print("\nTop 15 features by Mutual Information:")
    print(scores_df.head(15).to_string(index=False))

    print("\nBottom 10 features by Mutual Information:")
    print(scores_df.tail(10).to_string(index=False))

    print(f"\nSaved: {out_path}")
    print(f"MI score range: {scores_df['mi_score'].min():.4f} - {scores_df['mi_score'].max():.4f}")


if __name__ == "__main__":
    main()
