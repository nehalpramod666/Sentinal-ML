"""
Day 5 — Exploratory Data Analysis.

Generates three sets of plots from data/processed/train.csv:
  1. Class distribution (log-scale bar chart, given severe imbalance)
  2. BENIGN vs. Attack histograms for 12 representative features
  3. Correlation heatmap on the top 20 features by variance

All plots saved to reports/plots/. Run from the project root:
    python -m ml.eda
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = Path("reports/plots")


REPRESENTATIVE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Active Mean",
]


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    with open(PROCESSED_DIR / "label_mapping.json") as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    train_df["Label"] = train_df["Label_encoded"].map(label_mapping)
    train_df["is_attack"] = train_df["Label"] != "BENIGN"
    return train_df, label_mapping


def plot_class_distribution(df: pd.DataFrame, label_mapping: dict):
    counts = df["Label"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_ylabel("")
    ax.set_title("Class Distribution - CICIDS2017 (log scale due to severe imbalance)")
    ax.invert_yaxis()
    plt.tight_layout()
    out = PLOTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_feature_histograms(df: pd.DataFrame):
    available = [f for f in REPRESENTATIVE_FEATURES if f in df.columns]
    missing = [f for f in REPRESENTATIVE_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: features not found in data, skipping: {missing}")

    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(available):
        ax = axes[i]
        benign = df.loc[~df["is_attack"], feature]
        attack = df.loc[df["is_attack"], feature]

        lo, hi = df[feature].quantile([0.01, 0.99])
        bins = 40

        ax.hist(
            benign.clip(lo, hi),
            bins=bins,
            alpha=0.5,
            label="BENIGN",
            color="steelblue",
            density=True,
        )
        ax.hist(
            attack.clip(lo, hi),
            bins=bins,
            alpha=0.5,
            label="ATTACK",
            color="firebrick",
            density=True,
        )
        ax.set_title(feature, fontsize=10)
        ax.legend(fontsize=8)

    for j in range(len(available), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(
        "BENIGN vs. Attack Feature Distribution (clipped to 1st-99th percentile)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "feature_histograms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["Label_encoded"]]

    top_variance_features = (
        df[numeric_cols].var().sort_values(ascending=False).head(20).index.tolist()
    )
    corr = df[top_variance_features].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap — Top 20 Features by Variance", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    return top_variance_features, corr


def report_high_correlation_pairs(features: list, corr: pd.DataFrame, threshold: float = 0.9):
    print(f"\nFeature pairs with |correlation| > {threshold}:")
    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1 :]:
            r = corr.loc[f1, f2]
            if abs(r) > threshold:
                pairs.append((f1, f2, r))
                print(f"  {f1:30s} <-> {f2:30s}  r = {r:.3f}")
    if not pairs:
        print("  none")
    return pairs


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, label_mapping = load_data()
    print(f"Loaded {len(df):,} rows")

    print("\nGenerating class distribution plot...")
    plot_class_distribution(df, label_mapping)

    print("\nGenerating feature histograms...")
    plot_feature_histograms(df)

    print("\nGenerating correlation heatmap...")
    top_features, corr = plot_correlation_heatmap(df)

    high_corr_pairs = report_high_correlation_pairs(top_features, corr)

    print(
        f"\nEDA complete. {len(high_corr_pairs)} highly correlated pairs found among top-variance features."
    )
    print(
        "These are candidates for redundancy — relevant context for Day 6-9 (MI + ACO feature selection)."
    )


if __name__ == "__main__":
    main()
