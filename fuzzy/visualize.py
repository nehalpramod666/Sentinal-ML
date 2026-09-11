"""
Day 13 — Fuzzy visualization.

Produces two plots:
  1. fuzzy_membership_functions.png — the trapezoidal/triangular membership
     curves for all 3 inputs and the output, so the fuzzy sets defined in
     Day 11 are visually inspectable, not just numeric.
  2. risk_distribution.png — risk score histogram and risk level counts
     across a larger (2,000-row) stratified sample of real test data,
     split by true traffic type (BENIGN vs. Attack), extending Day 12's
     84-row spot-check into a fuller picture.

Run from the project root:
    python -m fuzzy.visualize
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuzzy.engine import confidence, probability, risk, traffic_density
from fuzzy.integration import load_model_and_metadata, score_row

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = Path("reports/plots")
SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def plot_membership_functions():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    variables = [
        (axes[0, 0], probability, "Probability (P(attack))"),
        (axes[0, 1], confidence, "Confidence (top1-top2 margin)"),
        (axes[1, 0], traffic_density, "Traffic Density (normalized)"),
        (axes[1, 1], risk, "Risk (output, 0-100)"),
    ]

    for ax, var, title in variables:
        for term_name in var.terms:
            mf = var[term_name].mf
            ax.plot(var.universe, mf, label=term_name, linewidth=2)
            ax.fill_between(var.universe, mf, alpha=0.15)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Membership degree")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Fuzzy Membership Functions", fontsize=4, y=1.00)
    plt.tight_layout()
    out = PLOTS_DIR / "fuzzy_membership_functions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def run_pipeline_on_sample():
    """Score a larger stratified sample of real test rows through the full
    model -> fuzzy pipeline, reusing Day 12's score_row logic directly."""
    model, metadata = load_model_and_metadata()
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    label_mapping = {int(k): v for k, v in metadata["label_mapping"].items()}
    test_df["_label_name"] = test_df["Label_encoded"].map(label_mapping)

    n_classes = test_df["_label_name"].nunique()
    per_class_quota = max(1, SAMPLE_SIZE // n_classes)

    sample_parts = []
    for _, group in test_df.groupby("_label_name"):
        sample_parts.append(
            group.sample(n=min(per_class_quota, len(group)), random_state=RANDOM_STATE)
        )
    sample_df = pd.concat(sample_parts).reset_index(drop=True)

    print(f"Scoring {len(sample_df):,} rows through the full pipeline...")
    results = [score_row(model, metadata, row) for _, row in sample_df.iterrows()]
    return pd.DataFrame(results)


def plot_risk_distribution(results_df: pd.DataFrame):
    results_df["true_is_attack"] = results_df["true_label"] != "BENIGN"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    benign_scores = results_df.loc[~results_df["true_is_attack"], "risk_score"]
    attack_scores = results_df.loc[results_df["true_is_attack"], "risk_score"]
    bins = np.arange(0, 105, 5)
    ax.hist(
        benign_scores,
        bins=bins,
        alpha=0.6,
        label=f"True BENIGN (N={len(benign_scores)})",
        color="steelblue",
    )
    ax.hist(
        attack_scores,
        bins=bins,
        alpha=0.6,
        label=f"True Attack (n={len(attack_scores)})",
        color="firebrick",
    )
    ax.set_xlabel("Risk score (0-100)")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution by True Traffic Type")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    level_order = ["Low", "Medium", "High", "Critical"]
    counts = (
        results_df.groupby(["risk_level", "true_is_attack"])
        .size()
        .unstack(fill_value=0)
        .reindex(level_order)
    )
    counts.columns = ["True BENIGN", "True Attack"] if False in counts.columns else counts.columns
    counts.plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "firebrick"])
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Count")
    ax.set_title("Risk Level Counts by True Traffic Type")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    out = PLOTS_DIR / "risk_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    return counts


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting fuzzy membership functions...")
    plot_membership_functions()

    print("\nRunning full pipeline on a larger sample for distribution analysis...")
    results_df = run_pipeline_on_sample()

    print("\nPlotting risk distribution...")
    counts = plot_risk_distribution(results_df)

    print("\nRisk level counts by true traffic type:")
    print(counts)

    results_path = PLOTS_DIR.parent / "fuzzy_risk_distribution_sample.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
