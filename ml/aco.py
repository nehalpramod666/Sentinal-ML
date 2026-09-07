"""
Day 8 — Ant Colony Optimization for feature selection.

Implements the design from docs/aco_design.md: binary per-feature inclusion
decided by pheromone + heuristic (MI), fitness = macro F1 blended with a
compactness bonus, MMAS-style bounded pheromone with evaporation and
elitist reinforcement.

Fitness evaluation trains GaussianNB on a stratified subsample of the
training data (not the full 2.26M rows) — same rationale as
ml/fast_heuristic.py: this runs hundreds of times (n_ants * n_iterations),
so it must be fast, at the cost of not using every row.

Outputs (all in reports/):
  - selected_features.csv  : final best feature subset
  - aco_history.csv        : per-iteration best/avg fitness and subset size
  - aco_pheromone_final.csv: final pheromone value per feature

Run from the project root:
    python -m ml.aco
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42

FITNESS_SAMPLE_SIZE = 1_000_000


@dataclass
class ACOConfig:
    n_ants: int = 20
    n_iterations: int = 30
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.2
    q: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 0.99
    w_quality: float = 0.8
    w_compact: float = 0.2


def load_heuristic(feature_names: list[str]) -> np.ndarray:
    scores_df = pd.read_csv(REPORTS_DIR / "feature_scores.csv").set_index("feature")
    mi = scores_df.loc[feature_names, "mi_score"].to_numpy()
    max_mi = mi.max()
    return mi / max_mi if max_mi > 0 else np.ones_like(mi)


def load_fitness_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_full = train_df.drop(columns=["Label_encoded"])
    y_full = train_df["Label_encoded"]

    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=FITNESS_SAMPLE_SIZE, stratify=y_full, random_state=RANDOM_STATE
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=RANDOM_STATE
    )
    return X_tr, X_val, y_tr, y_val


def construct_ant(
    tau: np.ndarray, eta: np.ndarray, alpha: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    numerator = (tau**alpha) * (eta**beta)
    denominator = numerator + ((1 - tau) ** alpha) * ((1 - eta) ** beta)
    p_include = np.divide(
        numerator, denominator, out=np.full_like(numerator, 0.5), where=denominator != 0
    )

    draws = rng.random(len(tau))
    selected = draws < p_include

    if not selected.any():
        selected[np.argmax(eta)] = True
    return selected


def evaluate_fitness(
    selected_mask: np.ndarray,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: ACOConfig,
) -> tuple[float, float]:
    selected_cols = X_tr.columns[selected_mask]
    model = GaussianNB()
    model.fit(X_tr[selected_cols], y_tr)
    y_pred = model.predict(X_val[selected_cols])
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    compactness = 1 - (selected_mask.sum() / len(selected_mask))
    fitness = config.w_quality * macro_f1 + config.w_compact * compactness
    return fitness, macro_f1


def update_pheromone(
    tau: np.ndarray, ant_masks: list[np.ndarray], ant_fitness: list[float], config: ACOConfig
) -> np.ndarray:
    """Section 7: evaporation, then elitist-weighted reinforcement, then clamp.

    Deposit is normalized by n_ants so that pheromone reflects average
    solution quality per iteration, not raw agreement count — without this,
    any feature selected by more than a handful of ants saturates to
    tau_max within one or two iterations regardless of actual fitness.
    """
    tau = tau * (1 - config.rho)

    n_ants = len(ant_masks)
    for mask, fitness in zip(ant_masks, ant_fitness, strict=False):
        tau[mask] += (config.q * fitness) / n_ants

    return np.clip(tau, config.tau_min, config.tau_max)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config = ACOConfig()
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading fitness evaluation data (stratified subsample)...")
    X_tr, X_val, y_tr, y_val = load_fitness_data()
    feature_names = list(X_tr.columns)
    n_features = len(feature_names)
    print(f"Fitness eval: {len(X_tr):,} train / {len(X_val):,} val rows, {n_features} features")

    print("Loading MI ranking as heuristic (eta)...")
    eta = load_heuristic(feature_names)

    tau = np.full(n_features, 0.5)

    history = []
    best_fitness_ever = -np.inf
    best_mask_ever = None
    best_macro_f1_ever = None

    print(f"\nRunning ACO: {config.n_ants} ants x {config.n_iterations} iterations...")
    for iteration in range(1, config.n_iterations + 1):
        ant_masks = []
        ant_fitness = []
        ant_macro_f1 = []

        for _ in range(config.n_ants):
            mask = construct_ant(tau, eta, config.alpha, config.beta, rng)
            fitness, macro_f1 = evaluate_fitness(mask, X_tr, y_tr, X_val, y_val, config)
            ant_masks.append(mask)
            ant_fitness.append(fitness)
            ant_macro_f1.append(macro_f1)

        tau = update_pheromone(tau, ant_masks, ant_fitness, config)

        iter_best_idx = int(np.argmax(ant_fitness))
        iter_best_fitness = ant_fitness[iter_best_idx]
        iter_best_n_features = int(ant_masks[iter_best_idx].sum())
        iter_avg_fitness = float(np.mean(ant_fitness))

        if iter_best_fitness > best_fitness_ever:
            best_fitness_ever = iter_best_fitness
            best_mask_ever = ant_masks[iter_best_idx].copy()
            best_macro_f1_ever = ant_macro_f1[iter_best_idx]

        history.append(
            {
                "iteration": iteration,
                "best_fitness": iter_best_fitness,
                "avg_fitness": iter_avg_fitness,
                "best_n_features": iter_best_n_features,
                "global_best_fitness": best_fitness_ever,
            }
        )

        print(
            f"Iter {iteration:2d}/{config.n_iterations} "
            f"best={iter_best_fitness:.4f} avg={iter_avg_fitness:.4f}  "
            f"n_features={iter_best_n_features} global_best={best_fitness_ever:.4f}"
        )
    selected_features = [f for f, sel in zip(feature_names, best_mask_ever, strict=False) if sel]

    print(f"\nBest subset: {len(selected_features)}/{n_features} features")
    print(f"Best fitness: {best_fitness_ever:.4f} (macro F1: {best_macro_f1_ever:.4f})")
    print(f"Selected features: {selected_features}")

    pd.DataFrame({"feature": selected_features}).to_csv(
        REPORTS_DIR / "selected_features.csv", index=False
    )
    pd.DataFrame(history).to_csv(REPORTS_DIR / "aco_history.csv", index=False)
    pd.DataFrame({"feature": feature_names, "final_pheromone": tau}).sort_values(
        "final_pheromone", ascending=False
    ).to_csv(REPORTS_DIR / "aco_pheromone_final.csv", index=False)

    summary = {
        "n_features_total": n_features,
        "n_features_selected": len(selected_features),
        "best_fitness": round(float(best_fitness_ever), 4),
        "best_macro_f1": round(float(best_macro_f1_ever), 4),
        "config": vars(config),
    }
    (REPORTS_DIR / "aco_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        "\nSaved: reports/selected_features.csv, aco_history.csv, aco_pheromone_final.csv, aco_summary.json"
    )


if __name__ == "__main__":
    main()
