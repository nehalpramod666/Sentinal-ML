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


def load_fitness_data(sample_size: int = FITNESS_SAMPLE_SIZE):
    """Stratified subsample of train.csv, split further into an internal
    train/validation split used to score every ant's candidate subset.

    At small sample sizes, extremely rare classes (e.g. Heartbleed: 9 rows
    in the full training set) can end up with 0-1 examples in the drawn
    subsample, which breaks stratified splitting (a class needs at least 2
    members to appear in both train and validation). Such classes are
    dropped from the fitness-evaluation data with a printed warning — this
    only affects ACO's internal fitness signal, not the final model
    (Day 10+), which still trains/evaluates on the full class set.
    """
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_full = train_df.drop(columns=["Label_encoded"])
    y_full = train_df["Label_encoded"]

    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=sample_size, stratify=y_full, random_state=RANDOM_STATE
    )

    class_counts = y_sample.value_counts()
    too_rare = class_counts[class_counts < 2].index.tolist()
    if too_rare:
        print(
            f"WARNING: dropping {len(too_rare)} class(es) with <2 samples in this "
            f"{sample_size:,}-row subsample (too rare to stratify-split): {too_rare}"
        )
        keep_mask = ~y_sample.isin(too_rare)
        X_sample = X_sample[keep_mask]
        y_sample = y_sample[keep_mask]

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


def run_aco(
    config: ACOConfig,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    eta: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Run ACO once with the given config and pre-loaded fitness data.

    Returns a dict with best_mask, best_fitness, best_macro_f1, history
    (list of per-iteration stats), and final_pheromone. Factored out of
    main() so ml/aco_tuning.py (Day 9) can call this repeatedly with
    different configs and different (typically smaller, faster) fitness
    data without duplicating the ACO loop itself.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    feature_names = list(X_tr.columns)
    n_features = len(feature_names)
    tau = np.full(n_features, 0.5)

    history = []
    best_fitness_ever = -np.inf
    best_mask_ever = None
    best_macro_f1_ever = None

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
                "best_n_features": int(ant_masks[iter_best_idx].sum()),
                "global_best_fitness": best_fitness_ever,
            }
        )

        if verbose:
            print(
                f"Iter {iteration:2d}/{config.n_iterations} "
                f"best={iter_best_fitness:.4f} avg={iter_avg_fitness:.4f}   "
                f"global_best={best_fitness_ever:.4f}"
            )
    return {
        "best_mask": best_mask_ever,
        "best_fitness": best_fitness_ever,
        "best_macro_f1": best_macro_f1_ever,
        "history": history,
        "final_pheromone": tau,
        "feature_names": feature_names,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config = ACOConfig()

    print("Loading fitness evaluation data (stratified subsample)...")
    X_tr, X_val, y_tr, y_val = load_fitness_data()
    print(f"Fitness eval: {len(X_tr):,} train / {len(X_val):,} val rows, {X_tr.shape[1]} features")

    print("Loading MI ranking as heuristic (eta)...")
    eta = load_heuristic(list(X_tr.columns))

    print(f"\nRunning ACO: {config.n_ants} ants x {config.n_iterations} iterations...")
    result = run_aco(config, X_tr, y_tr, X_val, y_val, eta)

    selected_features = [
        f for f, sel in zip(result["feature_names"], result["best_mask"], strict=False) if sel
    ]

    print(f"\nBest subst: {len(selected_features)}/{len(result['feature_name'])} features")
    print(f"Best fitness: {result['best_fitness']:.4f} (macro F1: {result['best_macro_f1']:.4f})")
    print(f"Selected features: {selected_features}")

    pd.DataFrame({"feature": selected_features}).to_csv(
        REPORTS_DIR / "selected_features.csv", index=False
    )
    pd.DataFrame(result["history"]).to_csv(REPORTS_DIR / "aco_history.csv", index=False)
    pd.DataFrame(
        {"feature": result["feature_names"], "final_pheromone": result["final_pheromone"]}
    ).sort_values("final_pheromone", ascending=False).to_csv(
        REPORTS_DIR / "aco_pheromone_final.csv", index=False
    )

    summary = {
        "n_features_total": len(result["feature_names"]),
        "n_features_selected": len(selected_features),
        "best_fitness": round(float(result["best_fitness"]), 4),
        "best_macro_f1": round(float(result["best_macro_f1"]), 4),
        "config": vars(config),
    }
    (REPORTS_DIR / "aco_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        "\nSaved: reports/selected_features.csv, aco_history.csv, aco_pheromone_final.csv, aco_summary.json"
    )


if __name__ == "__main__":
    main()
