"""
Day 9 — ACO hyperparameter tuning.

Runs multiple ACO configurations (varying n_ants, n_iterations, alpha,
beta, rho) on a smaller, faster fitness sample, logs results to
reports/aco_tuning_experiments.csv, then re-runs the winning configuration
at full scale (1,000,000-row fitness sample, matching Day 8's canonical
setup) and overwrites the official reports/selected_features.csv etc.

Run from the project root:
    python -m ml.aco_tuning
"""

import time
from pathlib import Path

import pandas as pd

from ml.aco import ACOConfig, load_fitness_data, load_heuristic, run_aco

REPORTS_DIR = Path("reports")

TUNING_SAMPLE_SIZE = 100_000
FINAL_SAMPLE_SIZE = 1_000_000

EXPERIMENTS = [
    {"name": "baseline", "n_ants": 20, "n_iterations": 30, "alpha": 1.0, "beta": 2.0, "rho": 0.2},
    {
        "name": "more_iterations",
        "n_ants": 20,
        "n_iterations": 60,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.2,
    },
    {
        "name": "higher_alpha",
        "n_ants": 20,
        "n_iterations": 30,
        "alpha": 2.0,
        "beta": 2.0,
        "rho": 0.2,
    },
    {"name": "lower_beta", "n_ants": 20, "n_iterations": 30, "alpha": 1.0, "beta": 1.0, "rho": 0.2},
    {"name": "more_ants", "n_ants": 40, "n_iterations": 30, "alpha": 1.0, "beta": 2.0, "rho": 0.2},
    {
        "name": "slower_evaporation",
        "n_ants": 20,
        "n_iterations": 30,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
    },
    {
        "name": "faster_evaporation",
        "n_ants": 20,
        "n_iterations": 30,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.4,
    },
]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tuning fitness data (stratified {TUNING_SAMPLE_SIZE:,}-row sample)...")
    X_tr, X_val, y_tr, y_val = load_fitness_data(sample_size=TUNING_SAMPLE_SIZE)
    eta = load_heuristic(list(X_tr.columns))
    print(f"Tuning eval: {len(X_tr):,} train / {len(X_val):,} val rows\n")

    results = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        config = ACOConfig(
            n_ants=exp["n_ants"],
            n_iterations=exp["n_iterations"],
            alpha=exp["alpha"],
            beta=exp["beta"],
            rho=exp["rho"],
        )
        print(f"--- Experiment: {name} ({exp}) ---")
        start = time.time()
        result = run_aco(config, X_tr, y_tr, X_val, y_val, eta, verbose=False)
        elapsed = time.time() - start

        n_selected = int(result["best_mask"].sum())
        print(
            f" best_fitness={result['best_fitness']:.4f}"
            f" macro_f1={result['best_macro_f1']:.4f}"
            f" n_features={n_selected} runtime={elapsed:.1f}s\n"
        )

        results.append(
            {
                "experiment": name,
                "n_ants": exp["n_ants"],
                "n_iterations": exp["n_iterations"],
                "alpha": exp["alpha"],
                "beta": exp["beta"],
                "rho": exp["rho"],
                "best_fitness": round(float(result["best_fitness"]), 4),
                "best_macro_f1": round(float(result["best_macro_f1"]), 4),
                "n_features_selected": n_selected,
                "runtime_seconds": round(elapsed, 1),
            }
        )

    results_df = pd.DataFrame(results).sort_values("best_fitness", ascending=False)
    exp_path = REPORTS_DIR / "aco_tuning_experiments.csv"
    results_df.to_csv(exp_path, index=False)
    print(f"Saved: {exp_path}\n")
    print(results_df.to_string(index=False))

    best_exp = results_df.iloc[0]
    print(f"\nBest config: {best_exp['experiment']} (fitness={best_exp['best_fitness']:.4f})")
    print(f"\nRe-running best config at full scale ({FINAL_SAMPLE_SIZE:,}-row sample)...")

    final_config = ACOConfig(
        n_ants=int(best_exp["n_ants"]),
        n_iterations=int(best_exp["n_iterations"]),
        alpha=float(best_exp["alpha"]),
        beta=float(best_exp["beta"]),
        rho=float(best_exp["rho"]),
    )
    X_tr_full, X_val_full, y_tr_full, y_val_full = load_fitness_data(sample_size=FINAL_SAMPLE_SIZE)
    eta_full = load_heuristic(list(X_tr_full.columns))

    final_result = run_aco(
        final_config, X_tr_full, y_tr_full, X_val_full, y_val_full, eta_full, verbose=True
    )

    selected_features = [
        f
        for f, sel in zip(final_result["feature_names"], final_result["best_mask"], strict=False)
        if sel
    ]
    print(f"\nFinal best subset: {len(selected_features)} features")
    print(
        f"Final best fitness: {final_result['best_fitness']:.4f}  (macro F1: {final_result['best_macro_f1']:.4f})"
    )

    pd.DataFrame({"feature": selected_features}).to_csv(
        REPORTS_DIR / "selected_features.csv", index=False
    )
    pd.DataFrame(final_result["history"]).to_csv(REPORTS_DIR / "aco_history.csv", index=False)
    pd.DataFrame(
        {
            "feature": final_result["feature_names"],
            "final_pheromone": final_result["final_pheromone"],
        }
    ).sort_values("final_pheromone", ascending=False).to_csv(
        REPORTS_DIR / "aco_pheromone_final.csv", index=False
    )

    print("\nOverwrote: reports/selected_features.csv, aco_history.csv, aco_pheromone_final.csv")
    print("with the tuned, full-scale result.")


if __name__ == "__main__":
    main()
