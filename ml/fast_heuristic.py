"""
Fast, subsampled heuristic information for ACO (Day 7-9).

ACO evaluates many candidate feature subsets across many ants and
iterations. Using the full-dataset MI computation (~15-20 min, see
ml/mutual_information.py) per iteration would make ACO impractically slow.

This module provides a stratified-subsample MI computation instead, used
only as ACO's internal heuristic (eta) signal — not as a substitute for the
canonical Day 6 ranking. Known limitation: with this dataset's extreme class
imbalance, a 100k-row stratified sample will contain few or zero examples of
the rarest classes (e.g. Heartbleed: 9 rows in the full training set), so
this heuristic may under-represent rare-class-specific signal. That's an
acceptable tradeoff here because ACO's pheromone mechanism lets useful
features get reinforced over many iterations even if any single heuristic
pass is imperfect — unlike the Day 6 ranking, which is a one-time reported
result and needs to be as accurate as practical.

Not run standalone; imported by ml/aco.py.
"""

import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 100_000


def compute_fast_mi(
    X: DataFrame, y: pd.Series, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> pd.Series:
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X, y

    scores = mutual_info_classif(
        X_sample, y_sample, discrete_features=False, random_state=RANDOM_STATE
    )
    return pd.Series(scores, index=X.columns, name="mi_score")
