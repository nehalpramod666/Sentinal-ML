# ACO Feature Selection — Design Document

**Day 7 deliverable.** Design only — implementation is Day 8 (`ml/aco.py`).

## 1. Problem framing

Feature selection is framed as a **binary subset selection problem**, not a
path-construction problem (the more common ACO formulation for routing/TSP).
Each of the 77 features is either included or excluded from a candidate
subset; there is no ordering or sequence between features, so ants don't
"walk a graph" — instead, each ant independently decides inclusion/exclusion
for every feature based on that feature's pheromone level and heuristic
value. This is the standard adaptation of ACO to feature selection (as
opposed to ACO's original TSP formulation).

## 2. Ant representation

Each ant's candidate solution is a **binary vector** of length 77:

ant_solution = [1, 0, 1, 1, 0, ..., 1] # 1 = feature included, 0 = excluded


Per your decision, subset size is **variable, not fixed** — an ant might end
up selecting 15 features or 45; nothing forces a specific count. This lets
ACO discover the natural size of an informative subset rather than assuming
it upfront (contrast with, say, "always pick the top 20 by some score" —
that would presuppose the answer this experiment is meant to find).

## 3. Heuristic values (eta)

Each feature `i` has a heuristic value `eta_i` derived from its Mutual
Information score (Day 6), **normalized to [0, 1]**:

eta_i = mi_score_i / max(mi_score)


Using `reports/feature_scores.csv` (canonical, full-dataset MI) as the
static heuristic reference table, and `ml/fast_heuristic.py`'s subsampled
computation if/when heuristics need to be recomputed on a resampled
training subset during ACO runs (not required if using the static table —
see Section 7).

## 4. Pheromone representation

Since this is a per-feature binary decision (not a graph edge), pheromone is
a **single vector** `tau`, one value per feature, representing the learned
"desirability of including this feature":

tau_i in [tau_min, tau_max] # bounded, e.g. [0.01, 0.99]


**Initialization:** all `tau_i` start at a neutral 0.5 (no prior bias toward
inclusion or exclusion) — the MI-derived heuristic provides the initial
guidance instead, and pheromone updates shift the balance as ants report
back fitness results.

**Bounding (`tau_min`/`tau_max`):** borrowed from Max-Min Ant System (MMAS),
a well-established ACO variant. Without bounds, pheromone can collapse to
0 or 1 for some features early on and never recover even if later evidence
suggests otherwise (premature convergence / stagnation) — clamping avoids
this and is a standard, citable design choice if asked about it later.

## 5. Feature inclusion probability

For each feature `i`, an ant decides inclusion via a probability that
combines pheromone and heuristic, both weighted by exponents `alpha` (weight
given to pheromone/learned experience) and `beta` (weight given to
heuristic/MI):

P_include(i) = (tau_i^alpha * eta_i^beta)
----------------------------------------------------
(tau_i^alpha * eta_i^beta) + ((1-tau_i)^alpha * (1-eta_i)^beta)


The ant draws a random number in [0,1]; if it's below `P_include(i)`, feature
`i` is included. This is done **independently per feature**, which is what
produces variable-size subsets (per your decision) rather than a fixed-count
selection.

`alpha` and `beta` are exactly the hyperparameters Day 9 tunes.

## 6. Fitness function

Each ant's candidate subset is evaluated by training a Gaussian Naive Bayes
classifier on just those features (validation split) and scoring it. Per
your decision, fitness is a **weighted blend of classification quality and
subset compactness**:

fitness = w_quality * quality_score + w_compact * (1 - n_selected / n_total)


Where `w_quality + w_compact = 1` (e.g. 0.8 / 0.2 as a starting point,
tunable on Day 9), `n_selected` is how many features this ant chose, and
`n_total` = 77.

**Design decision on `quality_score`:** using **macro F1**, not raw accuracy,
despite the original brief's phrasing ("higher validation accuracy"). Reason:
Day 4's baseline already demonstrated that on this severely imbalanced
dataset (80% BENIGN), accuracy can look reasonable or even be optimized while
badly failing on minority/attack classes — exactly the traffic this project
is meant to detect. Macro F1 weights all 15 classes equally regardless of
size, so ACO is rewarded for subsets that help distinguish rare attacks too,
not just subsets that predict BENIGN well. This is a one-line swap in Day 8's
code (`quality_score = accuracy` instead of `macro_f1`) if a different
framing is wanted later — flagging the choice explicitly rather than
changing the metric silently.

**Practical note carried over from Day 6:** fitness evaluation happens once
per ant per iteration, and with many ants over many iterations, this cannot
run GaussianNB training on the full 2.26M-row training set each time (same
scaling problem as MI). Fitness evaluation will use a **stratified subsample**
of the training data (separate from the full-dataset canonical results),
consistent with the fast/canonical split established in `ml/fast_heuristic.py`.

## 7. Pheromone update rule

After all ants in an iteration report their fitness, pheromone updates in
two steps, per standard ACO:

**Evaporation** (all features, prevents unbounded pheromone growth and lets
the colony "forget" outdated information):

tau_i <- (1 - rho) * tau_i

where `rho` is the evaporation rate (Day 9 hyperparameter, e.g. 0.1-0.3).

**Reinforcement** (only for ants that included feature `i`, scaled by that
ant's fitness — better subsets contribute more):

tau_i <- tau_i + sum over ants k that included feature i of (Q * fitness_k)

where `Q` is a scaling constant. After both steps, `tau_i` is clamped to
`[tau_min, tau_max]`.

## 8. Stopping criteria

ACO runs for a fixed number of iterations (Day 9 tunes this), tracking the
best subset (highest fitness) seen across all ants and iterations —
**elitism**: even if a later iteration's ants do worse on average, the
best-ever subset is retained as the final answer rather than whatever the
last iteration happened to produce.

## 9. Hyperparameters summary (tuned Day 9)

| Parameter | Meaning | Starting value |
|---|---|---|
| `n_ants` | Ants per iteration | 20 |
| `n_iterations` | Total iterations | 30 |
| `alpha` | Pheromone influence weight | 1.0 |
| `beta` | Heuristic (MI) influence weight | 2.0 |
| `rho` | Evaporation rate | 0.2 |
| `Q` | Pheromone deposit scaling constant | 1.0 |
| `tau_min` / `tau_max` | Pheromone bounds | 0.01 / 0.99 |
| `w_quality` / `w_compact` | Fitness blend weights | 0.8 / 0.2 |

These are starting points, not final values — Day 9 is specifically about
experimentally tuning these and logging results.

## 10. What Day 8 implements

- `Ant` construction logic (Section 5's probability formula)
- Fitness evaluation function (Section 6, GaussianNB on subsampled data)
- Pheromone update loop (Section 7)
- Main ACO loop tying it together, with elitism (Section 8)
- Logging of best subset, best fitness, and pheromone values per iteration
  (needed for Day 13's "Feature Selection Evolution" and "Pheromone Heatmap"
  visualizations from the original project brief)
