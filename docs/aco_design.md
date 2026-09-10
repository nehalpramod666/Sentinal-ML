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
## Implementation notes (Day 8)

Implemented in `ml/aco.py` per this design. One bug found and fixed during
initial runs:

**Pheromone saturation bug**: the initial `update_pheromone` implementation
deposited `Q * fitness` per ant that selected a feature, without normalizing
by ant count. With 20 ants and even moderate agreement on a feature, this
caused pheromone to hit `tau_max` within 1-2 iterations regardless of true
feature quality, collapsing the search (avg subset size ballooned from 30 to
consistently 55-58 features, and iteration-2-onward fitness dropped ~25%
below iteration 1 and never recovered — evidence the search had effectively
stopped discriminating between features). Fixed by dividing the deposit by
`n_ants`, matching standard ACO practice where deposit magnitude should
reflect average solution quality, not raw agreement count.

**Post-fix results** (20 ants, 30 iterations, alpha=1.0, beta=2.0, rho=0.2):
- Best subset: 30/77 features, fitness 0.3073 (macro F1 0.2316)
- Final pheromone distribution: mean 0.512, with 38 features >0.9 and 36
  features <0.1 — a bimodal, polarized distribution consistent with genuine
  convergence on a binary selection problem (clear "keep" vs. "drop" groups),
  not a saturation artifact (compare: pre-fix mean was 0.761 with 77% of
  features pinned at the ceiling)
- Average fitness per iteration rose from 0.2692 (iter 1) to ~0.277-0.278
  (iters 20-30), showing the search population improving over time
- **Open question for Day 9**: the single best-ever fitness (0.3073) was set
  at iteration 1 and never exceeded, even as the average improved. Plausible
  explanation: iteration 1's ants draw mostly from the MI heuristic (tau
  starts uniform at 0.5), which may already be a strong starting point;
  pheromone-guided search over 30 iterations improved the population's
  average without yet finding a single ant that beats that early result.
  Day 9 will test whether more iterations, higher alpha (more pheromone
  influence), or different beta values close this gap.

## Hyperparameter tuning (Day 9)

Ran 7 configurations on a 100,000-row stratified tuning sample (faster than
the 1,000,000-row canonical sample), varying one parameter at a time from
the Day 8 baseline (n_ants=20, n_iterations=30, alpha=1.0, beta=2.0, rho=0.2).
Full results: `reports/aco_tuning_experiments.csv`.

| Experiment | n_ants | n_iter | alpha | beta | rho | Fitness | Macro F1 | # Features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **lower_beta** | 20 | 30 | 1.0 | **1.0** | 0.2 | **0.2992** | 0.2571 | 41 |
| faster_evaporation | 20 | 30 | 1.0 | 2.0 | **0.4** | 0.2921 | 0.2092 | 29 |
| more_ants | **40** | 30 | 1.0 | 2.0 | 0.2 | 0.2909 | 0.2305 | 36 |
| baseline | 20 | 30 | 1.0 | 2.0 | 0.2 | 0.2854 | 0.2399 | 41 |
| more_iterations | 20 | **60** | 1.0 | 2.0 | 0.2 | 0.2854 | 0.2399 | 41 |
| higher_alpha | 20 | 30 | **2.0** | 2.0 | 0.2 | 0.2833 | 0.2178 | 35 |
| slower_evaporation | 20 | 30 | 1.0 | 2.0 | **0.1** | 0.2833 | 0.2178 | 35 |

**Finding**: lowering `beta` from 2.0 to 1.0 (reducing MI-heuristic influence,
increasing relative weight of pheromone/learned experience) produced the
largest single improvement of any tested change — larger than doubling ant
count or doubling iterations. This suggests the raw MI ranking, while a
reasonable starting heuristic, was somewhat over-constraining the search;
letting the colony's own learning carry more relative weight found better
subsets.

### Full-scale confirmation (lower_beta config, 1,000,000-row sample)

Re-ran the winning config at the canonical 1M-row scale:
- `global_best` climbed across iterations 1-4 (0.2931 -> 0.3012 -> 0.3038 ->
  0.3045), then plateaued for the remaining 26 iterations without further
  improvement
- Final: 35/77 features, fitness 0.3045, macro F1 0.2443

**Comparison to Day 8's un-tuned result** (0.3073, set at iteration 1, never
improved thereafter): tuning changed *how* convergence happens — genuine
multi-iteration improvement (iterations 1-4) instead of a single early spike
— but the final fitness (0.3045) is marginally *below* Day 8's original best
(0.3073). This is an honest, non-clean-win result: beta=1.0 improved the
*search dynamics* (visible learning over iterations, better average fitness
across the population) without improving the *single best subset found*, at
least within 30 iterations. A longer run (60+ iterations) with beta=1.0,
not yet tested at full scale due to time cost, is a natural next experiment
if further tuning were pursued.

### Rare-class handling during tuning

At the 100,000-row tuning sample size, 2 classes (Heartbleed, and one other
ultra-rare class) had fewer than 2 examples in the drawn subsample and were
automatically dropped from that experiment's fitness evaluation (see
`load_fitness_data`'s rare-class guard in `ml/aco.py`). This only affects
ACO's internal fitness signal during the tuning sweep — the final 1M-row run
and all downstream model training (Day 10+) use the full class set.

### Selected feature subset (final, post-tuning)

`reports/selected_features.csv` — 35 features, superseding Day 8's 30-feature
result. Will be compared against the Day 4 full-feature baseline (78/77
features, accuracy 0.1635) on Day 10.
