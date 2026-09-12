# Dataset: CICIDS2017

**Source:** Canadian Institute for Cybersecurity — https://www.unb.ca/cic/datasets/ids-2017.html
**File used:** `MachineLearningCSV.zip` (CICFlowMeter-processed flows, not raw PCAP)
**Citation:** Sharafaldin, I., Lashkari, A.H., and Ghorbani, A.A., "Toward Generating a New
Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.

## Overview

- **8 CSV files**, one per capture session (Monday–Friday, July 3–7, 2017)
- **2,830,743 total rows**
- **79 columns** — 78 numerical flow features + 1 categorical `Label` column
- Column set is **consistent across all 8 files** (verified via `scripts/inspect_dataset.py`)
- Column names have leading whitespace in the raw CSVs (e.g. `" Flow Duration"`) — strip on load

## Files

| File | Rows | Labels present |
|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | BENIGN only |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | BENIGN, FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | BENIGN, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | BENIGN, Web Attack (Brute Force/XSS/SQL Injection) |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | BENIGN, Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | BENIGN, Bot |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | BENIGN, PortScan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | BENIGN, DDoS |

## Label distribution (combined, all 8 files)

| Label | Count | % of total |
|---|---:|---:|
| BENIGN | 2,273,097 | 80.30% |
| DoS Hulk | 231,073 | 8.16% |
| PortScan | 158,930 | 5.61% |
| DDoS | 128,027 | 4.52% |
| DoS GoldenEye | 10,293 | 0.36% |
| FTP-Patator | 7,938 | 0.28% |
| SSH-Patator | 5,897 | 0.21% |
| DoS slowloris | 5,796 | 0.20% |
| DoS Slowhttptest | 5,499 | 0.19% |
| Bot | 1,966 | 0.07% |
| Web Attack – Brute Force | 1,507 | 0.05% |
| Web Attack – XSS | 652 | 0.02% |
| Infiltration | 36 | 0.00% |
| Web Attack – SQL Injection | 21 | 0.00% |
| Heartbleed | 11 | 0.00% |

**Class imbalance is severe**: BENIGN is 80.3% of the data; the four rarest attack
types (Infiltration, Web Attack – SQL Injection, Heartbleed, and even Bot/Web Attack – XSS)
have under 2,000 samples combined. This has direct implications for:
- **Preprocessing (Day 3):** stratified train/test split is mandatory to keep rare
  classes represented in both sets.
- **Baseline model (Day 4):** accuracy alone will be a misleading metric; report
  precision/recall/F1 per class, not just overall accuracy.
- **Fuzzy risk scoring (Week 2):** rare-but-severe attacks (Heartbleed, Infiltration)
  are exactly the kind of thing binary classification underserves — worth calling
  out explicitly in the final write-up as the motivating case for graded risk levels.

## Data quality issues found

1. **Missing values** — isolated to a single column, `Flow Bytes/s`, across all files:

   | File | Missing rows |
   |---|---:|
   | Wednesday-workingHours.pcap_ISCX.csv | 1,008 |
   | Tuesday-WorkingHours.pcap_ISCX.csv | 201 |
   | Monday-WorkingHours.pcap_ISCX.csv | 64 |
   | Friday-WorkingHours-Morning.pcap_ISCX.csv | 28 |
   | Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 15 |
   | Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 18 |
   | Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 20 |
   | Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 4 |

   Total: 1,358 rows (~0.05% of the dataset) — negligible; safe to drop these rows
   rather than impute.

2. **Mislabeled encoding on 3 Web Attack labels** — `Web Attack` labels contain a
   `�` replacement character where a dash should be (`Web Attack \ufffd Brute Force`
   instead of `Web Attack – Brute Force`). Root cause: source CSV encoded in
   Windows-1252/Latin-1, read as UTF-8. **Fix in preprocessing**: either re-read
   affected files with `encoding="cp1252"`, or normalize `Label` strings by
   replacing `�` and stripping to a canonical form before encoding labels.

3. **Column name whitespace** — most column names have a leading space (CICFlowMeter
   quirk). Strip with `df.columns = df.columns.str.strip()` immediately after load.

## Feature categories (from the 78 numerical columns)

Flow-level statistics extracted by CICFlowMeter, grouped roughly as:
- **Identifiers/timing:** Flow ID components, Timestamp, Flow Duration
- **Packet counts/sizes:** Total Fwd/Bwd Packets, packet length min/max/mean/std
- **Rate features:** Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s
- **Inter-arrival time (IAT):** Flow IAT mean/std/max/min, Fwd/Bwd IAT stats
- **TCP flags:** FIN/SYN/RST/PSH/ACK/URG/CWE/ECE flag counts
- **Window/segment:** Init_Win_bytes, min_seg_size, Active/Idle time stats

Full column list is in `scripts/inspect_dataset.py` output — not reproduced here to
keep this doc scannable; run the script for the authoritative list against your copy.

## Next steps (Day 3 — Preprocessing)

- [ ] Strip column name whitespace
- [ ] Drop the 1,358 rows with missing `Flow Bytes/s`
- [ ] Normalize the 3 mojibake `Web Attack` labels
- [ ] Encode `Label` (multiclass: keep all 15 classes, or binary BENIGN/ATTACK — decide before Day 4 baseline)
- [ ] Concatenate all 8 files into one DataFrame
- [ ] Stratified train/test split (recommend 80/20, `stratify=y`)
- [ ] Save `data/processed/train.csv` and `data/processed/test.csv`

## Design decisions (Day 2)

- **Label encoding: multiclass (all 15 classes retained).** Chosen over binary
  BENIGN/ATTACK to preserve attack-type granularity, which the fuzzy risk-scoring
  stage depends on for producing graded (not flat binary) severity.
- **Ultra-rare classes (Infiltration: 36, Heartbleed: 11) kept as-is**, no merging
  or dropping. Caveat: an 80/20 stratified split leaves only ~2 test samples for
  Heartbleed, so per-class precision/recall for the rarest classes will be
  statistically noisy — flag this explicitly when reporting Day 4/10 results
  rather than treating those numbers as reliable.
## Preprocessing results (Day 3)

- Rows after dropping missing `Flow Bytes/s`: 2,829,385
- Inf/-inf values found: 3,018 across 1,509 rows (Flow Bytes/s and Flow Packets/s
  both become inf on the same zero-duration flows — 2 inf cells per bad row)
- Final dataset: 2,827,876 rows (2,262,300 train / 565,576 test), stratified 80/20
- No identifier columns (Flow ID, Source/Destination IP, Timestamp) were present
  in the MachineLearningCSV variant — nothing to drop there
- Outputs: `data/processed/train.csv`, `data/processed/test.csv`,
  `data/processed/label_mapping.json`

## Baseline model results (Day 4)

Gaussian Naive Bayes, all 78 features, no feature selection:

- **Accuracy: 0.1430** — worse than the 0.803 majority-class (all-BENIGN) baseline
- **Macro F1: 0.1956** (Precision 0.24 / Recall 0.62)
- **BENIGN recall: 0.05** — the model misclassifies ~95% of normal traffic as an
  attack, dragging down precision across most attack classes
- Root cause: GaussianNB assumes per-class Gaussian feature distributions; BENIGN
  is a broad, heterogeneous mix of legitimate traffic types with wide/multi-modal
  distributions, which lose out to attack classes' narrower, more repetitive
  traffic patterns under NB's independence + normality assumptions
- **This result motivates the project's core design**: ACO feature selection
  (Week 2) is expected to substantially outperform this full-feature baseline by
  removing noisy/uninformative dimensions that worsen NB's distributional
  assumptions; fuzzy risk scoring is motivated by exactly this kind of
  low-confidence, easily-confused binary-ish output.
- Full metrics, confusion matrix, and per-class breakdown saved in
  `reports/baseline_results.json`
## Preprocessing fix (Day 4 follow-up): duplicate column and impossible values

Feature-scale investigation (post Day-4 baseline) surfaced two data quality issues:

- **Duplicate column**: `Fwd Header Length` appeared twice in the raw CSVs
  (pandas auto-renamed the second occurrence to `Fwd Header Length.1` on load).
  Verified identical across all rows — dropped the duplicate. Features: 78 -> 77.
- **Impossible negative values**: 150 rows (0.005% of the combined dataset) had
  physically impossible negative values in `Flow Duration`, `Flow Bytes/s`, or
  `Fwd Header Length` (e.g. `Fwd Header Length` as low as -3.22e10) — a known
  CICFlowMeter generation artifact (likely integer underflow), not a
  preprocessing bug. Dropped these rows.

`ml/preprocessing.py` updated with `drop_duplicate_columns()` and
`drop_impossible_values()`; train/test regenerated (2,262,180 / 565,546 rows).

### Re-run baseline (Day 4, corrected data)

| Metric | Before fix | After fix |
|---|---:|---:|
| Accuracy | 0.1430 | 0.1635 |
| BENIGN recall | 0.05 | 0.08 |
| Macro F1 | 0.1956 | 0.1977 |
| Weighted F1 | 0.1679 | 0.2068 |

**Conclusion**: the fix produced a small, real improvement, but the core failure
mode is unchanged — BENIGN recall remains far below its 80% class share, still
worse than a majority-class baseline. This confirms the earlier hypothesis: the
problem is GaussianNB's Gaussian/independence assumptions breaking down on
BENIGN's broad, heterogeneous traffic distribution, not data corruption. Data
cleanliness was a contributing factor at the margin, not the root cause.
## EDA findings (Day 5)

Plots generated: `reports/plots/class_distribution.png`,
`feature_histograms.png`, `correlation_heatmap.png` (top 20 features by variance).

### Class imbalance
Visually confirms the Day 2 label counts — BENIGN dominates at 80.3%, with a
long tail of attack classes spanning 4 orders of magnitude down to single-digit
sample counts (Heartbleed, Infiltration). Log-scale y-axis was necessary; a
linear scale makes every class but BENIGN and DoS Hulk invisible.

### Feature redundancy (correlation heatmap, top 20 by variance)

22 feature pairs found with |r| > 0.9 among the top-variance features, clustering
into one dominant redundant group — all timing/duration-related:

- `Flow Duration` <-> `Fwd IAT Total`: r = 0.999 (near-identical)
- A tightly interlinked cluster: `Fwd IAT Max`, `Flow IAT Max`, `Idle Max`,
  `Idle Mean`, `Idle Min`, `Fwd IAT Std`, `Flow IAT Std` — all pairwise r > 0.9
- `Bwd IAT Mean` <-> `Bwd IAT Min`: r = 0.933

**Implication for Day 6-9**: this is direct empirical evidence that the raw
78/77-feature space contains substantial redundancy concentrated in
timing-based features. Expect Mutual Information ranking and ACO feature
selection to both down-weight or drop several of these in favor of one
representative timing feature, rather than needing near-identical duplicates.

### BENIGN vs. Attack distribution overlap

Histograms of 12 representative features (packet stats, timing, rate, flags),
clipped to 1st-99th percentile, show substantial distributional overlap between
BENIGN and Attack traffic on most individual features — consistent with the
Day 4 baseline finding that no single feature (and apparently not the full
linear combination GaussianNB assumes) cleanly separates the two classes.
This is the visual counterpart to the low BENIGN recall (0.08) seen in the
baseline: BENIGN's distribution is broad and overlaps heavily with several
attack types' distributions on individual features, requiring either feature
selection to isolate more separable dimensions (ACO) or a model that doesn't
assume Gaussian per-feature independence.
## Mutual Information ranking (Day 6)

Computed via `mutual_info_classif` (KNN-based estimator) on the full training
set (2,262,180 rows x 77 features). **Runtime: ~15-20 minutes** — full ranking
kept as this run's canonical result since it already completed, but going
forward (ACO tuning, Day 7-9) MI/heuristic computations will use a stratified
100,000-row subsample for practicality; full-dataset ranking is not required
again since Day 6's results already establish the reference ranking.

### Top 5 features by MI
| Rank | Feature | MI Score |
|---|---|---:|
| 1 | Average Packet Size | 0.5887 |
| 2 | Packet Length Mean | 0.5609 |
| 3 | Packet Length Std | 0.5580 |
| 4 | Packet Length Variance | 0.5570 |
| 5 | Subflow Bwd Bytes | 0.4986 |

Packet-size statistics dominate the top of the ranking, suggesting attack
traffic in this dataset is more distinguishable by packet size patterns than
by timing patterns — despite timing features (Flow Duration, IAT) having the
highest raw variance (see Day 5 EDA). Variance and MI are measuring different
things: MI is target-aware, variance is not. This is a useful contrast to
highlight in the final write-up.

### Zero/near-zero MI features — dead or near-dead columns

Five features are **exactly constant** (single unique value, 0) across all
2,262,180 rows: `Bwd PSH Flags`, `Bwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`,
`Fwd Avg Bulk Rate`, `Fwd Avg Packets/Bulk`. Root cause: CICFlowMeter's "bulk
transfer" and backward-PSH metrics essentially never populate for this
dataset's traffic patterns. These carry zero information and are guaranteed
candidates for ACO to deselect — worth flagging explicitly rather than
silently letting ACO discover it, since it validates the algorithm is working
as expected if these end up excluded from every ant's selected subset.

Two more (`RST Flag Count`, `CWE Flag Count`) are non-constant but extremely
rare (<0.03% of rows non-zero), rounding to ~0 MI. Kept in the feature set —
borderline, low information, but not literally constant.

Full ranking: `reports/feature_scores.csv`.

## ACO-selected model vs. full-feature baseline (Day 10)

GaussianNB trained on ACO's 35-feature subset (Day 9, tuned), evaluated on
the same held-out test set as the Day 4 baseline. Full results:
`reports/aco_model_results.json`.

| Metric | Baseline (77 features) | ACO-selected (35 features) | Delta |
|---|---:|---:|---:|
| Accuracy | 0.1635 | 0.1845 | +0.0210 |
| Precision (macro) | 0.2593 | 0.2747 | +0.0154 |
| Recall (macro) | 0.5220 | 0.4903 | -0.0317 |
| F1 (macro) | 0.1977 | 0.2095 | +0.0118 |
| F1 (weighted) | 0.2068 | 0.2305 | +0.0237 |
| Train time | 5.19s | 1.69s | -67.4% |
| Inference time | 5.48s | 2.46s | -55.2% |
| ROC AUC (macro, OvR) | not computed | 0.8896 | — |

**Efficiency gains are strong and unambiguous**: 3.1x faster training, 2.2x
faster inference, using 45% of the original features (35/77) — directly
validating the project's stated objectives of reduced training time and
improved inference efficiency. ROC AUC of 0.8896 is a strong absolute score,
indicating the model's predicted class probabilities do rank true classes
well even where its hard classification decisions (argmax) do not.

**Quality metrics improved modestly but genuinely**: accuracy, macro
precision, macro F1, and weighted F1 all increased. However, **macro recall
decreased** (0.522 -> 0.490), and critically, **BENIGN recall remains
severely impaired** (0.08 baseline -> 0.09 ACO-selected) — the model still
misclassifies over 90% of legitimate traffic as some attack type either way.

**Interpretation**: ACO feature selection successfully addressed the
project's efficiency goals and produced small, consistent quality gains, but
did **not** resolve GaussianNB's core failure mode. This is expected, not a
flaw in the ACO implementation: the underlying issue (Day 4) is that
GaussianNB's per-class Gaussian/independence assumptions break down on
BENIGN's broad, heterogeneous traffic distribution — a *model* limitation
that persists regardless of which feature subset is used, since no subset
of the original 77 features changes what BENIGN traffic actually looks like
statistically. This result is the direct empirical justification for the
project's fuzzy risk-scoring layer (Week 2 remainder): rather than relying
on a single classifier's unreliable hard decision boundary, converting
prediction probabilities into graded risk levels (Low/Medium/High/Critical)
provides a more honest and actionable signal precisely in the regime where
GaussianNB's binary-ish classification is this unreliable.

## Fuzzy risk inference engine (Day 11)

Implemented `fuzzy/engine.py`: a Mamdani fuzzy inference system combining
three inputs (probability, confidence, traffic_density) into a risk score
(0-100) and categorical level (Low/Medium/High/Critical).

**Bug found and fixed — incomplete rule coverage**: the initial hand-written
16-rule set left gaps in the 27-combination input space (3 terms x 3
variables); specifically, `probability=medium & confidence=low &
traffic_density=high` had no matching rule, causing `sim.output` to be
empty for that input region (`KeyError: 'risk'`). Fixed by generating all
27 rules programmatically via an additive scoring scheme (low=0, medium=1,
high=2 per input; summed score maps to output level), guaranteeing complete
coverage by construction rather than relying on manually enumerating every
case.

**Design limitation found via scenario testing**: using P(top class) as the
`probability` input produced backwards results — a confident, high-probability
BENIGN prediction scored as High risk, identical to a confident attack
prediction, since both have high probability and high confidence regardless
of which class was predicted. Fixed by redefining `probability` as
`P(attack) = 1 - P(BENIGN)`, directly encoding attack-likelihood rather than
raw prediction strength. This will be computed from real model output on
Day 12.

Sanity-check scenarios (synthetic inputs, `python -m fuzzy.engine`):

| Scenario | Prob | Conf | Density | Score | Level |
|---|---:|---:|---:|---:|---|
| Confident attack, heavy traffic | 0.95 | 0.90 | 0.90 | 92.22 | Critical |
| Confident attack, light traffic | 0.95 | 0.90 | 0.10 | 71.06 | High |
| Ambiguous prediction, heavy traffic | 0.55 | 0.20 | 0.85 | 42.50 | Medium |
| Ambiguous prediction, light traffic | 0.55 | 0.20 | 0.15 | 14.09 | Low |
| Everything moderate | 0.50 | 0.50 | 0.50 | 42.50 | Medium |

Traffic density's amplifying effect is visible in the "ambiguous prediction"
pair: identical probability/confidence, but heavy traffic pushes Medium vs.
light traffic staying Low — matching the intended design where high-volume
ambiguous traffic warrants more scrutiny than low-volume ambiguous traffic.
## Fuzzy-model integration (Day 12)

Trained the final GaussianNB model on ACO's 35 selected features (full
training set) and saved it (`models/model.pkl`, `models/model_metadata.json`).
Built `fuzzy/integration.py`, running real test-set predictions through the
corrected P(attack) calculation, confidence margin, and normalized traffic
density into Day 11's fuzzy engine end-to-end. Sample output:
`reports/fuzzy_integration_sample.csv` (84 stratified rows).

**Pipeline correctness verified**: risk scores traced by hand against the
27-rule scoring table match exactly (e.g. prob_attack=high + confidence=high
+ density=low -> score 4 -> "High", 71.06 — the modal outcome across the
sample). No plumbing errors.

**Finding: GaussianNB's predict_proba is nearly saturated (binary) on this
dataset.** Across the 84-row sample, `prob_attack` = P(attack) = 1-P(BENIGN)
was almost always exactly 0.0000 or 1.0000, with only a handful of
intermediate values (e.g. 0.0379). This is a known Naive Bayes pathology:
because NB computes the joint likelihood as a *product* of per-feature
likelihoods (the "naive" independence assumption), even mild per-feature
signal compounds multiplicatively across 35 features into near-total
certainty at the joint posterior level — regardless of whether that
certainty is actually warranted. This is consistent with, and likely
compounded by, the extreme feature-scale disparity documented in Day 5's
EDA (features spanning single digits to tens of millions).

**Practical consequence**: in the current fuzzy system, `confidence` (the
top-1/top-2 margin) carries most of the real differentiating signal between
risk levels, since `probability` behaves close to a binary switch rather
than the graded estimate it was designed to be. `traffic_density` still
functions as designed (the amplifying effect from Day 11 is visible: e.g.
identical prob_attack and confidence, but density=1.0 pushing risk from
High to Critical in the FTP-Patator and DoS Slowhttptest rows).

**Not fixed in this pass** — flagged as a known limitation and candidate for
future work: NB probability calibration (e.g. Platt scaling / isotonic
regression via `CalibratedClassifierCV`) could produce smoother, better-
calibrated probability estimates without changing the underlying
classification decisions. Out of scope for the current 4-week timeline, but
worth noting explicitly rather than presenting `prob_attack` as more
nuanced than it actually is.

### Risk level vs. true traffic type (84-row sample)

| True type | Critical | High | Medium |
|---|---:|---:|---:|
| BENIGN | 0 | 5 | 1 |
| Attack | 2 | 70 | 6 |

Consistent with Day 10's finding: real BENIGN traffic is frequently flagged
High risk (5 of 6 BENIGN rows in this sample), reflecting the underlying
model's known BENIGN-recall weakness flowing through the fuzzy layer
faithfully rather than being corrected by it — fuzzy inference reshapes a
prediction into a graded, actionable signal, but it cannot fix errors in
the underlying prediction itself.

## Fuzzy visualization and distribution analysis (Day 13)

Generated `reports/plots/fuzzy_membership_functions.png` (all 3 input sets
and the output set, visually confirming the trapezoidal/triangular shapes
from Day 11's design) and `reports/plots/risk_distribution.png` (risk score
histogram and risk level counts, from a 1,606-row stratified sample of real
test data — smaller than the requested 2,000 due to per-class sampling caps
on rare classes).

### Risk level distribution by true traffic type (1,606-row sample)

| Risk level | True BENIGN (n=133) | True Attack (n=1,473) |
|---|---:|---:|
| Low | 1 | 4 |
| Medium | 18 | 64 |
| High | 114 | 1,394 |
| Critical | 0 | 11 |

**Finding: the fuzzy layer shows limited discrimination between BENIGN and
Attack traffic in its current form.** 86% of true-BENIGN samples (114/133)
and 95% of true-Attack samples (1,394/1,473) both land in "High" — nearly
identical distributions across ground-truth classes. This is a direct
consequence of Day 12's probability-saturation finding: since `prob_attack`
is almost always exactly 0 or 1 and `confidence` is almost always very
high (GaussianNB is highly "confident" even on misclassifications), most
rows land in the same corner of the fuzzy input space
(`probability=high, confidence=high`), which the rule table maps to "High"
for most `traffic_density` values.

**Honest interpretation**: this is not a flaw in the fuzzy engine's design
or rule logic — the membership functions and rule table behave exactly as
specified (verified in Day 11/12). The limitation is upstream: fuzzy
inference can only be as discriminating as the inputs it receives, and
GaussianNB's near-binary, poorly-calibrated probability output doesn't give
the fuzzy system enough graded signal to work with. The project's fuzzy
layer successfully demonstrates the *architecture* (probability + confidence
+ density -> graded risk), and traffic_density's amplifying effect remains
visible (the 11 Critical cases are exactly the high-density rows from
Day 12's spot-check), but achieving genuinely differentiated, actionable
risk levels in practice would require addressing NB's probability
calibration first (see Day 12's noted future-work item: Platt
scaling/isotonic regression via `CalibratedClassifierCV`).

This is a legitimate and valuable finding for the project write-up: it
demonstrates the full pipeline is correctly implemented end-to-end, while
also surfacing a real, specific, and well-understood limitation with a
concrete, named path to improving it — a stronger and more credible result
than an inflated claim that fuzzy scoring "solved" the classification
problem.

## FastAPI inference service (Day 15)

Built `api/main.py` (FastAPI app) and `api/schemas.py` (Pydantic
request/response models), exposing three endpoints:

- `GET /health` — returns `{"status": "ok", "model_loaded": true}`
- `GET /model-info` — returns the 35 ACO-selected feature names, label
  mapping, and density feature name (so API consumers know exactly what
  `/predict` expects without reading source code)
- `POST /predict` — accepts a `{"features": {...}}` payload with all 35
  ACO-selected feature values, returns predicted label, prob_attack,
  confidence, traffic_density, risk_score, and risk_level

**Verified correct, not just running**: `scripts/test_api.py` posts a real
test-set row to the live API and compares the result field-by-field against
`fuzzy.integration.score_row()`'s direct computation for the same input —
all 6 output fields matched exactly (PASS). This confirms `/predict` is a
faithful HTTP wrapper around the same tested scoring logic used throughout
the project, not a parallel reimplementation that could silently drift.

Interactive API docs available at `/docs` (FastAPI's auto-generated Swagger
UI) once the server is running.
