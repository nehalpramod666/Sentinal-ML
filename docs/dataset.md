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
