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
