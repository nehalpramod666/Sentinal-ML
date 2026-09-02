"""
Day 3 — CICIDS2017 preprocessing pipeline.

Loads all 8 raw CSVs, cleans column names, fixes the mojibake in Web Attack
labels, drops rows with missing Flow Bytes/s, removes a duplicate column,
drops rows with physically impossible negative values, encodes labels, and
writes a stratified 80/20 train/test split to data/processed/.

Run from the project root:
    python -m ml.preprocessing
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Maps the mojibake replacement character back to a clean, canonical label.
# Root cause: source CSVs are Windows-1252/Latin-1 encoded; a plain UTF-8
# read turns the original en-dash into U+FFFD.
LABEL_FIXES = {
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack - SQL Injection",
}

# Columns that should never be negative (a negative value is physically
# impossible, not just a rare edge case) — used to drop corrupted rows.
NON_NEGATIVE_COLUMNS = ["Flow Duration", "Flow Bytes/s", "Fwd Header Length"]


def load_raw_data() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR.resolve()}. "
            "Download MachineLearningCSV.zip from unb.ca/cic/datasets/ids-2017.html "
            "and place the CSVs in data/raw/."
        )

    frames = []
    for f in files:
        print(f"Loading {f.name} ...")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["Label"] = df["Label"].replace(LABEL_FIXES)
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["Flow Bytes/s"])
    dropped = before - len(df)
    print(f"Dropped {dropped:,} rows with missing Flow Bytes/s")
    return df


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # CICIDS2017's raw CSVs contain "Fwd Header Length" twice; pandas
    # renames the second occurrence to "Fwd Header Length.1" on load.
    # Verified identical across all rows (see docs/dataset.md) — drop the
    # duplicate rather than train on the same feature twice.
    if "Fwd Header Length.1" in df.columns:
        print("Dropping duplicate column: Fwd Header Length.1")
        df = df.drop(columns=["Fwd Header Length.1"])
    return df


def drop_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    # A handful of rows have physically impossible negative values (negative
    # duration, negative bytes/sec, negative header length) — a known
    # CICFlowMeter generation artifact, not a preprocessing bug. Affects
    # ~0.005% of rows, so dropping is safe and far simpler than imputing.
    before = len(df)
    mask = pd.Series(False, index=df.index)
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            mask |= df[col] < 0
    df = df[~mask]
    dropped = before - len(df)
    print(f"Dropped {dropped:,} rows with impossible negative values in {NON_NEGATIVE_COLUMNS}")
    return df


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Flow ID, source/destination IP, and timestamp are identifiers, not
    # generalizable traffic-behavior features — including them would let the
    # model memorize specific hosts/times instead of learning attack patterns.
    drop_cols = [
        c for c in ["Flow ID", "Source IP", "Destination IP", "Timestamp"] if c in df.columns
    ]
    if drop_cols:
        print(f"Dropping identifier columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoder = LabelEncoder()
    df["Label_encoded"] = encoder.fit_transform(df["Label"])
    mapping = {int(i): label for i, label in enumerate(encoder.classes_)}
    print(f"Encoded {len(mapping)} classes")
    return df, mapping


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    df = clean_labels(df)
    df = drop_missing(df)
    df = drop_duplicate_columns(df)
    df = drop_impossible_values(df)
    df = drop_non_feature_columns(df)
    df, label_mapping = encode_labels(df)

    # Handle any remaining inf/-inf values (Flow Bytes/s and Flow Packets/s
    # can divide by zero-duration flows) by replacing with NaN then dropping.
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    inf_count = (
        ((df[numeric_cols] == float("inf")) | (df[numeric_cols] == float("-inf"))).sum().sum()
    )
    if inf_count > 0:
        print(f"Found {inf_count:,} inf/-inf values, replacing with NaN and dropping")
        df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)
        df = df.dropna()

    X = df.drop(columns=["Label", "Label_encoded"])
    y = df["Label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = X_train.copy()
    train_df["Label_encoded"] = y_train
    test_df = X_test.copy()
    test_df["Label_encoded"] = y_test

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    mapping_path = PROCESSED_DIR / "label_mapping.json"
    mapping_path.write_text(json.dumps(label_mapping, indent=2))

    print(f"\nSaved: {train_path} ({len(train_df):,} rows)")
    print(f"Saved: {test_path} ({len(test_df):,} rows)")
    print(f"Saved: {mapping_path}")

    print("\nTrain label distribution:")
    print(y_train.map(label_mapping).value_counts())
    print("\nTest label distribution:")
    print(y_test.map(label_mapping).value_counts())


if __name__ == "__main__":
    main()
