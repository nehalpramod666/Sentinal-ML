"""
Day 2 — CICIDS2017 dataset inspection.

Scans every CSV in data/raw/, reports shape, columns, dtypes, label
distribution, and missing values per file, then prints a combined summary.
Run from the project root:
    python scripts/inspect_dataset.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def inspect_file(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    label_col = "Label" if "Label" in df.columns else None
    label_counts = df[label_col].value_counts().to_dict() if label_col else {}

    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()

    return {
        "file": path.name,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "label_counts": label_counts,
        "missing": missing,
    }


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {RAW_DIR.resolve()}")
        print("Download MachineLearningCSV.zip from unb.ca/cic/datasets/ids-2017.html,")
        print("unzip it, and place the CSVs in data/raw/.")
        return

    all_columns = None
    combined_labels = {}
    total_rows = 0

    for f in files:
        print(f"\n{'=' * 70}")
        print(f"Inspecting: {f.name}")
        result = inspect_file(f)

        print(f"Rows: {result['rows']:,} | Columns: {result['cols']}")

        if all_columns is None:
            all_columns = set(result["columns"])
        else:
            diff = all_columns.symmetric_difference(result["columns"])
            if diff:
                print(f" WARNING: column mismatch vs. other files: {diff}")
        print("Label distributions:")
        for label, count in sorted(result["label_counts"].items(), key=lambda x: -x[-1]):
            print(f" {label:30s} {count:>10,}")
            combined_labels[label] = combined_labels.get(label, 0) + count

        if result["missing"]:
            print("Missing values:")
            for col, count in result["missing"].items():
                print(f" {col:30s} {count:>10,}")
        else:
            print("Missing values: none")

        total_rows += result["rows"]

    print(f"\n{'=' * 70}")
    print("COMBINED SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files: {len(files)}")
    print(f"Total rows:  {total_rows:,}")
    print(f"Consistent columns across all files: {all_columns is not None}")
    print("\nCombined label distribution:")

    for label, count in sorted(combined_labels.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_rows
        print(f" {label:30s} {count:>10,} ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
