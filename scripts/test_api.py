"""
Day 15 — API correctness check.

Pulls one real row from data/processed/test.csv, posts it to the running
API's /predict endpoint, and prints the result alongside what
fuzzy.integration's score_row() computes directly for the same row — they
should match exactly, confirming the API wraps the same logic correctly
rather than diverging from it.

Requires the API to already be running (uvicorn api.main:app --reload)
in another terminal.

Run from the project root:
    python scripts/test_api.py
"""

import json
from pathlib import Path

import pandas as pd
import requests

from fuzzy.integration import load_model_and_metadata, score_row

API_URL = "http://127.0.0.1:8000"
PROCESSES_DIR = Path("data/processed")


def main():
    print("Loading a real test row...")
    test_df = pd.read_csv(PROCESSES_DIR / "test.csv")
    model, metadata = load_model_and_metadata()
    selected_features = metadata["selected_features"]

    label_mapping = {int(k): v for k, v in metadata["label_mapping"].items()}
    test_df["_label_name"] = test_df["Label_encoded"].map(label_mapping)
    attack_rows = test_df[test_df["_label_name"] != "BENIGN"]
    row = attack_rows.iloc[0]

    print(f"True label: {row['_label_name']}\n")

    print("--- Direct call (fuzzy.integration.score_row) ---")
    direct_result = score_row(model, metadata, row)
    print(json.dumps(direct_result, indent=2))

    print("\n--- API call (POST /predict) ---")
    feature_payload = {f: float(row[f]) for f in selected_features}
    response = requests.post(f"{API_URL}/predict", json={"features": feature_payload})

    if response.status_code != 200:
        print(f"API ERROR {response.status_code}: {response.text}")
        return

    api_result = response.json()
    print(json.dumps(api_result, indent=2))

    print("\n--- Comparison ---")
    checks = [
        ("predicted_label", direct_result["predicted_label"], api_result["predicted_label"]),
        ("prob_attack", direct_result["prob_attack"], api_result["prob_attack"]),
        ("confidence", direct_result["confidence"], api_result["confidence"]),
        ("traffic_density", direct_result["traffic_density"], api_result["traffic_density"]),
        ("risk_score", direct_result["risk_score"], api_result["risk_score"]),
        ("risk_level", direct_result["risk_level"], api_result["risk_level"]),
    ]
    all_match = True
    for field, direct_val, api_val in checks:
        match = direct_val == api_val
        all_match &= match
        status = "OK" if match else "MISMATCH"
        print(f" {field:<20} direct={direct_val!r:<15} api={api_val!r:<15} [{status}]")

    print(f"\n{'PASS: API matches direct call exactly' if all_match else 'FAIL: missmatch found'}")


if __name__ == "__main__":
    main()
