"""
Day 24 — API stress testing.

Hits the running API's /predict endpoint with real feature payloads
(sampled from test.csv) at increasing concurrency levels, measuring
latency percentiles and throughput at each level to see how the API
degrades under load.

Requires the API to already be running:
    uvicorn api.main:app --reload --port 8080

Run from the project root:
    python -m scripts.stress_test
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8080"
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

CONCURRENCY_LEVELS = [1, 5, 20, 50]
REQUESTS_PER_LEVEL = 100


def load_sample_payloads(n: int) -> list[dict]:
    """Pull n real rows from test.csv as feature payloads, cycling through
    different rows (not the same row n times) for a more realistic mix."""
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    selected = pd.read_csv(REPORTS_DIR / "selected_features.csv")["feature"].tolist()

    sample_rows = test_df.sample(
        n=min(n, len(test_df)), random_state=42, replace=(n > len(test_df))
    )
    payloads = [{f: float(row[f]) for f in selected} for _, row in sample_rows.iterrows()]
    return payloads


def send_request(payload: dict) -> dict:
    start = time.time()
    try:
        response = requests.post(f"{API_URL}/predict", json={"features": payload}, timeout=10)
        elapsed_ms = (time.time() - start) * 1000
        return {
            "success": response.status_code == 200,
            "elapsed_ms": elapsed_ms,
            "status_code": response.status_code,
        }
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time.time() - start) * 1000
        return {"success": False, "elapsed_ms": elapsed_ms, "status_code": None, "error": str(e)}


def run_load_level(concurrency: int, payloads: list[dict]) -> dict:
    print(f"\n--- Concurrency: {concurrency} ({len(payloads)} requests) ---")
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, p) for p in payloads]
        for future in as_completed(futures):
            results.append(future.result())

    total_elapsed = time.time() - start_time

    latencies = [r["elapsed_ms"] for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    if not latencies:
        print("  All requests failed!")
        return {
            "concurrency": concurrency,
            "n_requests": len(payloads),
            "n_errors": len(errors),
            "total_seconds": round(total_elapsed, 2),
        }

    latencies.sort()
    summary = {
        "concurrency": concurrency,
        "n_requests": len(payloads),
        "n_errors": len(errors),
        "total_seconds": round(total_elapsed, 2),
        "throughput_req_per_sec": round(len(payloads) / total_elapsed, 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "latency_p99_ms": (
            round(latencies[int(len(latencies) * 0.99)], 2)
            if len(latencies) >= 100
            else round(latencies[-1], 2)
        ),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
    }

    print(f"  Throughput: {summary['throughput_req_per_sec']} req/s")
    print(
        f"  Latency (ms) — median: {summary['latency_median_ms']}  p95: {summary['latency_p95_ms']}  p99: {summary['latency_p99_ms']}  max: {summary['latency_max_ms']}"
    )
    if errors:
        print(f"  Errors: {len(errors)}/{len(payloads)}")

    return summary


def main():
    print("Checking API is reachable...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API not reachable at {API_URL} ({e})")
        print("Start it first: uvicorn api.main:app --reload --port 8080")
        return

    all_results = []
    for concurrency in CONCURRENCY_LEVELS:
        level_payloads = load_sample_payloads(REQUESTS_PER_LEVEL)
        result = run_load_level(concurrency, level_payloads)
        all_results.append(result)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'Concurrency':>12} {'Throughput':>12} {'Median ms':>10} {'P95 ms':>8} {'P99 ms':>8} {'Errors':>8}"
    )
    for r in all_results:
        print(
            f"{r['concurrency']:>12} {r.get('throughput_req_per_sec', 'N/A'):>12} "
            f"{r.get('latency_median_ms', 'N/A'):>10} {r.get('latency_p95_ms', 'N/A'):>8} "
            f"{r.get('latency_p99_ms', 'N/A'):>8} {r['n_errors']:>8}"
        )

    out_path = REPORTS_DIR / "stress_test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
