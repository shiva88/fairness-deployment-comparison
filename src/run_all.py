"""
run_all.py - drive the full 240-run experimental matrix.

3 datasets × 2 models × 4 methods × 10 seeds = 240 runs.

Each run is invoked as a subprocess so each gets a fresh Python interpreter,
guarding against accidental state leakage between runs.

Usage:
    python src/run_all.py
    python src/run_all.py --out results/full_results.csv
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DATASETS = ["adult", "compas", "german"]
MODELS = ["lr", "rf"]
METHODS = ["baseline", "reweighing", "dir", "eqodds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/full_results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    total = len(DATASETS) * len(MODELS) * len(METHODS) * len(SEEDS)
    print(f"Total runs to execute: {total}")
    print(f"Output: {args.out}\n")

    completed = 0
    failed = []
    t_start = time.perf_counter()

    for dataset in DATASETS:
        for model in MODELS:
            for method in METHODS:
                for seed in SEEDS:
                    completed += 1
                    cmd = [
                        sys.executable, "src/run_one.py",
                        "--dataset", dataset,
                        "--model", model,
                        "--method", method,
                        "--seed", str(seed),
                        "--out", args.out,
                    ]
                    label = f"[{completed}/{total}] {dataset} {model} {method} seed={seed}"
                    print(label)
                    if args.dry_run:
                        print("  " + " ".join(cmd))
                        continue
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"  FAILED: {result.stderr[-500:]}")
                        failed.append((dataset, model, method, seed))

    elapsed = time.perf_counter() - t_start
    print(f"\nFinished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Completed: {completed - len(failed)}/{total}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
