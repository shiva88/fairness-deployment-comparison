"""
run_more_seeds_german.py
--------------------------
Extended seed budget for German Credit (R2 concern #6).

R2 concern: "Increase the number of random seeds beyond 10 or provide
statistical justification. German Credit's small size (n=1000) produces high
single-seed variance."

Runs 30 total seeds (the original 10 + 20 more) on German Credit only,
across all 4 methods and 2 models. Adult and COMPAS keep their existing 10
seeds since their run-to-run variance is already low.

Runtime: ~30-45 minutes.

Usage (full run):
    python run_more_seeds_german.py

Usage (smoke test):
    python run_more_seeds_german.py --seeds 42 --methods baseline --models lr

Output:
    results/german_extended_seeds.csv
    (same column format as run_one.py's full_results.csv)
"""
import argparse
import os
import time
import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")

import aif360
import sklearn

from run_one import (
    DATASET_LOADERS,
    split_dataset,
    METHOD_RUNNERS,
    write_row,
)

# Original 10 + 20 additional seeds, all deterministic and reproducible
EXTENDED_SEEDS = [
    # Original 10 (already reported in the manuscript)
    42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768,
    # 20 new seeds
    65537, 131072, 262144, 524288, 1048576,
    2097152, 4194304, 8388608, 16777216, 33554432,
    67108864, 134217728, 268435456, 536870912, 999999999,
    111, 222, 333, 444, 555,
]

DEFAULT_METHODS = ["baseline", "reweighing", "dir", "eqodds"]
DEFAULT_MODELS = ["lr", "rf"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=EXTENDED_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["lr", "rf"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--out", default="results/german_extended_seeds.csv")
    args = parser.parse_args()

    total = len(args.seeds) * len(args.models) * len(args.methods)
    print(f"Planned runs: {total}  (German only, {len(args.seeds)} seeds)")
    print(f"  methods: {args.methods}")
    print(f"  models: {args.models}")
    print(f"Writing to: {args.out}\n")

    info = DATASET_LOADERS["german"]()
    bld = info["dataset"]
    prot_attr = info["protected_attribute"]
    priv = info["privileged_groups"]
    unpriv = info["unprivileged_groups"]

    done = 0
    start = time.time()
    for seed in args.seeds:
        train_bld, val_bld, test_bld = split_dataset(bld, prot_attr, seed)
        for model_name in args.models:
            for method_name in args.methods:
                try:
                    np.random.seed(seed)
                    runner = METHOD_RUNNERS[method_name]
                    result = runner(
                        bld, train_bld, val_bld, test_bld,
                        model_name, seed, priv, unpriv, prot_attr, seed,
                    )
                    row = {
                        "dataset": "german",
                        "model": model_name,
                        "method": method_name,
                        "seed": seed,
                        "split_seed": seed,
                        "model_seed": seed,
                        "protected_attribute": prot_attr,
                        "privileged_group": str(priv),
                        "unprivileged_group": str(unpriv),
                        "n_train": train_bld.features.shape[0],
                        "n_val": val_bld.features.shape[0],
                        "n_test": test_bld.features.shape[0],
                        "aif360_version": aif360.__version__,
                        "sklearn_version": sklearn.__version__,
                        "notes": "german_extended_30seeds",
                        **result,
                    }
                    write_row(row, args.out)
                    done += 1
                    elapsed = time.time() - start
                    rate = done / max(elapsed, 0.01)
                    eta = (total - done) / max(rate, 0.01)
                    print(f"[{done}/{total}] german/{model_name}/{method_name}/"
                          f"s={seed}: acc={result['accuracy']:.3f} "
                          f"|SPD|={abs(result['spd']):.3f} "
                          f"ETA {eta/60:.1f}min")
                except Exception as e:
                    done += 1
                    print(f"  ERROR german/{model_name}/{method_name}/"
                          f"s={seed}: {e}")

    print(f"\nComplete. Total time: {(time.time()-start)/60:.1f} min")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
