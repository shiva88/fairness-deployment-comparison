"""
run_repair_sweep.py
--------------------
DIR repair-level sensitivity analysis (R2 concern #3).

Uses run_one.py's tested functions directly. Only repair_level varies.
Writes results in the same CSV format as run_one.py.

Usage (full sweep - ~2 hours):
    python run_repair_sweep.py

Usage (quick smoke test - ~2 minutes):
    python run_repair_sweep.py --datasets german --models lr --seeds 42

Output:
    results/dir_repair_sweep.csv
"""
import argparse
import os
import time

import numpy as np
os.environ.setdefault("PYTHONHASHSEED", "0")

import aif360
import sklearn
from sklearn.preprocessing import StandardScaler

from run_one import (
    DATASET_LOADERS,
    split_dataset,
    make_model,
    needs_scaling,
    compute_metrics,
    measure_latency,
    write_row,
)
from dir_train_only import TrainOnlyDisparateImpactRemover

DEFAULT_SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_REPAIR_LEVELS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_DATASETS = ["adult", "compas", "german"]
DEFAULT_MODELS = ["lr", "rf"]
N_QUANTILES = 101


def run_dir_at_repair_level(train_bld, test_bld, model_name, model_seed,
                             privileged_groups, unprivileged_groups,
                             protected_attribute, repair_level):
    t0 = time.perf_counter()
    dir_op = TrainOnlyDisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=protected_attribute,
        n_quantiles=N_QUANTILES,
    )
    dir_op.fit(train_bld)
    dir_fit_time_sec = time.perf_counter() - t0

    train_rep = dir_op.transform(train_bld)
    test_rep = dir_op.transform(test_bld)

    feature_idx = train_rep.feature_names.index(protected_attribute)
    X_train = np.delete(train_rep.features, feature_idx, axis=1)
    y_train = train_rep.labels.ravel()
    X_test_repaired = np.delete(test_rep.features, feature_idx, axis=1)

    scaler = None
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_repaired = scaler.transform(X_test_repaired)

    model = make_model(model_name, model_seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = dir_fit_time_sec + (time.perf_counter() - t0)

    y_pred = model.predict(X_test_repaired)
    metrics = compute_metrics(test_bld, y_pred, privileged_groups, unprivileged_groups)

    def predict_fn(X_raw_chunk):
        X_repaired = dir_op.transform_array(X_raw_chunk)
        X_chunk = np.delete(X_repaired, feature_idx, axis=1)
        if scaler is not None:
            X_chunk = scaler.transform(X_chunk)
        return model.predict(X_chunk)

    single_ms, batch_ms = measure_latency(predict_fn, test_bld.features)

    return {
        **metrics,
        "train_time_sec": train_time_sec,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": 24,
        "implementation_dependencies_added": 1,
        "implementation_hyperparams": 2,
        "implementation_stage": "preprocessing",
        "implementation_complexity_score": 3,
        "method_hyperparams": (
            f"repair_level={repair_level},"
            f"n_quantiles={N_QUANTILES},"
            f"train_only_fit=True"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=list(DATASET_LOADERS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["lr", "rf"])
    parser.add_argument("--repair-levels", nargs="+", type=float,
                        default=DEFAULT_REPAIR_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="results/dir_repair_sweep.csv")
    args = parser.parse_args()

    total = (len(args.datasets) * len(args.models)
             * len(args.repair_levels) * len(args.seeds))
    print(f"Planned runs: {total}")
    print(f"  datasets: {args.datasets}")
    print(f"  models: {args.models}")
    print(f"  repair_levels: {args.repair_levels}")
    print(f"  seeds: {args.seeds}")
    print(f"Writing to: {args.out}\n")

    done = 0
    start = time.time()
    for dataset_name in args.datasets:
        info = DATASET_LOADERS[dataset_name]()
        bld = info["dataset"]
        prot_attr = info["protected_attribute"]
        priv = info["privileged_groups"]
        unpriv = info["unprivileged_groups"]

        for seed in args.seeds:
            train_bld, val_bld, test_bld = split_dataset(bld, prot_attr, seed)
            for model_name in args.models:
                for repair_level in args.repair_levels:
                    try:
                        np.random.seed(seed)
                        result = run_dir_at_repair_level(
                            train_bld, test_bld, model_name, seed,
                            priv, unpriv, prot_attr, repair_level,
                        )
                        row = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": "dir",
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
                            "notes": f"repair_sweep rl={repair_level}",
                            **result,
                        }
                        write_row(row, args.out)
                        done += 1
                        elapsed = time.time() - start
                        rate = done / max(elapsed, 0.01)
                        eta = (total - done) / max(rate, 0.01)
                        print(f"[{done}/{total}] {dataset_name}/{model_name}/"
                              f"rl={repair_level}/s={seed}: "
                              f"acc={result['accuracy']:.3f} "
                              f"|SPD|={abs(result['spd']):.3f} "
                              f"ETA {eta/60:.1f}min")
                    except Exception as e:
                        done += 1
                        print(f"  ERROR {dataset_name}/{model_name}/"
                              f"rl={repair_level}/s={seed}: {e}")

    print(f"\nComplete. Total time: {(time.time()-start)/60:.1f} min")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
