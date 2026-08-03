"""
run_multi_batch_latency.py (v2 - fixes EqOdds crash on small test sets)
------------------------------------------------------------------------
Same as v1 but the EqOdds predict_fn now handles the case where the batch
size exceeds the test set size (which happened on German with bs=256 and
bs=1024, since German has only 200 test rows).

Fix: when n_chunk > n_test, wrap indices modulo n_test.

Usage (rerun ONLY the failed cells - German EqOdds - ~2 minutes):
    python run_multi_batch_latency.py --datasets german --methods eqodds

Usage (full run):
    python run_multi_batch_latency.py

Output:
    results/multi_batch_latency.csv
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("PYTHONHASHSEED", "0")

import aif360
import sklearn
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

from run_one import (
    DATASET_LOADERS,
    split_dataset,
    make_model,
    needs_scaling,
)
from dir_train_only import TrainOnlyDisparateImpactRemover

DEFAULT_SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_DATASETS = ["adult", "compas", "german"]
DEFAULT_MODELS = ["lr", "rf"]
DEFAULT_METHODS = ["baseline", "reweighing", "dir", "eqodds"]
DEFAULT_BATCH_SIZES = [1, 32, 128, 256, 1024]

N_WARMUP = 20
N_MEASUREMENTS = 100
N_BOOT = 2000

CSV_COLUMNS = [
    "dataset", "model", "method", "seed", "batch_size",
    "latency_ms_median", "latency_ms_mean", "latency_ms_std",
    "latency_ci_lo", "latency_ci_hi",
    "throughput_per_sec",
    "aif360_version", "sklearn_version", "notes",
]


def build_predict_fn(train_bld, val_bld, test_bld, model_name, seed,
                     priv, unpriv, protected_attribute, method):
    if method == "baseline":
        X_train = train_bld.features
        y_train = train_bld.labels.ravel()
        scaler = None
        if needs_scaling(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        model = make_model(model_name, seed)
        model.fit(X_train, y_train)

        def predict_fn(X_chunk):
            if scaler is not None:
                X_chunk = scaler.transform(X_chunk)
            return model.predict(X_chunk)

        return predict_fn, test_bld.features

    if method == "reweighing":
        rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
        train_rw = rw.fit_transform(train_bld)
        X_train = train_rw.features
        y_train = train_rw.labels.ravel()
        sw = train_rw.instance_weights.ravel()
        scaler = None
        if needs_scaling(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        model = make_model(model_name, seed)
        model.fit(X_train, y_train, sample_weight=sw)

        def predict_fn(X_chunk):
            if scaler is not None:
                X_chunk = scaler.transform(X_chunk)
            return model.predict(X_chunk)

        return predict_fn, test_bld.features

    if method == "dir":
        dir_op = TrainOnlyDisparateImpactRemover(
            repair_level=1.0,
            sensitive_attribute=protected_attribute,
            n_quantiles=101,
        )
        dir_op.fit(train_bld)
        train_rep = dir_op.transform(train_bld)
        fi = train_rep.feature_names.index(protected_attribute)
        X_train = np.delete(train_rep.features, fi, axis=1)
        y_train = train_rep.labels.ravel()
        scaler = None
        if needs_scaling(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        model = make_model(model_name, seed)
        model.fit(X_train, y_train)

        def predict_fn(X_raw_chunk):
            X_rep = dir_op.transform_array(X_raw_chunk)
            X_chunk = np.delete(X_rep, fi, axis=1)
            if scaler is not None:
                X_chunk = scaler.transform(X_chunk)
            return model.predict(X_chunk)

        return predict_fn, test_bld.features

    if method == "eqodds":
        X_train = train_bld.features
        y_train = train_bld.labels.ravel()
        X_val = val_bld.features
        scaler = None
        if needs_scaling(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = make_model(model_name, seed)
        model.fit(X_train, y_train)

        val_pred_bld = val_bld.copy()
        val_pred_bld.labels = model.predict(X_val).reshape(-1, 1).astype(float)

        eq = EqOddsPostprocessing(unprivileged_groups=unpriv,
                                    privileged_groups=priv, seed=seed)
        eq = eq.fit(val_bld, val_pred_bld)

        n_test = len(test_bld.features)

        def predict_fn(X_chunk):
            if scaler is not None:
                X_chunk = scaler.transform(X_chunk)
            raw = model.predict(X_chunk).reshape(-1, 1).astype(float)
            n_chunk = len(X_chunk)
            # Handle case where batch is bigger than test set:
            # wrap indices modulo n_test
            if n_chunk <= n_test:
                indices = list(range(n_chunk))
            else:
                indices = [i % n_test for i in range(n_chunk)]
            cb = test_bld.subset(indices)
            cb.labels = raw
            return eq.predict(cb).labels.ravel()

        return predict_fn, test_bld.features

    raise ValueError(method)


def measure_at_batch(predict_fn, X_test_raw, batch_size, rng):
    n_test = len(X_test_raw)
    if batch_size <= n_test:
        idx = rng.choice(n_test, size=batch_size, replace=False)
    else:
        idx = rng.choice(n_test, size=batch_size, replace=True)
    X_batch = X_test_raw[idx]

    for _ in range(N_WARMUP):
        predict_fn(X_batch)

    times = np.empty(N_MEASUREMENTS)
    for i in range(N_MEASUREMENTS):
        t0 = time.perf_counter()
        predict_fn(X_batch)
        times[i] = (time.perf_counter() - t0) * 1000.0

    med = float(np.median(times))
    mean = float(np.mean(times))
    std = float(np.std(times, ddof=1))

    boot = rng.choice(times, size=(N_BOOT, N_MEASUREMENTS), replace=True)
    boot_meds = np.median(boot, axis=1)
    ci_lo = float(np.percentile(boot_meds, 2.5))
    ci_hi = float(np.percentile(boot_meds, 97.5))

    throughput = batch_size / (med / 1000.0) if med > 0 else float("inf")

    return {
        "latency_ms_median": med,
        "latency_ms_mean": mean,
        "latency_ms_std": std,
        "latency_ci_lo": ci_lo,
        "latency_ci_hi": ci_hi,
        "throughput_per_sec": throughput,
    }


def write_row(row, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    ordered = {col: row.get(col, "") for col in CSV_COLUMNS}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(ordered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=list(DATASET_LOADERS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["lr", "rf"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--out", default="results/multi_batch_latency.csv")
    args = parser.parse_args()

    total = (len(args.datasets) * len(args.models)
             * len(args.methods) * len(args.seeds))
    print(f"Planned cells: {total}")
    print(f"Batch sizes: {args.batch_sizes}")
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
                for method_name in args.methods:
                    try:
                        np.random.seed(seed)
                        rng = np.random.default_rng(seed)
                        predict_fn, X_test_raw = build_predict_fn(
                            train_bld, val_bld, test_bld,
                            model_name, seed, priv, unpriv, prot_attr,
                            method_name,
                        )
                        last_stats = None
                        for bs in args.batch_sizes:
                            stats = measure_at_batch(predict_fn, X_test_raw, bs, rng)
                            last_stats = stats
                            row = {
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method_name,
                                "seed": seed,
                                "batch_size": bs,
                                **stats,
                                "aif360_version": aif360.__version__,
                                "sklearn_version": sklearn.__version__,
                                "notes": "multi_batch",
                            }
                            write_row(row, args.out)

                        done += 1
                        elapsed = time.time() - start
                        rate = done / max(elapsed, 0.01)
                        eta = (total - done) / max(rate, 0.01)
                        print(f"[{done}/{total}] {dataset_name}/{model_name}/"
                              f"{method_name}/s={seed}: "
                              f"bs1024={last_stats['latency_ms_median']:.2f}ms "
                              f"ETA {eta/60:.1f}min")
                    except Exception as e:
                        done += 1
                        print(f"  ERROR {dataset_name}/{model_name}/"
                              f"{method_name}/s={seed}: {e}")

    print(f"\nComplete. Total time: {(time.time()-start)/60:.1f} min")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
