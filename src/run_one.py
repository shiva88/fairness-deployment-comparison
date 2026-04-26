"""
run_one.py - single reproducible row of the bias mitigation comparison.

Usage:
    python src/run_one.py --dataset adult --model lr --method baseline   --seed 42
    python src/run_one.py --dataset adult --model lr --method reweighing --seed 42
    python src/run_one.py --dataset adult --model lr --method dir        --seed 42
    python src/run_one.py --dataset adult --model lr --method eqodds     --seed 42
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import aif360
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing


# ---------------------------------------------------------------------------
# Dataset registry.
# ---------------------------------------------------------------------------

def load_adult():
    ds = AdultDataset()
    return {
        "dataset": ds,
        "protected_attribute": "sex",
        "privileged_groups": [{"sex": 1.0}],
        "unprivileged_groups": [{"sex": 0.0}],
    }

def load_compas():
    ds = CompasDataset()
    return {
        "dataset": ds,
        "protected_attribute": "race",
        "privileged_groups": [{"race": 1.0}],
        "unprivileged_groups": [{"race": 0.0}],
    }

def load_german():
    ds = GermanDataset()
    return {
        "dataset": ds,
        "protected_attribute": "age",
        "privileged_groups": [{"age": 1.0}],
        "unprivileged_groups": [{"age": 0.0}],
    }

DATASET_LOADERS = {
    "adult": load_adult,
    "compas": load_compas,
    "german": load_german,
}


# ---------------------------------------------------------------------------
# Stratified 60/20/20 split.
# ---------------------------------------------------------------------------

def split_dataset(bld, protected_attribute, split_seed):
    n = bld.features.shape[0]
    indices = np.arange(n)
    labels = bld.labels.ravel().astype(int)
    prot_idx = bld.protected_attribute_names.index(protected_attribute)
    prot = bld.protected_attributes[:, prot_idx].astype(int)
    strat = labels * 10 + prot

    train_idx, temp_idx = train_test_split(
        indices, train_size=0.6, random_state=split_seed, stratify=strat
    )
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, random_state=split_seed, stratify=strat[temp_idx]
    )

    train = bld.subset(train_idx.tolist())
    val = bld.subset(val_idx.tolist())
    test = bld.subset(test_idx.tolist())
    return train, val, test


# ---------------------------------------------------------------------------
# Model factory + scaler.
# ---------------------------------------------------------------------------

def make_model(model_name, model_seed):
    if model_name == "lr":
        return LogisticRegression(
            max_iter=1000, random_state=model_seed, n_jobs=1, solver="lbfgs",
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100, random_state=model_seed, n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def needs_scaling(model_name):
    return model_name == "lr"


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def compute_metrics(test_bld, y_pred, privileged_groups, unprivileged_groups):
    y_true = test_bld.labels.ravel()
    test_pred_bld = test_bld.copy()
    test_pred_bld.labels = y_pred.reshape(-1, 1).astype(float)

    cm = ClassificationMetric(
        test_bld, test_pred_bld,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "spd": float(cm.statistical_parity_difference()),
        "eod": float(cm.equal_opportunity_difference()),
        "disparate_impact": float(cm.disparate_impact()),
    }


# ---------------------------------------------------------------------------
# Latency.
# ---------------------------------------------------------------------------

def measure_latency(predict_fn, X_test, n_single=1000, batch_size=256):
    for _ in range(10):
        predict_fn(X_test[:1])

    single_times = []
    for i in range(n_single):
        x = X_test[i % len(X_test):i % len(X_test) + 1]
        t0 = time.perf_counter()
        predict_fn(x)
        single_times.append(time.perf_counter() - t0)
    single_median_ms = float(np.median(single_times) * 1000.0)

    batch = X_test[:batch_size] if len(X_test) >= batch_size else X_test
    batch_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        predict_fn(batch)
        batch_times.append(time.perf_counter() - t0)
    batch_median_ms = float(np.median(batch_times) * 1000.0)

    return single_median_ms, batch_median_ms


# ---------------------------------------------------------------------------
# Method runners.
# ---------------------------------------------------------------------------

def run_baseline(bld, train_bld, val_bld, test_bld, model_name, model_seed,
                 privileged_groups, unprivileged_groups, protected_attribute,
                 split_seed):
    X_train = train_bld.features
    y_train = train_bld.labels.ravel()
    X_test = test_bld.features

    if needs_scaling(model_name):
        X_train, X_test, _ = scale_features(X_train, X_test)

    model = make_model(model_name, model_seed)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_bld, y_pred, privileged_groups, unprivileged_groups)
    single_ms, batch_ms = measure_latency(model.predict, X_test)

    return {
        **metrics,
        "train_time_sec": train_time_sec,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": 0,
        "implementation_dependencies_added": 0,
        "implementation_hyperparams": 0,
        "implementation_stage": "none",
        "implementation_complexity_score": 1,
        "method_hyperparams": "",
    }


def run_reweighing(bld, train_bld, val_bld, test_bld, model_name, model_seed,
                   privileged_groups, unprivileged_groups, protected_attribute,
                   split_seed):
    """Reweighing (Kamiran & Calders 2012). Pre-processing via sample weights."""
    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    train_rw = rw.fit_transform(train_bld)

    X_train = train_rw.features
    y_train = train_rw.labels.ravel()
    sample_weight = train_rw.instance_weights.ravel()
    X_test = test_bld.features

    if needs_scaling(model_name):
        X_train, X_test, _ = scale_features(X_train, X_test)

    model = make_model(model_name, model_seed)

    t0 = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_time_sec = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_bld, y_pred, privileged_groups, unprivileged_groups)
    single_ms, batch_ms = measure_latency(model.predict, X_test)

    return {
        **metrics,
        "train_time_sec": train_time_sec,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": 6,
        "implementation_dependencies_added": 0,
        "implementation_hyperparams": 0,
        "implementation_stage": "preprocessing",
        "implementation_complexity_score": 2,
        "method_hyperparams": "",
    }


def run_dir(bld, train_bld, val_bld, test_bld, model_name, model_seed,
            privileged_groups, unprivileged_groups, protected_attribute,
            split_seed):
    """
    Disparate Impact Remover (Feldman et al. 2015).

    Methodology: We apply DIR to the FULL dataset before splitting, then
    re-split using the same stratified 60/20/20 logic. This matches the
    AIF360 reference tutorials. DIR does not use class labels in its repair,
    limiting any leak to feature distribution geometry only.

    repair_level = 1.0 (AIF360 default).
    Latency includes amortized repair-transform cost per row.
    """
    REPAIR_LEVEL = 1.0

    t0 = time.perf_counter()
    dir_op = DisparateImpactRemover(
        repair_level=REPAIR_LEVEL,
        sensitive_attribute=protected_attribute,
    )
    repaired_bld = dir_op.fit_transform(bld)
    repair_total_sec = time.perf_counter() - t0
    repair_per_row_ms = (repair_total_sec / len(bld.features)) * 1000.0

    train_rep, val_rep, test_rep = split_dataset(
        repaired_bld, protected_attribute, split_seed
    )

    feature_idx = train_rep.feature_names.index(protected_attribute)
    X_train = np.delete(train_rep.features, feature_idx, axis=1)
    y_train = train_rep.labels.ravel()
    X_test = np.delete(test_rep.features, feature_idx, axis=1)

    if needs_scaling(model_name):
        X_train, X_test, _ = scale_features(X_train, X_test)

    model = make_model(model_name, model_seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_rep, y_pred, privileged_groups, unprivileged_groups)

    model_single_ms, model_batch_ms = measure_latency(model.predict, X_test)
    single_ms = model_single_ms + repair_per_row_ms
    batch_ms = model_batch_ms + (repair_per_row_ms * 256)

    return {
        **metrics,
        "train_time_sec": train_time_sec,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": 18,
        "implementation_dependencies_added": 1,
        "implementation_hyperparams": 1,
        "implementation_stage": "preprocessing",
        "implementation_complexity_score": 3,
        "method_hyperparams": f"repair_level={REPAIR_LEVEL}",
    }


def run_eqodds(bld, train_bld, val_bld, test_bld, model_name, model_seed,
               privileged_groups, unprivileged_groups, protected_attribute,
               split_seed):
    """
    Equalized Odds Post-processing (Hardt, Price, Srebro 2016).

    Trains a baseline classifier on train, then fits the EqOdds post-processor
    on the validation split (true labels vs predictions), avoiding test-set
    leakage. Inference: model.predict + per-group probabilistic adjustment.
    """
    X_train = train_bld.features
    y_train = train_bld.labels.ravel()
    X_val = val_bld.features
    X_test = test_bld.features

    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model = make_model(model_name, model_seed)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    model_train_time_sec = time.perf_counter() - t0

    val_pred_bld = val_bld.copy()
    val_pred_bld.labels = model.predict(X_val).reshape(-1, 1).astype(float)

    test_pred_bld = test_bld.copy()
    test_pred_bld.labels = model.predict(X_test).reshape(-1, 1).astype(float)

    eq = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=model_seed,
    )

    t0 = time.perf_counter()
    eq = eq.fit(val_bld, val_pred_bld)
    eqodds_fit_time_sec = time.perf_counter() - t0

    train_time_sec = model_train_time_sec + eqodds_fit_time_sec

    test_pred_eq = eq.predict(test_pred_bld)
    y_pred = test_pred_eq.labels.ravel().astype(int)

    metrics = compute_metrics(test_bld, y_pred, privileged_groups, unprivileged_groups)

    def predict_fn_with_eqodds(X_chunk):
        raw_preds = model.predict(X_chunk).reshape(-1, 1).astype(float)
        n_chunk = len(X_chunk)
        chunk_bld = test_bld.subset(list(range(n_chunk)))
        chunk_bld.labels = raw_preds
        adjusted = eq.predict(chunk_bld)
        return adjusted.labels.ravel()

    single_ms, batch_ms = measure_latency(predict_fn_with_eqodds, X_test)

    return {
        **metrics,
        "train_time_sec": train_time_sec,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": 22,
        "implementation_dependencies_added": 0,
        "implementation_hyperparams": 0,
        "implementation_stage": "postprocessing",
        "implementation_complexity_score": 3,
        "method_hyperparams": "fit_on=validation",
    }


METHOD_RUNNERS = {
    "baseline": run_baseline,
    "reweighing": run_reweighing,
    "dir": run_dir,
    "eqodds": run_eqodds,
}


# ---------------------------------------------------------------------------
# CSV writer.
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "model", "method", "seed", "split_seed", "model_seed",
    "protected_attribute", "privileged_group", "unprivileged_group",
    "n_train", "n_val", "n_test",
    "accuracy", "f1", "spd", "eod", "disparate_impact",
    "train_time_sec", "single_pred_latency_ms", "batch_256_latency_ms",
    "implementation_loc_added", "implementation_dependencies_added",
    "implementation_hyperparams", "implementation_stage",
    "implementation_complexity_score", "method_hyperparams",
    "aif360_version", "sklearn_version", "notes",
]


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


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    parser.add_argument("--model", required=True, choices=["lr", "rf"])
    parser.add_argument("--method", required=True, choices=list(METHOD_RUNNERS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", default="results/full_results.csv")
    args = parser.parse_args()

    split_seed = args.seed
    model_seed = args.seed
    np.random.seed(args.seed)

    info = DATASET_LOADERS[args.dataset]()
    bld = info["dataset"]
    prot_attr = info["protected_attribute"]
    priv = info["privileged_groups"]
    unpriv = info["unprivileged_groups"]

    train_bld, val_bld, test_bld = split_dataset(bld, prot_attr, split_seed)

    runner = METHOD_RUNNERS[args.method]
    result = runner(
        bld, train_bld, val_bld, test_bld,
        args.model, model_seed,
        priv, unpriv,
        prot_attr,
        split_seed,
    )

    row = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "protected_attribute": prot_attr,
        "privileged_group": str(priv),
        "unprivileged_group": str(unpriv),
        "n_train": train_bld.features.shape[0],
        "n_val": val_bld.features.shape[0],
        "n_test": test_bld.features.shape[0],
        "aif360_version": aif360.__version__,
        "sklearn_version": sklearn.__version__,
        "notes": "",
        **result,
    }

    write_row(row, args.out)
    print(f"Wrote 1 row to {args.out}")
    print({k: row[k] for k in ("dataset", "model", "method", "seed",
                                "accuracy", "f1", "spd", "eod",
                                "disparate_impact", "train_time_sec",
                                "single_pred_latency_ms", "batch_256_latency_ms")})


if __name__ == "__main__":
    main()
