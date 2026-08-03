"""
run_hyperparameter_tuning.py
------------------------------
Hyperparameter tuning for all four methods (R2 concern #4).

For each cell (dataset x model x method x seed), searches a modest grid of
classifier hyperparameters on the VALIDATION fold and reports the best-tuned
result on the TEST fold.

Grid:
    LR:  C in {0.1, 1.0, 10.0}                             = 3 combinations
    RF:  n_estimators x max_depth x class_weight
         {100, 200} x {None, 10} x {None, 'balanced'}     = 8 combinations

Usage (full sweep - ~2-4 hours):
    python run_hyperparameter_tuning.py

Usage (smoke test - 30 seconds):
    python run_hyperparameter_tuning.py --datasets german --models lr --methods baseline --seeds 42

Output:
    results/tuned_results.csv
"""
import argparse
import os
import time
import json

import numpy as np
os.environ.setdefault("PYTHONHASHSEED", "0")

import aif360
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

from run_one import (
    DATASET_LOADERS,
    split_dataset,
    needs_scaling,
    compute_metrics,
    measure_latency,
    write_row,
)
from dir_train_only import TrainOnlyDisparateImpactRemover

DEFAULT_SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_DATASETS = ["adult", "compas", "german"]
DEFAULT_MODELS = ["lr", "rf"]
DEFAULT_METHODS = ["baseline", "reweighing", "dir", "eqodds"]
N_QUANTILES = 101

LR_GRID = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]

RF_GRID = [
    {"n_estimators": 100, "max_depth": None, "class_weight": None},
    {"n_estimators": 100, "max_depth": None, "class_weight": "balanced"},
    {"n_estimators": 100, "max_depth": 10,   "class_weight": None},
    {"n_estimators": 100, "max_depth": 10,   "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": None, "class_weight": None},
    {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 10,   "class_weight": None},
    {"n_estimators": 200, "max_depth": 10,   "class_weight": "balanced"},
]


def make_model_tuned(model_name, seed, params):
    if model_name == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1,
                                    solver="lbfgs", C=params["C"])
    if model_name == "rf":
        return RandomForestClassifier(
            random_state=seed, n_jobs=1,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight=params["class_weight"],
        )
    raise ValueError(model_name)


def tune_on_validation(X_train, y_train, X_val, y_val, model_name, seed,
                        sample_weight=None):
    best_acc, best_params = -1.0, None
    grid = LR_GRID if model_name == "lr" else RF_GRID
    for params in grid:
        model = make_model_tuned(model_name, seed, params)
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_params, best_acc


def _row(metrics, train_time, single_ms, batch_ms, params, val_acc,
         loc, deps, hparams, stage, score, extra=None):
    hd = {**params, "val_acc": val_acc}
    if extra:
        hd.update(extra)
    return {
        **metrics,
        "train_time_sec": train_time,
        "single_pred_latency_ms": single_ms,
        "batch_256_latency_ms": batch_ms,
        "implementation_loc_added": loc,
        "implementation_dependencies_added": deps,
        "implementation_hyperparams": hparams,
        "implementation_stage": stage,
        "implementation_complexity_score": score,
        "method_hyperparams": json.dumps(hd),
    }


def tuned_baseline(train_bld, val_bld, test_bld, model_name, seed, priv, unpriv):
    X_tr = train_bld.features
    y_tr = train_bld.labels.ravel()
    X_va = val_bld.features
    y_va = val_bld.labels.ravel()
    X_te = test_bld.features

    scaler = None
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    best, val_acc = tune_on_validation(X_tr, y_tr, X_va, y_va, model_name, seed)
    model = make_model_tuned(model_name, seed, best)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    tt = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    metrics = compute_metrics(test_bld, y_pred, priv, unpriv)
    sm, bm = measure_latency(model.predict, X_te)
    return _row(metrics, tt, sm, bm, best, val_acc, 0, 0, len(best), "none", 1)


def tuned_reweighing(train_bld, val_bld, test_bld, model_name, seed, priv, unpriv):
    rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
    train_rw = rw.fit_transform(train_bld)

    X_tr = train_rw.features
    y_tr = train_rw.labels.ravel()
    sw = train_rw.instance_weights.ravel()
    X_va = val_bld.features
    y_va = val_bld.labels.ravel()
    X_te = test_bld.features

    scaler = None
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    best, val_acc = tune_on_validation(X_tr, y_tr, X_va, y_va, model_name, seed,
                                         sample_weight=sw)
    model = make_model_tuned(model_name, seed, best)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, sample_weight=sw)
    tt = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    metrics = compute_metrics(test_bld, y_pred, priv, unpriv)
    sm, bm = measure_latency(model.predict, X_te)
    return _row(metrics, tt, sm, bm, best, val_acc, 6, 0, len(best),
                "preprocessing", 2)


def tuned_dir(train_bld, val_bld, test_bld, model_name, seed, priv, unpriv,
               protected_attribute):
    t0 = time.perf_counter()
    dir_op = TrainOnlyDisparateImpactRemover(
        repair_level=1.0,
        sensitive_attribute=protected_attribute,
        n_quantiles=N_QUANTILES,
    )
    dir_op.fit(train_bld)
    dir_fit = time.perf_counter() - t0

    tr_rep = dir_op.transform(train_bld)
    va_rep = dir_op.transform(val_bld)
    te_rep = dir_op.transform(test_bld)

    fi = tr_rep.feature_names.index(protected_attribute)
    X_tr = np.delete(tr_rep.features, fi, axis=1)
    y_tr = tr_rep.labels.ravel()
    X_va = np.delete(va_rep.features, fi, axis=1)
    y_va = va_rep.labels.ravel()
    X_te = np.delete(te_rep.features, fi, axis=1)

    scaler = None
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    best, val_acc = tune_on_validation(X_tr, y_tr, X_va, y_va, model_name, seed)
    model = make_model_tuned(model_name, seed, best)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    tt = dir_fit + (time.perf_counter() - t0)

    y_pred = model.predict(X_te)
    metrics = compute_metrics(test_bld, y_pred, priv, unpriv)

    def predict_fn(X_raw_chunk):
        X_rep = dir_op.transform_array(X_raw_chunk)
        X_ch = np.delete(X_rep, fi, axis=1)
        if scaler is not None:
            X_ch = scaler.transform(X_ch)
        return model.predict(X_ch)

    sm, bm = measure_latency(predict_fn, test_bld.features)
    return _row(metrics, tt, sm, bm, best, val_acc, 24, 1, len(best) + 2,
                "preprocessing", 3,
                extra={"repair_level": 1.0, "n_quantiles": N_QUANTILES})


def tuned_eqodds(train_bld, val_bld, test_bld, model_name, seed, priv, unpriv):
    X_tr = train_bld.features
    y_tr = train_bld.labels.ravel()
    X_va = val_bld.features
    y_va = val_bld.labels.ravel()
    X_te = test_bld.features

    scaler = None
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    best, val_acc = tune_on_validation(X_tr, y_tr, X_va, y_va, model_name, seed)
    model = make_model_tuned(model_name, seed, best)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    model_tt = time.perf_counter() - t0

    val_pred = val_bld.copy()
    val_pred.labels = model.predict(X_va).reshape(-1, 1).astype(float)
    test_pred = test_bld.copy()
    test_pred.labels = model.predict(X_te).reshape(-1, 1).astype(float)

    eq = EqOddsPostprocessing(unprivileged_groups=unpriv,
                                privileged_groups=priv, seed=seed)
    t0 = time.perf_counter()
    eq = eq.fit(val_bld, val_pred)
    eq_fit = time.perf_counter() - t0
    tt = model_tt + eq_fit

    test_pred_eq = eq.predict(test_pred)
    y_pred = test_pred_eq.labels.ravel().astype(int)
    metrics = compute_metrics(test_bld, y_pred, priv, unpriv)

    def predict_fn(X_chunk):
        rp = model.predict(X_chunk).reshape(-1, 1).astype(float)
        cb = test_bld.subset(list(range(len(X_chunk))))
        cb.labels = rp
        return eq.predict(cb).labels.ravel()

    sm, bm = measure_latency(predict_fn, X_te)
    return _row(metrics, tt, sm, bm, best, val_acc, 22, 0, len(best),
                "postprocessing", 3)


METHOD_RUNNERS = {
    "baseline": lambda tr, va, te, m, s, p, u, pa: tuned_baseline(tr, va, te, m, s, p, u),
    "reweighing": lambda tr, va, te, m, s, p, u, pa: tuned_reweighing(tr, va, te, m, s, p, u),
    "dir": lambda tr, va, te, m, s, p, u, pa: tuned_dir(tr, va, te, m, s, p, u, pa),
    "eqodds": lambda tr, va, te, m, s, p, u, pa: tuned_eqodds(tr, va, te, m, s, p, u),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=list(DATASET_LOADERS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["lr", "rf"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="results/tuned_results.csv")
    args = parser.parse_args()

    total = (len(args.datasets) * len(args.models)
             * len(args.methods) * len(args.seeds))
    print(f"Planned runs: {total}")
    print(f"  datasets: {args.datasets}")
    print(f"  models: {args.models}")
    print(f"  methods: {args.methods}")
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
                for method_name in args.methods:
                    try:
                        np.random.seed(seed)
                        result = METHOD_RUNNERS[method_name](
                            train_bld, val_bld, test_bld,
                            model_name, seed, priv, unpriv, prot_attr,
                        )
                        row = {
                            "dataset": dataset_name,
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
                            "notes": "tuned_hp",
                            **result,
                        }
                        write_row(row, args.out)
                        done += 1
                        elapsed = time.time() - start
                        rate = done / max(elapsed, 0.01)
                        eta = (total - done) / max(rate, 0.01)
                        print(f"[{done}/{total}] {dataset_name}/{model_name}/"
                              f"{method_name}/s={seed}: "
                              f"acc={result['accuracy']:.3f} "
                              f"|SPD|={abs(result['spd']):.3f} "
                              f"ETA {eta/60:.1f}min")
                    except Exception as e:
                        done += 1
                        print(f"  ERROR {dataset_name}/{model_name}/"
                              f"{method_name}/s={seed}: {e}")

    print(f"\nComplete. Total time: {(time.time()-start)/60:.1f} min")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
