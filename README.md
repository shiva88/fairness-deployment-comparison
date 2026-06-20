# Deployment-Aware Comparison of Lightweight Bias Mitigation Techniques

Reproducibility package for the paper:

> **Deployment-Aware Comparison of Lightweight Bias Mitigation Techniques in Machine Learning Pipelines**
> Shivaraman Parthasarathy (Independent Researcher)
> IEEE Access (under review), Manuscript ID Access-2026-19472

This repository contains the code, frozen environment specification, and full result matrices for a 240-run empirical comparison of Reweighing, Disparate Impact Remover (DIR), and Equalized Odds post-processing against an unmitigated baseline, across three benchmark datasets (Adult, COMPAS, German Credit) and two classifiers (Logistic Regression, Random Forest), with an additional 60-run corrected DIR sub-protocol.

---

## Key methodological finding: AIF360 transductive DIR leakage

The AIF360 `DisparateImpactRemover` class exposes only a `fit_transform(dataset)` API. When applied before the train/test split — the pattern shown in the AIF360 reference tutorials — the test rows participate in computing the per-protected-group empirical CDFs and the median target distribution that drive the quantile-mapping repair. The repaired test data is then evaluated against a distribution that the test rows themselves helped define, producing test-time fairness measurements that overstate DIR's effectiveness.

This repository provides **`dir_train_only.py`**, a drop-in replacement that preserves the Feldman et al. (2015) quantile-mapping algorithm but fits the per-group CDFs and median target distribution from training data only and applies an out-of-sample quantile-rank lookup at inference time.

Under the corrected protocol on the three benchmarks here, DIR's |SPD| improvement (originally 6 of 6 cells, 4–38% reduction under the transductive protocol) drops to 3 of 6 cells with marginal effect, and |SPD| increases by 3–54% in the other 3 cells. See Section V.B of the paper for full discussion.

---

## Repository contents

    README.md                   This file
    requirements.txt            Frozen pip environment (Python 3.11.9)
    LICENSE                     MIT
    dir_train_only.py           Train-only Disparate Impact Remover module
    run_one.py                  Per-cell experiment runner (one dataset/model/method/seed)
    results/
      full_results.csv          240 rows: 3 datasets x 2 models x 4 methods x 10 seeds
      dir_train_only.csv        60 rows: corrected DIR sub-protocol

---

## Requirements

Exact environment used for all results in the paper:

- **OS:** Microsoft Windows 11 Home, build 26200
- **Hardware:** Intel Core i7-1165G7 @ 2.70 GHz (4 cores / 8 threads), 16 GB RAM
- **Python:** 3.11.9
- **Key libraries:** scikit-learn 1.5.2, AIF360 0.6.1, NumPy 1.26.4, Pandas 1.5.3 (full frozen pin list in `requirements.txt`)

Install on Windows:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

On Linux/macOS:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

---

## Reproducing the paper

### Full 240-run sweep (Tables 1–4)

Iterate `run_one.py` over the experimental matrix (3 datasets x 2 models x 4 methods x 10 seeds) and accumulate to `results/full_results.csv`. See the argument signature at the top of `run_one.py`.

### Corrected DIR sub-protocol (60 rows)

Same procedure restricted to `method=dir`, with DIR routed through `TrainOnlyDisparateImpactRemover` in `dir_train_only.py`. Output to `results/dir_train_only.csv`.

### Random seeds

All experiments use these 10 seeds:

    42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768

A single seed controls both the 60/20/20 stratified split (on label x protected attribute) and the model's `random_state`. `PYTHONHASHSEED=0` and `n_jobs=1` are enforced.

---

## Datasets

Loaded via AIF360's standard preprocessing:

| Dataset       | n      | Protected attr. | Positive label              | Prevalence |
|---------------|--------|-----------------|-----------------------------|------------|
| Adult         | 45,222 | sex             | income > $50K               | ~0.24      |
| COMPAS        | 6,167  | race            | no recidivism within 2 yrs  | ~0.54      |
| German Credit | 1,000  | age (cutoff 25) | good credit                 | ~0.70      |

---

## Result file columns

Both `results/full_results.csv` and `results/dir_train_only.csv` have the following columns:

    dataset, model, method, seed,
    accuracy, f1,
    spd, eod, disparate_impact,
    train_time_s, infer_single_ms, infer_batch256_ms,
    complexity_score

Notes:
- `spd` and `eod` are **signed** values per seed. Fairness magnitudes reported in the paper are `mean(|SPD|)` and `mean(|EOD|)` across the 10 seeds.
- `complexity_score` is the rubric score from Table 6 (Reweighing 2, DIR 3, EqOdds 3).

---

## Citation

If you use this code or results, please cite:

    @article{parthasarathy2026deployment,
      title   = {Deployment-Aware Comparison of Lightweight Bias Mitigation Techniques in Machine Learning Pipelines},
      author  = {Parthasarathy, Shivaraman},
      journal = {IEEE Access},
      year    = {2026},
      note    = {Under review, Manuscript ID Access-2026-19472}
    }

---

## License

MIT. See `LICENSE`.

## Contact

Shivaraman Parthasarathy — clicktoraman@gmail.com
ORCID: [0009-0006-3817-8535](https://orcid.org/0009-0006-3817-8535)
