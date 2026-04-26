# Deployment-Aware Comparison of Lightweight Bias Mitigation Techniques

Reproducibility code for the paper *"Deployment-Aware Comparison of Lightweight Bias Mitigation Techniques in Machine Learning Pipelines"* (IEEE Access, under review).

This repository contains everything needed to reproduce the 240-run experimental matrix, the aggregation tables, and the figures in the paper.

---

## What this code does

Evaluates **3 bias mitigation methods** (Reweighing, Disparate Impact Remover, Equalized Odds post-processing) plus a baseline against an unmitigated classifier, across:

- **3 datasets**: Adult (US Census), COMPAS (recidivism), German Credit
- **2 classifiers**: Logistic Regression, Random Forest
- **10 random seeds**: 42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768

Total: **3 × 2 × 4 × 10 = 240 runs**.

Reports performance (accuracy, F1), fairness (SPD, EOD, Disparate Impact), and **deployment cost** (training time, single-prediction latency, batch-256 latency, implementation complexity).

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/shiva88/fairness-deployment-comparison.git
cd fairness-deployment-comparison

python3.11 -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

**Python 3.11 is required.** AIF360 0.6.x does not work on Python 3.12 or 3.13 due to dependency constraints with `BlackBoxAuditing` and recent NumPy/pandas releases.

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs AIF360, scikit-learn, BlackBoxAuditing, and the pinned versions of NumPy and pandas that AIF360 needs.

### 3. Place the dataset files

AIF360 does not ship dataset files. Download them from UCI / ProPublica and place them where AIF360 expects:

**Adult** — three files into `.venv/lib/python3.11/site-packages/aif360/data/raw/adult/`:
- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

**COMPAS** — one file into `.venv/lib/python3.11/site-packages/aif360/data/raw/compas/`:
- https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

**German Credit** — two files into `.venv/lib/python3.11/site-packages/aif360/data/raw/german/`:
- https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc

On Windows, browsers may add `.txt` to the filenames and "block" the files. Rename to remove `.txt` and unblock via right-click → Properties → Unblock if needed.

### 4. Run a single experiment

```bash
python src/run_one.py --dataset adult --model lr --method baseline --seed 42
```

This appends one row to `results/full_results.csv`. Expected output:

```
Wrote 1 row to results/full_results.csv
{'dataset': 'adult', 'model': 'lr', 'method': 'baseline', 'seed': 42,
 'accuracy': 0.8528, 'f1': 0.6735, 'spd': -0.1750, 'eod': -0.0828,
 'disparate_impact': 0.3262, 'train_time_sec': 0.10, ...}
```

### 5. Run the full 240-run matrix

```bash
python src/run_all.py
```

Estimated time: 30–60 minutes on a modern laptop. The script appends rows to `results/full_results.csv` as it progresses.

### 6. Aggregate results and generate figures

```bash
python src/analyze.py
python src/make_figures.py
```

Outputs `results/summary_stats.csv`, `results/wilcoxon_tests.csv`, and PNGs in `figures/`.

---

## Repository structure

```
fairness-deployment-comparison/
├── README.md                  # this file
├── LICENSE                    # MIT
├── requirements.txt           # pinned dependencies
├── .gitignore
├── src/
│   ├── run_one.py             # single experimental run (one CSV row)
│   ├── run_all.py             # batch driver (240 runs)
│   ├── analyze.py             # aggregation + statistical tests
│   └── make_figures.py        # the 4 paper figures
├── results/
│   └── full_results.csv       # the 240-row experimental output
├── figures/                   # output PNGs
└── paper/                     # (optional) paper draft and assets
```

---

## Reproducibility guarantees

- **Determinism within a single run**: `random_state=seed` and `n_jobs=1` are set everywhere; running the same command twice produces byte-identical fairness/accuracy metrics.
- **Pinned environment**: `requirements.txt` lists exact versions. The original experiments used Python 3.11.9, AIF360 0.6.1, scikit-learn 1.5.2, NumPy 1.26.4, pandas 1.5.3.
- **No GPU**: all experiments run on CPU; results are reproducible on any machine that can install the dependencies.
- **Latency caveat**: train time and inference latency vary slightly across runs due to wall-clock noise. Fairness and accuracy metrics do not.

---

## Methodological notes

- **Train/Validation/Test split**: 60/20/20, stratified on the intersection of class label and protected attribute. The validation split is used only by Equalized Odds post-processing for calibration.
- **Disparate Impact Remover (DIR)** is applied to the full dataset before splitting, following AIF360's reference implementation. This introduces limited feature-distribution leakage without label exposure. DIR uses `repair_level=1.0`.
- **Equalized Odds** is fit on validation predictions (not test) to avoid leakage.
- **Inference latency for DIR** includes the amortized per-row repair-transform cost, not just `model.predict`.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{parthasarathy2026deployment,
  title   = {Deployment-Aware Comparison of Lightweight Bias Mitigation
             Techniques in Machine Learning Pipelines},
  author  = {Parthasarathy, Shivaraman},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT — see `LICENSE`.

The Adult, COMPAS, and German Credit datasets retain their original licenses (UCI Machine Learning Repository terms; ProPublica's published data is in the public domain).

---

## Contact

Shivaraman Parthasarathy — clicktoraman@gmail.com
ORCID: [0009-0006-3817-8535](https://orcid.org/0009-0006-3817-8535)

Issues and pull requests welcome.
