"""
make_figure5.py

Generates Figure 5 for the IEEE Access manuscript Access-2026-19472:
"Disparate Impact under the original (AIF360 transductive) DIR protocol and
the corrected (train-only) DIR protocol, by dataset and classifier."

Inputs:
    results/full_results.csv     — 240 rows; contains baseline + transductive DIR
    results/dir_train_only.csv   — 60 rows;  contains corrected (train-only) DIR

Output:
    figures/fig5_dir_di_comparison.png  (300 DPI, ~12x6 inches, fits IEEE 2-col width)

Usage from repo root:
    python make_figure5.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Configuration -----
FULL_CSV = "results/full_results.csv"
CORRECTED_CSV = "results/dir_train_only.csv"
OUT_PNG = "figures/fig5_dir_di_comparison.png"

DATASETS = [("adult", "Adult"), ("compas", "COMPAS"), ("german", "German Credit")]
MODELS = [("lr", "Logistic Regression"), ("rf", "Random Forest")]

# IEEE-friendly palette
COLOR_BASELINE = "#808080"      # gray
COLOR_TRANSDUCTIVE = "#E69F00"  # amber — the inflated one
COLOR_CORRECTED = "#D55E00"     # vermillion — the honest one

BAR_LABELS = ["Baseline", "DIR (transductive)", "DIR (train-only, corrected)"]

# ----- Load data -----
full = pd.read_csv(FULL_CSV)
corrected = pd.read_csv(CORRECTED_CSV)

baseline = full[full["method"] == "baseline"]
transductive_dir = full[full["method"] == "dir"]
corrected_dir = corrected[corrected["method"] == "dir"]


def cell_stats(df, ds, model):
    """Return (mean DI, std DI, n) for a given dataset/model subset."""
    sub = df[(df["dataset"] == ds) & (df["model"] == model)]
    di = sub["disparate_impact"].values
    return di.mean(), di.std(ddof=1), len(di)


# ----- Build figure -----
fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(12, 6.5),
    sharex=False, sharey=False,
)

x = np.arange(3)
width = 0.68

for r, (model_key, model_label) in enumerate(MODELS):
    for c, (ds_key, ds_label) in enumerate(DATASETS):
        ax = axes[r, c]

        m_b, s_b, n_b = cell_stats(baseline, ds_key, model_key)
        m_t, s_t, n_t = cell_stats(transductive_dir, ds_key, model_key)
        m_c, s_c, n_c = cell_stats(corrected_dir, ds_key, model_key)

        means = [m_b, m_t, m_c]
        stds = [s_b, s_t, s_c]
        colors = [COLOR_BASELINE, COLOR_TRANSDUCTIVE, COLOR_CORRECTED]

        bars = ax.bar(x, means, width, yerr=stds,
                      color=colors, edgecolor="black", linewidth=0.7,
                      capsize=4, error_kw={"elinewidth": 1.0, "ecolor": "#222222"})

        # DI = 1.0 reference (perfect parity)
        ax.axhline(1.0, color="#2E7D32", linestyle=":", linewidth=1.0, zorder=0)
        # DI = 0.8 reference (4/5 legal threshold)
        ax.axhline(0.8, color="#B71C1C", linestyle=":", linewidth=1.0, zorder=0)

        # Value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="medium")

        # Cosmetics
        ax.set_xticks(x)
        ax.set_xticklabels(["Base", "DIR\ntrans.", "DIR\ncorr."],
                            fontsize=9)
        ax.set_ylim(0, max(1.25, max(m + s for m, s in zip(means, stds)) + 0.15))
        ax.grid(axis="y", linestyle="-", alpha=0.25)
        ax.set_axisbelow(True)

        if c == 0:
            ax.set_ylabel(f"{model_label}\nDisparate Impact", fontsize=10)
        if r == 0:
            ax.set_title(ds_label, fontsize=12, fontweight="bold")

# Shared legend at top
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_BASELINE, edgecolor="black", label=BAR_LABELS[0]),
    plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TRANSDUCTIVE, edgecolor="black", label=BAR_LABELS[1]),
    plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CORRECTED, edgecolor="black", label=BAR_LABELS[2]),
    plt.Line2D([0], [0], color="#2E7D32", linestyle=":", linewidth=1.5, label="DI = 1.0 (parity)"),
    plt.Line2D([0], [0], color="#B71C1C", linestyle=":", linewidth=1.5, label="DI = 0.8 (4/5 rule)"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=10)

fig.suptitle(
    "Disparate Impact: Baseline vs. Transductive DIR vs. Train-Only (Corrected) DIR — closer to 1.0 is fairer",
    fontsize=12, fontweight="bold", y=1.05,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
print(f"Image size: {os.path.getsize(OUT_PNG)} bytes")
