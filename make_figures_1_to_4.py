"""
make_figures_1_to_4.py

Generates Figures 1-4 for IEEE Access manuscript Access-2026-19472.

All four figures use:
- baseline, reweighing, eqodds rows from results/full_results.csv
- DIR rows from results/dir_train_only.csv  (CORRECTED train-only protocol)

This is critical: using transductive DIR rows from full_results.csv would
contradict Tables 1-3 and Section V.B of the corrected manuscript.

Outputs:
    figures/fig1_spd.png             |SPD| by dataset and model
    figures/fig2_eod.png             |EOD| by dataset and model
    figures/fig3_latency.png         Batch-256 inference latency (log scale)
    figures/fig4_accuracy_eod.png    Accuracy vs. |EOD| trade-off

Usage from repo root:
    python make_figures_1_to_4.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Configuration -----
FULL_CSV = "results/full_results.csv"
CORRECTED_CSV = "results/dir_train_only.csv"
OUT_DIR = "figures"

DATASETS = [("adult", "Adult"), ("compas", "COMPAS"), ("german", "German Credit")]
MODELS = [("lr", "Logistic Regression"), ("rf", "Random Forest")]
METHODS = ["baseline", "reweighing", "dir", "eqodds"]
METHOD_LABELS = ["Baseline", "Reweighing", "DIR", "EqOdds"]
# IEEE-friendly palette
METHOD_COLORS = ["#808080", "#1F77B4", "#D62728", "#2CA02C"]

# ----- Load and combine: substitute corrected DIR -----
full = pd.read_csv(FULL_CSV)
corrected = pd.read_csv(CORRECTED_CSV)

# Drop the transductive DIR rows from full, replace with corrected ones
df = pd.concat([
    full[full["method"] != "dir"],
    corrected[corrected["method"] == "dir"]
], ignore_index=True)
df["abs_spd"] = df["spd"].abs()
df["abs_eod"] = df["eod"].abs()


def cell(df, ds, model, method, col):
    sub = df[(df["dataset"] == ds) & (df["model"] == model) & (df["method"] == method)]
    return sub[col].mean(), sub[col].std(ddof=1), len(sub)


def make_bar_grid(metric_col, ylabel, title, outpath, log_scale=False):
    """Generic 2x3 grouped-bar figure: 6 panels, 4 method bars each."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 7),
                              sharex=False, sharey=False)
    x = np.arange(4)
    width = 0.65

    for r, (model_key, model_label) in enumerate(MODELS):
        for c, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[r, c]
            means, stds = [], []
            for method in METHODS:
                m, s, _ = cell(df, ds_key, model_key, method, metric_col)
                means.append(m)
                stds.append(s)

            bars = ax.bar(x, means, width, yerr=stds,
                          color=METHOD_COLORS, edgecolor="black", linewidth=0.7,
                          capsize=4,
                          error_kw={"elinewidth": 1.0, "ecolor": "#222222"})

            # Value labels
            for i, (m, s) in enumerate(zip(means, stds)):
                if log_scale:
                    label_y = m * 1.15
                else:
                    label_y = m + s + (max(means) * 0.03)
                ax.text(i, label_y, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="medium")

            ax.set_xticks(x)
            ax.set_xticklabels(METHOD_LABELS, fontsize=9, rotation=20)

            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(0.1, min(means) * 0.5))
            else:
                upper = max(m + s for m, s in zip(means, stds)) * 1.25
                ax.set_ylim(0, max(upper, 0.05))

            ax.grid(axis="y", linestyle="-", alpha=0.25)
            ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(f"{model_label}\n{ylabel}", fontsize=10)
            if r == 0:
                ax.set_title(ds_label, fontsize=12, fontweight="bold")

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[i],
                      edgecolor="black", label=METHOD_LABELS[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=11)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def make_tradeoff_scatter():
    """Figure 4: Accuracy vs. |EOD| trade-off, one point per (dataset, model, method)."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 7),
                              sharex=False, sharey=False)

    for r, (model_key, model_label) in enumerate(MODELS):
        for c, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[r, c]
            for i, (method, color, label) in enumerate(zip(METHODS, METHOD_COLORS, METHOD_LABELS)):
                acc_m, acc_s, _ = cell(df, ds_key, model_key, method, "accuracy")
                eod_m, eod_s, _ = cell(df, ds_key, model_key, method, "abs_eod")
                ax.errorbar(acc_m, eod_m, xerr=acc_s, yerr=eod_s,
                            fmt="o", color=color, ecolor=color, markersize=11,
                            markeredgecolor="black", markeredgewidth=0.8,
                            capsize=3, elinewidth=1.0, label=label, zorder=3)

            ax.grid(linestyle="-", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_xlabel("Accuracy", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{model_label}\n|EOD|", fontsize=10)
            if r == 0:
                ax.set_title(ds_label, fontsize=12, fontweight="bold")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=METHOD_COLORS[i], markeredgecolor="black",
                   markersize=11, label=METHOD_LABELS[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=11)
    fig.suptitle(
        "Accuracy vs. |EOD| Trade-off — higher accuracy and lower |EOD| are both preferred",
        fontsize=13, fontweight="bold", y=1.05,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "fig4_accuracy_eod.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ----- Build all four -----
os.makedirs(OUT_DIR, exist_ok=True)

make_bar_grid(
    metric_col="abs_spd",
    ylabel="|SPD|",
    title="Statistical Parity Difference (|SPD|) by Dataset and Model — lower is fairer",
    outpath=os.path.join(OUT_DIR, "fig1_spd.png"),
)

make_bar_grid(
    metric_col="abs_eod",
    ylabel="|EOD|",
    title="Equal Opportunity Difference (|EOD|) by Dataset and Model — lower is fairer",
    outpath=os.path.join(OUT_DIR, "fig2_eod.png"),
)

make_bar_grid(
    metric_col="batch_256_latency_ms",
    ylabel="Batch-256 latency (ms)",
    title="Inference Latency (Batch-256, log scale) by Dataset and Model — lower is faster",
    outpath=os.path.join(OUT_DIR, "fig3_latency.png"),
    log_scale=True,
)

make_tradeoff_scatter()

print("\nAll four figures generated.")
