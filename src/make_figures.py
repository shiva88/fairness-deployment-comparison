"""
make_figures.py - generate the 4 figures used in the paper.

Outputs:
  figures/fig1_spd.png      |SPD| comparison (3 datasets × 2 models)
  figures/fig2_eod.png      |EOD| comparison (3 datasets × 2 models)
  figures/fig3_latency.png  Batch-256 latency, log scale
  figures/fig4_pareto.png   Accuracy vs |EOD| trade-off

Usage:
    python src/make_figures.py
    python src/make_figures.py --in results/full_results.csv --out-dir figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---- Style ----
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"

METHOD_ORDER = ["baseline", "reweighing", "dir", "eqodds"]
METHOD_LABELS = {"baseline": "Baseline", "reweighing": "Reweighing",
                 "dir": "DIR", "eqodds": "EqOdds"}
METHOD_COLORS = {
    "baseline":   "#7F7F7F",
    "reweighing": "#2E86AB",
    "dir":        "#E63946",
    "eqodds":     "#06A77D",
}
DATASETS = ["adult", "compas", "german"]
DATASET_LABELS = {"adult": "Adult", "compas": "COMPAS", "german": "German Credit"}
DATASET_MARKERS = {"adult": "o", "compas": "s", "german": "^"}
MODELS = ["lr", "rf"]
MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest"}


def load_agg(infile):
    df = pd.read_csv(infile)
    df = df.drop_duplicates(subset=["dataset", "model", "method", "seed"], keep="last")
    df["abs_spd"] = df["spd"].abs()
    df["abs_eod"] = df["eod"].abs()
    return df.groupby(["dataset", "model", "method"]).agg(
        spd_mean=("abs_spd", "mean"), spd_std=("abs_spd", "std"),
        eod_mean=("abs_eod", "mean"), eod_std=("abs_eod", "std"),
        lat_mean=("batch_256_latency_ms", "mean"),
        lat_std=("batch_256_latency_ms", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    ).reset_index()


def make_fairness_grid(agg, metric_mean, metric_std, ylabel_metric, title, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for col, ds in enumerate(DATASETS):
        for row, mdl in enumerate(MODELS):
            ax = axes[row, col]
            sub = agg[(agg["dataset"] == ds) & (agg["model"] == mdl)]
            sub = sub.set_index("method").reindex(METHOD_ORDER)
            x = np.arange(len(METHOD_ORDER))
            means = sub[metric_mean].values
            stds = sub[metric_std].values
            colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
            bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                          edgecolor="black", linewidth=0.6, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                               rotation=20, ha="right")
            top = max(0.30, (means + stds).max() * 1.2)
            ax.set_ylim(0, top)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[mdl]}\n{ylabel_metric}",
                              fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
            for bar, mean in zip(bars, means):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (means + stds).max() * 0.03,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(str(outpath).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def make_latency(agg, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for idx, mdl in enumerate(MODELS):
        ax = axes[idx]
        bar_pos, bar_h, bar_e, bar_c, bar_l, group_centers = [], [], [], [], [], []
        pos = 0
        for ds in DATASETS:
            group_start = pos
            for method in METHOD_ORDER:
                row = agg[
                    (agg["dataset"] == ds) & (agg["model"] == mdl) & (agg["method"] == method)
                ].iloc[0]
                bar_pos.append(pos)
                bar_h.append(row["lat_mean"])
                bar_e.append(row["lat_std"])
                bar_c.append(METHOD_COLORS[method])
                bar_l.append(METHOD_LABELS[method])
                pos += 1
            group_centers.append((group_start + pos - 1) / 2)
            pos += 1
        bars = ax.bar(bar_pos, bar_h, yerr=bar_e, capsize=3,
                      color=bar_c, edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 1000)
        ax.set_ylabel("Batch-256 latency (ms, log scale)", fontsize=10)
        ax.set_title(MODEL_LABELS[mdl], fontsize=11, fontweight="bold")
        ax.set_xticks(group_centers)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                           fontsize=10, fontweight="bold")
        for bar, label, h in zip(bars, bar_l, bar_h):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
                    label[:3], ha="center", va="bottom", fontsize=7, rotation=90)

    legend = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black",
                    label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    fig.legend(handles=legend, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)
    fig.suptitle("Inference Latency (batch-256, median over 20 calls) — note log scale",
                 fontsize=12, fontweight="bold", y=1.10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(str(outpath).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def make_pareto(agg, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for idx, mdl in enumerate(MODELS):
        ax = axes[idx]
        sub = agg[agg["model"] == mdl]
        for ds in DATASETS:
            for _, r in sub[sub["dataset"] == ds].iterrows():
                ax.errorbar(
                    r["acc_mean"], r["eod_mean"],
                    xerr=r["acc_std"], yerr=r["eod_std"],
                    marker=DATASET_MARKERS[ds], color=METHOD_COLORS[r["method"]],
                    markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                    capsize=2, linewidth=0.8, alpha=0.85,
                )
        ax.set_xlabel("Accuracy (mean across 10 seeds)", fontsize=10)
        ax.set_ylabel("|EOD| (mean across 10 seeds)", fontsize=10)
        ax.set_title(MODEL_LABELS[mdl], fontsize=11, fontweight="bold")
        ax.annotate("better\n(higher acc, lower |EOD|)",
                    xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8, style="italic", color="#444")

    legend_method = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black",
                           label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    legend_ds = [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="w",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label=DATASET_LABELS[d]) for d in DATASETS
    ]
    fig.legend(handles=legend_method + legend_ds, loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.suptitle("Accuracy vs. |EOD| trade-off — methods (color) and datasets (marker)",
                 fontsize=12, fontweight="bold", y=1.07)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(str(outpath).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="results/full_results.csv")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_agg(args.infile)

    make_fairness_grid(
        agg, "spd_mean", "spd_std", "|SPD|",
        "Statistical Parity Difference (|SPD|) by Dataset and Model — lower is fairer",
        out_dir / "fig1_spd.png",
    )
    make_fairness_grid(
        agg, "eod_mean", "eod_std", "|EOD|",
        "Equal Opportunity Difference (|EOD|) by Dataset and Model — lower is fairer",
        out_dir / "fig2_eod.png",
    )
    make_latency(agg, out_dir / "fig3_latency.png")
    make_pareto(agg, out_dir / "fig4_pareto.png")


if __name__ == "__main__":
    main()
