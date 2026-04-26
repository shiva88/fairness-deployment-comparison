"""
analyze.py - aggregate the 240-row results CSV into:
  - results/summary_stats.csv  (mean ± std for every cell)
  - results/wilcoxon_tests.csv (paired tests vs baseline)
  - per-dataset formatted tables (results/table_{dataset}.csv)

Usage:
    python src/analyze.py
    python src/analyze.py --in results/full_results.csv --out-dir results/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="results/full_results.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.infile)
    # Dedupe in case of accidental duplicate rows
    df = df.drop_duplicates(subset=["dataset", "model", "method", "seed"], keep="last")

    print(f"Loaded {len(df)} rows from {args.infile}")
    counts = df.groupby(["dataset", "model", "method"]).size()
    print(f"Cells: {len(counts)} (expected 24)")
    if (counts != 10).any():
        print("WARNING: not all cells have 10 seeds")
        print(counts[counts != 10])

    # ====================================================================
    # 1. Summary statistics: mean ± std for every cell
    # ====================================================================
    metric_cols = [
        "accuracy", "f1", "spd", "eod", "disparate_impact",
        "train_time_sec", "single_pred_latency_ms", "batch_256_latency_ms",
    ]
    summary = df.groupby(["dataset", "model", "method"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "summary_stats.csv")
    print(f"Saved: {out_dir / 'summary_stats.csv'}")

    # ====================================================================
    # 2. Per-dataset formatted tables ("0.848 ± 0.003" cells)
    # ====================================================================
    for ds in ["adult", "compas", "german"]:
        sub = df[df["dataset"] == ds]
        agg = sub.groupby(["model", "method"])[metric_cols].agg(["mean", "std"])
        formatted = pd.DataFrame(index=agg.index)
        for col in metric_cols:
            formatted[col] = (
                agg[(col, "mean")].apply(lambda x: f"{x:.4f}")
                + " ± "
                + agg[(col, "std")].apply(lambda x: f"{x:.4f}")
            )
        out_path = out_dir / f"table_{ds}.csv"
        formatted.to_csv(out_path)
        print(f"Saved: {out_path}")

    # ====================================================================
    # 3. Paired Wilcoxon signed-rank tests (each method vs baseline)
    # ====================================================================
    results = []
    for ds in ["adult", "compas", "german"]:
        for mdl in ["lr", "rf"]:
            baseline = df[
                (df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == "baseline")
            ].sort_values("seed")
            for method in ["reweighing", "dir", "eqodds"]:
                other = df[
                    (df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == method)
                ].sort_values("seed")
                if not (baseline["seed"].values == other["seed"].values).all():
                    print(f"WARNING: seed mismatch for {ds}/{mdl}/{method}")
                    continue
                row = {"dataset": ds, "model": mdl, "method": method}
                for metric in ["accuracy", "spd", "eod", "disparate_impact"]:
                    b = baseline[metric].values
                    o = other[metric].values
                    if metric in ["spd", "eod"]:
                        b = np.abs(b)
                        o = np.abs(o)
                    if np.all(b == o):
                        row[f"{metric}_p"] = 1.0
                    else:
                        try:
                            _, p = wilcoxon(b, o)
                            row[f"{metric}_p"] = float(p)
                        except Exception as e:
                            row[f"{metric}_p"] = float("nan")
                results.append(row)

    wdf = pd.DataFrame(results)
    wdf.to_csv(out_dir / "wilcoxon_tests.csv", index=False)
    print(f"Saved: {out_dir / 'wilcoxon_tests.csv'}")

    # ====================================================================
    # 4. Print headline findings
    # ====================================================================
    print("\n" + "=" * 80)
    print("HEADLINE FINDINGS")
    print("=" * 80)

    print("\nMean |SPD| reduction vs baseline (each method, each cell):")
    for ds in ["adult", "compas", "german"]:
        for mdl in ["lr", "rf"]:
            b = df[(df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == "baseline")]
            b_spd = np.abs(b["spd"]).mean()
            line = f"  {ds:6s} {mdl:2s} baseline |SPD|={b_spd:.3f}"
            for method in ["reweighing", "dir", "eqodds"]:
                m = df[(df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == method)]
                m_spd = np.abs(m["spd"]).mean()
                pct = 100 * (m_spd - b_spd) / b_spd
                line += f" | {method[:4]} {m_spd:.3f} ({pct:+.0f}%)"
            print(line)

    print("\nMean batch-256 latency overhead (× baseline):")
    for ds in ["adult", "compas", "german"]:
        for mdl in ["lr", "rf"]:
            b = df[(df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == "baseline")]
            b_lat = b["batch_256_latency_ms"].mean()
            line = f"  {ds:6s} {mdl:2s}"
            for method in ["reweighing", "dir", "eqodds"]:
                m = df[(df["dataset"] == ds) & (df["model"] == mdl) & (df["method"] == method)]
                m_lat = m["batch_256_latency_ms"].mean()
                line += f" | {method[:4]} {m_lat / b_lat:6.2f}x"
            print(line)


if __name__ == "__main__":
    main()
