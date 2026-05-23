#!/usr/bin/env python3
from __future__ import annotations

"""Plot and analyse BLuR-GA linkage-correctness results.

This script is intentionally restricted to eval_scope == "all_returned".
It consumes two aggregate CSV files produced by run_linkage_eval.py aggregate:

    linkage_eval_summary.csv      # run-level rows
    linkage_eval_by_method.csv    # dataset-method average rows

Example on Windows PowerShell
-----------------------------
python plot_linkage_all_returned_analysis.py `
  --summary "linkage_eval_summary(1).csv" `
  --by-method "linkage_eval_by_method(2).csv" `
  --outdir linkage_figures_all_returned

Outputs
-------
The script creates PNG/PDF figures and analysis CSVs under --outdir.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METHOD_ORDER = [
    "empirical_linkage_legacy",
    "blur_ga_main_pairwise_lasso",
    "blur_ga_pairwise_lasso",
    "top_cooccurrence",
    "random_graph",
]

METHOD_LABELS = {
    "empirical_linkage_legacy": "Empirical linkage",
    "blur_ga_main_pairwise_lasso": "Main+pairwise Lasso",
    "blur_ga_pairwise_lasso": "Pairwise Lasso",
    "top_cooccurrence": "Top co-occurrence",
    "random_graph": "Random graph",
}

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "average_precision": "Average precision",
    "spearman_abs_weight": "Spearman |weight|",
    "sign_accuracy": "Sign accuracy",
    "n_pred_edges": "Returned edges",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "true_positives": "True positives",
}

MISSING_COLUMNS = ["dataset", "family", "method", "method_label"]


def safe_read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV safely.

    This is mainly used for optional diagnostic files that may legitimately
    have zero rows. If such a file is empty/0-byte, return an empty DataFrame
    with the expected columns instead of raising pandas.errors.EmptyDataError.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def infer_family(dataset: str) -> str:
    """Infer a benchmark-family label from the dataset/file name.

    The previous version hard-coded only a few large benchmark names, so
    debug datasets such as ``ising_spin_glass_grid5x5_*``,
    ``maxsat_d35_*``, ``nk_landscape_d30_*``, and
    ``pairwise_qubo_d30_*`` were all plotted as ``Other``.
    This parser reads the family directly from the filename pattern and also
    keeps the most important size parameters in the label.
    """
    name = str(dataset)

    m = re.match(r"ising_spin_glass_grid(\d+)x(\d+)", name)
    if m:
        return f"Ising {m.group(1)}x{m.group(2)}"

    m = re.match(r"maxsat_d(\d+)_m(\d+)_k(\d+)", name)
    if m:
        return f"MaxSAT d{m.group(1)} m{m.group(2)} k{m.group(3)}"

    m = re.match(r"nk_landscape_d(\d+)_k(\d+)", name)
    if m:
        return f"NK d{m.group(1)} k{m.group(2)}"

    m = re.match(r"pairwise_qubo_d(\d+)_e(\d+)", name)
    if m:
        return f"QUBO d{m.group(1)} e{m.group(2)}"

    m = re.match(r"planted_xor_d(\d+)_n(\d+)_g([^_]+)", name)
    if m:
        return f"Planted XOR d{m.group(1)} g{m.group(3)}"

    m = re.match(r"gametes_style_d(\d+)_n(\d+)_loci(\d+)", name)
    if m:
        return f"GAMETES d{m.group(1)} loci{m.group(3)}"

    return "Other"


def infer_seed(dataset: str) -> str:
    m = re.search(r"seed(\d+)$", dataset)
    return m.group(1) if m else ""


def shorten_dataset(dataset: str) -> str:
    """Short labels for plots; keeps family + seed information."""
    family = infer_family(dataset)
    seed = infer_seed(dataset)
    return f"{family} s{seed}" if seed else family


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def ordered_methods(methods: Iterable[str]) -> list[str]:
    methods = list(methods)
    known = [m for m in METHOD_ORDER if m in methods]
    unknown = sorted(m for m in methods if m not in METHOD_ORDER)
    return known + unknown


def save_current_figure(outdir: Path, name: str, dpi: int = 300) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def assert_columns(df: pd.DataFrame, required: Iterable[str], file_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def load_and_prepare(summary_path: Path, by_method_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    by_method = pd.read_csv(by_method_path)

    required_common = [
        "dataset", "method", "eval_scope", "precision", "recall", "f1",
        "average_precision", "n_pred_edges", "true_positives",
        "false_positives", "false_negatives",
    ]
    assert_columns(summary, required_common, "summary CSV")
    assert_columns(by_method, required_common, "by-method CSV")

    summary = summary.loc[summary["eval_scope"].eq("all_returned")].copy()
    by_method = by_method.loc[by_method["eval_scope"].eq("all_returned")].copy()

    if summary.empty:
        raise ValueError("No rows with eval_scope == 'all_returned' in summary CSV.")
    if by_method.empty:
        raise ValueError("No rows with eval_scope == 'all_returned' in by-method CSV.")

    for df in (summary, by_method):
        df["family"] = df["dataset"].map(infer_family)
        df["dataset_short"] = df["dataset"].map(shorten_dataset)
        df["method_label"] = df["method"].map(method_label)
        # Edge-volume diagnostics. Recall is strongly affected by how many edges a method returns.
        df["fp_per_tp"] = np.where(df["true_positives"] > 0,
                                   df["false_positives"] / df["true_positives"],
                                   np.nan)
        df["fn_per_true"] = np.where(
            (df["true_positives"] + df["false_negatives"]) > 0,
            df["false_negatives"] / (df["true_positives"] + df["false_negatives"]),
            np.nan,
        )

    return summary, by_method


# ---------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------

def write_analysis_tables(summary: pd.DataFrame, by_method: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    methods = ordered_methods(by_method["method"].unique())
    datasets = sorted(by_method["dataset"].unique())

    dataset_family = (
        by_method[["dataset", "family", "dataset_short"]]
        .drop_duplicates()
        .sort_values(["family", "dataset"])
    )
    dataset_family.to_csv(outdir / "dataset_family_mapping.csv", index=False)

    # Missing dataset-method combinations are important when some runs failed.
    missing_rows = []
    for dataset in datasets:
        present = set(by_method.loc[by_method["dataset"].eq(dataset), "method"])
        for method in methods:
            if method not in present:
                missing_rows.append({
                    "dataset": dataset,
                    "family": infer_family(dataset),
                    "method": method,
                    "method_label": method_label(method),
                })
    pd.DataFrame(missing_rows, columns=MISSING_COLUMNS).to_csv(outdir / "missing_dataset_method_combinations.csv", index=False)

    overall = (
        by_method.groupby("method", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            average_precision_mean=("average_precision", "mean"),
            spearman_abs_weight_mean=("spearman_abs_weight", "mean") if "spearman_abs_weight" in by_method.columns else ("f1", "mean"),
            sign_accuracy_mean=("sign_accuracy", "mean") if "sign_accuracy" in by_method.columns else ("f1", "mean"),
            n_pred_edges_mean=("n_pred_edges", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            false_negatives_mean=("false_negatives", "mean"),
        )
    )
    overall["method_label"] = overall["method"].map(method_label)
    overall = overall.sort_values("f1_mean", ascending=False)
    overall.to_csv(outdir / "overall_by_method_all_returned.csv", index=False)

    family = (
        by_method.groupby(["family", "method"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            average_precision_mean=("average_precision", "mean"),
            n_pred_edges_mean=("n_pred_edges", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            false_negatives_mean=("false_negatives", "mean"),
        )
    )
    family["method_label"] = family["method"].map(method_label)
    family.to_csv(outdir / "family_by_method_all_returned.csv", index=False)

    # Best method per dataset by F1.
    best_idx = by_method.groupby("dataset")["f1"].idxmax()
    best = by_method.loc[best_idx].sort_values(["family", "dataset"])
    best.to_csv(outdir / "best_method_per_dataset_by_f1_all_returned.csv", index=False)

    # Run-level variability from summary.
    run_level = (
        summary.groupby(["family", "dataset", "method", "repeat_id", "outer_fold"], as_index=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            average_precision=("average_precision", "mean"),
            n_pred_edges=("n_pred_edges", "mean"),
        )
        if {"repeat_id", "outer_fold"}.issubset(summary.columns)
        else summary.copy()
    )
    run_level["method_label"] = run_level["method"].map(method_label)
    run_level.to_csv(outdir / "run_level_all_returned.csv", index=False)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_heatmap_metric(by_method: pd.DataFrame, outdir: Path, metric: str = "f1") -> None:
    methods = ordered_methods(by_method["method"].unique())
    method_labels = [method_label(m) for m in methods]

    dataset_order = (
        by_method[["dataset", "family", "dataset_short"]]
        .drop_duplicates()
        .sort_values(["family", "dataset"])
    )
    datasets = dataset_order["dataset"].tolist()
    ylabels = dataset_order["dataset_short"].tolist()

    pivot = by_method.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    mat = pivot.reindex(index=datasets, columns=methods).to_numpy(dtype=float)

    fig_height = max(6, 0.42 * len(datasets))
    fig_width = max(8, 1.8 * len(methods))
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label=METRIC_LABELS.get(metric, metric))
    plt.xticks(np.arange(len(methods)), method_labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(datasets)), ylabels)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} by dataset and method (all_returned)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text = "NA" if np.isnan(value) else f"{value:.2f}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    save_current_figure(outdir, f"heatmap_{metric}_by_dataset_method")


def plot_family_grouped_bars(by_method: pd.DataFrame, outdir: Path, metric: str = "f1") -> None:
    family = (
        by_method.groupby(["family", "method"], as_index=False)[metric]
        .mean()
    )
    families = sorted(family["family"].unique())
    methods = ordered_methods(family["method"].unique())

    x = np.arange(len(families))
    width = 0.8 / max(1, len(methods))

    plt.figure(figsize=(max(10, len(families) * 1.4), 5.5))
    for idx, method in enumerate(methods):
        vals = []
        for fam in families:
            row = family.loc[family["family"].eq(fam) & family["method"].eq(method), metric]
            vals.append(float(row.iloc[0]) if not row.empty else np.nan)
        plt.bar(x + (idx - (len(methods) - 1) / 2) * width, vals, width, label=method_label(method))

    plt.xticks(x, families, rotation=25, ha="right")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"Mean {METRIC_LABELS.get(metric, metric)} by benchmark family (all_returned)")
    plt.ylim(0, 1.05 if metric in {"precision", "recall", "f1", "average_precision"} else None)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir, f"family_grouped_bar_{metric}")


def plot_overall_method_bars(by_method: pd.DataFrame, outdir: Path) -> None:
    metrics = ["precision", "recall", "f1", "average_precision"]
    methods = ordered_methods(by_method["method"].unique())
    labels = [method_label(m) for m in methods]
    data = by_method.groupby("method")[metrics].mean().reindex(methods)

    x = np.arange(len(methods))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(11, 5.5))
    for idx, metric in enumerate(metrics):
        plt.bar(x + (idx - (len(metrics) - 1) / 2) * width, data[metric].to_numpy(), width, label=METRIC_LABELS[metric])

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean score")
    plt.ylim(0, 1.05)
    plt.title("Overall method comparison (all_returned only)")
    plt.legend(ncol=2)
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir, "overall_precision_recall_f1_ap_by_method")


def plot_precision_recall_scatter(by_method: pd.DataFrame, outdir: Path) -> None:
    methods = ordered_methods(by_method["method"].unique())

    plt.figure(figsize=(7.5, 6.5))
    for method in methods:
        sub = by_method.loc[by_method["method"].eq(method)]
        sizes = 35 + 90 * (sub["n_pred_edges"] / max(1.0, by_method["n_pred_edges"].max()))
        plt.scatter(sub["recall"], sub["precision"], s=sizes, alpha=0.75, label=method_label(method))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision--recall trade-off by dataset (all_returned)")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_current_figure(outdir, "precision_recall_scatter_all_returned")


def plot_edge_volume_tradeoff(by_method: pd.DataFrame, outdir: Path) -> None:
    methods = ordered_methods(by_method["method"].unique())

    plt.figure(figsize=(8, 5.8))
    for method in methods:
        sub = by_method.loc[by_method["method"].eq(method)]
        plt.scatter(sub["n_pred_edges"], sub["f1"], alpha=0.75, label=method_label(method))

    plt.xlabel("Number of returned edges")
    plt.ylabel("F1")
    plt.title("Returned-edge volume vs F1 (all_returned)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_current_figure(outdir, "returned_edges_vs_f1_all_returned")


def plot_error_decomposition(by_method: pd.DataFrame, outdir: Path) -> None:
    methods = ordered_methods(by_method["method"].unique())
    data = (
        by_method.groupby("method")[["true_positives", "false_positives", "false_negatives"]]
        .mean()
        .reindex(methods)
    )
    labels = [method_label(m) for m in methods]
    x = np.arange(len(methods))

    plt.figure(figsize=(10, 5.8))
    bottom = np.zeros(len(methods))
    for col in ["true_positives", "false_positives", "false_negatives"]:
        vals = data[col].to_numpy()
        plt.bar(x, vals, bottom=bottom, label=METRIC_LABELS[col])
        bottom += np.nan_to_num(vals)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean number of edges")
    plt.title("Mean TP / FP / FN decomposition (all_returned)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir, "tp_fp_fn_decomposition_by_method")


def plot_run_level_boxplots(summary: pd.DataFrame, outdir: Path, metric: str = "f1") -> None:
    methods = ordered_methods(summary["method"].unique())
    data = [summary.loc[summary["method"].eq(method), metric].dropna().to_numpy() for method in methods]
    labels = [method_label(m) for m in methods]

    plt.figure(figsize=(10, 5.8))
    
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"Run-level distribution of {METRIC_LABELS.get(metric, metric)} (all_returned)")
    if metric in {"precision", "recall", "f1", "average_precision"}:
        plt.ylim(-0.02, 1.05)
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir, f"run_level_boxplot_{metric}")


def plot_best_method_counts(by_method: pd.DataFrame, outdir: Path) -> None:
    idx = by_method.groupby("dataset")["f1"].idxmax()
    best = by_method.loc[idx]
    counts = best["method"].value_counts().reindex(ordered_methods(by_method["method"].unique()), fill_value=0)

    labels = [method_label(m) for m in counts.index]
    plt.figure(figsize=(8.5, 5))
    plt.bar(np.arange(len(counts)), counts.to_numpy())
    plt.xticks(np.arange(len(counts)), labels, rotation=30, ha="right")
    plt.ylabel("Number of datasets")
    plt.title("How often each method is best by F1 (all_returned)")
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir, "best_method_count_by_f1")


# ---------------------------------------------------------------------
# Textual report
# ---------------------------------------------------------------------

def write_markdown_report(by_method: pd.DataFrame, outdir: Path) -> None:
    overall = (
        by_method.groupby("method")[["precision", "recall", "f1", "average_precision", "n_pred_edges", "false_positives", "false_negatives"]]
        .mean()
        .reindex(ordered_methods(by_method["method"].unique()))
    )

    family = (
        by_method.groupby(["family", "method"])[["precision", "recall", "f1", "average_precision", "n_pred_edges", "false_positives", "false_negatives"]]
        .mean()
        .reset_index()
    )

    missing_path = outdir / "missing_dataset_method_combinations.csv"
    missing = safe_read_csv(missing_path, columns=MISSING_COLUMNS)

    lines = []
    lines.append("# Linkage-correctness analysis: all_returned only\n")
    lines.append("## Overall mean by method\n")
    lines.append(overall.round(4).to_markdown())
    lines.append("\n\n## Mean by benchmark family and method\n")
    lines.append(family.round(4).to_markdown(index=False))

    lines.append("\n\n## Important interpretation notes\n")
    lines.append("- `all_returned` evaluates every edge returned by a method, not only the top-k edges.")
    lines.append("- A method can have perfect or high recall but poor precision if it returns too many edges.")
    lines.append("- A method can have perfect precision but moderate recall if it is conservative and returns only a subset of true edges.")
    lines.append("- F1 is therefore the main balance metric for all_returned evaluation.")

    if not missing.empty:
        lines.append("\n\n## Missing dataset-method combinations\n")
        lines.append("Some method/dataset combinations are absent. Do not over-interpret the overall average until these jobs are rerun.")
        lines.append(missing.to_markdown(index=False))

    (outdir / "analysis_report_all_returned.md").write_text("\n".join(lines), encoding="utf-8")


def make_all_outputs(summary: pd.DataFrame, by_method: pd.DataFrame, outdir: Path) -> None:
    write_analysis_tables(summary, by_method, outdir)

    for metric in ["f1", "precision", "recall", "average_precision"]:
        plot_heatmap_metric(by_method, outdir, metric=metric)
        plot_family_grouped_bars(by_method, outdir, metric=metric)

    plot_overall_method_bars(by_method, outdir)
    plot_precision_recall_scatter(by_method, outdir)
    plot_edge_volume_tradeoff(by_method, outdir)
    plot_error_decomposition(by_method, outdir)
    plot_run_level_boxplots(summary, outdir, metric="f1")
    plot_run_level_boxplots(summary, outdir, metric="average_precision")
    plot_best_method_counts(by_method, outdir)
    write_markdown_report(by_method, outdir)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot linkage-correctness results using all_returned rows only.")
    parser.add_argument("--summary", required=True, type=Path, help="Path to linkage_eval_summary.csv")
    parser.add_argument("--by-method", required=True, type=Path, help="Path to linkage_eval_by_method.csv")
    parser.add_argument("--outdir", default=Path("linkage_figures_all_returned"), type=Path, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, by_method = load_and_prepare(args.summary, args.by_method)
    make_all_outputs(summary, by_method, args.outdir)

    print(f"Loaded run-level rows after all_returned filter: {len(summary)}")
    print(f"Loaded dataset-method rows after all_returned filter: {len(by_method)}")
    print(f"Datasets: {by_method['dataset'].nunique()}")
    print(f"Methods: {', '.join(ordered_methods(by_method['method'].unique()))}")
    print(f"Outputs written to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python analysis_interactive/analyze_linkage_correctness/plot_linkage_all_returned_analysis.py `
#   --summary "..\Results\results_linkage_test_theory_v3\linkage_eval_summary.csv" `
#   --by-method "..\Results\results_linkage_test_theory_v3\linkage_eval_by_method.csv" `
#   --outdir "..\Results\results_linkage_test_theory_v3\linkage_figures_all_returned"