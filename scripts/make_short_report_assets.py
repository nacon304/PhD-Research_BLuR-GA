#!/usr/bin/env python3
"""Create paper/report-ready CSV tables and figures for BLuR-GA comparisons.

This script is intentionally standalone.  It reads the existing aggregate files
created by `scripts/summarize_method_outputs.py` or `run_blur_ga.py aggregate`:

    <results-root>/comparison_summary/runs_long.csv
    <results-root>/comparison_summary/generation_traces_long.csv      optional
    <results-root>/comparison_summary/final_building_block_run_summary.csv optional

The goal is to generate compact assets for a short report:
1. Dataset profile table.
2. Performance / runtime / subset-size summary.
3. Baseline-delta heatmaps and trade-off plots.
4. EC-style convergence curves.
5. Linkage and building-block dynamics across generations.
6. Feature-selection and building-block stability summaries.

Examples
--------
Report 1: two proposed pairwise-lasso methods vs baseline

python scripts/make_short_report_assets.py \
  --results-root ../Results/results_final \
  --comparison-name report1_pairwise_lasso_vs_baseline \
  --baseline-method standard_ga \
  --include-methods standard_ga \
    blur_ga_pairwise_lasso_uniform_positive_binary_nopre \
    blur_ga_pairwise_lasso_uniform_positive_binary_pre

Report 2: two guided methods vs baseline and earlier pairwise-lasso methods

python scripts/make_short_report_assets.py \
  --results-root ../Results/results_fs_uniform_positive_small_extend \
  --comparison-name report2_guided_pair_vs_previous \
  --baseline-method standard_ga
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "standard_ga": "Standard GA",
    "empirical_linkage_legacy_uniform_positive": "GAwLL-BB uniform+",
    "blur_ga_pairwise_lasso_uniform_positive_binary_nopre": "PSLE-BB no-pre",
    "blur_ga_pairwise_lasso_uniform_positive_binary_pre": "PSLE-BB pre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_nopre": "PSLE-BB spin no-pre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_pre": "PSLE-BB spin pre",
    "linkage_guided_pair_uniform_signed_binary_nopre": "Guided pair no-pre",
    "linkage_guided_pair_uniform_signed_binary_pre": "Guided pair pre",
    "linkage_guided_pair_main_uniform_signed_binary_nopre": "Guided pair+main no-pre",
    "linkage_guided_pair_main_uniform_signed_binary_pre": "Guided pair+main pre",
}

METHOD_ORDER = [
    "standard_ga",
    "empirical_linkage_legacy_uniform_positive",
    "blur_ga_pairwise_lasso_uniform_positive_binary_nopre",
    "blur_ga_pairwise_lasso_uniform_positive_binary_pre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_nopre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_pre",
    "linkage_guided_pair_uniform_signed_binary_nopre",
    "linkage_guided_pair_uniform_signed_binary_pre",
    "linkage_guided_pair_main_uniform_signed_binary_nopre",
    "linkage_guided_pair_main_uniform_signed_binary_pre",
]

RUN_KEY = ["dataset", "repeat_id", "outer_fold"]
NUMERIC_RUN_COLS = [
    "outer_score", "best_inner_fitness", "subset_size", "n_edges",
    "generations", "runtime_seconds", "eval_count", "classifier_type",
    "ga_type", "repeat_id", "outer_fold", "run_id", "seed",
]
TRACE_METRICS = [
    "best_so_far_fitness", "mean_population_fitness", "best_so_far_subset_size",
    "mean_subset_size", "n_edges", "eval_count", "elapsed_seconds",
    "lr_first_fit_done", "bb_n_blocks", "bb_n_candidates", "bb_max_score",
    "bb_mean_block_size", "bb_positive_edges_in_blocks",
    "bb_negative_edges_in_blocks", "bb_mix_trials", "bb_mix_accepts",
    "bb_mix_accept_rate", "bb_mix_fitness_gain", "bb_mix_size_reduction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate short-report assets for BLuR-GA results.")
    p.add_argument("--results-root", required=True, help="Result root or comparison_summary directory.")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: <results-root>/report_assets/<comparison-name>.")
    p.add_argument("--comparison-name", default="short_report", help="Name used for output subfolder and report text.")
    p.add_argument("--baseline-method", default="standard_ga")
    p.add_argument("--include-methods", nargs="*", default=None, help="Optional method whitelist.")
    p.add_argument("--exclude-datasets", nargs="*", default=[])
    p.add_argument("--max-dataset-plots", type=int, default=8, help="Limit per-dataset convergence/detail plots.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-per-dataset-plots", action="store_true")
    p.add_argument("--skip-traces", action="store_true", help="Skip generation-trace reading and convergence/dynamics plots.")
    p.add_argument("--force-trace-rebuild", action="store_true", help="Re-read generation_traces_long.csv even if cached trace summaries already exist in the output table folder.")
    return p.parse_args()


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def clean_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def sort_methods(methods: Iterable[str]) -> list[str]:
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(set(methods), key=lambda m: (order.get(m, 999), m))


def resolve_summary_dir(root: Path) -> Path:
    if (root / "runs_long.csv").exists():
        return root
    summary = root / "comparison_summary"
    if (summary / "runs_long.csv").exists():
        return summary
    raise FileNotFoundError(f"Cannot find runs_long.csv under {root}")


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def parse_feature_set(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return set()
    return {int(x) for x in re.findall(r"\d+", s)}


def infer_n_features_from_runs(runs: pd.DataFrame, final_bb: pd.DataFrame) -> dict[str, float]:
    n_features: dict[str, float] = {}
    if not final_bb.empty and "n_features" in final_bb.columns:
        for ds, g in final_bb.groupby("dataset"):
            vals = pd.to_numeric(g["n_features"], errors="coerce").dropna()
            if not vals.empty:
                n_features[ds] = float(vals.max())
    for ds, g in runs.groupby("dataset"):
        if ds in n_features:
            continue
        lengths: list[int] = []
        if "best_chromosome" in g.columns:
            for value in g["best_chromosome"].dropna():
                s = str(value).strip()
                if set(s) <= {"0", "1"}:
                    lengths.append(len(s))
        if lengths:
            n_features[ds] = float(max(lengths))
            continue
        max_idx = 0
        if "best_selected_features" in g.columns:
            for value in g["best_selected_features"]:
                fs = parse_feature_set(value)
                if fs:
                    max_idx = max(max_idx, max(fs))
        n_features[ds] = float(max_idx) if max_idx > 0 else np.nan
    return n_features


def dimension_group(n_features: float) -> str:
    if not np.isfinite(n_features):
        return "unknown"
    if n_features >= 1000:
        return "high-dimensional"
    if n_features >= 100:
        return "medium-dimensional"
    return "low-dimensional"


def load_inputs(summary_dir: Path, include_methods: list[str] | None, exclude_datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(summary_dir / "runs_long.csv", dtype={"best_chromosome": str, "best_selected_features": str}, low_memory=False)
    runs = to_numeric(runs, NUMERIC_RUN_COLS)

    final_bb = read_csv_if_exists(summary_dir / "final_building_block_run_summary.csv", low_memory=False)
    final_bb = to_numeric(final_bb, [
        "repeat_id", "outer_fold", "run_id", "generation", "n_features",
        "n_graph_edges", "n_candidates", "n_qualified", "n_blocks",
        "selected_feature_coverage", "mean_block_size", "max_block_size",
        "mean_score", "max_score", "bb_mix_accept_rate", "bb_mix_fitness_gain",
    ])

    bb_long = read_csv_if_exists(summary_dir / "building_block_summary_long.csv", low_memory=False)
    bb_long = to_numeric(bb_long, [
        "repeat_id", "outer_fold", "run_id", "generation", "n_features",
        "n_graph_edges", "n_candidates", "n_qualified", "n_blocks",
        "selected_feature_coverage", "mean_block_size", "max_block_size",
        "mean_score", "max_score", "bb_mix_accept_rate", "bb_mix_fitness_gain",
    ])

    if include_methods:
        runs = runs[runs["method"].isin(include_methods)].copy()
        if not final_bb.empty:
            final_bb = final_bb[final_bb["method"].isin(include_methods)].copy()
        if not bb_long.empty:
            bb_long = bb_long[bb_long["method"].isin(include_methods)].copy()
    if exclude_datasets:
        runs = runs[~runs["dataset"].isin(exclude_datasets)].copy()
        if not final_bb.empty:
            final_bb = final_bb[~final_bb["dataset"].isin(exclude_datasets)].copy()
        if not bb_long.empty:
            bb_long = bb_long[~bb_long["dataset"].isin(exclude_datasets)].copy()

    n_features = infer_n_features_from_runs(runs, final_bb)
    runs["n_features"] = runs["dataset"].map(n_features)
    runs["subset_ratio"] = runs["subset_size"] / runs["n_features"]
    runs["method_label"] = runs["method"].map(method_label)
    return runs, final_bb, bb_long


def load_generation_traces(summary_dir: Path, include_methods: list[str] | None, exclude_datasets: list[str]) -> pd.DataFrame:
    path = summary_dir / "generation_traces_long.csv"
    if not path.exists():
        return pd.DataFrame()

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "run_id", "generation"] + TRACE_METRICS if c in header]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False):
        if include_methods:
            chunk = chunk[chunk["method"].isin(include_methods)]
        if exclude_datasets:
            chunk = chunk[~chunk["dataset"].isin(exclude_datasets)]
        if chunk.empty:
            continue
        numeric = [c for c in usecols if c not in {"dataset", "method"}]
        chunk = to_numeric(chunk, numeric)
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def dataset_profile(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in runs.groupby("dataset"):
        n_features = pd.to_numeric(g["n_features"], errors="coerce").max()
        rows.append({
            "dataset": ds,
            "dimension_group": dimension_group(n_features),
            "n_features": n_features,
            "n_methods": g["method"].nunique(),
            "n_runs_total": len(g),
            "runs_per_method_min": g.groupby("method").size().min(),
            "runs_per_method_max": g.groupby("method").size().max(),
            "baseline_outer_score_mean": pd.to_numeric(g.loc[g["method"].eq("standard_ga"), "outer_score"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows).sort_values(["dimension_group", "n_features", "dataset"])


def method_dataset_summary(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    agg = runs.groupby(["dataset", "method", "method_label"], as_index=False).agg(
        n_runs=("outer_score", "size"),
        outer_score_mean=("outer_score", "mean"),
        outer_score_std=("outer_score", "std"),
        subset_size_mean=("subset_size", "mean"),
        subset_size_std=("subset_size", "std"),
        subset_ratio_mean=("subset_ratio", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        runtime_seconds_std=("runtime_seconds", "std"),
        eval_count_mean=("eval_count", "mean"),
        n_edges_mean=("n_edges", "mean"),
        n_features=("n_features", "max"),
    )
    agg["score_rank"] = agg.groupby("dataset")["outer_score_mean"].rank(ascending=False, method="min")
    agg["runtime_rank"] = agg.groupby("dataset")["runtime_seconds_mean"].rank(ascending=True, method="min")
    agg["subset_rank"] = agg.groupby("dataset")["subset_size_mean"].rank(ascending=True, method="min")
    base = agg[agg["method"].eq(baseline)][["dataset", "outer_score_mean", "subset_size_mean", "runtime_seconds_mean", "eval_count_mean"]]
    base = base.rename(columns={
        "outer_score_mean": "baseline_outer_score_mean",
        "subset_size_mean": "baseline_subset_size_mean",
        "runtime_seconds_mean": "baseline_runtime_seconds_mean",
        "eval_count_mean": "baseline_eval_count_mean",
    })
    out = agg.merge(base, on="dataset", how="left")
    out["delta_score_vs_baseline"] = out["outer_score_mean"] - out["baseline_outer_score_mean"]
    out["delta_subset_vs_baseline"] = out["subset_size_mean"] - out["baseline_subset_size_mean"]
    out["runtime_ratio_vs_baseline"] = out["runtime_seconds_mean"] / out["baseline_runtime_seconds_mean"]
    out["eval_ratio_vs_baseline"] = out["eval_count_mean"] / out["baseline_eval_count_mean"]
    return out.sort_values(["dataset", "score_rank", "method"])


def overall_summary(dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in dm.groupby("method"):
        rows.append({
            "method": method,
            "method_label": method_label(method),
            "n_datasets": g["dataset"].nunique(),
            "avg_outer_score_mean": g["outer_score_mean"].mean(),
            "avg_delta_score_vs_baseline": g["delta_score_vs_baseline"].mean(),
            "avg_score_rank": g["score_rank"].mean(),
            "avg_subset_size_mean": g["subset_size_mean"].mean(),
            "avg_subset_ratio_mean": g["subset_ratio_mean"].mean(),
            "avg_runtime_seconds_mean": g["runtime_seconds_mean"].mean(),
            "avg_runtime_ratio_vs_baseline": g["runtime_ratio_vs_baseline"].replace([np.inf, -np.inf], np.nan).mean(),
            "avg_eval_count_mean": g["eval_count_mean"].mean(),
            "avg_n_edges_mean": g["n_edges_mean"].mean(),
            "n_best_score_datasets": int((g["score_rank"] == 1).sum()),
        })
    return pd.DataFrame(rows).sort_values(["avg_score_rank", "avg_delta_score_vs_baseline"], ascending=[True, False])


def paired_run_deltas(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    key = [c for c in RUN_KEY if c in runs.columns]
    metrics = ["outer_score", "subset_size", "runtime_seconds", "eval_count", "n_edges"]
    base = runs[runs["method"].eq(baseline)][key + metrics].rename(columns={m: f"baseline_{m}" for m in metrics})
    rows = []
    for method, g in runs[~runs["method"].eq(baseline)].groupby("method"):
        m = g[key + ["method", "method_label"] + metrics].merge(base, on=key, how="inner")
        if m.empty:
            continue
        for metric in metrics:
            m[f"delta_{metric}_vs_baseline"] = m[metric] - m[f"baseline_{metric}"]
            m[f"ratio_{metric}_vs_baseline"] = m[metric] / m[f"baseline_{metric}"]
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def feature_stability(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "best_selected_features" not in runs.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (dataset, method), g in runs.groupby(["dataset", "method"]):
        sets = [parse_feature_set(v) for v in g["best_selected_features"]]
        vals = []
        for a, b in itertools.combinations(sets, 2):
            union = len(a | b)
            vals.append(1.0 if union == 0 else len(a & b) / union)
        rows.append({
            "dataset": dataset,
            "method": method,
            "method_label": method_label(method),
            "n_runs": len(sets),
            "mean_selected_jaccard": float(np.mean(vals)) if vals else np.nan,
            "std_selected_jaccard": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "mean_subset_size": pd.to_numeric(g["subset_size"], errors="coerce").mean(),
        })
    by_ds = pd.DataFrame(rows)
    overall = by_ds.groupby(["method", "method_label"], as_index=False).agg(
        avg_selected_jaccard=("mean_selected_jaccard", "mean"),
        avg_subset_size=("mean_subset_size", "mean"),
        n_datasets=("dataset", "nunique"),
    ).sort_values("avg_selected_jaccard", ascending=False)
    return by_ds, overall


def building_block_stability(runs: pd.DataFrame, final_bb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if final_bb.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_cols = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "run_id"] if c in final_bb.columns]
    bb = final_bb.copy()
    metric_map = {
        "n_graph_edges": "final_graph_edges",
        "n_blocks": "final_bb_blocks",
        "mean_block_size": "final_mean_block_size",
        "max_block_size": "final_max_block_size",
        "mean_score": "final_mean_bb_score",
        "selected_feature_coverage": "final_selected_feature_coverage",
        "bb_mix_accept_rate": "final_bb_mix_accept_rate",
        "bb_mix_fitness_gain": "final_bb_mix_fitness_gain",
    }
    keep = key_cols + [c for c in metric_map if c in bb.columns]
    bb = bb[keep].rename(columns=metric_map)
    # Some summaries contain multiple rows per run. Keep the last generation-like row if present.
    if key_cols:
        bb = bb.drop_duplicates(subset=key_cols, keep="last")
    rows = []
    metrics = [c for c in bb.columns if c.startswith("final_")]
    for (dataset, method), g in bb.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "method_label": method_label(method), "n_runs": len(g)}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            mean = vals.mean()
            std = vals.std(ddof=1)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_cv"] = std / abs(mean) if mean and np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
        rows.append(row)
    by_ds = pd.DataFrame(rows)
    mean_cols = [c for c in by_ds.columns if c.endswith("_mean")]
    cv_cols = [c for c in by_ds.columns if c.endswith("_cv")]
    overall = by_ds.groupby(["method", "method_label"], as_index=False).agg(
        **{f"avg_{c}": (c, "mean") for c in mean_cols},
        **{f"avg_{c}": (c, "mean") for c in cv_cols},
        n_datasets=("dataset", "nunique"),
    )
    sort_col = "avg_final_bb_blocks_cv" if "avg_final_bb_blocks_cv" in overall.columns else "method"
    overall = overall.sort_values(sort_col, ascending=True)
    return by_ds, overall


def trace_generation_summary(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if traces.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = [c for c in TRACE_METRICS if c in traces.columns]
    by_ds = traces.groupby(["dataset", "method", "generation"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_runs=("best_so_far_fitness", "count") if "best_so_far_fitness" in metrics else ("generation", "count"),
    )
    overall = traces.groupby(["method", "generation"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_runs=("best_so_far_fitness", "count") if "best_so_far_fitness" in metrics else ("generation", "count"),
    )
    return by_ds, overall


def save_table_image(df: pd.DataFrame, out: Path, title: str, max_rows: int = 18, dpi: int = 200) -> None:
    if df.empty:
        return
    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}")
    fig_h = max(2.0, 0.38 * len(show) + 1.1)
    fig_w = min(18, max(8, 1.4 * len(show.columns)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=10)
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df: pd.DataFrame, value: str, label_col: str, title: str, xlabel: str, out: Path, ascending: bool, dpi: int) -> None:
    if df.empty or value not in df.columns:
        return
    d = df[[label_col, value]].dropna().sort_values(value, ascending=ascending)
    if d.empty:
        return
    fig_h = max(3.5, 0.45 * len(d) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    y = np.arange(len(d))
    vals = d[value].to_numpy(dtype=float)
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            text = f" {v:.4f}" if abs(v) < 10 else f" {v:.1f}"
            ax.text(v, i, text, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, label: str, title: str, xlabel: str, ylabel: str, out: Path, dpi: int) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    d = df[[x, y, label, "method"]].dropna()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.scatter(d[x], d[y], s=55)
    for _, r in d.iterrows():
        ax.annotate(str(r[label]), (r[x], r[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, title: str, out: Path, dpi: int, fmt: str = ".3f") -> None:
    if table.empty:
        return
    data = table.to_numpy(dtype=float)
    fig_w = max(7.5, 1.15 * len(table.columns))
    fig_h = max(4.0, 0.45 * len(table.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([method_label(c) for c in table.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    fig.colorbar(im, ax=ax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_generation_lines(summary: pd.DataFrame, metric_mean: str, metric_std: str | None, title: str, ylabel: str, out: Path, dpi: int) -> None:
    if summary.empty or metric_mean not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for method in sort_methods(summary["method"].unique()):
        g = summary[summary["method"].eq(method)].sort_values("generation")
        x = pd.to_numeric(g["generation"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[metric_mean], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, label=method_label(method), linewidth=2)
        if metric_std and metric_std in g.columns:
            std = pd.to_numeric(g[metric_std], errors="coerce").to_numpy(dtype=float)
            n = pd.to_numeric(g.get("n_runs", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(dtype=float)
            sem = std / np.sqrt(np.maximum(n, 1))
            if np.isfinite(sem).any():
                ax.fill_between(x, y - sem, y + sem, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path, dpi: int) -> None:
    if df.empty or metric not in df.columns:
        return
    methods = sort_methods(df["method"].unique())
    data = [pd.to_numeric(df.loc[df["method"].eq(m), metric], errors="coerce").dropna().to_numpy() for m in methods]
    if not any(len(x) for x in data):
        return
    fig, ax = plt.subplots(figsize=(max(8.8, 1.25 * len(methods)), 5.2))
    ax.boxplot(data, tick_labels=[method_label(m) for m in methods], showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def choose_dataset_plot_order(dm: pd.DataFrame, max_datasets: int) -> list[str]:
    if dm.empty:
        return []
    # Prefer datasets where methods differ most, since these are more informative for a short report.
    spread = dm.groupby("dataset")["outer_score_mean"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    return list(spread.head(max_datasets).index)


def write_recommendations(out: Path, comparison_name: str, profile: pd.DataFrame, overall: pd.DataFrame, traces_available: bool, bb_available: bool) -> None:
    methods = ", ".join(overall["method_label"].tolist()) if not overall.empty else "N/A"
    dataset_groups = profile["dimension_group"].value_counts().to_dict() if not profile.empty else {}
    best = overall.iloc[0]["method_label"] if not overall.empty else "N/A"
    lines = [
        f"# Short-report asset guide: {comparison_name}",
        "",
        "## Dataset block",
        "Use `tables/dataset_profile.csv` before the result section. It states which datasets are used, how many features they have, and whether the section is low-, medium-, or high-dimensional.",
        "",
        "## Core result figures",
        "1. `figures/delta_score_vs_baseline_heatmap.png`: clearest figure for showing where each proposed method improves or loses against Standard GA.",
        "2. `figures/score_vs_runtime.png`: separates accuracy gain from computational cost; useful because linkage/BB methods can be slower.",
        "3. `figures/score_vs_subset_size.png`: shows whether a score gain comes with more selected features.",
        "4. `figures/paired_delta_score_boxplot.png`: shows run-level consistency instead of only mean score.",
        "",
        "## EC-style convergence figures",
    ]
    if traces_available:
        lines += [
            "Use `figures/convergence_best_so_far_overall.png` as the main convergence plot. Use the per-dataset plots only for representative cases where methods differ visibly.",
            "Also inspect `figures/convergence_mean_population_overall.png` if you want to discuss population-level search pressure, not just best-so-far performance.",
        ]
    else:
        lines.append("No generation trace file was found, so convergence plots could not be generated.")
    lines += [
        "",
        "## Linkage / building-block figures",
    ]
    if bb_available:
        lines += [
            "Use `figures/linkage_edges_over_generation.png` to show when linkage information becomes available and how dense it is.",
            "Use `figures/bb_blocks_over_generation.png` and `figures/bb_mean_block_size_over_generation.png` to show whether each method creates many small BBs or fewer/larger BBs.",
            "Use `figures/bb_structural_stability.png` to discuss whether BB structure is stable across folds/runs. Lower coefficient of variation is more stable.",
        ]
    else:
        lines.append("No final/building-block summary was found, so structural BB plots could not be generated.")
    lines += [
        "",
        "## Stability figures",
        "Use `figures/feature_stability_jaccard.png` to show whether selected feature subsets are reproducible across runs/folds. Higher Jaccard means more stable selected subsets.",
        "",
        "## Auto notes from this run",
        f"- Methods included: {methods}.",
        f"- Dataset dimension groups: {dataset_groups}.",
        f"- Best average score-rank method in this generated summary: {best}.",
        "",
        "## Suggested short comment template",
        textwrap.dedent("""
        The proposed linkage/building-block variants improve mean performance on several datasets, but the gain should be read together with runtime and structural stability.  When BBs are built too early, too often, or from unstable linkage, they may increase computational cost or create noisy blocks.  Therefore, report the score delta, convergence behavior, number of edges/blocks, and feature/BB stability together rather than claiming improvement from accuracy alone.
        """).strip(),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_run_script_example(out: Path) -> None:
    out.write_text(textwrap.dedent("""\
    #!/usr/bin/env bash
    set -euo pipefail

    # Example usage for the two current result folders.
    python scripts/make_short_report_assets.py \
      --results-root ../Results/results_final \
      --comparison-name report1_pairwise_lasso_vs_baseline \
      --baseline-method standard_ga \
      --include-methods standard_ga \
        blur_ga_pairwise_lasso_uniform_positive_binary_nopre \
        blur_ga_pairwise_lasso_uniform_positive_binary_pre

    python scripts/make_short_report_assets.py \
      --results-root ../Results/results_fs_uniform_positive_small_extend \
      --comparison-name report2_guided_pair_vs_previous \
      --baseline-method standard_ga
    """), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_dir = resolve_summary_dir(Path(args.results_root))
    results_root = summary_dir.parent if summary_dir.name == "comparison_summary" else summary_dir
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "report_assets" / args.comparison_name
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs, final_bb, bb_long = load_inputs(summary_dir, args.include_methods, args.exclude_datasets)
    if runs.empty:
        raise ValueError("No runs left after filtering.")

    profile = dataset_profile(runs)
    dm = method_dataset_summary(runs, args.baseline_method)
    overall = overall_summary(dm)
    paired = paired_run_deltas(runs, args.baseline_method)
    feat_stability_ds, feat_stability_overall = feature_stability(runs)
    bb_stability_ds, bb_stability_overall = building_block_stability(runs, final_bb)

    trace_ds_path = table_dir / "generation_summary_by_dataset_method.csv"
    trace_overall_path = table_dir / "generation_summary_overall.csv"
    traces = pd.DataFrame()
    trace_rows_read: int | None = None
    if args.skip_traces:
        trace_ds, trace_overall = pd.DataFrame(), pd.DataFrame()
    elif (not args.force_trace_rebuild) and trace_ds_path.exists() and trace_overall_path.exists():
        trace_ds = pd.read_csv(trace_ds_path, low_memory=False)
        trace_overall = pd.read_csv(trace_overall_path, low_memory=False)
    else:
        traces = load_generation_traces(summary_dir, args.include_methods, args.exclude_datasets)
        trace_rows_read = len(traces)
        trace_ds, trace_overall = trace_generation_summary(traces)

    # Tables.
    profile.to_csv(table_dir / "dataset_profile.csv", index=False)
    dm.to_csv(table_dir / "method_dataset_summary.csv", index=False)
    overall.to_csv(table_dir / "method_overall_summary.csv", index=False)
    paired.to_csv(table_dir / "paired_run_deltas_vs_baseline.csv", index=False)
    feat_stability_ds.to_csv(table_dir / "feature_stability_by_dataset_method.csv", index=False)
    feat_stability_overall.to_csv(table_dir / "feature_stability_overall.csv", index=False)
    bb_stability_ds.to_csv(table_dir / "bb_structural_stability_by_dataset_method.csv", index=False)
    bb_stability_overall.to_csv(table_dir / "bb_structural_stability_overall.csv", index=False)
    if not trace_ds.empty:
        trace_ds.to_csv(table_dir / "generation_summary_by_dataset_method.csv", index=False)
    if not trace_overall.empty:
        trace_overall.to_csv(table_dir / "generation_summary_overall.csv", index=False)

    # Plot-ready pivots.
    score_pivot = dm.pivot(index="dataset", columns="method", values="outer_score_mean").reindex(columns=sort_methods(dm["method"].unique()))
    delta_pivot = dm[~dm["method"].eq(args.baseline_method)].pivot(index="dataset", columns="method", values="delta_score_vs_baseline").reindex(columns=[m for m in sort_methods(dm["method"].unique()) if m != args.baseline_method])
    rank_pivot = dm.pivot(index="dataset", columns="method", values="score_rank").reindex(columns=sort_methods(dm["method"].unique()))

    # Core figures.
    plot_heatmap(delta_pivot, "Mean score delta vs Standard GA", fig_dir / "delta_score_vs_baseline_heatmap.png", args.dpi, fmt="+.3f")
    plot_heatmap(score_pivot, "Mean outer score by dataset and method", fig_dir / "score_mean_heatmap.png", args.dpi, fmt=".3f")
    plot_heatmap(rank_pivot, "Score rank by dataset", fig_dir / "score_rank_heatmap.png", args.dpi, fmt=".0f")
    plot_bar(overall, "avg_delta_score_vs_baseline", "method_label", "Average score delta vs Standard GA", "Mean score delta", fig_dir / "avg_delta_score_vs_baseline.png", ascending=True, dpi=args.dpi)
    plot_bar(overall, "avg_score_rank", "method_label", "Average score rank across datasets", "Lower is better", fig_dir / "avg_score_rank.png", ascending=False, dpi=args.dpi)
    plot_bar(overall, "avg_runtime_seconds_mean", "method_label", "Average runtime", "Seconds", fig_dir / "avg_runtime_seconds.png", ascending=True, dpi=args.dpi)
    plot_bar(overall, "avg_subset_size_mean", "method_label", "Average selected subset size", "Number of selected features", fig_dir / "avg_subset_size.png", ascending=True, dpi=args.dpi)
    plot_scatter(overall, "avg_runtime_seconds_mean", "avg_outer_score_mean", "method_label", "Score-runtime trade-off", "Average runtime seconds", "Average outer score", fig_dir / "score_vs_runtime.png", dpi=args.dpi)
    plot_scatter(overall, "avg_subset_size_mean", "avg_outer_score_mean", "method_label", "Score-subset-size trade-off", "Average selected features", "Average outer score", fig_dir / "score_vs_subset_size.png", dpi=args.dpi)
    plot_scatter(overall, "avg_n_edges_mean", "avg_outer_score_mean", "method_label", "Score-linkage-density view", "Average number of linkage edges", "Average outer score", fig_dir / "score_vs_linkage_edges.png", dpi=args.dpi)
    plot_boxplot(paired, "delta_outer_score_vs_baseline", "Run-level score delta vs Standard GA", "Outer-score delta", fig_dir / "paired_delta_score_boxplot.png", dpi=args.dpi)
    plot_boxplot(paired, "delta_runtime_seconds_vs_baseline", "Run-level runtime delta vs Standard GA", "Runtime delta seconds", fig_dir / "paired_delta_runtime_boxplot.png", dpi=args.dpi)

    # Stability figures.
    if not feat_stability_overall.empty:
        plot_bar(feat_stability_overall, "avg_selected_jaccard", "method_label", "Feature-selection stability", "Mean Jaccard across runs/folds", fig_dir / "feature_stability_jaccard.png", ascending=True, dpi=args.dpi)
    if not bb_stability_overall.empty:
        cv_col = "avg_final_bb_blocks_cv"
        if cv_col in bb_stability_overall.columns:
            plot_bar(bb_stability_overall, cv_col, "method_label", "Building-block structural stability", "CV of final number of BBs; lower is more stable", fig_dir / "bb_structural_stability.png", ascending=False, dpi=args.dpi)
        mean_col = "avg_final_bb_blocks_mean"
        if mean_col in bb_stability_overall.columns:
            plot_bar(bb_stability_overall, mean_col, "method_label", "Average final building-block count", "Final BB count", fig_dir / "final_bb_count.png", ascending=True, dpi=args.dpi)

    # Convergence and BB dynamics.
    if not trace_overall.empty:
        plot_generation_lines(trace_overall, "best_so_far_fitness_mean", "best_so_far_fitness_std", "EC-style convergence: best-so-far fitness", "Best-so-far fitness", fig_dir / "convergence_best_so_far_overall.png", dpi=args.dpi)
        plot_generation_lines(trace_overall, "mean_population_fitness_mean", "mean_population_fitness_std", "Mean population fitness over generations", "Mean population fitness", fig_dir / "convergence_mean_population_overall.png", dpi=args.dpi)
        plot_generation_lines(trace_overall, "best_so_far_subset_size_mean", "best_so_far_subset_size_std", "Best-so-far subset size over generations", "Selected feature count", fig_dir / "convergence_subset_size_overall.png", dpi=args.dpi)
        plot_generation_lines(trace_overall, "n_edges_mean", "n_edges_std", "Linkage edges over generations", "Number of edges", fig_dir / "linkage_edges_over_generation.png", dpi=args.dpi)
        plot_generation_lines(trace_overall, "bb_n_blocks_mean", "bb_n_blocks_std", "Building blocks over generations", "Number of BBs", fig_dir / "bb_blocks_over_generation.png", dpi=args.dpi)
        plot_generation_lines(trace_overall, "bb_mean_block_size_mean", "bb_mean_block_size_std", "Mean BB size over generations", "Mean BB size", fig_dir / "bb_mean_block_size_over_generation.png", dpi=args.dpi)
        if "bb_mix_accept_rate_mean" in trace_overall.columns:
            plot_generation_lines(trace_overall, "bb_mix_accept_rate_mean", "bb_mix_accept_rate_std", "BB crossover accept rate over generations", "Accept rate", fig_dir / "bb_mix_accept_rate_over_generation.png", dpi=args.dpi)

        if not args.no_per_dataset_plots:
            for dataset in choose_dataset_plot_order(dm, args.max_dataset_plots):
                dsum = trace_ds[trace_ds["dataset"].eq(dataset)]
                if dsum.empty:
                    continue
                safe = clean_filename(dataset)
                plot_generation_lines(dsum, "best_so_far_fitness_mean", "best_so_far_fitness_std", f"Convergence on {dataset}", "Best-so-far fitness", fig_dir / f"convergence_{safe}.png", dpi=args.dpi)
                plot_generation_lines(dsum, "bb_n_blocks_mean", "bb_n_blocks_std", f"BB count on {dataset}", "Number of BBs", fig_dir / f"bb_blocks_{safe}.png", dpi=args.dpi)

    # Table snapshots as images for easy slide/report inclusion.
    save_table_image(profile[["dataset", "dimension_group", "n_features", "n_methods", "runs_per_method_min", "runs_per_method_max"]], fig_dir / "dataset_profile_table.png", "Dataset profile", dpi=args.dpi)
    display_cols = ["method_label", "n_datasets", "avg_outer_score_mean", "avg_delta_score_vs_baseline", "avg_score_rank", "avg_subset_size_mean", "avg_runtime_seconds_mean", "n_best_score_datasets"]
    display_cols = [c for c in display_cols if c in overall.columns]
    save_table_image(overall[display_cols], fig_dir / "overall_summary_table.png", "Overall summary", dpi=args.dpi)

    # Markdown guide.
    write_recommendations(out_dir / "README_short_report_assets.md", args.comparison_name, profile, overall, not trace_overall.empty, not final_bb.empty)
    write_run_script_example(out_dir / "run_examples.sh")

    print(f"Read summary: {summary_dir}")
    print(f"Wrote assets: {out_dir}")
    print(f"Datasets={runs['dataset'].nunique()} Methods={runs['method'].nunique()} Runs={len(runs)}")
    if not overall.empty:
        best = overall.iloc[0]
        print(f"Best avg rank: {best['method_label']} | avg score={best['avg_outer_score_mean']:.4f} | avg rank={best['avg_score_rank']:.2f}")
    if trace_rows_read is not None:
        print(f"Trace rows read={trace_rows_read}")
    elif not trace_overall.empty and not args.skip_traces:
        print("Trace summaries reused from output cache")


if __name__ == "__main__":
    main()
