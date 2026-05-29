#!/usr/bin/env python3
"""Summarize BLuR-GA experiment outputs for method comparison.

This script is intentionally independent from the training runner.  It can read
both output layouts used in the project:

1. Full evaluation root:
       results_root/<dataset>/<method>/nested_summary_c2_aX.csv
       results_root/<dataset>/<method>/rep00/fold00/run000/run_summary.csv

2. One dataset folder:
       dataset_dir/<method>/nested_summary_c2_aX.csv
       dataset_dir/<method>/rep00/fold00/run000/run_summary.csv

It writes paper-friendly CSV tables for:
- all runs in long format
- per-dataset method comparisons
- overall method comparisons across datasets
- pivot tables by score / subset size / runtime / rank
- optional pairwise comparison against a baseline method
- optional convergence, selected-feature, and building-block summaries

Example, full evaluation root:
    python scripts/summarize_method_outputs.py `
        --input ../Results/results_fs_uniform_positive_small_tabular `
        --mode all `
        --baseline-method standard_ga `
        --output-dir ../Results/results_fs_uniform_positive_small_tabular/comparison_summary

Example, one dataset folder:
    python scripts/summarize_method_outputs.py `
        --input "../Results/results_analysis_small_tab/vehicle_uci" `
        --mode dataset `
        --baseline-method standard_ga

Example, root input but only one dataset:
    python scripts/summarize_method_outputs.py `
        --input ../Results/results_fs_main `
        --dataset parkinsons_uci
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RUN_ID_COLS = ["dataset", "method", "classifier_type", "ga_type", "repeat_id", "outer_fold", "run_id"]
PAIR_ID_COLS = ["dataset", "classifier_type", "repeat_id", "outer_fold"]
DEFAULT_METRIC_COLS = [
    "outer_score",
    "best_inner_fitness",
    "subset_size",
    "runtime_seconds",
    "eval_count",
    "n_edges",
    "generations",
]


@dataclass(frozen=True)
class MethodDir:
    dataset: str
    method: str
    path: Path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:  # keep batch summary robust
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def _as_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _dedupe_cols(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for col in cols:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def _looks_like_method_dir(path: Path) -> bool:
    if any(path.glob("nested_summary*.csv")):
        return True
    if any(path.glob("run_summary*.csv")):
        return True
    if any(path.glob("rep*/fold*/run*/run_summary.csv")):
        return True
    return False


def _looks_like_dataset_dir(path: Path) -> bool:
    return any(child.is_dir() and _looks_like_method_dir(child) for child in path.iterdir() if child.is_dir())


def _detect_mode(input_path: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if _looks_like_dataset_dir(input_path):
        return "dataset"
    return "all"


def _discover_method_dirs(input_path: Path, mode: str, dataset_filter: str | None) -> list[MethodDir]:
    input_path = input_path.resolve()
    method_dirs: list[MethodDir] = []

    if _looks_like_method_dir(input_path):
        dataset = input_path.parent.name
        method = input_path.name
        method_dirs.append(MethodDir(dataset=dataset, method=method, path=input_path))
    elif mode == "dataset":
        dataset = input_path.name
        for child in sorted(input_path.iterdir()):
            if child.is_dir() and _looks_like_method_dir(child):
                method_dirs.append(MethodDir(dataset=dataset, method=child.name, path=child))
    else:  # all-root mode
        for dataset_dir in sorted(input_path.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name in {"comparison_summary", "summary", "plots", "figures"}:
                continue
            if dataset_filter and dataset_dir.name != dataset_filter:
                continue
            for method_dir in sorted(dataset_dir.iterdir()):
                if method_dir.is_dir() and _looks_like_method_dir(method_dir):
                    method_dirs.append(MethodDir(dataset=dataset_dir.name, method=method_dir.name, path=method_dir))

    if dataset_filter:
        method_dirs = [m for m in method_dirs if m.dataset == dataset_filter]
    return method_dirs


def _run_summary_paths(method_dir: Path) -> list[Path]:
    """Prefer aggregate nested_summary, otherwise use run_summary files.

    Avoid double counting if both compact aggregate files and per-run files are present.
    """
    nested = sorted(method_dir.glob("nested_summary*.csv"))
    if nested:
        return nested

    # Root-level per-run files created by artifact_layout=root/both.
    root_run = sorted(p for p in method_dir.glob("run_summary*.csv") if p.name != "run_summary.csv")
    if root_run:
        return root_run

    # HPC-safe nested files.
    nested_run = sorted(method_dir.glob("rep*/fold*/run*/run_summary.csv"))
    if nested_run:
        return nested_run

    # Last fallback.
    return sorted(method_dir.rglob("run_summary*.csv"))


def _load_run_summaries(method_dirs: list[MethodDir]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for info in method_dirs:
        paths = _run_summary_paths(info.path)
        if not paths:
            print(f"[WARN] No run summary found in {info.path}", file=sys.stderr)
            continue
        for path in paths:
            df = _safe_read_csv(path)
            if df.empty:
                continue
            df.insert(0, "method", info.method)
            if "dataset" not in df.columns:
                df.insert(0, "dataset", info.dataset)
            else:
                df["dataset"] = df["dataset"].fillna(info.dataset).astype(str)
                # Aggregate files usually already contain this, but force consistency with directory.
                df.loc[df["dataset"].eq("") | df["dataset"].isna(), "dataset"] = info.dataset
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True, sort=False)
    numeric_cols = DEFAULT_METRIC_COLS + ["classifier_type", "ga_type", "repeat_id", "outer_fold", "run_id", "seed"]
    runs = _as_numeric(runs, numeric_cols)
    # Some legacy files use duplicate aggregate + per-run. Remove exact run duplicates, prefer rows with chromosomes.
    subset_cols = [c for c in RUN_ID_COLS if c in runs.columns]
    if subset_cols:
        runs["_has_chromosome"] = runs.get("best_chromosome", pd.Series([""] * len(runs))).notna().astype(int)
        runs = runs.sort_values(subset_cols + ["_has_chromosome", "source_file"])
        runs = runs.drop_duplicates(subset=subset_cols, keep="last").drop(columns=["_has_chromosome"], errors="ignore")
    return runs.reset_index(drop=True)


def _summary_stats(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg: dict[str, list[str]] = {}
    for col in metric_cols:
        if col in df.columns:
            agg[col] = ["mean", "std", "min", "max", "median"]
    if not agg:
        return pd.DataFrame()
    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out = out.reset_index()
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="n_runs")
    out = counts.merge(out, on=group_cols, how="left")
    for col in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        sem_col = f"{col}_sem"
        if mean_col in out.columns and std_col in out.columns:
            out[sem_col] = out[std_col] / np.sqrt(out["n_runs"].clip(lower=1))
    return out


def _add_dataset_ranks(summary: pd.DataFrame, score_col: str, higher_is_better: bool) -> pd.DataFrame:
    out = summary.copy()
    score_mean = f"{score_col}_mean"
    if score_mean in out.columns:
        out["score_rank_in_dataset"] = out.groupby("dataset")[score_mean].rank(
            method="min", ascending=not higher_is_better
        )
    if "subset_size_mean" in out.columns:
        out["subset_size_rank_in_dataset"] = out.groupby("dataset")["subset_size_mean"].rank(
            method="min", ascending=True
        )
    if "runtime_seconds_mean" in out.columns:
        out["runtime_rank_in_dataset"] = out.groupby("dataset")["runtime_seconds_mean"].rank(
            method="min", ascending=True
        )
    return out


def _method_overall_summary(method_dataset_summary: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if method_dataset_summary.empty:
        return pd.DataFrame()
    metric_cols = []
    for col in [
        f"{score_col}_mean",
        "subset_size_mean",
        "runtime_seconds_mean",
        "eval_count_mean",
        "n_edges_mean",
        "generations_mean",
        "score_rank_in_dataset",
        "subset_size_rank_in_dataset",
        "runtime_rank_in_dataset",
    ]:
        if col in method_dataset_summary.columns:
            metric_cols.append(col)
    if not metric_cols:
        return pd.DataFrame()
    overall = method_dataset_summary.groupby("method", dropna=False).agg(
        n_datasets=("dataset", "nunique"),
        n_dataset_method_rows=("dataset", "size"),
        **{f"{col}_avg_over_datasets": (col, "mean") for col in metric_cols},
        **{f"{col}_std_over_datasets": (col, "std") for col in metric_cols},
    ).reset_index()
    sort_col = f"score_rank_in_dataset_avg_over_datasets"
    if sort_col in overall.columns:
        overall = overall.sort_values([sort_col, f"{score_col}_mean_avg_over_datasets"], ascending=[True, False])
    return overall


def _write_pivots(outdir: Path, method_dataset_summary: pd.DataFrame, score_col: str) -> None:
    if method_dataset_summary.empty:
        return
    pivot_specs = [
        (f"{score_col}_mean", "pivot_score_mean.csv"),
        (f"{score_col}_std", "pivot_score_std.csv"),
        ("subset_size_mean", "pivot_subset_size_mean.csv"),
        ("runtime_seconds_mean", "pivot_runtime_seconds_mean.csv"),
        ("eval_count_mean", "pivot_eval_count_mean.csv"),
        ("n_edges_mean", "pivot_n_edges_mean.csv"),
        ("score_rank_in_dataset", "pivot_score_rank.csv"),
    ]
    for value_col, filename in pivot_specs:
        if value_col not in method_dataset_summary.columns:
            continue
        pivot = method_dataset_summary.pivot_table(
            index="dataset", columns="method", values=value_col, aggfunc="first"
        ).reset_index()
        pivot.to_csv(outdir / filename, index=False)


def _paired_baseline_comparison(runs: pd.DataFrame, baseline_method: str, score_col: str, higher_is_better: bool) -> pd.DataFrame:
    if runs.empty or baseline_method not in set(runs.get("method", [])):
        return pd.DataFrame()
    required = [c for c in PAIR_ID_COLS if c in runs.columns]
    if len(required) < 3 or score_col not in runs.columns:
        return pd.DataFrame()

    metric_cols = [c for c in [score_col, "subset_size", "runtime_seconds", "eval_count", "n_edges", "generations"] if c in runs.columns]
    base = runs[runs["method"] == baseline_method][required + metric_cols].copy()
    base = base.rename(columns={c: f"baseline_{c}" for c in metric_cols})
    others = runs[runs["method"] != baseline_method][required + ["method"] + metric_cols].copy()
    merged = others.merge(base, on=required, how="inner")
    if merged.empty:
        return pd.DataFrame()
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"baseline_{col}"]
    if not higher_is_better and f"delta_{score_col}" in merged.columns:
        merged[f"improvement_{score_col}"] = -merged[f"delta_{score_col}"]
    elif f"delta_{score_col}" in merged.columns:
        merged[f"improvement_{score_col}"] = merged[f"delta_{score_col}"]
    if "subset_size" in metric_cols:
        merged["feature_reduction_vs_baseline"] = merged["baseline_subset_size"] - merged["subset_size"]
        merged["feature_reduction_pct_vs_baseline"] = np.where(
            merged["baseline_subset_size"].abs() > 0,
            merged["feature_reduction_vs_baseline"] / merged["baseline_subset_size"].abs() * 100.0,
            np.nan,
        )

    group_cols = ["dataset", "method"]
    agg_cols = [c for c in merged.columns if c.startswith("delta_") or c.startswith("improvement_") or c.startswith("feature_reduction")]
    out = merged.groupby(group_cols, dropna=False).agg(
        n_paired=(required[0], "size"),
        **{f"{c}_mean": (c, "mean") for c in agg_cols},
        **{f"{c}_std": (c, "std") for c in agg_cols},
    ).reset_index()
    sort_col = f"improvement_{score_col}_mean"
    if sort_col in out.columns:
        out = out.sort_values(["dataset", sort_col], ascending=[True, False])
    return out


def _load_generation_traces(method_dirs: list[MethodDir]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for info in method_dirs:
        # Prefer aggregate traces, else per-run traces.
        paths = sorted(info.path.glob("generation_trace*.csv"))
        compact = [p for p in paths if re.fullmatch(r"generation_trace_c\d+_a\d+\.csv", p.name)]
        if compact:
            paths = compact
        elif not paths:
            paths = sorted(info.path.glob("rep*/fold*/run*/generation_trace.csv"))
        for path in paths:
            df = _safe_read_csv(path)
            if df.empty:
                continue
            df.insert(0, "method", info.method)
            df.insert(0, "dataset", info.dataset)
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    traces = pd.concat(frames, ignore_index=True, sort=False)
    numeric_cols = [
        "repeat_id", "outer_fold", "run_id", "generation", "elapsed_seconds",
        "best_population_fitness", "mean_population_fitness", "std_population_fitness",
        "worst_population_fitness", "best_so_far_fitness", "best_population_subset_size",
        "best_so_far_subset_size", "mean_subset_size", "min_subset_size", "max_subset_size",
        "n_edges", "n_tested_pairs", "eval_count", "archive_size", "bb_n_blocks",
        "bb_mix_trials", "bb_mix_accepts", "bb_mix_accept_rate", "bb_mix_fitness_gain",
        "bb_mix_size_reduction",
    ]
    traces = _as_numeric(traces, numeric_cols)
    return traces


def _summarize_convergence(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if traces.empty or "generation" not in traces.columns:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "run_id"] if c in traces.columns]
    run_rows: list[dict[str, float | str | int]] = []
    for key, g in traces.sort_values("generation").groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        last = g.iloc[-1]
        first = g.iloc[0]
        row["first_generation"] = int(first.get("generation", np.nan)) if pd.notna(first.get("generation", np.nan)) else np.nan
        row["last_generation"] = int(last.get("generation", np.nan)) if pd.notna(last.get("generation", np.nan)) else np.nan
        for col in [
            "best_so_far_fitness", "best_population_fitness", "mean_population_fitness",
            "best_so_far_subset_size", "mean_subset_size", "n_edges", "n_tested_pairs",
            "archive_size", "eval_count", "bb_n_blocks", "bb_mix_accept_rate",
            "bb_mix_fitness_gain", "bb_mix_size_reduction",
        ]:
            if col in g.columns:
                row[f"final_{col}"] = last.get(col, np.nan)
        if "best_so_far_fitness" in g.columns:
            y = g["best_so_far_fitness"].astype(float).to_numpy()
            x = g["generation"].astype(float).to_numpy()
            row["auc_best_so_far_by_generation"] = float(np.trapezoid(y, x)) if len(g) > 1 else float(y[0])
            row["early_25pct_best_so_far"] = float(g.iloc[max(0, math.ceil(len(g) * 0.25) - 1)]["best_so_far_fitness"])
            row["early_50pct_best_so_far"] = float(g.iloc[max(0, math.ceil(len(g) * 0.50) - 1)]["best_so_far_fitness"])
        run_rows.append(row)
    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        return run_df, pd.DataFrame()
    metric_cols = [c for c in run_df.columns if c not in group_cols]
    summary = _summary_stats(run_df, ["dataset", "method"], metric_cols)
    return run_df, summary


def _load_selected_features(method_dirs: list[MethodDir]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for info in method_dirs:
        paths = sorted(info.path.glob("selected_features*.csv"))
        # Prefer aggregate file if available; per-run root files include rep tags and can double count.
        compact = [p for p in paths if re.fullmatch(r"selected_features_c\d+_a\d+\.csv", p.name)]
        if compact:
            paths = compact
        elif not paths:
            paths = sorted(info.path.glob("rep*/fold*/run*/selected_features.csv"))
        for path in paths:
            df = _safe_read_csv(path)
            if df.empty or "feature_index" not in df.columns:
                continue
            df.insert(0, "method", info.method)
            df.insert(0, "dataset", info.dataset)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    feats = pd.concat(frames, ignore_index=True, sort=False)
    feats = _as_numeric(feats, ["repeat_id", "outer_fold", "run_id", "feature_index"])
    return feats


def _summarize_selected_features(features: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    denom = runs.groupby(["dataset", "method"], dropna=False).size().reset_index(name="n_runs")
    counts = features.groupby(["dataset", "method", "feature_index"], dropna=False).size().reset_index(name="selected_count")
    out = counts.merge(denom, on=["dataset", "method"], how="left")
    out["selection_frequency"] = out["selected_count"] / out["n_runs"].replace(0, np.nan)
    return out.sort_values(["dataset", "method", "selection_frequency", "feature_index"], ascending=[True, True, False, True])


def _load_building_block_summaries(method_dirs: list[MethodDir]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for info in method_dirs:
        paths = sorted(info.path.glob("building_block_summary*.csv"))
        compact = [p for p in paths if re.fullmatch(r"building_block_summary_c\d+_a\d+\.csv", p.name)]
        if compact:
            paths = compact
        elif not paths:
            paths = sorted(info.path.glob("rep*/fold*/run*/building_block_summary.csv"))
        for path in paths:
            df = _safe_read_csv(path)
            if df.empty:
                continue
            df.insert(0, "method", info.method)
            df.insert(0, "dataset", info.dataset)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    bb = pd.concat(frames, ignore_index=True, sort=False)
    numeric_cols = [
        "repeat_id", "outer_fold", "run_id", "generation", "n_features", "n_graph_edges",
        "n_candidates", "n_blocks", "mean_block_size", "max_block_size", "mean_score",
        "max_score", "mean_internal_abs", "mean_internal_positive",
        "mean_internal_negative_abs", "mean_signed_balance", "n_positive_edges_in_blocks",
        "n_negative_edges_in_blocks", "n_sign_conflicts_in_blocks", "bb_mix_trials",
        "bb_mix_accepts", "bb_mix_accept_rate", "bb_mix_fitness_gain", "bb_mix_size_reduction",
    ]
    return _as_numeric(bb, numeric_cols)


def _summarize_final_building_blocks(bb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bb.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "run_id"] if c in bb.columns]
    if not group_cols or "generation" not in bb.columns:
        return pd.DataFrame(), pd.DataFrame()
    final_rows = []
    for key, g in bb.sort_values("generation").groupby(group_cols, dropna=False):
        final_rows.append(g.iloc[-1])
    final = pd.DataFrame(final_rows).reset_index(drop=True)
    metric_cols = [
        c for c in [
            "generation", "n_graph_edges", "n_candidates", "n_blocks", "mean_block_size",
            "max_block_size", "mean_score", "max_score", "mean_internal_abs",
            "mean_internal_positive", "mean_internal_negative_abs", "mean_signed_balance",
            "n_positive_edges_in_blocks", "n_negative_edges_in_blocks",
            "n_sign_conflicts_in_blocks", "bb_mix_trials", "bb_mix_accepts",
            "bb_mix_accept_rate", "bb_mix_fitness_gain", "bb_mix_size_reduction",
        ] if c in final.columns
    ]
    summary = _summary_stats(final, ["dataset", "method"], metric_cols)
    return final, summary


def _best_methods_table(method_dataset_summary: pd.DataFrame, score_col: str, higher_is_better: bool) -> pd.DataFrame:
    if method_dataset_summary.empty or f"{score_col}_mean" not in method_dataset_summary.columns:
        return pd.DataFrame()
    rows = []
    for dataset, g in method_dataset_summary.groupby("dataset", dropna=False):
        sort_cols = [f"{score_col}_mean"]
        ascending = [not higher_is_better]
        if "subset_size_mean" in g.columns:
            sort_cols.append("subset_size_mean")
            ascending.append(True)
        best = g.sort_values(sort_cols, ascending=ascending).iloc[0]
        row = {
            "dataset": dataset,
            "best_score_method": best["method"],
            f"best_{score_col}_mean": best[f"{score_col}_mean"],
        }
        if f"{score_col}_std" in best:
            row[f"best_{score_col}_std"] = best[f"{score_col}_std"]
        if "subset_size_mean" in best:
            row["best_subset_size_mean"] = best["subset_size_mean"]
        if "runtime_seconds_mean" in best:
            row["best_runtime_seconds_mean"] = best["runtime_seconds_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown_report(
    outdir: Path,
    input_path: Path,
    mode: str,
    runs: pd.DataFrame,
    method_dataset_summary: pd.DataFrame,
    overall: pd.DataFrame,
    baseline: str,
    score_col: str,
) -> None:
    lines: list[str] = []
    lines.append("# BLuR-GA Method Comparison Summary")
    lines.append("")
    lines.append(f"Input: `{input_path}`")
    lines.append(f"Mode: `{mode}`")
    lines.append(f"Baseline method: `{baseline}`")
    lines.append(f"Score column: `{score_col}`")
    lines.append("")
    if not runs.empty:
        lines.append(f"Total runs: **{len(runs)}**")
        lines.append(f"Datasets: **{runs['dataset'].nunique()}**")
        lines.append(f"Methods: **{runs['method'].nunique()}**")
        lines.append("")
    if not overall.empty:
        show_cols = [c for c in ["method", "n_datasets", f"{score_col}_mean_avg_over_datasets", "score_rank_in_dataset_avg_over_datasets", "subset_size_mean_avg_over_datasets", "runtime_seconds_mean_avg_over_datasets"] if c in overall.columns]
        lines.append("## Overall method summary")
        lines.append("")
        lines.append(overall[show_cols].head(20).to_markdown(index=False))
        lines.append("")
    if not method_dataset_summary.empty:
        lines.append("## Per-dataset outputs")
        lines.append("")
        lines.append("Main files to inspect:")
        lines.append("- `method_dataset_summary.csv`")
        lines.append("- `method_overall_summary.csv`")
        lines.append("- `pivot_score_mean.csv`")
        lines.append("- `pairwise_vs_baseline_by_dataset.csv`")
        lines.append("- `convergence_method_dataset_summary.csv` if generation traces exist")
        lines.append("- `selected_feature_frequency.csv` if selected feature files exist")
        lines.append("- `final_building_block_method_dataset_summary.csv` if building-block files exist")
    (outdir / "README_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize BLuR-GA outputs for method comparison.")
    p.add_argument("--input", "--results-root", dest="input", required=True, help="Full results root or one dataset result folder.")
    p.add_argument("--output-dir", default=None, help="Directory to write comparison CSVs. Default: <input>/comparison_summary.")
    p.add_argument("--mode", choices=["auto", "all", "dataset"], default="auto", help="Input interpretation. auto detects dataset vs full root.")
    p.add_argument("--dataset", default=None, help="Filter to one dataset when --input is a full results root.")
    p.add_argument("--baseline-method", default="standard_ga", help="Method used as baseline for paired delta tables.")
    p.add_argument("--score-col", default="outer_score", help="Predictive-performance score column in run summary files.")
    p.add_argument("--higher-is-better", type=str, default="true", choices=["true", "false", "True", "False", "1", "0"], help="Whether larger score is better.")
    p.add_argument("--include-traces", type=str, default="true", choices=["true", "false", "True", "False", "1", "0"], help="Summarize generation_trace files if present.")
    p.add_argument("--include-features", type=str, default="true", choices=["true", "false", "True", "False", "1", "0"], help="Summarize selected_features files if present.")
    p.add_argument("--include-building-blocks", type=str, default="true", choices=["true", "false", "True", "False", "1", "0"], help="Summarize building_block_summary files if present.")
    return p


def _str_to_bool(v: str) -> bool:
    return str(v).lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    mode = _detect_mode(input_path, args.mode)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path / "comparison_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    higher_is_better = _str_to_bool(args.higher_is_better)

    method_dirs = _discover_method_dirs(input_path, mode, args.dataset)
    if not method_dirs:
        raise FileNotFoundError(
            f"No method result folders found under {input_path}. "
            "Expected <dataset>/<method>/nested_summary*.csv or <dataset>/<method>/rep*/fold*/run*/run_summary.csv."
        )

    manifest = pd.DataFrame([{"dataset": m.dataset, "method": m.method, "method_dir": str(m.path)} for m in method_dirs])
    manifest.to_csv(output_dir / "discovered_method_dirs.csv", index=False)

    runs = _load_run_summaries(method_dirs)
    if runs.empty:
        raise FileNotFoundError("Found method folders, but no readable run/nested summary rows.")

    # Stable ordering and output.
    sort_cols = [c for c in ["dataset", "method", "classifier_type", "repeat_id", "outer_fold", "run_id"] if c in runs.columns]
    runs = runs.sort_values(sort_cols).reset_index(drop=True)
    runs.to_csv(output_dir / "runs_long.csv", index=False)

    metric_cols = [c for c in DEFAULT_METRIC_COLS if c in runs.columns]
    method_dataset = _summary_stats(runs, ["dataset", "method"], metric_cols)
    method_dataset = _add_dataset_ranks(method_dataset, args.score_col, higher_is_better)
    if f"{args.score_col}_mean" in method_dataset.columns:
        method_dataset = method_dataset.sort_values(["dataset", "score_rank_in_dataset", f"{args.score_col}_mean"], ascending=[True, True, not higher_is_better])
    method_dataset.to_csv(output_dir / "method_dataset_summary.csv", index=False)

    overall = _method_overall_summary(method_dataset, args.score_col)
    overall.to_csv(output_dir / "method_overall_summary.csv", index=False)
    _write_pivots(output_dir, method_dataset, args.score_col)

    best = _best_methods_table(method_dataset, args.score_col, higher_is_better)
    best.to_csv(output_dir / "best_methods_by_dataset.csv", index=False)

    paired = _paired_baseline_comparison(runs, args.baseline_method, args.score_col, higher_is_better)
    paired.to_csv(output_dir / "pairwise_vs_baseline_by_dataset.csv", index=False)

    if _str_to_bool(args.include_traces):
        traces = _load_generation_traces(method_dirs)
        if not traces.empty:
            traces.to_csv(output_dir / "generation_traces_long.csv", index=False)
            conv_run, conv_summary = _summarize_convergence(traces)
            conv_run.to_csv(output_dir / "convergence_run_summary.csv", index=False)
            conv_summary.to_csv(output_dir / "convergence_method_dataset_summary.csv", index=False)

    if _str_to_bool(args.include_features):
        features = _load_selected_features(method_dirs)
        if not features.empty:
            features.to_csv(output_dir / "selected_features_long.csv", index=False)
            feat_summary = _summarize_selected_features(features, runs)
            feat_summary.to_csv(output_dir / "selected_feature_frequency.csv", index=False)

    if _str_to_bool(args.include_building_blocks):
        bb = _load_building_block_summaries(method_dirs)
        if not bb.empty:
            bb.to_csv(output_dir / "building_block_summary_long.csv", index=False)
            bb_final, bb_final_summary = _summarize_final_building_blocks(bb)
            bb_final.to_csv(output_dir / "final_building_block_run_summary.csv", index=False)
            bb_final_summary.to_csv(output_dir / "final_building_block_method_dataset_summary.csv", index=False)

    metadata = {
        "input": str(input_path),
        "mode": mode,
        "dataset_filter": args.dataset,
        "output_dir": str(output_dir),
        "baseline_method": args.baseline_method,
        "score_col": args.score_col,
        "higher_is_better": higher_is_better,
        "n_runs": int(len(runs)),
        "n_datasets": int(runs["dataset"].nunique()) if "dataset" in runs.columns else None,
        "n_methods": int(runs["method"].nunique()) if "method" in runs.columns else None,
        "datasets": sorted(map(str, runs["dataset"].unique())) if "dataset" in runs.columns else [],
        "methods": sorted(map(str, runs["method"].unique())) if "method" in runs.columns else [],
    }
    (output_dir / "summary_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_markdown_report(output_dir, input_path, mode, runs, method_dataset, overall, args.baseline_method, args.score_col)

    print(f"Discovered {len(method_dirs)} method folder(s).")
    print(f"Loaded {len(runs)} run row(s) across {runs['dataset'].nunique()} dataset(s) and {runs['method'].nunique()} method(s).")
    print(f"Wrote summary files to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
