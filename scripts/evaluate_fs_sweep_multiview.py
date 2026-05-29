#!/usr/bin/env python3
"""Multi-view evaluation for BLuR-GA feature-selection sweeps.

This script is designed to show more than mean test score.  It compares methods
from several angles:

1. Predictive performance: outer score, ranks, paired deltas vs baseline.
2. Feature compactness: selected subset size and subset ratio.
3. Computational cost: runtime, evaluation count, slowdown vs baseline.
4. Search dynamics: convergence AUC, early 25%/50% best-so-far fitness.
5. Linkage/building-block behavior: edges, block counts, first BB generation.
6. Feature-selection stability: mean Jaccard similarity across folds.
7. Multi-objective view: Pareto counts and normalized composite score.
8. Pairwise method win/loss matrices.

It reads the output of `python run_blur_ga.py aggregate --results-root ...`, i.e.
`comparison_summary/runs_long.csv` plus optional aggregate files when present.

Examples
--------
python scripts/evaluate_fs_sweep_multiview.py `
  --results-root ../Results/results_fs_uniform_positive_small_tabular

python scripts/evaluate_fs_sweep_multiview.py `
  --results-root ../Results/results_fs_uniform_positive_small_tabular `
  --exclude-datasets lymphography_uci `
  --baseline-method standard_ga
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None


METHOD_LABELS = {
    "standard_ga": "Type 0: Standard GA",
    "empirical_linkage_legacy_uniform_positive": "Type 1: Uniform Positive",
    "blur_ga_pairwise_lasso_uniform_positive_binary_nopre": "Type 2: Binary, no pre-explore",
    "blur_ga_pairwise_lasso_uniform_positive_binary_pre": "Type 2: Binary, pre-explore",
    "blur_ga_pairwise_lasso_uniform_positive_spin_nopre": "Type 2: Spin, no pre-explore",
    "blur_ga_pairwise_lasso_uniform_positive_spin_pre": "Type 2: Spin, pre-explore",
}

METHOD_ORDER = [
    "standard_ga",
    "empirical_linkage_legacy_uniform_positive",
    "blur_ga_pairwise_lasso_uniform_positive_binary_nopre",
    "blur_ga_pairwise_lasso_uniform_positive_binary_pre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_nopre",
    "blur_ga_pairwise_lasso_uniform_positive_spin_pre",
]

RUN_KEY_COLS = ["dataset", "repeat_id", "outer_fold"]

BASE_RUN_METRICS = [
    "outer_score",
    "subset_size",
    "subset_ratio",
    "runtime_seconds",
    "eval_count",
    "n_edges",
    "generations",
]

CONV_METRICS = [
    "auc_best_so_far_by_generation",
    "early_25pct_best_so_far",
    "early_50pct_best_so_far",
    "final_best_so_far_fitness",
    "final_mean_population_fitness",
]

BB_METRICS = [
    "final_bb_n_blocks",
    "bb_first_generation_with_blocks",
    "bb_generations_with_blocks",
    "bb_mean_blocks_over_trace",
    "bb_mean_block_size_final",
    "bb_max_block_size_final",
    "bb_mean_score_final",
    "bb_positive_edges_final",
    "bb_negative_edges_final",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-view evaluation for BLuR-GA FS sweeps.")
    p.add_argument("--results-root", required=True, help="Result root or comparison_summary directory.")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: <results-root>/evaluation_multiview.")
    p.add_argument("--baseline-method", default="standard_ga")
    p.add_argument("--exclude-datasets", nargs="*", default=[])
    p.add_argument("--include-methods", nargs="*", default=None)
    p.add_argument("--score-tie-tol", type=float, default=1e-12)
    p.add_argument("--score-weight", type=float, default=0.55, help="Composite score weight for outer score.")
    p.add_argument("--subset-weight", type=float, default=0.15, help="Composite score weight for smaller subset.")
    p.add_argument("--runtime-weight", type=float, default=0.15, help="Composite score weight for shorter runtime.")
    p.add_argument("--eval-weight", type=float, default=0.15, help="Composite score weight for fewer evaluations.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def resolve_summary_dir(results_root: Path) -> Path:
    if (results_root / "runs_long.csv").exists():
        return results_root
    c = results_root / "comparison_summary"
    if (c / "runs_long.csv").exists():
        return c
    raise FileNotFoundError(f"Could not find comparison_summary/runs_long.csv under {results_root}")


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def short_label(method: str) -> str:
    return (
        method_label(method)
        .replace("Type ", "T")
        .replace("Uniform Positive", "Uniform+")
        .replace("pre-explore", "pre")
        .replace("no pre", "no-pre")
    )


def sort_methods(methods: Iterable[str]) -> list[str]:
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda m: (order.get(m, 999), m))


def read_optional_csv(summary_dir: Path, name: str) -> pd.DataFrame:
    p = summary_dir / name
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_key_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in RUN_KEY_COLS if c in df.columns]
    if "classifier_type" in df.columns:
        cols.append("classifier_type")
    return cols


def make_run_identity_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ["dataset", "method", "repeat_id", "outer_fold"] if c in df.columns]
    if "classifier_type" in df.columns:
        cols.append("classifier_type")
    return cols


def parse_selected_features(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return set()
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return set(nums)


def infer_dataset_n_features(runs: pd.DataFrame, final_bb: pd.DataFrame, traces: pd.DataFrame) -> dict[str, int]:
    n_features: dict[str, int] = {}
    if not final_bb.empty and "n_features" in final_bb.columns:
        for ds, g in final_bb.groupby("dataset"):
            vals = pd.to_numeric(g["n_features"], errors="coerce").dropna()
            if not vals.empty:
                n_features[ds] = int(vals.max())
    if not traces.empty and "n_features" in traces.columns:
        for ds, g in traces.groupby("dataset"):
            if ds in n_features:
                continue
            vals = pd.to_numeric(g["n_features"], errors="coerce").dropna()
            if not vals.empty:
                n_features[ds] = int(vals.max())
    for ds, g in runs.groupby("dataset"):
        if ds in n_features:
            continue
        max_idx = 0
        if "best_selected_features" in g.columns:
            for value in g["best_selected_features"]:
                fs = parse_selected_features(value)
                if fs:
                    max_idx = max(max_idx, max(fs))
        if max_idx > 0:
            n_features[ds] = max_idx
        else:
            # Last-resort fallback.  This can undercount if leading zeroes were lost.
            vals = [len(str(v)) for v in g.get("best_chromosome", pd.Series(dtype=object)).dropna()]
            if vals:
                n_features[ds] = max(vals)
    return n_features


def load_and_enrich(summary_dir: Path, include_methods: list[str] | None, exclude_datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(summary_dir / "runs_long.csv")
    traces = read_optional_csv(summary_dir, "generation_traces_long.csv")
    conv = read_optional_csv(summary_dir, "convergence_run_summary.csv")
    final_bb = read_optional_csv(summary_dir, "final_building_block_run_summary.csv")

    required = {"dataset", "method", "repeat_id", "outer_fold", "outer_score", "subset_size"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"runs_long.csv missing required columns: {missing}")

    if exclude_datasets:
        runs = runs[~runs["dataset"].isin(exclude_datasets)].copy()
        if not traces.empty:
            traces = traces[~traces["dataset"].isin(exclude_datasets)].copy()
        if not conv.empty:
            conv = conv[~conv["dataset"].isin(exclude_datasets)].copy()
        if not final_bb.empty:
            final_bb = final_bb[~final_bb["dataset"].isin(exclude_datasets)].copy()
    if include_methods:
        runs = runs[runs["method"].isin(include_methods)].copy()
        if not traces.empty:
            traces = traces[traces["method"].isin(include_methods)].copy()
        if not conv.empty:
            conv = conv[conv["method"].isin(include_methods)].copy()
        if not final_bb.empty:
            final_bb = final_bb[final_bb["method"].isin(include_methods)].copy()

    runs = numeric_cols(runs, ["outer_score", "subset_size", "runtime_seconds", "eval_count", "n_edges", "generations", "best_inner_fitness"])
    traces = numeric_cols(traces, ["generation", "bb_n_blocks", "best_so_far_fitness", "eval_count", "elapsed_seconds", "lr_first_fit_done", "pre_lr_explore_active", "n_edges", "bb_mean_block_size", "bb_max_score"])
    conv = numeric_cols(conv, CONV_METRICS + ["final_eval_count", "final_n_edges", "final_bb_n_blocks"])
    final_bb = numeric_cols(final_bb, ["generation", "n_features", "n_graph_edges", "n_blocks", "mean_block_size", "max_block_size", "mean_score", "n_positive_edges_in_blocks", "n_negative_edges_in_blocks"])

    n_features_map = infer_dataset_n_features(runs, final_bb, traces)
    runs["n_features"] = runs["dataset"].map(n_features_map).astype("float")
    runs["subset_ratio"] = runs["subset_size"] / runs["n_features"]
    runs["method_label"] = runs["method"].map(method_label)

    # Merge convergence metrics to runs.
    if not conv.empty:
        key = [c for c in make_run_identity_cols(runs) if c in conv.columns]
        use_cols = [c for c in key + CONV_METRICS if c in conv.columns]
        conv_small = conv[use_cols].drop_duplicates(key)
        runs = runs.merge(conv_small, on=key, how="left")

    # Summarize trace-level building-block activation.
    if not traces.empty:
        key = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "classifier_type"] if c in traces.columns]
        rows = []
        for keys, g in traces.groupby(key, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(key, keys))
            blocks = pd.to_numeric(g.get("bb_n_blocks", pd.Series(dtype=float)), errors="coerce")
            has_blocks = blocks.fillna(0) > 0
            gens = pd.to_numeric(g.get("generation", pd.Series(dtype=float)), errors="coerce")
            row["bb_first_generation_with_blocks"] = float(gens[has_blocks].min()) if has_blocks.any() else np.nan
            row["bb_generations_with_blocks"] = int(has_blocks.sum())
            row["bb_mean_blocks_over_trace"] = float(blocks.mean()) if len(blocks) else np.nan
            lr_done = pd.to_numeric(g.get("lr_first_fit_done", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0
            row["lr_first_fit_generation"] = float(gens[lr_done].min()) if lr_done.any() else np.nan
            pre_active = pd.to_numeric(g.get("pre_lr_explore_active", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0
            row["pre_lr_active_generations"] = int(pre_active.sum())
            rows.append(row)
        trace_run = pd.DataFrame(rows)
        merge_key = [c for c in make_run_identity_cols(runs) if c in trace_run.columns]
        if merge_key:
            runs = runs.merge(trace_run, on=merge_key, how="left")

    # Merge final BB metrics.
    if not final_bb.empty:
        key = [c for c in ["dataset", "method", "repeat_id", "outer_fold", "classifier_type"] if c in final_bb.columns and c in runs.columns]
        if key:
            bb_cols = key + [
                c for c in [
                    "n_blocks", "mean_block_size", "max_block_size", "mean_score",
                    "n_positive_edges_in_blocks", "n_negative_edges_in_blocks"
                ] if c in final_bb.columns
            ]
            bb_small = final_bb[bb_cols].drop_duplicates(key).rename(columns={
                "n_blocks": "final_bb_n_blocks",
                "mean_block_size": "bb_mean_block_size_final",
                "max_block_size": "bb_max_block_size_final",
                "mean_score": "bb_mean_score_final",
                "n_positive_edges_in_blocks": "bb_positive_edges_final",
                "n_negative_edges_in_blocks": "bb_negative_edges_final",
            })
            runs = runs.merge(bb_small, on=key, how="left")

    return runs, traces, conv, final_bb


def safe_sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def cohen_dz(delta: pd.Series) -> float:
    delta = pd.to_numeric(delta, errors="coerce").dropna()
    if len(delta) <= 1:
        return np.nan
    sd = delta.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(delta.mean() / sd)


def pvalue_wilcoxon_or_sign(delta: pd.Series) -> tuple[float, str]:
    d = pd.to_numeric(delta, errors="coerce").dropna()
    nz = d[np.abs(d) > 1e-15]
    if len(nz) == 0:
        return np.nan, "all_ties"
    if scipy_stats is not None and len(nz) >= 2:
        try:
            res = scipy_stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
            return float(res.pvalue), "wilcoxon"
        except Exception:
            pass
    n_pos = int((nz > 0).sum())
    n = int(len(nz))
    k = min(n_pos, n - n_pos)
    if scipy_stats is not None:
        try:
            return float(scipy_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue), "sign_binomial"
        except Exception:
            pass
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * prob)), "sign_binomial_manual"


def summarize_dataset_method(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in BASE_RUN_METRICS + CONV_METRICS + BB_METRICS + ["lr_first_fit_generation", "pre_lr_active_generations"] if c in runs.columns]
    rows = []
    for (dataset, method), g in runs.groupby(["dataset", "method"], dropna=False):
        row = {"dataset": dataset, "method": method, "method_label": method_label(method), "n_runs": len(g)}
        row["n_features"] = pd.to_numeric(g.get("n_features", pd.Series(dtype=float)), errors="coerce").max()
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1)
            row[f"{m}_median"] = vals.median()
            row[f"{m}_sem"] = safe_sem(vals)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["score_rank_in_dataset"] = out.groupby("dataset")["outer_score_mean"].rank(ascending=False, method="min")
    out["subset_rank_in_dataset"] = out.groupby("dataset")["subset_size_mean"].rank(ascending=True, method="min")
    out["runtime_rank_in_dataset"] = out.groupby("dataset")["runtime_seconds_mean"].rank(ascending=True, method="min") if "runtime_seconds_mean" in out.columns else np.nan
    out["eval_rank_in_dataset"] = out.groupby("dataset")["eval_count_mean"].rank(ascending=True, method="min") if "eval_count_mean" in out.columns else np.nan
    return out.sort_values(["dataset", "score_rank_in_dataset", "method"])


def minmax_series(s: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-15:
        return pd.Series(1.0, index=s.index)
    norm = (s - mn) / (mx - mn)
    return norm if higher_is_better else 1.0 - norm


def add_composite_scores(dm: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = dm.copy()
    pieces = []
    for dataset, g in out.groupby("dataset"):
        idx = g.index
        out.loc[idx, "norm_score"] = minmax_series(g["outer_score_mean"], True).values
        out.loc[idx, "norm_subset"] = minmax_series(g["subset_size_mean"], False).values
        out.loc[idx, "norm_runtime"] = minmax_series(g.get("runtime_seconds_mean", pd.Series(0.0, index=idx)), False).values
        out.loc[idx, "norm_eval"] = minmax_series(g.get("eval_count_mean", pd.Series(0.0, index=idx)), False).values
        pieces.append(idx)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Composite weights must sum to a positive number")
    for k in weights:
        weights[k] = weights[k] / total_weight
    out["composite_score"] = (
        weights["score"] * out["norm_score"]
        + weights["subset"] * out["norm_subset"]
        + weights["runtime"] * out["norm_runtime"]
        + weights["eval"] * out["norm_eval"]
    )
    out["composite_rank_in_dataset"] = out.groupby("dataset")["composite_score"].rank(ascending=False, method="min")
    return out


def summarize_overall(dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in dm.columns if c.endswith("_mean")]
    for method, g in dm.groupby("method"):
        row = {
            "method": method,
            "method_label": method_label(method),
            "n_datasets": g["dataset"].nunique(),
            "avg_score_rank": g["score_rank_in_dataset"].mean(),
            "median_score_rank": g["score_rank_in_dataset"].median(),
            "avg_subset_rank": g["subset_rank_in_dataset"].mean(),
            "avg_runtime_rank": g["runtime_rank_in_dataset"].mean() if "runtime_rank_in_dataset" in g.columns else np.nan,
            "avg_eval_rank": g["eval_rank_in_dataset"].mean() if "eval_rank_in_dataset" in g.columns else np.nan,
            "avg_composite_rank": g["composite_rank_in_dataset"].mean() if "composite_rank_in_dataset" in g.columns else np.nan,
            "avg_composite_score": g["composite_score"].mean() if "composite_score" in g.columns else np.nan,
        }
        for c in metric_cols:
            row[f"avg_{c}"] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    # Best-score counts with tie handling.
    best_counts = []
    for dataset, g in dm.groupby("dataset"):
        best = g["outer_score_mean"].max()
        for _, r in g.iterrows():
            best_counts.append({"method": r["method"], "is_best_score_dataset": r["outer_score_mean"] >= best - 1e-12})
    bc = pd.DataFrame(best_counts).groupby("method", as_index=False)["is_best_score_dataset"].sum()
    bc = bc.rename(columns={"is_best_score_dataset": "n_best_score_datasets"})
    out = out.merge(bc, on="method", how="left")
    return out.sort_values(["avg_score_rank", "avg_composite_rank", "avg_outer_score_mean"], ascending=[True, True, False])


def paired_vs_baseline(runs: pd.DataFrame, baseline: str, tie_tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = make_key_cols(runs)
    base = runs[runs["method"] == baseline]
    if base.empty:
        raise ValueError(f"Baseline method not found: {baseline}")
    metrics = [c for c in BASE_RUN_METRICS + CONV_METRICS + BB_METRICS if c in runs.columns]
    base_small = base[key + metrics].rename(columns={c: f"baseline_{c}" for c in metrics})
    all_rows = []
    for method, g in runs[runs["method"] != baseline].groupby("method"):
        m = g.merge(base_small, on=key, how="inner")
        if m.empty:
            continue
        m = m.copy()
        m["baseline_method"] = baseline
        for c in metrics:
            if c in m.columns and f"baseline_{c}" in m.columns:
                m[f"delta_{c}"] = m[c] - m[f"baseline_{c}"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    m[f"ratio_{c}"] = m[c] / m[f"baseline_{c}"]
        all_rows.append(m)
    long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if long.empty:
        return long, pd.DataFrame()

    rows = []
    delta_metrics = [c for c in [f"delta_{m}" for m in metrics] if c in long.columns]
    for method, g in long.groupby("method"):
        row = {
            "method": method,
            "method_label": method_label(method),
            "baseline_method": baseline,
            "n_paired_runs": len(g),
            "n_datasets": g["dataset"].nunique(),
        }
        score_delta = pd.to_numeric(g["delta_outer_score"], errors="coerce")
        row["score_wins"] = int((score_delta > tie_tol).sum())
        row["score_ties"] = int((np.abs(score_delta) <= tie_tol).sum())
        row["score_losses"] = int((score_delta < -tie_tol).sum())
        p, test = pvalue_wilcoxon_or_sign(score_delta)
        row["score_p_value"] = p
        row["score_test"] = test
        row["score_cohen_dz"] = cohen_dz(score_delta)
        for dcol in delta_metrics:
            vals = pd.to_numeric(g[dcol], errors="coerce")
            row[f"{dcol}_mean"] = vals.mean()
            row[f"{dcol}_median"] = vals.median()
            row[f"{dcol}_std"] = vals.std(ddof=1)
        if "ratio_runtime_seconds" in g.columns:
            row["runtime_ratio_mean"] = pd.to_numeric(g["ratio_runtime_seconds"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        if "ratio_eval_count" in g.columns:
            row["eval_count_ratio_mean"] = pd.to_numeric(g["ratio_eval_count"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        # Marginal score gain per additional cost.  Positive is good only when extra cost > 0.
        extra_eval = pd.to_numeric(g.get("delta_eval_count", pd.Series(dtype=float)), errors="coerce")
        extra_time = pd.to_numeric(g.get("delta_runtime_seconds", pd.Series(dtype=float)), errors="coerce")
        row["delta_score_per_1000_extra_evals"] = (score_delta / (extra_eval / 1000.0)).replace([np.inf, -np.inf], np.nan).mean()
        row["delta_score_per_extra_second"] = (score_delta / extra_time).replace([np.inf, -np.inf], np.nan).mean()
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["delta_outer_score_mean", "delta_runtime_seconds_mean"], ascending=[False, True])
    return long, summary


def pairwise_win_matrix(runs: pd.DataFrame, metric: str, higher_is_better: bool, tie_tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = make_key_cols(runs)
    methods = sort_methods(runs["method"].unique())
    wide = runs.pivot_table(index=key, columns="method", values=metric, aggfunc="mean")
    wins = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    ties = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    for a, b in itertools.product(methods, methods):
        if a == b:
            continue
        sub = wide[[a, b]].dropna()
        if sub.empty:
            continue
        diff = sub[a] - sub[b]
        if not higher_is_better:
            diff = -diff
        wins.loc[a, b] = int((diff > tie_tol).sum())
        ties.loc[a, b] = int((np.abs(diff) <= tie_tol).sum())
    wins.index = [method_label(m) for m in wins.index]
    wins.columns = [method_label(m) for m in wins.columns]
    ties.index = [method_label(m) for m in ties.index]
    ties.columns = [method_label(m) for m in ties.columns]
    return wins, ties


def pareto_counts(dm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, g in dm.groupby("dataset"):
        g = g.copy()
        for i, a in g.iterrows():
            dominated = False
            for j, b in g.iterrows():
                if i == j:
                    continue
                b_better_or_equal = (
                    b["outer_score_mean"] >= a["outer_score_mean"] - 1e-12
                    and b["subset_size_mean"] <= a["subset_size_mean"] + 1e-12
                    and b.get("runtime_seconds_mean", np.nan) <= a.get("runtime_seconds_mean", np.nan) + 1e-12
                )
                b_strict = (
                    b["outer_score_mean"] > a["outer_score_mean"] + 1e-12
                    or b["subset_size_mean"] < a["subset_size_mean"] - 1e-12
                    or b.get("runtime_seconds_mean", np.nan) < a.get("runtime_seconds_mean", np.nan) - 1e-12
                )
                if b_better_or_equal and b_strict:
                    dominated = True
                    break
            rows.append({
                "dataset": dataset,
                "method": a["method"],
                "method_label": method_label(a["method"]),
                "is_pareto": not dominated,
                "outer_score_mean": a["outer_score_mean"],
                "subset_size_mean": a["subset_size_mean"],
                "runtime_seconds_mean": a.get("runtime_seconds_mean", np.nan),
            })
    detail = pd.DataFrame(rows)
    counts = detail.groupby(["method", "method_label"], as_index=False)["is_pareto"].sum().rename(columns={"is_pareto": "pareto_dataset_count"})
    counts["pareto_dataset_count"] = counts["pareto_dataset_count"].astype(int)
    counts = counts.sort_values("pareto_dataset_count", ascending=False)
    return detail, counts


def feature_stability(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "best_selected_features" not in runs.columns:
        return pd.DataFrame()
    for (dataset, method), g in runs.groupby(["dataset", "method"]):
        sets = [parse_selected_features(v) for v in g["best_selected_features"]]
        pairs = list(itertools.combinations(sets, 2))
        if not pairs:
            jacc = np.nan
        else:
            vals = []
            for a, b in pairs:
                union = len(a | b)
                vals.append(1.0 if union == 0 else len(a & b) / union)
            jacc = float(np.mean(vals))
        rows.append({
            "dataset": dataset,
            "method": method,
            "method_label": method_label(method),
            "n_runs": len(g),
            "mean_selected_jaccard_across_folds": jacc,
            "mean_subset_size": pd.to_numeric(g["subset_size"], errors="coerce").mean(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["stability_rank_in_dataset"] = out.groupby("dataset")["mean_selected_jaccard_across_folds"].rank(ascending=False, method="min")
    return out


def ablation_summary(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("pre_effect_binary", "blur_ga_pairwise_lasso_uniform_positive_binary_pre", "blur_ga_pairwise_lasso_uniform_positive_binary_nopre", "Binary: pre - no-pre"),
        ("pre_effect_spin", "blur_ga_pairwise_lasso_uniform_positive_spin_pre", "blur_ga_pairwise_lasso_uniform_positive_spin_nopre", "Spin: pre - no-pre"),
        ("encoding_effect_nopre", "blur_ga_pairwise_lasso_uniform_positive_spin_nopre", "blur_ga_pairwise_lasso_uniform_positive_binary_nopre", "No-pre: spin - binary"),
        ("encoding_effect_pre", "blur_ga_pairwise_lasso_uniform_positive_spin_pre", "blur_ga_pairwise_lasso_uniform_positive_binary_pre", "Pre: spin - binary"),
        ("type1_effect", "empirical_linkage_legacy_uniform_positive", "standard_ga", "Type 1 - standard"),
        ("best_type2_binary_effect", "blur_ga_pairwise_lasso_uniform_positive_binary_nopre", "standard_ga", "Type 2 binary no-pre - standard"),
    ]
    key = make_key_cols(runs)
    rows = []
    metrics = [c for c in ["outer_score", "subset_size", "runtime_seconds", "eval_count", "auc_best_so_far_by_generation"] if c in runs.columns]
    available = set(runs["method"].unique())
    for name, a, b, label in comparisons:
        if a not in available or b not in available:
            continue
        A = runs[runs["method"] == a][key + metrics].rename(columns={m: f"a_{m}" for m in metrics})
        B = runs[runs["method"] == b][key + metrics].rename(columns={m: f"b_{m}" for m in metrics})
        mrg = A.merge(B, on=key, how="inner")
        if mrg.empty:
            continue
        dscore = mrg["a_outer_score"] - mrg["b_outer_score"]
        p, test = pvalue_wilcoxon_or_sign(dscore)
        row = {
            "comparison": name,
            "comparison_label": label,
            "method_a": a,
            "method_b": b,
            "n_paired_runs": len(mrg),
            "n_datasets": mrg["dataset"].nunique(),
            "delta_outer_score_mean": dscore.mean(),
            "wins_a": int((dscore > 1e-12).sum()),
            "ties": int((np.abs(dscore) <= 1e-12).sum()),
            "losses_a": int((dscore < -1e-12).sum()),
            "p_value": p,
            "test": test,
            "cohen_dz": cohen_dz(dscore),
        }
        for metric in metrics:
            row[f"delta_{metric}_mean"] = (mrg[f"a_{metric}"] - mrg[f"b_{metric}"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, value: str, label_col: str, title: str, xlabel: str, out: Path, ascending: bool = True) -> None:
    if df.empty or value not in df.columns:
        return
    plot_df = df.sort_values(value, ascending=ascending).copy()
    fig_h = max(4.5, 0.48 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(plot_df))
    vals = pd.to_numeric(plot_df[value], errors="coerce").to_numpy()
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[label_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for idx, val in enumerate(vals):
        if np.isfinite(val):
            ax.text(val, idx, f" {val:.4f}" if abs(val) < 10 else f" {val:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_scatter(overall: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, out: Path) -> None:
    if overall.empty or x not in overall.columns or y not in overall.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.scatter(overall[x], overall[y])
    for _, r in overall.iterrows():
        ax.annotate(short_label(r["method"]), (r[x], r[y]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, title: str, out: Path, fmt: str = ".3f") -> None:
    if table.empty:
        return
    data = table.to_numpy(dtype=float)
    fig_w = max(8, 1.3 * len(table.columns))
    fig_h = max(4.5, 0.45 * len(table.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([short_label(c) if c in METHOD_LABELS else str(c) for c in table.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([str(x) for x in table.index])
    fig.colorbar(im, ax=ax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_boxplot(paired: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path) -> None:
    if paired.empty or metric not in paired.columns:
        return
    methods = sort_methods(paired["method"].unique())
    data = [pd.to_numeric(paired.loc[paired["method"] == m, metric], errors="coerce").dropna().to_numpy() for m in methods]
    labels = [short_label(m) for m in methods]
    fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(methods)), 5.8))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def nemenyi_cd(avg_ranks: pd.DataFrame, out_dir: Path, alpha: float = 0.05) -> pd.DataFrame:
    # Approximate Nemenyi critical difference across datasets, mainly as a ranking aid.
    methods = avg_ranks["method"].nunique()
    n_datasets = int(avg_ranks["n_datasets"].max()) if "n_datasets" in avg_ranks.columns else 0
    if methods < 2 or n_datasets < 2:
        return pd.DataFrame()
    if scipy_stats is not None and hasattr(scipy_stats, "studentized_range"):
        q = float(scipy_stats.studentized_range.ppf(1 - alpha, methods, np.inf) / math.sqrt(2.0))
    else:
        # Conservative fallback for k around 4-10 at alpha=.05.
        q = 2.85
    cd = q * math.sqrt(methods * (methods + 1) / (6.0 * n_datasets))
    best_rank = avg_ranks["avg_score_rank"].min()
    out = avg_ranks[["method", "method_label", "avg_score_rank", "n_datasets"]].copy()
    out["nemenyi_cd_alpha_0.05"] = cd
    out["within_cd_of_best_rank"] = out["avg_score_rank"] <= best_rank + cd
    out.to_csv(out_dir / "critical_difference_ranking.csv", index=False)
    return out


def write_report(
    out: Path,
    runs: pd.DataFrame,
    overall: pd.DataFrame,
    paired_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    pareto: pd.DataFrame,
    stability_overall: pd.DataFrame,
    cd: pd.DataFrame,
    excluded: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Multi-view FS sweep evaluation")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Datasets: **{runs['dataset'].nunique()}**")
    lines.append(f"- Methods: **{runs['method'].nunique()}**")
    lines.append(f"- Fold-level runs: **{len(runs)}**")
    if excluded:
        lines.append(f"- Excluded datasets: `{', '.join(excluded)}`")
    lines.append("")

    lines.append("## Overall multi-metric summary")
    lines.append("")
    show_cols = [
        "method_label", "avg_outer_score_mean", "avg_score_rank", "n_best_score_datasets",
        "avg_subset_size_mean", "avg_subset_ratio_mean", "avg_runtime_seconds_mean",
        "avg_eval_count_mean", "avg_composite_score", "avg_composite_rank",
    ]
    show_cols = [c for c in show_cols if c in overall.columns]
    lines.append(overall[show_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Paired tests vs standard GA")
    lines.append("")
    if not paired_summary.empty:
        cols = [
            "method_label", "n_paired_runs", "delta_outer_score_mean", "score_wins", "score_ties", "score_losses",
            "delta_subset_size_mean", "delta_runtime_seconds_mean", "delta_eval_count_mean",
            "runtime_ratio_mean", "eval_count_ratio_mean", "score_p_value",
        ]
        cols = [c for c in cols if c in paired_summary.columns]
        lines.append(paired_summary[cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No paired baseline comparison was available.")
    lines.append("")

    lines.append("## Focused ablations")
    lines.append("")
    if not ablation.empty:
        cols = ["comparison_label", "delta_outer_score_mean", "wins_a", "ties", "losses_a", "delta_runtime_seconds_mean", "delta_eval_count_mean", "p_value"]
        cols = [c for c in cols if c in ablation.columns]
        lines.append(ablation[cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No focused ablation pairs were available.")
    lines.append("")

    lines.append("## Pareto view")
    lines.append("")
    if not pareto.empty:
        lines.append(pareto.to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No Pareto summary was available.")
    lines.append("")

    lines.append("## Feature-selection stability")
    lines.append("")
    if not stability_overall.empty:
        lines.append(stability_overall.to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No stability summary was available.")
    lines.append("")

    lines.append("## Critical-difference ranking aid")
    lines.append("")
    if not cd.empty:
        lines.append(cd.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("This is a screening aid based on average ranks across datasets, not a replacement for repeated-run statistical analysis.")
    else:
        lines.append("Not enough methods/datasets for a critical-difference summary.")
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    lines.append("Key CSV files: `overall_multiview_summary.csv`, `dataset_method_multiview_summary.csv`, `paired_tests_vs_baseline_multiview.csv`, `pairwise_score_win_matrix.csv`, `pareto_counts.csv`, `feature_stability_overall.csv`, `focused_ablation_multiview.csv`.")
    lines.append("Plots are under `plots/`.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    summary_dir = resolve_summary_dir(results_root)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = summary_dir.parent / "evaluation_multiview" if summary_dir.name == "comparison_summary" else summary_dir / "evaluation_multiview"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs, traces, conv, final_bb = load_and_enrich(summary_dir, args.include_methods, args.exclude_datasets)
    if runs.empty:
        raise ValueError("No runs left after filtering.")

    weights = {"score": args.score_weight, "subset": args.subset_weight, "runtime": args.runtime_weight, "eval": args.eval_weight}
    dm = summarize_dataset_method(runs)
    dm = add_composite_scores(dm, weights)
    overall = summarize_overall(dm)
    paired_long, paired_summary = paired_vs_baseline(runs, args.baseline_method, args.score_tie_tol)
    ablation = ablation_summary(runs)
    pareto_detail, pareto_count = pareto_counts(dm)
    stability = feature_stability(runs)
    if not stability.empty:
        stability_overall = stability.groupby(["method", "method_label"], as_index=False).agg(
            avg_selected_jaccard=("mean_selected_jaccard_across_folds", "mean"),
            avg_stability_rank=("stability_rank_in_dataset", "mean"),
            avg_subset_size=("mean_subset_size", "mean"),
        ).sort_values(["avg_selected_jaccard", "avg_stability_rank"], ascending=[False, True])
    else:
        stability_overall = pd.DataFrame()

    score_wins, score_ties = pairwise_win_matrix(runs, "outer_score", True, args.score_tie_tol)
    subset_wins, _ = pairwise_win_matrix(runs, "subset_size", False, args.score_tie_tol)
    runtime_wins, _ = pairwise_win_matrix(runs, "runtime_seconds", False, args.score_tie_tol)

    # Pivots.
    score_pivot = dm.pivot(index="dataset", columns="method", values="outer_score_mean").reindex(columns=sort_methods(dm["method"].unique()))
    delta_pivot = pd.DataFrame()
    if not paired_long.empty:
        delta_pivot = paired_long.pivot_table(index="dataset", columns="method", values="delta_outer_score", aggfunc="mean").reindex(columns=[m for m in sort_methods(paired_long["method"].unique())])

    # Save tables.
    runs.to_csv(out_dir / "filtered_runs_long_enriched.csv", index=False)
    dm.to_csv(out_dir / "dataset_method_multiview_summary.csv", index=False)
    overall.to_csv(out_dir / "overall_multiview_summary.csv", index=False)
    paired_long.to_csv(out_dir / "paired_deltas_vs_baseline_multiview.csv", index=False)
    paired_summary.to_csv(out_dir / "paired_tests_vs_baseline_multiview.csv", index=False)
    ablation.to_csv(out_dir / "focused_ablation_multiview.csv", index=False)
    pareto_detail.to_csv(out_dir / "pareto_by_dataset_method.csv", index=False)
    pareto_count.to_csv(out_dir / "pareto_counts.csv", index=False)
    stability.to_csv(out_dir / "feature_stability_by_dataset_method.csv", index=False)
    stability_overall.to_csv(out_dir / "feature_stability_overall.csv", index=False)
    score_wins.to_csv(out_dir / "pairwise_score_win_matrix.csv")
    score_ties.to_csv(out_dir / "pairwise_score_tie_matrix.csv")
    subset_wins.to_csv(out_dir / "pairwise_subset_compactness_win_matrix.csv")
    runtime_wins.to_csv(out_dir / "pairwise_runtime_win_matrix.csv")
    score_pivot.rename(columns=method_label).to_csv(out_dir / "pivot_score_mean.csv")
    if not delta_pivot.empty:
        delta_pivot.rename(columns=method_label).to_csv(out_dir / "pivot_delta_score_vs_baseline.csv")
    cd = nemenyi_cd(overall, out_dir)

    if not args.no_plots:
        plot_bar(overall, "avg_score_rank", "method_label", "Average score rank across datasets", "Lower rank is better", plot_dir / "avg_score_rank.png", ascending=False)
        plot_bar(overall, "avg_outer_score_mean", "method_label", "Average outer score", "Higher is better", plot_dir / "avg_outer_score.png", ascending=True)
        plot_bar(overall, "avg_composite_score", "method_label", "Composite score across performance/cost/subset", "Higher is better", plot_dir / "composite_score.png", ascending=True)
        plot_bar(pareto_count, "pareto_dataset_count", "method_label", "Pareto appearances by method", "Number of datasets on Pareto front", plot_dir / "pareto_counts.png", ascending=True)
        if not stability_overall.empty:
            plot_bar(stability_overall, "avg_selected_jaccard", "method_label", "Feature-selection stability", "Mean Jaccard similarity across folds", plot_dir / "feature_stability.png", ascending=True)
        plot_scatter(overall, "avg_subset_size_mean", "avg_outer_score_mean", "Score vs selected subset size", "Average selected features", "Average outer score", plot_dir / "score_vs_subset_size.png")
        plot_scatter(overall, "avg_runtime_seconds_mean", "avg_outer_score_mean", "Score vs runtime", "Average runtime seconds", "Average outer score", plot_dir / "score_vs_runtime.png")
        plot_scatter(overall, "avg_eval_count_mean", "avg_outer_score_mean", "Score vs evaluation count", "Average eval count", "Average outer score", plot_dir / "score_vs_eval_count.png")
        plot_boxplot(paired_long, "delta_outer_score", "Fold-level score delta vs baseline", "Outer score delta", plot_dir / "delta_score_boxplot_vs_baseline.png")
        plot_boxplot(paired_long, "delta_subset_size", "Fold-level subset-size delta vs baseline", "Selected feature delta", plot_dir / "delta_subset_boxplot_vs_baseline.png")
        plot_boxplot(paired_long, "delta_runtime_seconds", "Fold-level runtime delta vs baseline", "Runtime delta seconds", plot_dir / "delta_runtime_boxplot_vs_baseline.png")
        plot_heatmap(score_pivot, "Mean outer score by dataset and method", plot_dir / "heatmap_score_mean.png", fmt=".3f")
        if not delta_pivot.empty:
            plot_heatmap(delta_pivot, "Mean score delta vs baseline by dataset", plot_dir / "heatmap_delta_score_vs_baseline.png", fmt="+.3f")
        plot_heatmap(score_wins, "Pairwise score win counts", plot_dir / "pairwise_score_win_matrix.png", fmt=".0f")
        plot_heatmap(runtime_wins, "Pairwise runtime win counts", plot_dir / "pairwise_runtime_win_matrix.png", fmt=".0f")
        if "avg_auc_best_so_far_by_generation_mean" in overall.columns:
            plot_bar(overall, "avg_auc_best_so_far_by_generation_mean", "method_label", "Convergence AUC: best-so-far fitness", "Higher is better", plot_dir / "convergence_auc.png", ascending=True)
        if "avg_final_bb_n_blocks_mean" in overall.columns:
            plot_bar(overall, "avg_final_bb_n_blocks_mean", "method_label", "Final building-block count", "Average final BB count", plot_dir / "final_bb_n_blocks.png", ascending=True)

    write_report(out_dir / "README_multiview_evaluation.md", runs, overall, paired_summary, ablation, pareto_count, stability_overall, cd, args.exclude_datasets)

    print(f"Read: {summary_dir}")
    print(f"Wrote: {out_dir}")
    print(f"Datasets={runs['dataset'].nunique()} Methods={runs['method'].nunique()} Runs={len(runs)}")
    best = overall.iloc[0]
    print(f"Best avg score rank: {best['method_label']} | avg_score={best.get('avg_outer_score_mean', np.nan):.4f} | avg_rank={best['avg_score_rank']:.2f}")


if __name__ == "__main__":
    main()
