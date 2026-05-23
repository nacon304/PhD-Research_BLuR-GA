#!/usr/bin/env python3
from __future__ import annotations

"""Sign-accuracy and convergence analysis for BLuR-GA linkage experiments.

The linkage-correctness part is restricted to eval_scope == "all_returned".
The convergence part is optional and requires the original results directory that
contains run-level files such as generation_trace.csv and run_summary.csv.

PowerShell examples
-------------------
# Only sign/linkage metrics from aggregate CSVs:
python plot_linkage_sign_and_convergence_analysis.py `
  --summary "linkage_eval_summary(1).csv" `
  --by-method "linkage_eval_by_method(2).csv" `
  --outdir linkage_sign_convergence_figures

# Add convergence-speed plots if the full results folder is available:
python plot_linkage_sign_and_convergence_analysis.py `
  --summary "linkage_eval_summary(1).csv" `
  --by-method "linkage_eval_by_method(2).csv" `
  --results-root "../Results/results_linkage_test" `
  --outdir linkage_sign_convergence_figures
"""

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

# These methods have no meaningful learned signed coefficient in the current setup.
# - empirical_linkage_legacy: current evaluator returns 0 when the prediction file has no sign column.
# - top_cooccurrence/random_graph: sign is always +1 in the baseline generator, so high sign accuracy
#   mainly means the true benchmark has mostly positive edges, not that the baseline learned signs.
DEFAULT_SIGNED_METHODS = [
    "blur_ga_main_pairwise_lasso",
    "blur_ga_pairwise_lasso",
]

FAMILY_PREFIX_ORDER = [
    "Ising",
    "MaxSAT",
    "NK",
    "QUBO",
    "Planted XOR",
    "GAMETES",
    "Other",
]


def infer_family(dataset: str) -> str:
    """Infer a benchmark-family label from the dataset/file name.

    This reads the family and size parameters from filenames such as
    ``ising_spin_glass_grid5x5_posneg_seed0``,
    ``maxsat_d35_m90_k3_weighted_seed0``,
    ``nk_landscape_d30_k3_random_seed0``, and
    ``pairwise_qubo_d30_e30_posneg_noise0_seed0``.
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
    m = re.search(r"seed(\d+)$", str(dataset))
    return m.group(1) if m else ""


def shorten_dataset(dataset: str) -> str:
    family = infer_family(str(dataset))
    seed = infer_seed(str(dataset))
    return f"{family} s{seed}" if seed else family


def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def ordered_methods(methods: Iterable[str]) -> list[str]:
    methods = list(dict.fromkeys(map(str, methods)))
    known = [m for m in METHOD_ORDER if m in methods]
    unknown = sorted(m for m in methods if m not in METHOD_ORDER)
    return known + unknown


def family_sort_key(family: str) -> tuple[int, str]:
    family = str(family)
    for idx, prefix in enumerate(FAMILY_PREFIX_ORDER):
        if family == prefix or family.startswith(prefix + " "):
            return idx, family
    return len(FAMILY_PREFIX_ORDER), family


def ordered_families(families: Iterable[str]) -> list[str]:
    families = list(dict.fromkeys(map(str, families)))
    return sorted(families, key=family_sort_key)


def savefig(outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def require_columns(df: pd.DataFrame, cols: Iterable[str], file_label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def load_all_returned(summary_path: Path, by_method_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    by_method = pd.read_csv(by_method_path)
    required = [
        "dataset", "method", "eval_scope", "precision", "recall", "f1",
        "average_precision", "sign_accuracy", "n_eval_edges", "n_pred_edges",
        "true_positives", "false_positives", "false_negatives",
    ]
    require_columns(summary, required, "summary CSV")
    require_columns(by_method, required, "by-method CSV")

    summary = summary.loc[summary["eval_scope"].eq("all_returned")].copy()
    by_method = by_method.loc[by_method["eval_scope"].eq("all_returned")].copy()
    if summary.empty:
        raise ValueError("No eval_scope == 'all_returned' rows found in summary CSV.")
    if by_method.empty:
        raise ValueError("No eval_scope == 'all_returned' rows found in by-method CSV.")

    for df in (summary, by_method):
        df["family"] = df["dataset"].map(infer_family)
        df["dataset_short"] = df["dataset"].map(shorten_dataset)
        df["method_label"] = df["method"].map(method_label)
        df["sign_correct_est"] = df["sign_accuracy"] * df["true_positives"]
        if "n_features" in df.columns:
            df["candidate_pairs"] = df["n_features"] * (df["n_features"] - 1) / 2
        else:
            df["candidate_pairs"] = np.nan
        df["pred_density"] = df["n_pred_edges"] / df["candidate_pairs"]
        df["true_edges"] = df["true_positives"] + df["false_negatives"]
        df["true_density"] = df["true_edges"] / df["candidate_pairs"]

    return summary, by_method


def make_sign_views(summary: pd.DataFrame, by_method: pd.DataFrame, signed_methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw and sign-capable views.

    Raw view preserves evaluator output. Signed view keeps only methods whose sign
    accuracy is meaningful as learned signed coefficients.
    """
    raw = by_method.copy()
    signed = by_method.loc[by_method["method"].isin(signed_methods)].copy()
    return raw, signed


def weighted_mean_sign(df: pd.DataFrame) -> float:
    denom = df["true_positives"].sum()
    if denom <= 0:
        return np.nan
    return float(df["sign_correct_est"].sum() / denom)


# -----------------------------------------------------------------------------
# Sign-accuracy plots
# -----------------------------------------------------------------------------

def plot_sign_heatmap(by_method: pd.DataFrame, outdir: Path, name: str, title: str) -> None:
    methods = ordered_methods(by_method["method"].unique())
    dataset_order = (
        by_method[["dataset", "family", "dataset_short"]]
        .drop_duplicates()
        .sort_values(["family", "dataset"])
    )
    datasets = dataset_order["dataset"].tolist()
    dataset_labels = dataset_order["dataset_short"].tolist()

    table = (
        by_method.pivot_table(index="dataset", columns="method", values="sign_accuracy", aggfunc="mean")
        .reindex(index=datasets, columns=methods)
    )

    fig_w = max(9, 1.5 * len(methods))
    fig_h = max(5, 0.35 * len(datasets) + 2)
    plt.figure(figsize=(fig_w, fig_h))
    arr = table.to_numpy(dtype=float)
    im = plt.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Sign accuracy")
    plt.xticks(range(len(methods)), [method_label(m) for m in methods], rotation=35, ha="right")
    plt.yticks(range(len(datasets)), dataset_labels)
    plt.title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            text = "NA" if np.isnan(value) else f"{value:.2f}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8)
    savefig(outdir, name)


def plot_sign_bar_by_family(signed: pd.DataFrame, outdir: Path) -> None:
    if signed.empty:
        return
    # First aggregate within each dataset-method, then average datasets within each family.
    fam = (
        signed.groupby(["family", "method"], as_index=False)
        .apply(lambda g: pd.Series({
            "sign_accuracy_mean": g["sign_accuracy"].mean(),
            "sign_accuracy_weighted_by_tp": weighted_mean_sign(g),
            "mean_true_positives": g["true_positives"].mean(),
        }))
        .reset_index(drop=True)
    )
    fam.to_csv(outdir / "sign_accuracy_by_family_signed_methods.csv", index=False)

    families = ordered_families(fam["family"].unique())
    methods = ordered_methods(fam["method"].unique())
    x = np.arange(len(families))
    width = 0.8 / max(1, len(methods))
    plt.figure(figsize=(max(9, 1.3 * len(families)), 5.5))
    for idx, method in enumerate(methods):
        vals = []
        for family in families:
            row = fam.loc[(fam["family"].eq(family)) & (fam["method"].eq(method)), "sign_accuracy_weighted_by_tp"]
            vals.append(np.nan if row.empty else float(row.iloc[0]))
        plt.bar(x + (idx - (len(methods) - 1) / 2) * width, vals, width, label=method_label(method))
    plt.xticks(x, families, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("TP-weighted sign accuracy")
    plt.title("Sign accuracy by benchmark family: signed Lasso methods only")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    savefig(outdir, "sign_accuracy_by_family_signed_methods")


def plot_sign_vs_f1(by_method: pd.DataFrame, outdir: Path, signed_only: bool) -> None:
    df = by_method.copy()
    if df.empty:
        return
    methods = ordered_methods(df["method"].unique())
    plt.figure(figsize=(8.5, 6))
    for method in methods:
        sub = df[df["method"].eq(method)]
        if sub.empty:
            continue
        # Marker size reflects how many true edges are involved; low-TP sign accuracy is less reliable.
        sizes = 30 + 2.0 * np.sqrt(np.maximum(sub["true_positives"].to_numpy(dtype=float), 0.0))
        plt.scatter(sub["sign_accuracy"], sub["f1"], s=sizes, alpha=0.7, label=method_label(method))
    plt.xlabel("Sign accuracy")
    plt.ylabel("F1 on returned edges")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.title("Sign accuracy vs support recovery F1" + (" (signed methods only)" if signed_only else " (raw evaluator output)"))
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(outdir, "sign_accuracy_vs_f1_signed_methods" if signed_only else "sign_accuracy_vs_f1_raw")


def plot_sign_boxplot(summary: pd.DataFrame, outdir: Path, signed_methods: list[str]) -> None:
    df = summary.loc[summary["method"].isin(signed_methods)].copy()
    if df.empty:
        return
    methods = ordered_methods(df["method"].unique())
    data = [df.loc[df["method"].eq(m), "sign_accuracy"].dropna().to_numpy() for m in methods]
    labels = [method_label(m) for m in methods]
    plt.figure(figsize=(8, 5))
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Run-level sign accuracy")
    plt.ylim(-0.02, 1.02)
    plt.xticks(rotation=20, ha="right")
    plt.title("Run-level variability of sign accuracy: signed Lasso methods only")
    plt.grid(axis="y", alpha=0.3)
    savefig(outdir, "run_level_sign_accuracy_boxplot_signed_methods")


def write_sign_tables(summary: pd.DataFrame, by_method: pd.DataFrame, signed_methods: list[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_family = (
        by_method[["dataset", "family", "dataset_short"]]
        .drop_duplicates()
        .sort_values(["family", "dataset"])
    )
    dataset_family.to_csv(outdir / "dataset_family_mapping.csv", index=False)

    raw_overall = (
        by_method.groupby("method", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            sign_accuracy_mean=("sign_accuracy", "mean"),
            sign_accuracy_min=("sign_accuracy", "min"),
            sign_accuracy_max=("sign_accuracy", "max"),
            f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            true_positives_mean=("true_positives", "mean"),
        )
    )
    raw_overall["method_label"] = raw_overall["method"].map(method_label)
    raw_overall.to_csv(outdir / "sign_accuracy_overall_raw_all_methods.csv", index=False)

    signed = by_method.loc[by_method["method"].isin(signed_methods)].copy()
    signed_overall_rows = []
    for method, g in signed.groupby("method"):
        signed_overall_rows.append({
            "method": method,
            "method_label": method_label(method),
            "n_datasets": g["dataset"].nunique(),
            "sign_accuracy_mean": g["sign_accuracy"].mean(),
            "sign_accuracy_weighted_by_tp": weighted_mean_sign(g),
            "f1_mean": g["f1"].mean(),
            "precision_mean": g["precision"].mean(),
            "recall_mean": g["recall"].mean(),
            "true_positives_sum": g["true_positives"].sum(),
        })
    pd.DataFrame(signed_overall_rows).to_csv(outdir / "sign_accuracy_overall_signed_methods.csv", index=False)

    signed_dataset = signed[[
        "dataset", "family", "method", "method_label", "sign_accuracy", "f1", "precision", "recall",
        "true_positives", "false_positives", "false_negatives", "n_pred_edges"
    ]].sort_values(["family", "dataset", "method"])
    signed_dataset.to_csv(outdir / "sign_accuracy_by_dataset_signed_methods.csv", index=False)

    # Dataset rows where sign is high but F1 is low: sign is correct on a small/poor support, not enough by itself.
    sign_high_support_low = signed.loc[(signed["sign_accuracy"] >= 0.8) & (signed["f1"] < 0.5)].copy()
    sign_high_support_low.to_csv(outdir / "cases_high_sign_accuracy_but_low_f1.csv", index=False)

    # Dataset rows where support is decent but sign is weak: edge support may be right, but coefficient direction is unstable.
    support_ok_sign_weak = signed.loc[(signed["f1"] >= 0.5) & (signed["sign_accuracy"] < 0.7)].copy()
    support_ok_sign_weak.to_csv(outdir / "cases_ok_f1_but_weak_sign_accuracy.csv", index=False)


# -----------------------------------------------------------------------------
# Optional convergence analysis from generation_trace.csv and run_summary.csv
# -----------------------------------------------------------------------------

def parse_run_path(path: Path, results_root: Path) -> dict[str, object]:
    """Parse linkage/<dataset>/<method>/repXX/foldYY/runZZZ/file.csv."""
    rel = path.relative_to(results_root)
    parts = rel.parts
    info = {"dataset": "", "method": "", "repeat_id": np.nan, "outer_fold": np.nan, "run_id": np.nan}
    if len(parts) >= 7 and parts[0] == "linkage":
        info["dataset"] = parts[1]
        info["method"] = parts[2]
        for part in parts:
            if part.startswith("rep"):
                info["repeat_id"] = int(part.replace("rep", ""))
            elif part.startswith("fold"):
                info["outer_fold"] = int(part.replace("fold", ""))
            elif re.fullmatch(r"run\d+", part):
                info["run_id"] = int(part.replace("run", ""))
    return info


def read_generation_traces(results_root: Path) -> pd.DataFrame:
    rows = []
    files = sorted((results_root / "linkage").glob("**/generation_trace.csv")) if (results_root / "linkage").exists() else []
    for p in files:
        if p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "generation" not in df.columns:
            continue
        info = parse_run_path(p, results_root)
        for key, value in info.items():
            if key not in df.columns or df[key].isna().all() or key in {"dataset", "method"}:
                df[key] = value
        df["dataset"] = info["dataset"]
        df["method"] = info["method"]
        df["family"] = df["dataset"].map(infer_family)
        df["method_label"] = df["method"].map(method_label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def read_run_summaries(results_root: Path) -> pd.DataFrame:
    rows = []
    files = sorted((results_root / "linkage").glob("**/run_summary.csv")) if (results_root / "linkage").exists() else []
    for p in files:
        if p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        info = parse_run_path(p, results_root)
        for key, value in info.items():
            if key not in df.columns or key in {"dataset", "method"}:
                df[key] = value
        df["dataset"] = info["dataset"]
        df["method"] = info["method"]
        df["family"] = df["dataset"].map(infer_family)
        df["method_label"] = df["method"].map(method_label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_convergence_progress(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace
    df = trace.copy()
    keys = ["dataset", "method", "repeat_id", "outer_fold", "run_id"]
    metric = "best_so_far_fitness"
    if metric not in df.columns:
        raise ValueError("generation_trace.csv must contain best_so_far_fitness for convergence analysis.")

    def _progress(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("generation").copy()
        start = float(g[metric].iloc[0])
        final = float(g[metric].iloc[-1])
        denom = final - start
        if abs(denom) < 1e-12:
            g["relative_progress"] = np.nan
        else:
            g["relative_progress"] = (g[metric] - start) / denom
            g["relative_progress"] = g["relative_progress"].clip(lower=0.0, upper=1.0)
        return g

    return df.groupby(keys, group_keys=False).apply(_progress)


def convergence_speed_table(trace: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    rows = []
    keys = ["dataset", "method", "repeat_id", "outer_fold", "run_id"]
    for key, g in trace.groupby(keys):
        g = g.sort_values("generation")
        base = {
            "dataset": key[0], "method": key[1], "repeat_id": key[2], "outer_fold": key[3], "run_id": key[4],
            "family": infer_family(key[0]), "method_label": method_label(key[1]),
            "initial_best_so_far": float(g["best_so_far_fitness"].iloc[0]),
            "final_best_so_far": float(g["best_so_far_fitness"].iloc[-1]),
            "final_generation": float(g["generation"].iloc[-1]),
            "final_eval_count": float(g["eval_count"].iloc[-1]) if "eval_count" in g.columns else np.nan,
            "final_elapsed_seconds": float(g["elapsed_seconds"].iloc[-1]) if "elapsed_seconds" in g.columns else np.nan,
        }
        for thr in thresholds:
            col_prefix = f"to_{int(round(thr * 100))}pct"
            hit = g.loc[g["relative_progress"] >= thr]
            if hit.empty:
                base[f"generation_{col_prefix}"] = np.nan
                base[f"eval_count_{col_prefix}"] = np.nan
                base[f"elapsed_seconds_{col_prefix}"] = np.nan
            else:
                first = hit.iloc[0]
                base[f"generation_{col_prefix}"] = float(first["generation"])
                base[f"eval_count_{col_prefix}"] = float(first["eval_count"]) if "eval_count" in g.columns else np.nan
                base[f"elapsed_seconds_{col_prefix}"] = float(first["elapsed_seconds"]) if "elapsed_seconds" in g.columns else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def plot_mean_convergence(trace: pd.DataFrame, outdir: Path, x_col: str, y_col: str, name: str, title: str) -> None:
    if trace.empty or x_col not in trace.columns or y_col not in trace.columns:
        return
    methods = ordered_methods(trace["method"].unique())
    plt.figure(figsize=(9, 5.5))
    for method in methods:
        sub = trace.loc[trace["method"].eq(method)].copy()
        if sub.empty:
            continue
        curve = sub.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
        plt.plot(curve[x_col], curve[y_col], label=method_label(method))
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(outdir, name)


def plot_family_convergence(trace: pd.DataFrame, outdir: Path) -> None:
    if trace.empty:
        return
    for family in ordered_families(trace["family"].unique()):
        subfam = trace.loc[trace["family"].eq(family)]
        if subfam.empty:
            continue
        safe_family = re.sub(r"[^A-Za-z0-9]+", "_", family).strip("_").lower()
        plot_mean_convergence(
            subfam,
            outdir,
            x_col="generation",
            y_col="best_so_far_fitness",
            name=f"convergence_best_fitness_by_generation_{safe_family}",
            title=f"Mean best-so-far fitness by generation: {family}",
        )
        if "relative_progress" in subfam.columns:
            plot_mean_convergence(
                subfam,
                outdir,
                x_col="generation",
                y_col="relative_progress",
                name=f"convergence_relative_progress_by_generation_{safe_family}",
                title=f"Mean relative progress by generation: {family}",
            )


def plot_speed_bars(speed: pd.DataFrame, outdir: Path, metric_col: str, name: str, ylabel: str, title: str) -> None:
    if speed.empty or metric_col not in speed.columns:
        return
    agg = (
        speed.groupby("method", as_index=False)[metric_col]
        .mean()
        .sort_values(metric_col, ascending=True)
    )
    if agg.empty:
        return
    labels = [method_label(m) for m in agg["method"]]
    x = np.arange(len(agg))
    plt.figure(figsize=(8.5, 5))
    plt.bar(x, agg[metric_col])
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    savefig(outdir, name)


def plot_runtime_vs_final_fitness(speed: pd.DataFrame, outdir: Path) -> None:
    if speed.empty or "final_elapsed_seconds" not in speed.columns:
        return
    methods = ordered_methods(speed["method"].unique())
    plt.figure(figsize=(8.5, 6))
    for method in methods:
        sub = speed[speed["method"].eq(method)]
        if sub.empty:
            continue
        plt.scatter(sub["final_elapsed_seconds"], sub["final_best_so_far"], alpha=0.7, label=method_label(method))
    plt.xlabel("Runtime seconds")
    plt.ylabel("Final best-so-far fitness")
    plt.title("Runtime vs final best-so-far fitness")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(outdir, "runtime_seconds_vs_final_best_fitness")


def run_convergence_analysis(results_root: Path, outdir: Path) -> None:
    trace = read_generation_traces(results_root)
    summaries = read_run_summaries(results_root)
    if trace.empty:
        (outdir / "convergence_not_available.txt").write_text(
            "No non-empty generation_trace.csv files were found. Convergence-speed analysis needs the full results-root, not only aggregate CSVs.\n",
            encoding="utf-8",
        )
        return
    trace = add_convergence_progress(trace)
    trace.to_csv(outdir / "generation_trace_all_runs_compact.csv", index=False)
    if not summaries.empty:
        summaries.to_csv(outdir / "run_summary_all_runs_compact.csv", index=False)

    speed = convergence_speed_table(trace, thresholds=[0.80, 0.90, 0.95])
    speed.to_csv(outdir / "convergence_speed_by_run.csv", index=False)

    if not speed.empty:
        agg_speed = (
            speed.groupby(["family", "method"], as_index=False)
            .agg(
                n_runs=("run_id", "count"),
                final_best_so_far_mean=("final_best_so_far", "mean"),
                final_generation_mean=("final_generation", "mean"),
                final_eval_count_mean=("final_eval_count", "mean"),
                final_elapsed_seconds_mean=("final_elapsed_seconds", "mean"),
                generation_to_95pct_mean=("generation_to_95pct", "mean"),
                eval_count_to_95pct_mean=("eval_count_to_95pct", "mean"),
                elapsed_seconds_to_95pct_mean=("elapsed_seconds_to_95pct", "mean"),
            )
        )
        agg_speed["method_label"] = agg_speed["method"].map(method_label)
        agg_speed.to_csv(outdir / "convergence_speed_by_family_method.csv", index=False)

    plot_mean_convergence(
        trace,
        outdir,
        x_col="generation",
        y_col="best_so_far_fitness",
        name="convergence_best_fitness_by_generation_all_datasets",
        title="Mean best-so-far fitness by generation: all datasets",
    )
    if "relative_progress" in trace.columns:
        plot_mean_convergence(
            trace,
            outdir,
            x_col="generation",
            y_col="relative_progress",
            name="convergence_relative_progress_by_generation_all_datasets",
            title="Mean relative progress toward final fitness: all datasets",
        )
    if "eval_count" in trace.columns:
        plot_mean_convergence(
            trace,
            outdir,
            x_col="eval_count",
            y_col="best_so_far_fitness",
            name="convergence_best_fitness_by_eval_count_all_datasets",
            title="Mean best-so-far fitness by evaluation count: all datasets",
        )
    if "elapsed_seconds" in trace.columns:
        plot_mean_convergence(
            trace,
            outdir,
            x_col="elapsed_seconds",
            y_col="best_so_far_fitness",
            name="convergence_best_fitness_by_seconds_all_datasets",
            title="Mean best-so-far fitness by elapsed seconds: all datasets",
        )

    plot_family_convergence(trace, outdir)
    if not speed.empty:
        plot_speed_bars(
            speed,
            outdir,
            metric_col="generation_to_95pct",
            name="speed_generation_to_95pct_final_by_method",
            ylabel="Generation to 95% of final improvement",
            title="Search-speed comparison: fewer generations is faster",
        )
        plot_speed_bars(
            speed,
            outdir,
            metric_col="eval_count_to_95pct",
            name="speed_eval_count_to_95pct_final_by_method",
            ylabel="Evaluations to 95% of final improvement",
            title="Evaluation efficiency: fewer evaluations is faster",
        )
        plot_speed_bars(
            speed,
            outdir,
            metric_col="elapsed_seconds_to_95pct",
            name="speed_seconds_to_95pct_final_by_method",
            ylabel="Seconds to 95% of final improvement",
            title="Wall-clock convergence speed: lower is faster",
        )
        plot_speed_bars(
            speed,
            outdir,
            metric_col="final_elapsed_seconds",
            name="runtime_seconds_final_by_method",
            ylabel="Final runtime seconds",
            title="Total runtime by method",
        )
        plot_runtime_vs_final_fitness(speed, outdir)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(by_method: pd.DataFrame, signed: pd.DataFrame, outdir: Path, convergence_available: bool) -> None:
    overall = (
        by_method.groupby("method", as_index=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            average_precision=("average_precision", "mean"),
            sign_accuracy_raw=("sign_accuracy", "mean"),
            n_pred_edges=("n_pred_edges", "mean"),
        )
        .sort_values("f1", ascending=False)
    )
    signed_rows = []
    for method, g in signed.groupby("method"):
        signed_rows.append({
            "method": method,
            "method_label": method_label(method),
            "mean_sign_accuracy": g["sign_accuracy"].mean(),
            "tp_weighted_sign_accuracy": weighted_mean_sign(g),
            "mean_f1": g["f1"].mean(),
            "mean_recall": g["recall"].mean(),
            "mean_precision": g["precision"].mean(),
        })
    signed_overall = pd.DataFrame(signed_rows)

    lines = []
    lines.append("# Sign accuracy and convergence analysis")
    lines.append("")
    lines.append("All linkage-correctness results in this report use `eval_scope == all_returned`.")
    lines.append("")
    lines.append("## How to read sign accuracy")
    lines.append("")
    lines.append("`sign_accuracy` is computed only on predicted edges that are also true edges. It checks whether the predicted sign equals the ground-truth sign. Therefore it is a direction/sign-quality metric, not a support-recovery metric.")
    lines.append("")
    lines.append("Important caveat: a method can have high sign accuracy and still be poor if it finds only a small or noisy subset of true edges. Always read sign accuracy together with precision, recall, F1, and the number of true positives.")
    lines.append("")
    lines.append("In the current evaluator, `empirical_linkage_legacy` may show `sign_accuracy = 0` because its predicted-edge output does not provide a meaningful sign column. This should be interpreted as not applicable, not as evidence that all signs are wrong.")
    lines.append("")
    lines.append("`top_cooccurrence` and `random_graph` use a constant positive sign in the baseline generator. If a benchmark has mostly positive true edges, their raw sign accuracy can look high even though they did not learn signed epistatic effects.")
    lines.append("")
    lines.append("## Overall all-returned metrics")
    lines.append("")
    for row in overall.itertuples(index=False):
        lines.append(
            f"- {method_label(row.method)}: F1={row.f1:.3f}, precision={row.precision:.3f}, "
            f"recall={row.recall:.3f}, AP={row.average_precision:.3f}, "
            f"raw sign accuracy={row.sign_accuracy_raw:.3f}, returned edges={row.n_pred_edges:.1f}."
        )
    lines.append("")
    lines.append("## Signed-method sign accuracy")
    lines.append("")
    if signed_overall.empty:
        lines.append("No signed methods were available after filtering.")
    else:
        for row in signed_overall.sort_values("tp_weighted_sign_accuracy", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.method_label}: mean sign accuracy={row.mean_sign_accuracy:.3f}, "
                f"TP-weighted sign accuracy={row.tp_weighted_sign_accuracy:.3f}, "
                f"F1={row.mean_f1:.3f}, precision={row.mean_precision:.3f}, recall={row.mean_recall:.3f}."
            )
    lines.append("")
    lines.append("## Metrics that should not be skipped")
    lines.append("")
    lines.append("- `average_precision`: ranking quality of the learned edge scores. This is useful because all-returned F1 depends on the chosen sparsity/threshold, while AP evaluates whether true edges are ranked above false edges.")
    lines.append("- `spearman_abs_weight`: whether predicted edge magnitudes are rank-correlated with true edge magnitudes. This matters if you claim that stronger learned edges correspond to stronger true interactions.")
    lines.append("- `n_pred_edges` / `n_eval_edges`: sparsity of the returned graph. Dense methods often get higher recall but lower precision.")
    lines.append("- `true_positives`, `false_positives`, `false_negatives`: these explain whether a method fails by adding too many wrong edges or by missing too many true edges.")
    lines.append("- Run-level variability across repeat/fold: a method with high mean but high variance may be unstable.")
    lines.append("- Missing dataset-method combinations: failed or empty runs can bias the overall mean.")
    lines.append("")
    lines.append("## Convergence-speed analysis")
    lines.append("")
    if convergence_available:
        lines.append("Convergence plots were generated from `generation_trace.csv`. The main speed metrics are generation/evaluation/seconds to reach 95% of the run's final improvement.")
    else:
        lines.append("The aggregate CSVs alone do not contain generation-level search traces, so they cannot measure convergence speed. To analyse convergence, rerun this script with `--results-root` pointing to the full results folder that contains `generation_trace.csv` and `run_summary.csv`.")
    lines.append("")
    (outdir / "analysis_report_sign_and_convergence.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path, help="Path to linkage_eval_summary.csv")
    parser.add_argument("--by-method", required=True, type=Path, help="Path to linkage_eval_by_method.csv")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for figures and analysis CSVs")
    parser.add_argument("--results-root", type=Path, default=None, help="Optional full results root containing linkage/**/generation_trace.csv")
    parser.add_argument(
        "--signed-methods",
        nargs="+",
        default=DEFAULT_SIGNED_METHODS,
        help="Methods whose sign_accuracy should be interpreted as learned signed coefficients.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary, by_method = load_all_returned(args.summary, args.by_method)
    raw, signed = make_sign_views(summary, by_method, args.signed_methods)

    write_sign_tables(summary, by_method, args.signed_methods, args.outdir)

    plot_sign_heatmap(
        raw,
        args.outdir,
        name="sign_accuracy_heatmap_raw_all_methods",
        title="Raw sign accuracy by dataset and method",
    )
    plot_sign_heatmap(
        signed,
        args.outdir,
        name="sign_accuracy_heatmap_signed_methods_only",
        title="Sign accuracy by dataset: signed Lasso methods only",
    )
    plot_sign_bar_by_family(signed, args.outdir)
    plot_sign_vs_f1(raw, args.outdir, signed_only=False)
    plot_sign_vs_f1(signed, args.outdir, signed_only=True)
    plot_sign_boxplot(summary, args.outdir, args.signed_methods)

    convergence_available = False
    if args.results_root is not None:
        run_convergence_analysis(args.results_root, args.outdir)
        convergence_available = not (args.outdir / "convergence_not_available.txt").exists()

    write_report(by_method, signed, args.outdir, convergence_available)
    print(f"Done. Outputs written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python analysis_interactive/analyze_linkage_correctness/plot_linkage_sign_and_convergence_analysis.py `
#   --summary "../Results/results_linkage_test_theory_v3/linkage_eval_summary.csv" `
#   --by-method "../Results/results_linkage_test_theory_v3/linkage_eval_by_method.csv" `
#   --results-root "../Results/results_linkage_test_theory_v3" `
#   --outdir ../Results/results_linkage_test_theory_v3/linkage_sign_convergence_figures