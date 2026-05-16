#!/usr/bin/env python3
from __future__ import annotations

"""HPC-safe linkage-correctness runner for BLuR-GA.

One HPC task should mean exactly:

    one repeat x one outer fold x one dataset x one method

The runner therefore writes only per-run artifacts by default.  Use the explicit
``aggregate`` subcommand after all array jobs finish to build global summaries.

Main workflows
--------------
Run one task directly::

    python run_linkage_eval.py \
      --dataset-root ../Dataset/prepared_linkage_benchmark \
      --output-root Results/results_linkage \
      --datasets pairwise_qubo_d100_e150_posneg_noise0_seed0 \
      --ga-types 6 --repeat-id 0 --outer-fold-id 0

Create and run an HPC task manifest::

    python run_linkage_eval.py make-tasks ... --tasks-out tasks_linkage.csv
    sbatch --array=0-<N-1> hpc/job_slurm_linkage_eval_task.sh tasks_linkage.csv
    python run_linkage_eval.py aggregate --results-root Results/results_linkage
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from blur_ga.config import ExperimentConfig, GAConfig
from blur_ga.data import DatParser
from blur_ga.evaluation.baselines import cooccurrence_edges, random_edges
from blur_ga.evaluation.linkage import evig_to_edges, evaluate_predicted_edges, load_true_edges, write_dataframe
from blur_ga.evaluation.synthetic_experiment import run_synthetic_landscape
from blur_ga.experiment import NestedCVExperiment
from blur_ga.output_policy import _normalise_groups
from blur_ga.results import EvaluatedRun, ResultWriter

SUBCOMMANDS = {"run", "make-tasks", "run-task", "aggregate"}

METHOD_FOLDER_NAMES: dict[int, str] = {
    0: "standard_ga",
    1: "empirical_linkage_legacy",
    2: "blur_ga_stage1_pairwise",
    3: "blur_ga_stage2_main_pairwise",
    4: "blur_ga_stage3_sparse",
    5: "blur_ga_stage4_sparse_excess",
    6: "blur_ga_pairwise_lasso",
    7: "blur_ga_main_pairwise_lasso",
}


@dataclass(frozen=True)
class LinkageTask:
    dataset: str
    ga_type: int
    repeat_id: int
    outer_fold_id: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def method_folder_name(ga_type: int) -> str:
    return METHOD_FOLDER_NAMES.get(int(ga_type), f"method_{int(ga_type)}")


def read_manifest(dataset_root: str | Path) -> list[dict[str, str]]:
    manifest = Path(dataset_root) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_manifest_rows(dataset_root: str | Path, datasets: list[str]) -> list[dict[str, str]]:
    rows = read_manifest(dataset_root)
    if datasets == ["all"]:
        return rows
    wanted = set(datasets)
    selected = [r for r in rows if r.get("dataset_key") in wanted]
    missing = wanted - {r.get("dataset_key") for r in selected}
    if missing:
        raise KeyError(f"Dataset(s) not found in manifest: {sorted(missing)}")
    return selected


def build_ga_config(args: argparse.Namespace, ga_type: int) -> GAConfig:
    ll_ratio = args.p_cross if args.ll_crossover_ratio is None else args.ll_crossover_ratio
    return GAConfig(
        classifier_type=args.classifier,
        ga_type=int(ga_type),
        popsize=args.popsize,
        crossover_probability=args.p_cross,
        ll_crossover_ratio=ll_ratio,
        mutation_probability=args.mutation_prob,
        tournament_size=args.tournament_size,
        tau_reset=args.tau_reset,
        resetpop_rate=args.resetpop_rate,
        local_search=args.local_search,
        stop=args.stop,
        max_gen=args.max_gen,
        max_time=args.max_time,
        max_evals=args.max_evals,
        edge_epsilon=args.edge_epsilon,
        save_generation_trace=args.save_generation_trace,
        save_linkage_events=args.save_linkage_events,
        save_graph_snapshots=args.save_graph_snapshots,
        graph_snapshot_interval=args.graph_snapshot_interval,
        graph_snapshot_top_k=args.graph_snapshot_top_k,
        graph_snapshot_min_weight=args.graph_snapshot_min_weight,
        cache_fitness=not args.no_fitness_cache,
        lr_gap_gen=args.lr_gap_gen,
        lr_min_samples=args.lr_min_samples,
        lr_ridge_alpha=args.lr_ridge_alpha,
        lr_sparse_alpha=args.lr_sparse_alpha,
        lr_l1_ratio=args.lr_l1_ratio,
        lr_stability_subsamples=args.lr_stability_subsamples,
        lr_stability_fraction=args.lr_stability_fraction,
        lr_edge_min_weight=args.lr_edge_min_weight,
        lr_edge_top_k=args.lr_edge_top_k,
        lr_excess_window=args.lr_excess_window,
    )


def run_dir_for(output_root: Path, row: dict[str, str], ga_type: int, repeat_id: int, outer_fold: int, run_id: int) -> tuple[Path, Path]:
    dataset_key = str(row["dataset_key"])
    method_dir = output_root / "linkage" / dataset_key / method_folder_name(ga_type)
    run_dir = method_dir / f"rep{repeat_id:02d}" / f"fold{outer_fold:02d}" / f"run{run_id:03d}"
    return method_dir, run_dir


def compact_run_prefix(ga_type: int) -> str:
    # Dataset and method are already encoded in the directory path.
    return f"c0_a{int(ga_type)}"


def save_archive(result, path: Path, *, repeat_id: int, outer_fold: int, run_id: int) -> None:
    X = result.archive_chromosomes
    y = result.archive_fitness
    gens = result.archive_generations
    if X is None or y is None or gens is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat_id", "outer_fold", "run_id", "archive_index", "generation", "fitness", "chromosome", "selected_features"])
        for idx, (chrom, fit, gen) in enumerate(zip(X, y, gens)):
            chrom_str = "".join(str(int(v)) for v in chrom)
            selected = ";".join(str(int(i)) for i in np.flatnonzero(np.asarray(chrom) == 1))
            writer.writerow([repeat_id, outer_fold, run_id, idx, int(gen), f"{float(fit):.14f}", chrom_str, selected])


def write_linkage_outputs(
    *,
    row: dict[str, str],
    ga_type: int,
    result,
    method_name: str,
    repeat_id: int,
    outer_fold: int,
    run_id: int,
    seed: int,
    output_root: Path,
    dataset_root: Path,
    k_values: list[int] | None,
    include_baselines: bool,
    include_all_returned: bool,
) -> pd.DataFrame:
    _, run_dir = run_dir_for(output_root, row, ga_type, repeat_id, outer_fold, run_id)
    dataset_key = str(row["dataset_key"])
    true_path = dataset_root / str(row["ground_truth_edges_path"])
    true_edges = load_true_edges(true_path)
    n_features = int(float(row.get("n_features") or 0))

    pred = evig_to_edges(result.evig, method=method_name, dataset=dataset_key, generation=result.generations)
    write_dataframe(pred, run_dir / "predicted_edges.csv")
    write_dataframe(true_edges, run_dir / "ground_truth_edges_used.csv")
    metrics = evaluate_predicted_edges(
        pred,
        true_edges,
        dataset=dataset_key,
        method=method_name,
        repeat_id=repeat_id,
        outer_fold=outer_fold,
        run_id=run_id,
        seed=seed,
        n_features=n_features,
        k_values=k_values,
        include_all_returned=include_all_returned,
    )
    write_dataframe(metrics, run_dir / "linkage_eval.csv")
    save_archive(result, run_dir / "archive.csv", repeat_id=repeat_id, outer_fold=outer_fold, run_id=run_id)

    if include_baselines and result.archive_chromosomes is not None and result.archive_fitness is not None:
        baseline_frames = [
            random_edges(n_features, max(1, len(pred)), dataset=dataset_key, seed=seed),
            cooccurrence_edges(result.archive_chromosomes, result.archive_fitness, dataset=dataset_key),
        ]
        for bpred in baseline_frames:
            bmethod = "baseline" if bpred.empty else str(bpred["method"].iloc[0])
            write_dataframe(bpred, run_dir / f"predicted_edges_{bmethod}.csv")
            bmetrics = evaluate_predicted_edges(
                bpred,
                true_edges,
                dataset=dataset_key,
                method=bmethod,
                repeat_id=repeat_id,
                outer_fold=outer_fold,
                run_id=run_id,
                seed=seed,
                n_features=n_features,
                k_values=k_values,
                include_all_returned=include_all_returned,
            )
            write_dataframe(bmetrics, run_dir / f"linkage_eval_{bmethod}.csv")
            metrics = pd.concat([metrics, bmetrics], ignore_index=True)
    return metrics


def _linkage_eval_files(output_root: Path) -> list[Path]:
    linkage_root = output_root / "linkage"
    if not linkage_root.exists():
        return []
    return sorted(
        p for p in linkage_root.glob("**/linkage_eval*.csv")
        if p.name not in {"linkage_eval_summary.csv", "linkage_eval_by_method.csv"}
    )


def aggregate_linkage_results(output_root: str | Path) -> tuple[Path, Path] | None:
    output_root = Path(output_root)
    files = _linkage_eval_files(output_root)
    if not files:
        return None
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    out = output_root / "linkage_eval_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    numeric_candidates = [
        "precision", "recall", "f1", "precision_strict",
        "precision_at_k", "recall_at_k", "f1_at_k", "precision_at_k_strict",
        "average_precision", "spearman_abs_weight", "sign_accuracy",
        "n_eval_edges", "n_pred_edges", "true_positives", "false_positives", "false_negatives",
    ]
    numeric = [c for c in numeric_candidates if c in df.columns]
    group_cols = [c for c in ["dataset", "method", "eval_scope", "k_requested", "k_effective"] if c in df.columns]
    by_method = output_root / "linkage_eval_by_method.csv"
    if group_cols and numeric:
        df.groupby(group_cols, dropna=False, as_index=False)[numeric].mean().to_csv(by_method, index=False)
    else:
        df.to_csv(by_method, index=False)
    return out, by_method


def run_one(args: argparse.Namespace, row: dict[str, str], ga_type: int, repeat_id: int, outer_fold: int) -> pd.DataFrame:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    dataset_key = str(row["dataset_key"])
    evaluator_type = str(row.get("evaluator_type") or "synthetic_landscape")
    output_groups = (
        _normalise_groups(args.output_groups, str(row.get("eval_group") or "linkage"))
        if args.output_groups
        else _normalise_groups(row.get("output_groups"), str(row.get("eval_group") or "linkage"))
    )
    ga_cfg = build_ga_config(args, ga_type)

    if evaluator_type == "synthetic_landscape":
        run = run_synthetic_landscape(
            problem_json=dataset_root / row["problem_path"],
            ga_config=ga_cfg,
            repeat_id=repeat_id,
            outer_fold=outer_fold,
            base_seed=args.seed,
            n_outer_folds=args.outer_folds,
        )
        result = run.run_result
        run_id = run.run_id
        seed = run.seed
        method_dir, _ = run_dir_for(output_root, row, ga_type, repeat_id, outer_fold, run_id)
        writer = ResultWriter(
            method_dir,
            dataset_key,
            0,
            ga_type,
            artifact_layout=args.artifact_layout,
            prefix_override=compact_run_prefix(ga_type),
        )
        erow = EvaluatedRun(
            dataset=dataset_key,
            classifier_type=0,
            ga_type=ga_type,
            repeat_id=repeat_id,
            outer_fold=outer_fold,
            run_id=run_id,
            seed=seed,
            best_inner_fitness=result.best_inner_fitness,
            outer_score=result.best_inner_fitness,
            subset_size=result.subset_size,
            n_edges=result.n_edges,
            generations=result.generations,
            runtime_seconds=result.runtime_seconds,
            eval_count=result.eval_count,
            chromosome=result.best_chromosome,
            run_result=result,
        )
        writer.add(erow)
        writer.save_run_artifacts(erow)
    else:
        # Supervised synthetic linkage datasets reuse the existing nested-CV pipeline.
        data_path = dataset_root / str(row.get("prepared_folder", ""))
        dataset = DatParser.read(data_path, search_dir=dataset_root)
        n_outer = int(float(row.get("n_outer_folds") or args.outer_folds))
        method_dir, _ = run_dir_for(output_root, row, ga_type, repeat_id, outer_fold, repeat_id * n_outer + outer_fold)
        exp_cfg = ExperimentConfig(
            outer_folds=n_outer,
            inner_folds=args.inner_folds,
            repeats=args.repeats,
            seed=args.seed,
            shuffle=args.shuffle,
            output_dir=method_dir,
            artifact_layout=args.artifact_layout,
            write_aggregate_outputs=False,
        )
        writer = NestedCVExperiment(dataset, ga_cfg, exp_cfg).run(outer_fold_id=outer_fold, repeat_id=repeat_id)
        if not writer.rows:
            raise RuntimeError(f"No supervised run row produced for {dataset_key}")
        erow = writer.rows[0]
        result = erow.run_result
        assert result is not None
        run_id = erow.run_id
        seed = erow.seed

    if "linkage" not in output_groups:
        return pd.DataFrame()
    return write_linkage_outputs(
        row=row,
        ga_type=ga_type,
        result=result,
        method_name=method_folder_name(ga_type),
        repeat_id=repeat_id,
        outer_fold=outer_fold,
        run_id=run_id,
        seed=seed,
        output_root=output_root,
        dataset_root=dataset_root,
        k_values=args.k_values,
        include_baselines=args.include_baselines,
        include_all_returned=args.include_all_returned,
    )


def add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset-root", required=True, help="Root folder containing manifest.csv and linkage/ prepared datasets.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--ga-types", nargs="+", type=int, default=[1, 2])
    p.add_argument("--classifier", type=int, choices=[1, 2], default=1, help="Only used by supervised synthetic datasets.")
    p.add_argument("--repeat-id", type=int, default=0)
    p.add_argument("--outer-fold-id", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--outer-folds", type=int, default=1)
    p.add_argument("--inner-folds", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shuffle", type=str2bool, default=True)
    p.add_argument("--output-groups", default=None, help="Comma-separated override, e.g. linkage,performance.")
    p.add_argument("--artifact-layout", choices=["both", "nested", "root"], default="nested")
    p.add_argument("--include-baselines", type=str2bool, default=True)
    p.add_argument("--include-all-returned", type=str2bool, default=True, help="Always write all-returned edge correctness row.")
    p.add_argument("--k-values", nargs="+", type=int, default=None, help="Optional top-k metrics. All-returned metrics are written separately by default.")
    p.add_argument("--write-aggregate-summary", type=str2bool, default=False, help="For local debugging only. HPC tasks should keep this false and run the aggregate subcommand later.")
    p.add_argument("--popsize", type=int, default=40)
    p.add_argument("--p-cross", type=float, default=0.4)
    p.add_argument("--ll-crossover-ratio", type=float, default=None)
    p.add_argument("--mutation-prob", type=float, default=None)
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--tau-reset", type=int, default=50)
    p.add_argument("--resetpop-rate", type=float, default=0.99)
    p.add_argument("--local-search", type=str2bool, default=False)
    p.add_argument("--stop", choices=["gen", "time", "eval"], default="gen")
    p.add_argument("--max-gen", type=int, default=20)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--max-evals", type=int, default=None)
    p.add_argument("--edge-epsilon", type=float, default=1e-6)
    p.add_argument("--save-generation-trace", type=str2bool, default=True)
    p.add_argument("--save-linkage-events", type=str2bool, default=False)
    p.add_argument("--save-graph-snapshots", type=str2bool, default=True)
    p.add_argument("--graph-snapshot-interval", type=int, default=1)
    p.add_argument("--graph-snapshot-top-k", type=int, default=None)
    p.add_argument("--graph-snapshot-min-weight", type=float, default=0.0)
    p.add_argument("--no-fitness-cache", action="store_true")
    p.add_argument("--lr-gap-gen", type=int, default=2)
    p.add_argument("--lr-min-samples", type=int, default=20)
    p.add_argument("--lr-ridge-alpha", type=float, default=1e-6)
    p.add_argument("--lr-sparse-alpha", type=float, default=0.001)
    p.add_argument("--lr-l1-ratio", type=float, default=0.95)
    p.add_argument("--lr-stability-subsamples", type=int, default=0)
    p.add_argument("--lr-stability-fraction", type=float, default=0.75)
    p.add_argument("--lr-edge-min-weight", type=float, default=0.0)
    p.add_argument("--lr-edge-top-k", type=int, default=None)
    p.add_argument("--lr-excess-window", type=int, default=5)


def build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one or more linkage-correctness tasks. Default output is HPC-safe per-run artifacts only.")
    add_run_options(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BLuR-GA linkage-correctness runner and HPC orchestrator.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run selected linkage task(s). Equivalent to the legacy no-subcommand mode.")
    add_run_options(run)

    make = sub.add_parser("make-tasks", help="Create a CSV manifest: one row = one repeat x one outer fold x one dataset x one method.")
    add_run_options(make)
    make.add_argument("--tasks-out", required=True)
    make.add_argument("--dry-run", action="store_true")

    task = sub.add_parser("run-task", help="Run one row from a linkage task manifest.")
    task.add_argument("--tasks", required=True)
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--dry-run", action="store_true")

    agg = sub.add_parser("aggregate", help="Rebuild linkage summaries after all per-run jobs have finished.")
    agg.add_argument("--results-root", required=True)
    return parser


def _run_command_from_args(args: argparse.Namespace, *, dataset: str, ga_type: int, repeat_id: int, outer_fold_id: int) -> list[str]:
    cmd = [
        "run",
        "--dataset-root", str(args.dataset_root),
        "--output-root", str(args.output_root),
        "--datasets", str(dataset),
        "--ga-types", str(int(ga_type)),
        "--classifier", str(args.classifier),
        "--repeat-id", str(int(repeat_id)),
        "--outer-fold-id", str(int(outer_fold_id)),
        "--repeats", str(args.repeats),
        "--outer-folds", str(args.outer_folds),
        "--inner-folds", str(args.inner_folds),
        "--seed", str(args.seed),
        "--shuffle", str(bool(args.shuffle)).lower(),
        "--artifact-layout", str(args.artifact_layout),
        "--include-baselines", str(bool(args.include_baselines)).lower(),
        "--include-all-returned", str(bool(args.include_all_returned)).lower(),
        "--write-aggregate-summary", "false",
        "--popsize", str(args.popsize),
        "--p-cross", str(args.p_cross),
        "--tournament-size", str(args.tournament_size),
        "--tau-reset", str(args.tau_reset),
        "--resetpop-rate", str(args.resetpop_rate),
        "--local-search", str(bool(args.local_search)).lower(),
        "--stop", str(args.stop),
        "--max-gen", str(args.max_gen),
        "--edge-epsilon", str(args.edge_epsilon),
        "--save-generation-trace", str(bool(args.save_generation_trace)).lower(),
        "--save-linkage-events", str(bool(args.save_linkage_events)).lower(),
        "--save-graph-snapshots", str(bool(args.save_graph_snapshots)).lower(),
        "--graph-snapshot-interval", str(args.graph_snapshot_interval),
        "--graph-snapshot-min-weight", str(args.graph_snapshot_min_weight),
        "--lr-gap-gen", str(args.lr_gap_gen),
        "--lr-min-samples", str(args.lr_min_samples),
        "--lr-ridge-alpha", str(args.lr_ridge_alpha),
        "--lr-sparse-alpha", str(args.lr_sparse_alpha),
        "--lr-l1-ratio", str(args.lr_l1_ratio),
        "--lr-stability-subsamples", str(args.lr_stability_subsamples),
        "--lr-stability-fraction", str(args.lr_stability_fraction),
        "--lr-edge-min-weight", str(args.lr_edge_min_weight),
        "--lr-excess-window", str(args.lr_excess_window),
    ]
    optional_pairs = [
        ("--output-groups", args.output_groups),
        ("--k-values", args.k_values),
        ("--ll-crossover-ratio", args.ll_crossover_ratio),
        ("--mutation-prob", args.mutation_prob),
        ("--max-time", args.max_time),
        ("--max-evals", args.max_evals),
        ("--graph-snapshot-top-k", args.graph_snapshot_top_k),
        ("--lr-edge-top-k", args.lr_edge_top_k),
    ]
    for flag, value in optional_pairs:
        if value is None:
            continue
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    if args.no_fitness_cache:
        cmd.append("--no-fitness-cache")
    return cmd


def cmd_run(args: argparse.Namespace) -> int:
    rows = select_manifest_rows(args.dataset_root, args.datasets)
    print(f"Prepared {len(rows) * len(args.ga_types)} linkage task(s).")
    all_metrics: list[pd.DataFrame] = []
    for row in rows:
        for ga_type in args.ga_types:
            print(f"[linkage] dataset={row['dataset_key']} ga_type={ga_type} repeat={args.repeat_id} fold={args.outer_fold_id}")
            metrics = run_one(args, row, ga_type, args.repeat_id, args.outer_fold_id)
            if not metrics.empty:
                all_metrics.append(metrics)
                cols = [c for c in ["dataset", "method", "eval_scope", "k_requested", "n_eval_edges", "precision", "recall", "f1", "precision_strict", "average_precision", "n_pred_edges"] if c in metrics.columns]
                print(metrics[cols].to_string(index=False))
    if args.write_aggregate_summary:
        summary = aggregate_linkage_results(Path(args.output_root))
        if summary:
            print(f"Wrote aggregate linkage summary: {summary[0]}")
    return 0


def cmd_make_tasks(args: argparse.Namespace) -> int:
    args.artifact_layout = "nested" if args.artifact_layout == "both" else args.artifact_layout
    args.write_aggregate_summary = False
    rows = select_manifest_rows(args.dataset_root, args.datasets)
    tasks: list[LinkageTask] = []
    for row in rows:
        dataset_key = str(row["dataset_key"])
        for ga_type in args.ga_types:
            for rep in range(int(args.repeats)):
                for fold in range(int(args.outer_folds)):
                    tasks.append(LinkageTask(dataset_key, int(ga_type), rep, fold))
    out = Path(args.tasks_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "dataset", "ga_type", "repeat_id", "outer_fold_id", "command_json"])
        for task_id, task in enumerate(tasks):
            cmd = _run_command_from_args(args, dataset=task.dataset, ga_type=task.ga_type, repeat_id=task.repeat_id, outer_fold_id=task.outer_fold_id)
            writer.writerow([task_id, task.dataset, task.ga_type, task.repeat_id, task.outer_fold_id, json.dumps(cmd)])
    print(f"Wrote {len(tasks)} linkage task(s): {out}")
    if args.dry_run and tasks:
        first = tasks[0]
        example = _run_command_from_args(args, dataset=first.dataset, ga_type=first.ga_type, repeat_id=first.repeat_id, outer_fold_id=first.outer_fold_id)
        print("Example command:")
        print("python run_linkage_eval.py " + " ".join(example))
    return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    with Path(args.tasks).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if int(r["task_id"]) == int(args.task_id)]
    if not matches:
        raise ValueError(f"task_id={args.task_id} not found in {args.tasks}")
    cmd = json.loads(matches[0]["command_json"])
    print("python run_linkage_eval.py " + " ".join(cmd))
    if args.dry_run:
        return 0
    return main(cmd)


def cmd_aggregate(args: argparse.Namespace) -> int:
    summary = aggregate_linkage_results(args.results_root)
    if not summary:
        raise FileNotFoundError(f"No linkage_eval*.csv files found under {Path(args.results_root) / 'linkage'}")
    print(f"Aggregated linkage metrics: {summary[0]}")
    print(f"Aggregated by method:       {summary[1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        # Unknown positional command: let argparse produce a useful error.
        args = build_parser().parse_args(argv)
    elif argv and argv[0] in SUBCOMMANDS:
        args = build_parser().parse_args(argv)
    else:
        # Backward-compatible no-subcommand mode:
        #   python run_linkage_eval.py --dataset-root ...
        args = build_run_parser().parse_args(argv)
        args.cmd = "run"

    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "make-tasks":
        return cmd_make_tasks(args)
    if args.cmd == "run-task":
        return cmd_run_task(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

# python run_linkage_eval.py `
#   --dataset-root ../Dataset/prepared_linkage_benchmark_mini_debug `
#   --output-root Results/results_linkage_test `
#   --datasets all `
#   --ga-types 1 6 7 `
#   --repeat-id 0 `
#   --outer-fold-id 0 `
#   --popsize 100 `
#   --max-gen 200 `
#   --lr-gap-gen 1 `
#   --lr-min-samples 10 `
#   --save-generation-trace true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1 `
#   --include-baselines true `
#   --k-values 100

# python run_linkage_eval.py aggregate `
#   --results-root Results/results_linkage_test

# $RUN = "Results/results_linkage_test/linkage/maxsat_d35_m90_k3_weighted_seed0/blur_ga_main_pairwise_lasso/rep00/fold00/run000"
# $RUN = "Results/results_linkage_test/linkage/maxsat_d35_m90_k3_weighted_seed0/blur_ga_pairwise_lasso/rep00/fold00/run000"
# $RUN = "Results/results_linkage_test/linkage/maxsat_d35_m90_k3_weighted_seed0/empirical_linkage_legacy/rep00/fold00/run000"

# python analysis_interactive/make_graph_ui.py `
#   --snapshots "$RUN/graph_snapshots.csv" `
#   --trace "$RUN/generation_trace.csv" `
#   --selected-features "$RUN/selected_features.csv" `
#   --run-id 0 `
#   --repeat-id 0 `
#   --outer-fold 0 `
#   --output "$RUN/graph_evolution_ui.html" `
#   --layout spring `
#   --step 1