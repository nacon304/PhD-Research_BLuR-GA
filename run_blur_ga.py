#!/usr/bin/env python3
"""Unified BLuR-GA launcher.

Compatibility is preserved:
    python run_blur_ga.py anneal_task363614 1 1 ...

New orchestration modes:
    python run_blur_ga.py batch ...       # local/sequential smoke or full batch
    python run_blur_ga.py make-tasks ...  # make a manifest for SLURM array jobs
    python run_blur_ga.py run-task ...    # run one manifest row
    python run_blur_ga.py aggregate ...   # rebuild root summaries from per-run files
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import run_blur_ga_oop

SUBCOMMANDS = {"batch", "make-tasks", "run-task", "aggregate"}

PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "popsize": 20,
        "max_gen": 5,
        "repeats": 1,
        "outer_folds": 3,
        "inner_folds": 2,
        "save_generation_trace": True,
        "save_graph_snapshots": True,
        "graph_snapshot_interval": 1,
        "graph_snapshot_top_k": 80,
        "save_linkage_events": False,
        "build_building_blocks": True,
        "bb_weight_mode": "absolute",
        "bb_gawll_update_interval": 1,
        "bb_snapshot_interval": 1,
        "artifact_layout": "root",
        "write_aggregate_outputs": True,
    },
    "analysis": {
        "popsize": 100,
        "max_gen": 100,
        "repeats": 2,
        "outer_folds": 2,
        "inner_folds": 3,
        "save_generation_trace": True,
        "save_graph_snapshots": True,
        "graph_snapshot_interval": 1,
        # "graph_snapshot_top_k": 120,
        "save_linkage_events": False,
        "build_building_blocks": True,
        "bb_weight_mode": "absolute",
        "bb_gawll_update_interval": 1,
        "bb_snapshot_interval": 1,
        "artifact_layout": "root",
        "write_aggregate_outputs": True,
    },
    "nesi_full": {
        "popsize": 100,
        "max_gen": 100,
        "repeats": 10,
        "outer_folds": 3,
        "inner_folds": 3,
        "save_generation_trace": True,
        "save_graph_snapshots": False,
        "save_linkage_events": False,
        "artifact_layout": "nested",
        "write_aggregate_outputs": False,
    },
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


@dataclass(frozen=True)
class Job:
    dataset: str
    classifier: int
    ga_type: int
    output_dir: Path
    repeat_id: int | None = None
    outer_fold_id: int | None = None


METHOD_FOLDER_NAMES: dict[int, str] = {
    0: "standard_ga",
    1: "empirical_linkage_legacy",
    2: "blur_ga_pairwise_lasso",
    3: "blur_ga_main_pairwise_lasso",
}


def method_folder_name(
    ga_type: int,
    *,
    bb_search_mode: str = "none",
    bb_weight_mode: str = "absolute",
    lr_encoding: str = "binary",
    pre_lr_explore: bool | str | None = None,
) -> str:
    """Human-readable method folder used under --output-root.

    File names keep the compact _a<ga_type> suffix, while directories use
    descriptive method labels.  When a BB-search operator is active, the folder
    name is suffixed so uniform and pattern_refine runs do not overwrite each
    other even if they share the same ga_type.  For regression-linkage methods,
    the LR encoding and pre-LR exploration flag are also included so binary/spin
    and pre/no-pre ablations cannot overwrite each other.
    """
    base = METHOD_FOLDER_NAMES.get(int(ga_type), f"method_{int(ga_type)}")
    mode = str(bb_search_mode or "none").strip().lower()
    if int(ga_type) in {1, 2, 3} and mode != "none":
        suffix = f"{mode}_{str(bb_weight_mode).strip().lower()}"
        if int(ga_type) in {2, 3}:
            suffix = f"{suffix}_{str(lr_encoding).strip().lower()}"
            if pre_lr_explore is not None:
                pre_bool = str(pre_lr_explore).strip().lower() in {"1", "true", "yes", "y"}
                suffix = f"{suffix}_{'pre' if pre_bool else 'nopre'}"
        return f"{base}_{suffix}"
    return base


def available_datasets(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    names: list[str] = []
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and ((child / "dataset.npz").exists() or ((child / "X_preprocessed.csv").exists() and (child / "y.csv").exists())):
            names.append(child.name)
        elif child.is_file() and child.suffix == ".dat":
            names.append(child.stem)
    if not names:
        raise FileNotFoundError(f"No datasets found in {data_dir}. Expected subfolders with dataset.npz, X_preprocessed.csv + y.csv, or .dat files.")
    return names


def read_dataset_file(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split(",")[0].strip())
    return names


def read_dataset_group(data_dir: Path, group_name: str) -> list[str]:
    group_path = data_dir / "dataset_groups" / f"{group_name}.txt"
    if not group_path.exists():
        available = sorted(p.stem for p in (data_dir / "dataset_groups").glob("*.txt")) if (data_dir / "dataset_groups").exists() else []
        raise FileNotFoundError(
            f"Dataset group {group_name!r} not found at {group_path}. "
            f"Available groups: {available or 'none'}"
        )
    return read_dataset_file(group_path)


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    data_dir = Path(args.data_dir)
    all_names = available_datasets(data_dir)
    dataset_group = getattr(args, "dataset_group", None)
    if args.dataset_file:
        requested = read_dataset_file(Path(args.dataset_file))
    elif dataset_group:
        requested = read_dataset_group(data_dir, str(dataset_group))
    elif args.datasets == ["all"]:
        requested = all_names
    else:
        requested = args.datasets
    exclude = set(args.exclude_datasets or [])
    out = [d for d in requested if d not in exclude]
    missing = [d for d in out if d not in all_names]
    if missing:
        raise ValueError(f"Dataset(s) not found in {args.data_dir}: {missing}")
    if args.limit_datasets is not None:
        out = out[: int(args.limit_datasets)]
    if not out:
        raise ValueError("No datasets selected.")
    return out


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return args
    preset = PRESETS[preset_name]
    # argparse stores None for optional arguments that were not supplied.  Only
    # fill those from the preset so explicit CLI values always win.
    for key, value in preset.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return args


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "classifiers": [1],
        "ga_types": [1],
        "repeats": 1,
        "outer_folds": 3,
        "inner_folds": 3,
        "seed": 1,
        "shuffle": True,
        "popsize": 100,
        "p_cross": 0.4,
        "ll_crossover_ratio": None,
        "mutation_prob": None,
        "tournament_size": 3,
        "tau_reset": 50,
        "resetpop_rate": 0.99,
        "local_search": False,
        "stop": "gen",
        "max_gen": 100,
        "max_time": None,
        "max_evals": None,
        "edge_epsilon": 1e-6,
        "save_generation_trace": True,
        "save_linkage_events": False,
        "save_graph_snapshots": True,
        "graph_snapshot_interval": 1,
        "graph_snapshot_top_k": None,
        "graph_snapshot_min_weight": 0.0,
        "no_fitness_cache": False,
        "lr_gap_gen": 1,
        "lr_min_samples": 10,
        "lr_sparse_alpha": 0.001,
        "lr_auto_alpha": True,
        "lr_alpha_c": 0.2,
        "lr_delta": 0.05,
        "lr_auto_min_samples": True,
        "lr_expected_edges": 60,
        "lr_refit_expected_edges": 30,
        "lr_min_samples_c": 2.0,
        "lr_stability_subsamples": 0,
        "lr_stability_fraction": 0.75,
        "lr_edge_min_weight": 0.0,
        "lr_edge_top_k": None,
        "lr_solver": "auto",
        "lr_encoding": "binary",
        "lr_matrix_free_threshold": 100,
        "lr_matrix_free_max_iter": 8,
        "lr_matrix_free_tol": 1e-4,
        "lr_matrix_free_step_scale": 0.5,
        "lr_matrix_free_dtype": "float32",
        "artifact_layout": "root",
        "write_aggregate_outputs": True,
        "build_building_blocks": False,
        "bb_weight_mode": "absolute",
        "bb_search_mode": "none",
        "bb_gawll_update_interval": None,
        "bb_min_block_size": 2,
        "bb_max_block_size": None,
        "bb_max_blocks": 20,
        "bb_snapshot_interval": 1,
        "bb_external_penalty": 0.25,
        "bb_size_penalty": 0.01,
        "bb_pattern_top_fraction": 0.30,
        "bb_pattern_min_support": 2,
        "bb_refine_fraction": 0.20,
        "bb_max_refine_trials": 10,
        "bb_accept_equal_sparser": True,
        "bb_shuffle_blocks": True,
        "bb_signed_repair_probability": 0.75,
        "pre_lr_explore": False,
        "pre_lr_mutation_multiplier": 3.0,
        "pre_lr_immigrant_rate": 0.0,
    }
    for key, value in defaults.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_jobs(args: argparse.Namespace, *, split_folds: bool) -> list[Job]:
    datasets = resolve_datasets(args)
    jobs: list[Job] = []
    for dataset in datasets:
        for classifier in args.classifiers:
            for ga_type in args.ga_types:
                out_dir = Path(args.output_root) / dataset / method_folder_name(
                    int(ga_type),
                    bb_search_mode=str(getattr(args, "bb_search_mode", "none")),
                    bb_weight_mode=str(getattr(args, "bb_weight_mode", "absolute")),
                    lr_encoding=str(getattr(args, "lr_encoding", "binary")),
                    pre_lr_explore=getattr(args, "pre_lr_explore", None),
                )
                if split_folds:
                    for rep in range(args.repeats):
                        for fold in range(args.outer_folds):
                            jobs.append(Job(dataset, int(classifier), int(ga_type), out_dir, rep, fold))
                else:
                    jobs.append(Job(dataset, int(classifier), int(ga_type), out_dir))
    return jobs


def common_oop_args(args: argparse.Namespace, job: Job) -> list[str]:
    # Batch-level BB flags are global, but standard GA (ga_type=0) has no
    # linkage graph and therefore cannot consume BB search operators.  Normalize
    # them per job so mixed batches such as ``--ga-types 0 1 2 3`` can compare
    # standard GA against BB-aware methods without failing configuration checks.
    linkage_capable = job.ga_type in (1, 2, 3)
    build_building_blocks = bool(args.build_building_blocks) and linkage_capable
    bb_search_mode = str(args.bb_search_mode) if linkage_capable else "none"

    cmd = [
        str(job.dataset),
        str(job.classifier),
        str(job.ga_type),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(job.output_dir),
        "--outer-folds", str(args.outer_folds),
        "--inner-folds", str(args.inner_folds),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--shuffle", str(bool(args.shuffle)).lower(),
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
        "--lr-sparse-alpha", str(args.lr_sparse_alpha),
        "--lr-auto-alpha", str(bool(args.lr_auto_alpha)).lower(),
        "--lr-alpha-c", str(args.lr_alpha_c),
        "--lr-delta", str(args.lr_delta),
        "--lr-auto-min-samples", str(bool(args.lr_auto_min_samples)).lower(),
        "--lr-min-samples-c", str(args.lr_min_samples_c),
        "--lr-stability-subsamples", str(args.lr_stability_subsamples),
        "--lr-stability-fraction", str(args.lr_stability_fraction),
        "--lr-edge-min-weight", str(args.lr_edge_min_weight),
        "--lr-encoding", str(args.lr_encoding),
        "--lr-solver", str(args.lr_solver),
        "--lr-matrix-free-threshold", str(args.lr_matrix_free_threshold),
        "--lr-matrix-free-max-iter", str(args.lr_matrix_free_max_iter),
        "--lr-matrix-free-tol", str(args.lr_matrix_free_tol),
        "--lr-matrix-free-step-scale", str(args.lr_matrix_free_step_scale),
        "--lr-matrix-free-dtype", str(args.lr_matrix_free_dtype),
        "--build-building-blocks", str(build_building_blocks).lower(),
        "--bb-weight-mode", str(args.bb_weight_mode),
        "--bb-search-mode", bb_search_mode,
        "--bb-min-block-size", str(args.bb_min_block_size),
        "--bb-max-blocks", str(args.bb_max_blocks),
        "--bb-snapshot-interval", str(args.bb_snapshot_interval),
        "--bb-external-penalty", str(args.bb_external_penalty),
        "--bb-size-penalty", str(args.bb_size_penalty),
        "--bb-pattern-top-fraction", str(args.bb_pattern_top_fraction),
        "--bb-pattern-min-support", str(args.bb_pattern_min_support),
        "--bb-refine-fraction", str(args.bb_refine_fraction),
        "--bb-accept-equal-sparser", str(bool(args.bb_accept_equal_sparser)).lower(),
        "--bb-shuffle-blocks", str(bool(args.bb_shuffle_blocks)).lower(),
        "--bb-signed-repair-probability", str(args.bb_signed_repair_probability),
        "--pre-lr-explore", str(bool(args.pre_lr_explore)).lower(),
        "--pre-lr-mutation-multiplier", str(args.pre_lr_mutation_multiplier),
        "--pre-lr-immigrant-rate", str(args.pre_lr_immigrant_rate),
        "--artifact-layout", str(args.artifact_layout),
        "--write-aggregate-outputs", str(bool(args.write_aggregate_outputs)).lower(),
    ]
    optional_pairs = [
        ("--ll-crossover-ratio", args.ll_crossover_ratio),
        ("--mutation-prob", args.mutation_prob),
        ("--max-time", args.max_time),
        ("--max-evals", args.max_evals),
        ("--graph-snapshot-top-k", args.graph_snapshot_top_k),
        ("--lr-edge-top-k", args.lr_edge_top_k),
        ("--lr-expected-edges", args.lr_expected_edges),
        ("--lr-refit-expected-edges", args.lr_refit_expected_edges),
        ("--bb-gawll-update-interval", args.bb_gawll_update_interval),
        ("--bb-max-block-size", args.bb_max_block_size),
        ("--bb-max-refine-trials", args.bb_max_refine_trials),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.no_fitness_cache:
        cmd.append("--no-fitness-cache")
    if job.repeat_id is not None:
        cmd.extend(["--repeat-id", str(job.repeat_id)])
    if job.outer_fold_id is not None:
        cmd.extend(["--outer-fold-id", str(job.outer_fold_id)])
    return cmd


def run_oop_args(oop_args: list[str]) -> None:
    run_blur_ga_oop.main(oop_args)


def add_shared_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--datasets", nargs="+", default=["all"], help="Dataset names, or 'all'.")
    p.add_argument("--dataset-group", default=None, help="Read dataset names from <data-dir>/dataset_groups/<group>.txt, e.g. small, medium, highdim, all.")
    p.add_argument("--dataset-file", default=None, help="Text/CSV file; first comma-separated field is treated as dataset name.")
    p.add_argument("--exclude-datasets", nargs="*", default=[])
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--classifiers", nargs="+", type=int, default=None)
    p.add_argument("--ga-types", nargs="+", type=int, default=None)
    p.add_argument("--repeats", "--repeat", dest="repeats", type=int, default=None)
    p.add_argument("--outer-folds", type=int, default=None)
    p.add_argument("--inner-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--shuffle", type=str2bool, default=None)
    p.add_argument("--popsize", type=int, default=None)
    p.add_argument("--p-cross", type=float, default=None)
    p.add_argument("--ll-crossover-ratio", type=float, default=None)
    p.add_argument("--mutation-prob", type=float, default=None)
    p.add_argument("--tournament-size", type=int, default=None)
    p.add_argument("--tau-reset", type=int, default=None)
    p.add_argument("--resetpop-rate", type=float, default=None)
    p.add_argument("--local-search", type=str2bool, default=None)
    p.add_argument("--stop", choices=["gen", "time", "eval"], default=None)
    p.add_argument("--max-gen", type=int, default=None)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--max-evals", type=int, default=None)
    p.add_argument("--edge-epsilon", type=float, default=None)
    p.add_argument("--save-generation-trace", type=str2bool, default=None)
    p.add_argument("--save-linkage-events", type=str2bool, default=None)
    p.add_argument("--save-graph-snapshots", type=str2bool, default=None)
    p.add_argument("--graph-snapshot-interval", type=int, default=None)
    p.add_argument("--graph-snapshot-top-k", type=int, default=None)
    p.add_argument("--graph-snapshot-min-weight", type=float, default=None)
    p.add_argument("--no-fitness-cache", action="store_true")
    p.add_argument("--lr-gap-gen", type=int, default=None)
    p.add_argument("--lr-min-samples", type=int, default=None)
    p.add_argument("--lr-sparse-alpha", type=float, default=None)
    p.add_argument("--lr-auto-alpha", type=str2bool, default=None)
    p.add_argument("--lr-alpha-c", type=float, default=None)
    p.add_argument("--lr-delta", type=float, default=None)
    p.add_argument("--lr-auto-min-samples", type=str2bool, default=None)
    p.add_argument("--lr-expected-edges", type=int, default=None)
    p.add_argument("--lr-refit-expected-edges", type=int, default=None)
    p.add_argument("--lr-min-samples-c", type=float, default=None)
    p.add_argument("--lr-stability-subsamples", type=int, default=None)
    p.add_argument("--lr-stability-fraction", type=float, default=None)
    p.add_argument("--lr-edge-min-weight", type=float, default=None)
    p.add_argument("--lr-edge-top-k", type=int, default=None)
    p.add_argument("--lr-encoding", choices=["binary", "spin"], default=None)
    p.add_argument("--lr-solver", choices=["auto", "sklearn_full", "dual_ridge", "matrix_free"], default=None)
    p.add_argument("--lr-matrix-free-threshold", type=int, default=None)
    p.add_argument("--lr-matrix-free-max-iter", type=int, default=None)
    p.add_argument("--lr-matrix-free-tol", type=float, default=None)
    p.add_argument("--lr-matrix-free-step-scale", type=float, default=None)
    p.add_argument("--lr-matrix-free-dtype", choices=["float64", "float32"], default=None)
    p.add_argument("--build-building-blocks", type=str2bool, default=None)
    p.add_argument("--bb-weight-mode", choices=["absolute", "signed", "positive"], default=None)
    p.add_argument("--bb-search-mode", choices=["none", "uniform", "pattern_refine"], default=None)
    p.add_argument("--bb-gawll-update-interval", type=int, default=None)
    p.add_argument("--bb-min-block-size", type=int, default=None)
    p.add_argument("--bb-max-block-size", type=int, default=None)
    p.add_argument("--bb-max-blocks", type=int, default=None)
    p.add_argument("--bb-snapshot-interval", type=int, default=None)
    p.add_argument("--bb-external-penalty", type=float, default=None)
    p.add_argument("--bb-size-penalty", type=float, default=None)
    p.add_argument("--bb-pattern-top-fraction", type=float, default=None)
    p.add_argument("--bb-pattern-min-support", type=int, default=None)
    p.add_argument("--bb-refine-fraction", type=float, default=None)
    p.add_argument("--bb-max-refine-trials", type=int, default=None)
    p.add_argument("--bb-accept-equal-sparser", type=str2bool, default=None)
    p.add_argument("--bb-shuffle-blocks", type=str2bool, default=None)
    p.add_argument("--bb-signed-repair-probability", type=float, default=None)
    p.add_argument("--pre-lr-explore", type=str2bool, default=None)
    p.add_argument("--pre-lr-mutation-multiplier", type=float, default=None)
    p.add_argument("--pre-lr-immigrant-rate", type=float, default=None)
    p.add_argument("--artifact-layout", choices=["both", "nested", "root"], default=None)
    p.add_argument("--write-aggregate-outputs", type=str2bool, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BLuR-GA launcher and batch/HPC orchestrator.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    batch = sub.add_parser("batch", help="Run selected datasets/methods sequentially in one Python command.")
    add_shared_options(batch)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--continue-on-error", action="store_true")

    make = sub.add_parser("make-tasks", help="Create a CSV manifest for SLURM array jobs.")
    add_shared_options(make)
    make.add_argument("--tasks-out", required=True)

    task = sub.add_parser("run-task", help="Run one row from a task manifest.")
    task.add_argument("--tasks", required=True)
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--dry-run", action="store_true")

    agg = sub.add_parser("aggregate", help="Rebuild root summary/vector/trace files from nested per-run artifacts.")
    agg.add_argument("--results-root", required=True)
    return parser


def cmd_batch(args: argparse.Namespace) -> int:
    args = fill_defaults(apply_preset(args))
    jobs = build_jobs(args, split_folds=False)
    print(f"Prepared {len(jobs)} dataset-method job(s).")
    failures = 0
    for idx, job in enumerate(jobs):
        oop_args = common_oop_args(args, job)
        print(f"[job {idx}] dataset={job.dataset} c={job.classifier} a={job.ga_type} out={job.output_dir}")
        print("  python run_blur_ga_oop.py " + " ".join(oop_args))
        if args.dry_run:
            continue
        try:
            run_oop_args(oop_args)
        except Exception as exc:
            failures += 1
            print(f"ERROR in job {idx}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    return 1 if failures else 0


def cmd_make_tasks(args: argparse.Namespace) -> int:
    args = fill_defaults(apply_preset(args))
    # Array jobs should be race-safe by default because several tasks may write into the same method folder.
    if args.artifact_layout != "nested":
        args.artifact_layout = "nested"
    if args.write_aggregate_outputs is True:
        args.write_aggregate_outputs = False
    jobs = build_jobs(args, split_folds=True)
    out = Path(args.tasks_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "dataset", "classifier", "ga_type", "repeat_id", "outer_fold_id", "output_dir", "command_json"])
        for task_id, job in enumerate(jobs):
            oop_args = common_oop_args(args, job)
            writer.writerow([
                task_id,
                job.dataset,
                job.classifier,
                job.ga_type,
                job.repeat_id,
                job.outer_fold_id,
                str(job.output_dir),
                json.dumps(oop_args),
            ])
    print(f"Wrote {len(jobs)} task(s): {out}")
    return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    with Path(args.tasks).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if int(r["task_id"]) == int(args.task_id)]
    if not matches:
        raise ValueError(f"task_id={args.task_id} not found in {args.tasks}")
    oop_args = json.loads(matches[0]["command_json"])
    print("python run_blur_ga_oop.py " + " ".join(oop_args))
    if args.dry_run:
        return 0
    run_oop_args(oop_args)
    return 0


def _read_one_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_rep_dir(name: str) -> bool:
    return name.startswith("rep") or name.startswith("repeat_")


def _is_fold_dir(name: str) -> bool:
    return name.startswith("fold") or name.startswith("outer_fold_")


def _is_run_dir(name: str) -> bool:
    return name.startswith("run")


def _is_nested_run_summary(path: Path) -> bool:
    parts = path.parts
    if len(parts) < 4:
        return False
    if path.name != "run_summary.csv" and not path.name.startswith("run_summary_"):
        return False
    return _is_run_dir(parts[-2]) and _is_fold_dir(parts[-3]) and _is_rep_dir(parts[-4])


def _method_dir_from_run_summary(path: Path) -> Path:
    if _is_nested_run_summary(path):
        return path.parents[3]
    return path.parent


def _prefix_from_summary_rows(rows: list[dict[str, str]], fallback_path: Path) -> str:
    if rows:
        first = rows[0]
        c = first.get("classifier_type")
        a = first.get("ga_type")
        if c not in (None, "") and a not in (None, ""):
            return f"c{int(float(c))}_a{int(float(a))}"
    stem = fallback_path.stem
    if stem.startswith("run_summary_"):
        body = stem[len("run_summary_"):]
        # New compact root files look like run_summary_c2_a1_rep00_fold00_run000.
        if "_rep" in body:
            return body.split("_rep", 1)[0]
        # Old root files looked like run_summary_<dataset>_c2_a1_r0.
        if "_c" in body and "_a" in body:
            tail = body.rsplit("_c", 1)[-1].rsplit("_r", 1)[0]
            return "c" + tail
        return body.rsplit("_r", 1)[0]
    return "c0_a0"


def _chrom_to_bind(chrom: str) -> str:
    return ", ".join(ch for ch in chrom.strip())


def _trace_path_for_summary(summary_path: Path) -> Path:
    return _artifact_path_for_summary(summary_path, "generation_trace")


def _artifact_path_for_summary(summary_path: Path, base: str) -> Path:
    if summary_path.name == "run_summary.csv":
        return summary_path.with_name(f"{base}.csv")
    if summary_path.name.startswith("run_summary_"):
        return summary_path.with_name(summary_path.name.replace("run_summary_", f"{base}_", 1))
    return summary_path.with_name(f"{base}.csv")


def _merge_optional_artifacts(summary_paths: list[Path], method_dir: Path, prefix: str, base: str) -> None:
    paths = [p for p in (_artifact_path_for_summary(s, base) for s in summary_paths) if p.exists()]
    if not paths:
        return
    rows: list[dict[str, str]] = []
    header: list[str] = []
    for path in paths:
        loaded = _read_one_csv(path)
        if not loaded:
            continue
        for col in loaded[0].keys():
            if col not in header:
                header.append(col)
        rows.extend(loaded)
    if not rows:
        return
    sort_cols = ["repeat_id", "outer_fold", "run_id", "generation", "block_id"]
    def key(row: dict[str, str]) -> tuple[int, ...]:
        return tuple(int(float(row.get(c, 0) or 0)) for c in sort_cols if c in row)
    rows.sort(key=key)
    with (method_dir / f"{base}_{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in header})


def cmd_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.results_root)
    nested = [p for p in root.rglob("run_summary.csv") if _is_nested_run_summary(p)]
    nested.extend(p for p in root.rglob("run_summary_*.csv") if _is_nested_run_summary(p))
    paths = sorted(set(nested))
    if not paths:
        paths = sorted(root.rglob("run_summary_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No run_summary CSV files found under {root}")

    groups: dict[tuple[Path, str], list[tuple[Path, list[dict[str, str]]]]] = {}
    for p in paths:
        rows = _read_one_csv(p)
        if not rows:
            continue
        prefix = _prefix_from_summary_rows(rows, p)
        groups.setdefault((_method_dir_from_run_summary(p), prefix), []).append((p, rows))

    for (method_dir, prefix), grouped in sorted(groups.items(), key=lambda kv: str(kv[0][0] / kv[0][1])):
        rows: list[dict[str, str]] = []
        summary_paths: list[Path] = []
        for p, loaded in grouped:
            summary_paths.append(p)
            rows.extend(loaded)
        rows.sort(key=lambda r: (int(r["repeat_id"]), int(r["outer_fold"]), int(r["run_id"])))
        method_dir.mkdir(parents=True, exist_ok=True)

        summary_cols = [
            "dataset", "classifier_type", "ga_type", "repeat_id", "outer_fold", "run_id", "seed",
            "best_inner_fitness", "outer_score", "subset_size", "n_edges", "generations", "runtime_seconds", "eval_count",
        ]
        with (method_dir / f"nested_summary_{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in summary_cols})

        def write_vector(name: str, key: str, fmt=lambda x: x) -> None:
            (method_dir / f"{name}_{prefix}.csv").write_text(", ".join(fmt(r[key]) for r in rows), encoding="utf-8")

        write_vector("bfi", "best_inner_fitness")
        write_vector("test_score", "outer_score")
        write_vector("time", "runtime_seconds", lambda x: f"{float(x):.2f}")
        write_vector("gen", "generations")
        write_vector("evals", "eval_count")
        if rows and int(float(rows[0].get("ga_type", "0"))) != 0:
            write_vector("nedges", "n_edges", lambda x: f"{float(x):.1f}")
        (method_dir / f"bind_{prefix}.csv").write_text("\n".join(_chrom_to_bind(r["best_chromosome"]) for r in rows), encoding="utf-8")

        with (method_dir / f"selected_features_{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["repeat_id", "outer_fold", "run_id", "feature_index"])
            for r in rows:
                feats = r.get("best_selected_features", "")
                if feats:
                    for feat in feats.split(";"):
                        if feat != "":
                            w.writerow([r["repeat_id"], r["outer_fold"], r["run_id"], int(feat)])

        trace_paths: list[Path] = []
        for p in summary_paths:
            t = _trace_path_for_summary(p)
            if t.exists():
                trace_paths.append(t)
        if trace_paths:
            trace_rows: list[dict[str, str]] = []
            header: list[str] = []
            for t in trace_paths:
                tr = _read_one_csv(t)
                if tr:
                    for col in tr[0].keys():
                        if col not in header:
                            header.append(col)
                    trace_rows.extend(tr)
            trace_rows.sort(key=lambda r: (int(r.get("repeat_id", 0)), int(r.get("outer_fold", 0)), int(r.get("run_id", 0)), int(r.get("generation", 0))))
            with (method_dir / f"generation_trace_{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in trace_rows:
                    w.writerow({c: r.get(c, "") for c in header})

        _merge_optional_artifacts(summary_paths, method_dir, prefix, "building_block_summary")
        _merge_optional_artifacts(summary_paths, method_dir, prefix, "building_blocks")

        print(f"Aggregated {len(rows)} run(s): {method_dir / ('nested_summary_' + prefix + '.csv')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: old positional command goes straight to the OOP runner.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        run_blur_ga_oop.main(argv)
        return 0
    args = build_parser().parse_args(argv)
    if args.cmd == "batch":
        return cmd_batch(args)
    if args.cmd == "make-tasks":
        return cmd_make_tasks(args)
    if args.cmd == "run-task":
        return cmd_run_task(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

"""
python run_blur_ga.py batch `
  --preset analysis `
  --data-dir ../Dataset/prepared_feature_selection `
  --output-root ../Results/results_analysis_med_high `
  --datasets colon_asu `
  --classifiers 2 `
  --ga-types 0 1 2 `
  --repeat 1 `
  --inner-folds 4 `
  --outer-folds 5 `
  --bb-pattern-top-fraction 0.20 `
  --build-building-blocks true `
  --bb-weight-mode positive `
  --bb-search-mode uniform `
  --bb-gawll-update-interval 10 `
  --pre-lr-explore true `
  --pre-lr-mutation-multiplier 3.0 `
  --pre-lr-immigrant-rate 0.10 `
  --lr-expected-edges 60 `
  --lr-refit-expected-edges 30 `
  --save-generation-trace true `
  --save-graph-snapshots false `
  --save-linkage-events false `
  --artifact-layout nested `
  --write-aggregate-outputs false

python run_blur_ga.py aggregate --results-root ../Results/results_analysis_test_v2

python analysis_interactive/make_graph_ui.py `
  --snapshots ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/graph_snapshots_c2_a1_rep00_fold00_run000.csv `
  --trace ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/generation_trace_c2_a1.csv `
  --selected-features ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/selected_features_c2_a1.csv `
  --run-id 0 `
  --output ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/graph_evolution_ui.html `
  --layout-k-scale 2.0 `
  --initial-spacing 1.25 `
  --layout spring

python analysis_interactive/make_graph_ui.py `
  --snapshots ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/graph_snapshots_c2_a2_rep00_fold00_run000.csv `
  --trace ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/generation_trace_c2_a2.csv `
  --selected-features ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/selected_features_c2_a2.csv `
  --run-id 0 `
  --output ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/graph_evolution_ui.html `
  --layout-k-scale 2.0 `
  --initial-spacing 1.25 `
  --layout spring

python analysis_interactive/make_graph_ui.py `
  --snapshots ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/graph_snapshots_c2_a3_rep00_fold00_run000.csv `
  --trace ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/generation_trace_c2_a3.csv `
  --selected-features ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/selected_features_c2_a3.csv `
  --run-id 0 `
  --output ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/graph_evolution_ui.html `
  --layout-k-scale 2.0 `
  --initial-spacing 1.25 `
  --layout spring
"""