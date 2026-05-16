#!/usr/bin/env python3
from __future__ import annotations

"""Run linkage-correctness evaluation on prepared linkage benchmark datasets.

HPC unit: one dataset x one method/ga_type x one outer_fold x one repeat.
The output layer is intentionally split from the older performance outputs:

  <output-root>/linkage/<dataset>/<method>/repeat_XX/outer_fold_XX/run_XXX/
      run_summary_*.csv              # old-compatible per-run summary
      eVIG_*.csv, eVIG_edges_*.csv    # old-compatible graph artifacts
      graph_snapshots_*.csv           # old-compatible HTML/UI input
      linkage_eval_*.csv              # new linkage correctness metrics
      predicted_edges_*.csv           # learned edge scores
      ground_truth_edges_used_*.csv   # exact ground truth used by metrics

Use --output-groups to override dataset metadata.  This lets a real-case dataset
emit performance, linkage, interpretability, etc. later without changing the HPC
task definition.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from blur_ga.config import ExperimentConfig, GAConfig
from blur_ga.data import DatParser
from blur_ga.experiment import NestedCVExperiment
from blur_ga.results import EvaluatedRun, ResultWriter
from blur_ga.output_policy import _normalise_groups
from blur_ga.evaluation.synthetic_experiment import run_synthetic_landscape
from blur_ga.evaluation.linkage import evig_to_edges, evaluate_predicted_edges, load_true_edges, write_dataframe
from blur_ga.evaluation.baselines import cooccurrence_edges, random_edges

METHOD_FOLDER_NAMES: dict[int, str] = {
    0: "standard_ga",
    1: "empirical_linkage_legacy",
    2: "blur_ga_stage1_pairwise",
    3: "blur_ga_stage2_main_pairwise",
    4: "blur_ga_stage3_sparse",
    5: "blur_ga_stage4_sparse_excess",
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
    """Short per-run filename prefix for linkage outputs.

    The dataset and method are already encoded in the directory path.  Keeping
    long dataset keys inside every CSV filename can exceed Windows MAX_PATH
    for normal project folders, so linkage evaluation uses this compact stem.
    Existing performance outputs still use the legacy full prefix.
    """
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
) -> pd.DataFrame:
    _, run_dir = run_dir_for(output_root, row, ga_type, repeat_id, outer_fold, run_id)
    dataset_key = str(row["dataset_key"])
    true_path = dataset_root / str(row["ground_truth_edges_path"])
    true_edges = load_true_edges(true_path)
    n_features = int(float(row.get("n_features") or 0))
    prefix = compact_run_prefix(ga_type)

    pred = evig_to_edges(result.evig, method=method_name, dataset=dataset_key, generation=result.generations)
    pred_path = run_dir / "predicted_edges.csv"
    eval_path = run_dir / "linkage_eval.csv"
    truth_path = run_dir / "ground_truth_edges_used.csv"
    archive_path = run_dir / "archive.csv"
    write_dataframe(pred, pred_path)
    write_dataframe(true_edges, truth_path)
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
    )
    write_dataframe(metrics, eval_path)
    save_archive(result, archive_path, repeat_id=repeat_id, outer_fold=outer_fold, run_id=run_id)

    if include_baselines and result.archive_chromosomes is not None and result.archive_fitness is not None:
        baseline_frames = [
            random_edges(n_features, max(1, len(pred)), dataset=dataset_key, seed=seed),
            cooccurrence_edges(result.archive_chromosomes, result.archive_fitness, dataset=dataset_key),
        ]
        for bpred in baseline_frames:
            if bpred.empty:
                bmethod = "baseline"
            else:
                bmethod = str(bpred["method"].iloc[0])
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
            )
            write_dataframe(bmetrics, run_dir / f"linkage_eval_{bmethod}.csv")
            metrics = pd.concat([metrics, bmetrics], ignore_index=True)
    return metrics


def write_aggregate_summary(output_root: Path) -> Path | None:
    files = []
    if (output_root / "linkage").exists():
        files = [p for p in (output_root / "linkage").glob("**/linkage_eval*.csv") if p.name != "linkage_eval_summary.csv"]
    if not files:
        return None
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    out = output_root / "linkage_eval_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    by_method = output_root / "linkage_eval_by_method.csv"
    numeric = ["precision_at_k", "recall_at_k", "f1_at_k", "average_precision", "spearman_abs_weight", "sign_accuracy", "n_pred_edges"]
    df.groupby(["dataset", "method", "k"], as_index=False)[numeric].mean().to_csv(by_method, index=False)
    return out


def run_one(args: argparse.Namespace, row: dict[str, str], ga_type: int, repeat_id: int, outer_fold: int) -> pd.DataFrame:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    dataset_key = str(row["dataset_key"])
    evaluator_type = str(row.get("evaluator_type") or "synthetic_landscape")
    output_groups = _normalise_groups(args.output_groups, str(row.get("eval_group") or "linkage")) if args.output_groups else _normalise_groups(row.get("output_groups"), str(row.get("eval_group") or "linkage"))
    ga_cfg = build_ga_config(args, ga_type)

    if evaluator_type == "synthetic_landscape":
        run = run_synthetic_landscape(
            problem_json=dataset_root / row["problem_path"],
            ga_config=ga_cfg,
            repeat_id=repeat_id,
            outer_fold=outer_fold,
            base_seed=args.seed,
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
    metrics = write_linkage_outputs(
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
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run linkage correctness evaluation for prepared BLuR-GA linkage benchmarks.")
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
    p.add_argument("--k-values", nargs="+", type=int, default=None)
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
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = select_manifest_rows(args.dataset_root, args.datasets)
    all_metrics: list[pd.DataFrame] = []
    print(f"Prepared {len(rows) * len(args.ga_types)} linkage task(s).")
    for row in rows:
        for ga_type in args.ga_types:
            print(f"[linkage] dataset={row['dataset_key']} ga_type={ga_type} repeat={args.repeat_id} fold={args.outer_fold_id}")
            metrics = run_one(args, row, ga_type, args.repeat_id, args.outer_fold_id)
            if not metrics.empty:
                all_metrics.append(metrics)
                print(metrics[["dataset", "method", "k", "precision_at_k", "recall_at_k", "f1_at_k", "average_precision", "n_pred_edges"]].to_string(index=False))
    summary = write_aggregate_summary(Path(args.output_root))
    if summary:
        print(f"Wrote aggregate linkage summary: {summary}")


if __name__ == "__main__":
    main()

# python run_linkage_eval.py `
# --dataset-root ../Dataset/prepared_linkage_benchmark `
# --output-root Results/results_linkage_test `
# --datasets pairwise_qubo_d100_e150_posneg_noise0_seed0 `
# --ga-types 1 2 `
# --repeat-id 0 `
# --outer-fold-id 0 `
# --popsize 20 `
# --max-gen 5 `
# --lr-gap-gen 1 `
# --lr-min-samples 10 `
# --save-generation-trace true `
# --save-graph-snapshots true `
# --graph-snapshot-interval 1 `
# --include-baselines true `
# --k-values 50 150