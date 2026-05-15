#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from blur_ga.config import ExperimentConfig, GAConfig
from blur_ga.data import DatParser
from blur_ga.experiment import NestedCVExperiment


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean BLuR-GA runner with nested cross-validation. "
            "Fitness is computed only on inner validation folds; outer test folds are used once for final evaluation."
        )
    )
    parser.add_argument("problem", help="Dataset name without .dat, or explicit .dat path")
    parser.add_argument("classifier", type=int, choices=[1, 2], help="1: KNN-3, 2: KNN-5")
    parser.add_argument("ga_type", type=int, choices=[0, 1, 2, 3, 4, 5], help="0: standard GA, 1: empirical linkage GA, 2: pairwise LR, 3: main+pairwise LR, 4: sparse, 5: sparse+excess response")
    parser.add_argument("--data-dir", default=".", help="Directory containing <problem>.dat")
    parser.add_argument("--output-dir", default="results_nested", help="Directory for result CSV files")
    parser.add_argument("--artifact-layout", choices=["both", "nested", "root"], default="both", help="Where to write per-run artifacts. both preserves old root files and nested HPC-safe files.")
    parser.add_argument("--write-aggregate-outputs", type=str2bool, default=True, help="Write root aggregate files such as nested_summary, bind, generation_trace. Use false for parallel HPC array tasks and aggregate later.")
    parser.add_argument("--outer-folds", type=int, default=3, help="Number of outer CV folds")
    parser.add_argument("--inner-folds", type=int, default=3, help="Number of inner CV folds")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated outer CV runs")
    parser.add_argument("--outer-fold-id", type=int, default=None, help="Run only one outer fold, 0-based. Useful for HPC arrays.")
    parser.add_argument("--repeat-id", type=int, default=None, help="Run only one repeat, 0-based. Useful for HPC arrays.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle before CV splitting")
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--p-cross", type=float, default=0.4, help="Standard GA crossover probability")
    parser.add_argument("--ll-crossover-ratio", type=float, default=None, help="Linkage-learning crossover offspring ratio. Defaults to --p-cross.")
    parser.add_argument("--mutation-prob", type=float, default=None, help="Default: 1 / n_features")
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--tau-reset", type=int, default=50)
    parser.add_argument("--resetpop-rate", type=float, default=0.99)
    parser.add_argument("--local-search", type=str2bool, default=False)
    parser.add_argument("--stop", choices=["gen", "time", "eval"], default="gen")
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--edge-epsilon", type=float, default=1e-6)
    parser.add_argument("--save-generation-trace", type=str2bool, default=False)
    parser.add_argument("--save-linkage-events", type=str2bool, default=False, help="For linkage-aware variants, save every tested pair event from linkage mutation")
    parser.add_argument("--save-graph-snapshots", type=str2bool, default=False, help="For linkage-aware variants, save cumulative edge lists at generation snapshots")
    parser.add_argument("--graph-snapshot-interval", type=int, default=1, help="Save graph snapshot every N generations")
    parser.add_argument("--graph-snapshot-top-k", type=int, default=None, help="Optionally save only top-k edges per snapshot")
    parser.add_argument("--graph-snapshot-min-weight", type=float, default=0.0, help="Only save snapshot edges with weight >= this value")
    parser.add_argument("--no-fitness-cache", action="store_true", help="Disable chromosome-level fitness cache")

    # Regression-linkage options for ga_type 2--5.
    parser.add_argument("--lr-gap-gen", type=int, default=5, help="Fit/update the LR linkage graph every N generations")
    parser.add_argument("--lr-min-samples", type=int, default=20, help="Minimum unique evaluated solutions before fitting LR linkage")
    parser.add_argument("--lr-ridge-alpha", type=float, default=1e-6, help="Small ridge penalty for ga_type 2/3; use 0 for pure OLS-like fitting")
    parser.add_argument("--lr-sparse-alpha", type=float, default=0.001, help="ElasticNet/LASSO alpha for ga_type 4/5")
    parser.add_argument("--lr-l1-ratio", type=float, default=0.95, help="ElasticNet L1 ratio for ga_type 4/5; 1.0 approximates LASSO")
    parser.add_argument("--lr-stability-subsamples", type=int, default=0, help="Number of subsampling refits for stability selection in ga_type 4/5")
    parser.add_argument("--lr-stability-fraction", type=float, default=0.75, help="Fraction of archive used per stability-selection subsample")
    parser.add_argument("--lr-edge-min-weight", type=float, default=0.0, help="Drop LR edges with final importance below this value")
    parser.add_argument("--lr-edge-top-k", type=int, default=None, help="Keep only the top-K LR edges in the final graph")
    parser.add_argument("--lr-excess-window", type=int, default=5, help="Rolling generation window for baseline-standardized excess response")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = DatParser.read(args.problem, search_dir=args.data_dir)
    ll_ratio = args.p_cross if args.ll_crossover_ratio is None else args.ll_crossover_ratio
    ga_cfg = GAConfig(
        classifier_type=args.classifier,
        ga_type=args.ga_type,
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
    exp_cfg = ExperimentConfig(
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        repeats=args.repeats,
        seed=args.seed,
        shuffle=args.shuffle,
        output_dir=Path(args.output_dir),
        artifact_layout=args.artifact_layout,
        write_aggregate_outputs=args.write_aggregate_outputs,
    )
    print("\n ***** BLuR-GA / Nested CV Runner *****")
    print(f"Dataset: {dataset.name} | samples={dataset.n_samples} | features={dataset.n_features}")
    print(f"GA type: {ga_cfg.ga_type} | classifier: KNN-{ga_cfg.knn_k}")
    if ga_cfg.ga_type in {2, 3, 4, 5}:
        print(f"LR linkage: gap_gen={ga_cfg.lr_gap_gen}, min_samples={ga_cfg.lr_min_samples}, edge_top_k={ga_cfg.lr_edge_top_k}")
    print(f"Nested CV: outer={exp_cfg.outer_folds}, inner={exp_cfg.inner_folds}, repeats={exp_cfg.repeats}")
    print(f"Output: artifact_layout={exp_cfg.artifact_layout}, aggregate_outputs={exp_cfg.write_aggregate_outputs}")
    if args.outer_fold_id is not None:
        print(f"HPC fold mode: repeat={args.repeat_id if args.repeat_id is not None else 'all'}, outer_fold={args.outer_fold_id}")
    experiment = NestedCVExperiment(dataset, ga_cfg, exp_cfg)
    writer = experiment.run(outer_fold_id=args.outer_fold_id, repeat_id=args.repeat_id)
    print(f"Done. Wrote {len(writer.rows)} evaluated run(s) to: {writer.output_dir}")
    if writer.rows:
        mean_outer = sum(r.outer_score for r in writer.rows) / len(writer.rows)
        mean_subset = sum(r.subset_size for r in writer.rows) / len(writer.rows)
        print(f"Mean outer score: {mean_outer:.6f} | mean subset size: {mean_subset:.2f}")


if __name__ == "__main__":
    main()

# python run_blur_ga.py anneal_task363614 1 1 `
#   --data-dir ../Dataset/prepared_tabarena `
#   --output-dir ps_empirical_linkage `
#   --outer-folds 3 `
#   --inner-folds 3 `
#   --repeats 3 `
#   --save-generation-trace true `
#   --save-linkage-events true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1

# python run_blur_ga.py anneal_task363614 1 2 `
#   --data-dir ../Dataset/prepared_tabarena `
#   --output-dir ps_stage2_pairwise `
#   --outer-folds 3 `
#   --inner-folds 3 `
#   --repeats 3 `
#   --lr-gap-gen 1 `
#   --lr-min-samples 1000 `
#   --save-generation-trace true `
#   --save-linkage-events true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1
#   --lr-edge-top-k 120 `

# python run_blur_ga.py anneal_task363614 1 3 `
#   --data-dir ../Dataset/prepared_tabarena `
#   --output-dir ps_stage3_main_pair `
#   --outer-folds 3 `
#   --inner-folds 3 `
#   --repeats 3 `
#   --lr-gap-gen 1 `
#   --lr-min-samples 1000 `
#   --save-generation-trace true `
#   --save-linkage-events true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1
#   --lr-edge-top-k 120 `

# python run_blur_ga.py anneal_task363614 1 4 `
#   --data-dir ../Dataset/prepared_tabarena `
#   --output-dir ps_stage4_sparse `
#   --outer-folds 3 `
#   --inner-folds 3 `
#   --repeats 3 `
#   --lr-gap-gen 1 `
#   --lr-min-samples 20 `
#   --lr-sparse-alpha 0.05 `
#   --lr-l1-ratio 0.7 `
#   --lr-edge-top-k 120 `
#   --save-generation-trace true `
#   --save-linkage-events true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1

# python run_blur_ga.py anneal_task363614 1 5 `
#   --data-dir ../Dataset/prepared_tabarena `
#   --output-dir ps_stage5_final `
#   --outer-folds 3 `
#   --inner-folds 3 `
#   --repeats 3 `
#   --lr-gap-gen 1 `
#   --lr-min-samples 30 `
#   --lr-sparse-alpha 0.5 `
#   --lr-l1-ratio 0.8 `
#   --lr-edge-min-weight 0.05 `
#   --save-generation-trace true `
#   --save-linkage-events true `
#   --save-graph-snapshots true `
#   --graph-snapshot-interval 1

# python analysis_interactive/make_graph_ui.py `
#   --snapshots ps_empirical_linkage/graph_snapshots_anneal_task363614_c1_a1_r0.csv `
#   --trace ps_empirical_linkage/generation_trace_anneal_task363614_c1_a1.csv `
#   --selected-features ps_empirical_linkage/selected_features_anneal_task363614_c1_a1.csv `
#   --run-id 0 `
#   --output ps_empirical_linkage/graph_evolution_ui.html `
#   --layout spring

# python analysis_interactive/make_graph_ui.py `
#   --snapshots ps_stage2_pairwise/graph_snapshots_anneal_task363614_c1_a2_r0.csv `
#   --trace ps_stage2_pairwise/generation_trace_anneal_task363614_c1_a2.csv `
#   --selected-features ps_stage2_pairwise/selected_features_anneal_task363614_c1_a2.csv `
#   --run-id 0 `
#   --output ps_stage2_pairwise/graph_evolution_ui.html `
#   --layout spring

# python analysis_interactive/make_graph_ui.py `
#   --snapshots ps_stage3_main_pair/graph_snapshots_anneal_task363614_c1_a3_r0.csv `
#   --trace ps_stage3_main_pair/generation_trace_anneal_task363614_c1_a3.csv `
#   --selected-features ps_stage3_main_pair/selected_features_anneal_task363614_c1_a3.csv `
#   --run-id 0 `
#   --output ps_stage3_main_pair/graph_evolution_ui.html `
#   --layout spring