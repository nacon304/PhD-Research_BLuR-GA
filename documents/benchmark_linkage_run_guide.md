# BLuR-GA Linkage-Correctness Benchmark Guide

This guide is for the synthetic linkage-correctness lane only. It is intentionally separated from real-case performance evaluation and future building-block evaluation.

## 1. Generate benchmark datasets

Active implemented families in the current preset are:

- `pairwise`
- `ising`
- `nk`
- `maxsat`

```bash
python scripts/prepare_linkage_benchmarks.py \
  --output-root ../Dataset/prepared_linkage_benchmark \
  --preset main \
  --families all \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --overwrite-manifest
```

The generated layout is:

```text
prepared_linkage_benchmark/
  manifest.csv
  linkage/
    <dataset_key>/
      metadata.json
      problem.json
      feature_names.json
      node_labels.csv
      ground_truth_edges.csv
      ground_truth_matrix.csv
      ground_truth_graph_snapshots.csv
```

## 2. Plot ground-truth graphs

```bash
python scripts/plot_ground_truth_graphs.py \
  --dataset-root ../Dataset/prepared_linkage_benchmark \
  --layout auto \
  --scale 1.8 \
  --signed \
  --write-metrics \
  --format png
```

## 3. Run one linkage-evaluation task locally

```bash
python run_linkage_eval.py \
  --dataset-root ../Dataset/prepared_linkage_benchmark \
  --output-root Results/results_linkage_test \
  --datasets pairwise_qubo_d100_e150_posneg_noise0_seed0 \
  --ga-types 6 \
  --repeat-id 0 \
  --outer-fold-id 0 \
  --popsize 100 \
  --max-gen 200 \
  --lr-gap-gen 1 \
  --lr-min-samples 10 \
  --save-generation-trace true \
  --save-graph-snapshots true \
  --graph-snapshot-interval 1 \
  --include-baselines true \
  --k-values 50 150
```

The per-task output is race-safe:

```text
Results/results_linkage_test/linkage/<dataset_key>/<method>/rep00/fold00/run000/
  run_summary.csv
  generation_trace.csv
  selected_features.csv
  graph_snapshots.csv
  eVIG.csv
  eVIG_edges.csv
  predicted_edges.csv
  ground_truth_edges_used.csv
  linkage_eval.csv
```

No global summary is written during the task by default.

## 4. Aggregate after all tasks finish

```bash
python run_linkage_eval.py aggregate --results-root Results/results_linkage_test
```

This writes:

```text
Results/results_linkage_test/linkage_eval_summary.csv
Results/results_linkage_test/linkage_eval_by_method.csv
```

## 5. HPC task-manifest workflow

```bash
python run_linkage_eval.py make-tasks \
  --dataset-root ../Dataset/prepared_linkage_benchmark \
  --output-root Results/results_linkage_hpc \
  --datasets all \
  --ga-types 1 2 3 6 7 \
  --repeats 10 \
  --outer-folds 1 \
  --popsize 100 \
  --max-gen 200 \
  --lr-gap-gen 1 \
  --lr-min-samples 10 \
  --include-baselines true \
  --k-values 50 150 \
  --tasks-out tasks_linkage.csv

sbatch --array=0-<N-1> hpc/job_slurm_linkage_eval_task.sh tasks_linkage.csv
python run_linkage_eval.py aggregate --results-root Results/results_linkage_hpc
```

## 6. Metric scopes

`linkage_eval.csv` contains:

- `eval_scope=all_returned`: evaluate all edges returned by the method against ground truth.
- `eval_scope=top_k`: evaluate requested top-k prefixes.

For `top_k`, `n_eval_edges` can be smaller than `k_requested` when a sparse method returns fewer edges. `precision` uses the evaluated returned edges as denominator, while `precision_strict` uses the requested/effective k as denominator.
