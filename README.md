# BLuR-GA - NeSI/HPC-ready runner

**BLuR-GA: Building-block Linkage using Regression for Genetic Algorithms**.

This repository currently supports three separated experiment lanes:

1. **Linkage correctness** on synthetic ground-truth linkage datasets via `run_linkage_eval.py`.
2. **Building-block evaluation** as a future/independent dataset group.
3. **Performance / real-case feature-selection evaluation** on prepared tabular datasets via `run_blur_ga.py`.

The intended HPC unit is:

```text
one repeat x one outer fold x one dataset x one method
```

Per-task runs write nested, race-safe artifacts only. Aggregate summaries are rebuilt after all tasks finish.

## Main entry points

- `run_blur_ga.py`: local/HPC launcher for supervised feature-selection performance experiments.
- `run_linkage_eval.py`: local/HPC launcher for linkage-correctness experiments.
- `run_blur_ga_oop.py`: direct nested-CV runner used internally by `run_blur_ga.py`.
- `analysis_interactive/make_graph_ui.py`: dependency-light interactive HTML graph viewer.
- `hpc/job_slurm_blur_ga_task.sh`: SLURM array runner for `run_blur_ga.py` task manifests.
- `hpc/job_slurm_linkage_eval_task.sh`: SLURM array runner for `run_linkage_eval.py` task manifests.

## Method folders under `--output-root`

| `ga_type` | Method folder                  | Meaning                                    |
| --------: | ------------------------------ | ------------------------------------------ |
| 0 | `standard_ga`                  | Standard GA baseline                       |
| 1 | `empirical_linkage_legacy`     | Legacy empirical-linkage variant           |
| 2 | `blur_ga_stage1_pairwise`      | Pairwise-only regression linkage           |
| 3 | `blur_ga_stage2_main_pairwise` | Main effects + pairwise regression linkage |
| 4 | `blur_ga_stage3_sparse`        | Sparse ElasticNet/LASSO-style linkage      |
| 5 | `blur_ga_stage4_sparse_excess` | Sparse linkage with excess-fitness response|
| 6 | `blur_ga_pairwise_lasso`       | Pairwise Lasso linkage                     |
| 7 | `blur_ga_main_pairwise_lasso`  | Main-controlled pairwise Lasso linkage     |

## Output layout

Nested per-run layout is the default for both launchers:

```text
<output-root>/<dataset>/<method>/rep00/fold00/run000/
<output-root>/linkage/<dataset>/<method>/rep00/fold00/run000/
```

Per-run files use compact names such as:

```text
run_summary.csv
generation_trace.csv
selected_features.csv
graph_snapshots.csv
eVIG.csv
eVIG_edges.csv
linkage_eval.csv              # linkage runner only
predicted_edges.csv           # linkage runner only
ground_truth_edges_used.csv   # linkage runner only
```

After all array jobs finish, use aggregate commands to rebuild root summaries.

## Environment

```bash
python -m venv .venv_blur_ga
source .venv_blur_ga/bin/activate
pip install -r requirements_nesi.txt
```

Runtime dependencies are intentionally small: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `networkx`.

## Local performance smoke test: one anneal run

Use direct fold mode when you want exactly one repeat x one outer fold x one dataset x one method. `outer_folds` must still be at least 2 because this is a nested-CV experiment.

```bash
python run_blur_ga.py anneal_task363614 2 2 \
  --data-dir ../Dataset/prepared_tabarena \
  --output-dir Results/results_anneal_smoke/anneal_task363614/blur_ga_stage1_pairwise \
  --outer-folds 2 \
  --inner-folds 2 \
  --repeats 1 \
  --repeat-id 0 \
  --outer-fold-id 0 \
  --popsize 20 \
  --max-gen 5 \
  --lr-gap-gen 1 \
  --lr-min-samples 10 \
  --save-generation-trace true \
  --save-graph-snapshots true \
  --graph-snapshot-interval 1 \
  --artifact-layout nested \
  --write-aggregate-outputs false

python run_blur_ga.py aggregate --results-root Results/results_anneal_smoke
```

Graph HTML example for the nested run:

```bash
python analysis_interactive/make_graph_ui.py \
  --snapshots Results/results_anneal_smoke/anneal_task363614/blur_ga_stage1_pairwise/rep00/fold00/run000/graph_snapshots.csv \
  --trace Results/results_anneal_smoke/anneal_task363614/blur_ga_stage1_pairwise/rep00/fold00/run000/generation_trace.csv \
  --selected-features Results/results_anneal_smoke/anneal_task363614/blur_ga_stage1_pairwise/rep00/fold00/run000/selected_features.csv \
  --run-id 0 \
  --output Results/results_anneal_smoke/anneal_task363614/blur_ga_stage1_pairwise/graph_evolution_ui.html \
  --layout spring
```

## Local linkage-correctness smoke test

```bash
python run_linkage_eval.py \
  --dataset-root ../Dataset/prepared_linkage_benchmark_mini_debug \
  --output-root Results/results_linkage_smoke \
  --datasets pairwise_qubo_d30_e30_posneg_noise0_seed0 \
  --ga-types 6 7 \
  --repeat-id 0 \
  --outer-fold-id 0 \
  --repeats 1 \
  --outer-folds 1 \
  --popsize 20 \
  --max-gen 5 \
  --lr-gap-gen 1 \
  --lr-min-samples 10 \
  --include-baselines true \
  --k-values 10 30

python run_linkage_eval.py aggregate --results-root Results/results_linkage_smoke
```

`linkage_eval.csv` now always contains an `all_returned` row by default. Requested `--k-values` add separate `top_k` rows. This avoids forcing a fixed `k` when a sparse method returns fewer edges.

## HPC workflow: performance / real-case datasets

```bash
python run_blur_ga.py make-tasks \
  --preset nesi_full \
  --data-dir ../Dataset/prepared_tabarena \
  --output-root results_nesi \
  --datasets all \
  --classifiers 1 \
  --ga-types 0 1 2 3 4 5 6 7 \
  --repeats 10 \
  --outer-folds 3 \
  --inner-folds 3 \
  --tasks-out tasks_tabarena.csv

sbatch --array=0-<N-1> hpc/job_slurm_blur_ga_task.sh tasks_tabarena.csv
python run_blur_ga.py aggregate --results-root results_nesi
```

## HPC workflow: linkage correctness

```bash
python run_linkage_eval.py make-tasks \
  --dataset-root ../Dataset/prepared_linkage_benchmark \
  --output-root results_linkage \
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
python run_linkage_eval.py aggregate --results-root results_linkage
```

## Linkage metric interpretation

`linkage_eval.csv` has two evaluation scopes:

- `eval_scope=all_returned`: evaluates every edge returned by the method. This is the default sparse-graph correctness score.
- `eval_scope=top_k`: evaluates requested top-k prefixes. If a method returns fewer than `k` edges, `n_eval_edges` shows how many were actually available. `precision` is computed over returned/evaluated edges, while `precision_strict` is computed over the requested/effective `k`.

This gives both sparse-quality and rank-at-k views without mixing their meanings.
