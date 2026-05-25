# BLuR-GA project documentation - cleaned mode version

This document describes the cleaned project after removing unused GA modes.

## Active mode map

| `ga_type` | Method folder | Core behavior | Previous id |
|---:|---|---|---:|
| 0 | `standard_ga` | Standard GA baseline: tournament selection, crossover, mutation, optional reset/local search. | 0 |
| 1 | `empirical_linkage_legacy` | Legacy empirical linkage GA using `EmpiricalVIG`. | 1 |
| 2 | `blur_ga_pairwise_lasso` | Regression linkage using pairwise Lasso on bipolar pairwise terms `u_i u_j`. | 6 |
| 3 | `blur_ga_main_pairwise_lasso` | Regression linkage using partial main+pairwise Lasso; intercept and main effects are controls, pairwise terms become edges. | 7 |

Removed legacy modes: old `2`, `3`, `4`, and `5`.

## Main files

| File | Role |
|---|---|
| `run_blur_ga.py` | Batch/local/HPC launcher for feature-selection performance experiments. |
| `run_blur_ga_oop.py` | Direct nested-CV runner used by the launcher. |
| `run_linkage_eval.py` | Linkage-correctness runner for synthetic benchmark datasets. |
| `blur_ga/config.py` | Configuration dataclasses and validation. |
| `blur_ga/engine.py` | GA search loop, archive collection, empirical/regression graph updates. |
| `blur_ga/lr_linkage.py` | `RegressionVIG`, pairwise Lasso, and partial main+pairwise Lasso. |
| `blur_ga/results.py` | Per-run and aggregate CSV output writer. |

## Regression linkage pipeline

For `ga_type=2` and `ga_type=3`, the GA stores unique evaluated chromosomes and their fitness values in an archive. Every `--lr-gap-gen` generations, if the archive has enough new samples, `RegressionLinkageLearner` fits a regression model and updates `RegressionVIG`.

The response is the centered fitness value:

```text
y_reg = fitness - mean(fitness)
```

Pairwise Lasso (`ga_type=2`) builds bipolar features:

```text
u_i = 2 x_i - 1
Z_ij = u_i u_j
```

Main+pairwise Lasso (`ga_type=3`) first residualizes both `y` and all `Z_ij` columns against `[1, u_1, ..., u_d]`, then fits Lasso on the residualized pairwise design.

## Important CLI options

| Option | Meaning |
|---|---|
| `--ga-types` | Select methods from `0 1 2 3`. |
| `--lr-gap-gen` | Generation interval between regression graph refits. |
| `--lr-min-samples` | Minimum unique evaluated archive samples before fitting. |
| `--lr-sparse-alpha` | Manual Lasso penalty when auto-alpha is disabled. |
| `--lr-auto-alpha` | Use the PSLE/MPSLE-inspired lambda rule. |
| `--lr-alpha-c` | Multiplier for the auto-alpha lambda. |
| `--lr-delta` | Failure-probability parameter used by auto alpha/min-sample rules. |
| `--lr-auto-min-samples` | Enforce a theory-inspired minimum archive size. |
| `--lr-expected-edges` | Expected sparsity level for the auto min-sample rule. |
| `--lr-stability-subsamples` | Optional stability-selection subsampling refits. |
| `--lr-edge-top-k` | Keep only the top K regression edges in `RegressionVIG`. |

Removed CLI options: `--lr-ridge-alpha`, `--lr-l1-ratio`, and `--lr-excess-window`.

## Output naming

The directory uses the descriptive method folder, while files keep compact `a<ga_type>` suffixes. After cleanup, `a2` means pairwise Lasso and `a3` means main+pairwise Lasso.

```text
<output-root>/<dataset>/blur_ga_pairwise_lasso/rep00/fold00/run000/
<output-root>/<dataset>/blur_ga_main_pairwise_lasso/rep00/fold00/run000/
```

Common per-run files include:

| File | Meaning |
|---|---|
| `run_summary.csv` | Run metadata, best inner fitness, outer score, selected subset size. |
| `selected_features.csv` | Final selected binary mask. |
| `generation_trace.csv` | Best/mean fitness and graph metadata by generation. |
| `archive.csv` | Unique evaluated chromosomes and fitness values. |
| `eVIG.csv` | Learned graph adjacency matrix for linkage methods. |
| `eVIG_coefficients.csv` | Signed regression coefficient matrix for Lasso modes. |
| `eVIG_edges.csv` | Edge list with weights, coefficients, stability, and generation metadata. |
| `graph_snapshots.csv` | Graph evolution snapshots when enabled. |

## Example performance run

```bash
python run_blur_ga.py batch \
  --data-dir ../Dataset/prepared_tabarena \
  --output-root ../Results/results_analysis_test_v2 \
  --datasets anneal_task363614 \
  --classifiers 2 \
  --ga-types 0 1 2 3 \
  --repeats 1 --outer-folds 3 --inner-folds 3 \
  --popsize 100 --max-gen 200 \
  --save-generation-trace true --save-graph-snapshots true
```

## Example linkage run

```bash
python run_linkage_eval.py run \
  --dataset-root ../Dataset/prepared_linkage_benchmark_mini_debug \
  --output-root ../Results/results_linkage_test \
  --datasets ising_spin_glass_grid5x5_posneg_seed0 \
  --ga-types 0 1 2 3 \
  --repeat-id 0 --outer-fold-id 0 \
  --popsize 40 --max-gen 20 \
  --lr-gap-gen 1 --lr-min-samples 10
```
