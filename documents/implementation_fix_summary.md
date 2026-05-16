# Implementation Fix Summary

Date: 2026-05-16

## 1. Consistent output layout

The project now consistently uses nested per-run folders:

```text
repXX/foldXX/runXXX/
```

for both performance runs and linkage-correctness runs. Documentation, README examples, and graph-UI examples were updated accordingly.

## 2. Linkage-correctness metrics

`blur_ga/evaluation/linkage.py` now reports two separate metric scopes:

- `eval_scope=all_returned`: evaluate all edges returned by the method against the true edges.
- `eval_scope=top_k`: evaluate requested top-k prefixes from `--k-values`.

Important columns:

- `n_eval_edges`: number of predicted edges actually evaluated in the row.
- `precision`: `TP / n_eval_edges`.
- `recall`: `TP / n_true_edges`.
- `f1`: harmonic mean of `precision` and `recall`.
- `precision_strict`: for `top_k`, `TP / k_effective`; useful when sparse methods return fewer than the requested k.
- Backward-compatible aliases are preserved: `precision_at_k`, `recall_at_k`, `f1_at_k`, and `precision_at_k_strict`.

## 3. HPC-safe linkage runner

`run_linkage_eval.py` is now split into explicit modes:

```bash
python run_linkage_eval.py ...                  # backward-compatible run mode
python run_linkage_eval.py run ...              # explicit run mode
python run_linkage_eval.py make-tasks ...       # one task = repeat x outer fold x dataset x method
python run_linkage_eval.py run-task ...         # run one task-manifest row
python run_linkage_eval.py aggregate ...        # aggregate after all tasks finish
```

Per-task linkage runs no longer write global summaries by default. This avoids write races on HPC. Global files are only produced by:

```bash
python run_linkage_eval.py aggregate --results-root <output-root>
```

which writes:

```text
<linkage-output-root>/linkage_eval_summary.csv
<linkage-output-root>/linkage_eval_by_method.csv
```

## 4. HPC scripts

Added:

```text
hpc/job_slurm_linkage_eval_task.sh
```

Updated:

```text
hpc/submit_tabarena_manifest.sh
hpc/job_slurm_blur_ga_nested_fold.sh
```

to include `ga_type=6` and `ga_type=7` Lasso methods.

## 5. Synthetic landscape run IDs

`blur_ga/evaluation/synthetic_experiment.py` now computes synthetic-landscape `run_id` as:

```python
repeat_id * n_outer_folds + outer_fold
```

instead of assuming one outer fold internally.

## 6. Verified smoke checks

The following checks were run successfully in the sandbox:

- Python compile check for all project `.py` files.
- Linkage run on `pairwise_qubo_d30_e30_posneg_noise0_seed0` with `ga_type=6` and `ga_type=7`.
- Linkage aggregate command.
- Linkage task-manifest generation and `run-task --dry-run`.
- Anneal one-fold performance run on `anneal_task363614`, `classifier=2`, `ga_type=2`.
- Performance aggregate command.
- Graph HTML generation for the anneal run via `analysis_interactive/make_graph_ui.py`.
