# BLuR-GA - NeSI/HPC-ready runner

**BLuR-GA: Building-block Linkage using Regression for Genetic Algorithms**.

This cleaned version keeps only the active GA modes and removes the unused dense-ridge / ElasticNet / excess-response experimental modes.

## Active `ga_type` mapping

| `ga_type` | Method folder | Meaning | Previous id |
|---:|---|---|---:|
| 0 | `standard_ga` | Standard GA baseline | 0 |
| 1 | `empirical_linkage_legacy` | Legacy empirical-linkage variant | 1 |
| 2 | `blur_ga_pairwise_lasso` | Theory-aligned pairwise Lasso linkage with bipolar pair terms | 6 |
| 3 | `blur_ga_main_pairwise_lasso` | Theory-aligned partial main+pairwise Lasso; main effects are unpenalized controls | 7 |

Removed modes: old `2`, `3`, `4`, and `5`.

## Main entry points

- `run_blur_ga.py`: local/HPC launcher for supervised feature-selection performance experiments.
- `run_linkage_eval.py`: local/HPC launcher for linkage-correctness experiments.
- `run_blur_ga_oop.py`: direct nested-CV runner used internally by `run_blur_ga.py`.
- `analysis_interactive/make_graph_ui.py`: interactive graph viewer.

## Example: performance smoke run

```bash
python run_blur_ga.py batch \
  --data-dir ../Dataset/prepared_tabarena \
  --output-root ../Results/results_anneal_smoke \
  --datasets anneal_task363614 \
  --classifiers 2 \
  --ga-types 0 1 2 3 \
  --repeats 1 --outer-folds 3 --inner-folds 2 \
  --popsize 20 --max-gen 5
```

## Example: linkage-correctness run

```bash
python run_linkage_eval.py run \
  --dataset-root ../Dataset/prepared_linkage_benchmark_mini_debug \
  --output-root ../Results/results_linkage_smoke \
  --datasets ising_spin_glass_grid5x5_posneg_seed0 \
  --ga-types 0 1 2 3 \
  --repeat-id 0 --outer-fold-id 0 \
  --popsize 20 --max-gen 5 \
  --lr-gap-gen 1 --lr-min-samples 5 --lr-auto-min-samples false
```

## Output layout

Nested per-run layout is the default for HPC-safe execution:

```text
<output-root>/<dataset>/<method_folder>/repXX/foldYY/runZZZ/
```

File names still include compact suffixes such as `c2_a2`, where `a2` now means pairwise Lasso.
