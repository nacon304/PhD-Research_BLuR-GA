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

## Optional signed LTGA building-block diagnostics

The active linkage modes `ga_type=1`, `ga_type=2`, and `ga_type=3` can now extract LTGA-style building blocks from the current linkage graph without changing the search operator:

```bash
python run_blur_ga.py batch \
  --data-dir ../Dataset/prepared_linkage_benchmark_mini_debug/linkage \
  --output-root ../Results/results_signed_ltga_blocks \
  --datasets gametes_style_d30_n800_loci2_maf0.25_seed0 \
  --classifiers 1 \
  --ga-types 1 2 3 \
  --repeats 1 --outer-folds 2 --inner-folds 2 \
  --popsize 20 --max-gen 5 \
  --build-building-blocks true \
  --bb-weight-mode absolute \
  --bb-snapshot-interval 1
```

`--bb-weight-mode absolute` clusters by absolute linkage strength while recording both positive and negative signed relationships inside each block. `--bb-weight-mode signed` also uses absolute strength, then checks whether positive edges imply same-state relations and negative edges imply opposite-state relations. `--bb-weight-mode positive` keeps only positive/same-state linkages when building the LTGA tree.

Main building-block outputs:

- `building_block_summary_c*_a*.csv`: compact per-generation diagnostics.
- `building_blocks_c*_a*.csv`: selected LTGA blocks, capped by `--bb-max-blocks`.
- `analysis_interactive/analyze_building_blocks.py`: plots block counts, score, signed strengths, top blocks, and, when given `--graph-snapshots`, overlays selected blocks on the linkage graph for a chosen generation.

The default local output layout is now `root`, meaning per-run artifacts are kept in the method folder with tags such as `rep00_fold00_run000` instead of deeply nested run folders.  For array jobs, `make-tasks` still forces `nested` output to avoid concurrent file collisions, and `aggregate` rebuilds compact root summaries afterwards.

## Feature-selection evaluation datasets

Use `scripts/prepare_feature_selection_datasets.py` to download/process the small, medium, and high-dimensional datasets used by the feature-selection performance evaluation. The generated folder is directly readable by `run_blur_ga.py`; pass `--dataset-group small`, `--dataset-group medium`, or `--dataset-group highdim` to run one group. See `documents/feature_selection_dataset_preparation.md` for the full workflow.

