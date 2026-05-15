# BLuR-GA - NeSI-ready runner

**BLuR-GA: Building-block Linkage using Regression for Genetic Algorithms**.  
This package was renamed away from the old baseline-specific naming because the regression-linkage stages are a separate idea.

## Main entry points

- `run_blur_ga.py`: unified local/HPC launcher.
- `run_blur_ga_oop.py`: direct nested-CV runner used internally by the launcher.
- `blur_ga/`: core implementation package.
- `analysis_interactive/make_graph_ui.py`: interactive graph viewer.
- `hpc/job_slurm_blur_ga_task.sh`: preferred SLURM array runner from a task manifest.
- `hpc/submit_tabarena_manifest.sh`: convenience submitter using a prepared TabArena manifest.

## Method folders under `--output-root`

The root-level launcher now writes descriptive method folders instead of `a1`, `a2`, etc.  
The CSV file names still keep `_a<ga_type>` so existing analysis code that reads file names still works.

| `ga_type` | Method folder                  | Meaning                                                   |
| --------: | ------------------------------ | --------------------------------------------------------- |
|         0 | `standard_ga`                  | Standard GA baseline                                      |
|         1 | `empirical_linkage_legacy`     | Legacy empirical-linkage variant                          |
|         2 | `blur_ga_stage1_pairwise`      | Pairwise-only regression linkage                          |
|         3 | `blur_ga_stage2_main_pairwise` | Main effects + pairwise regression linkage                |
|         4 | `blur_ga_stage3_sparse`        | Sparse ElasticNet/LASSO-style linkage                     |
|         5 | `blur_ga_stage4_sparse_excess` | Sparse linkage with baseline-standardized excess response |

## Environment on NeSI

Use either `requirements_nesi.txt` with pip or `environment_nesi.yml` with conda/mamba.

```bash
python -m venv .venv_blur_ga
source .venv_blur_ga/bin/activate
pip install -r requirements_nesi.txt
```

Or:

```bash
mamba env create -f environment_nesi.yml
conda activate blur_ga
```

Runtime dependencies are minimal: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `networkx`.

## Local smoke test

The included tiny synthetic dataset is only for checking the pipeline.

```bash
python run_blur_ga.py batch \
  --data-dir tests/data \
  --output-root test_outputs/blur_ga_smoke \
  --datasets synthetic_tiny \
  --classifiers 1 \
  --ga-types 2 \
  --repeats 1 \
  --outer-folds 2 \
  --inner-folds 2 \
  --popsize 8 \
  --max-gen 3 \
  --lr-gap-gen 1 \
  --lr-min-samples 4 \
  --lr-edge-top-k 10 \
  --save-generation-trace true \
  --save-graph-snapshots true \
  --graph-snapshot-interval 1
```

Expected output folder:

```text
test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/
```

## Graph UI example

```bash
python analysis_interactive/make_graph_ui.py \
  --snapshots test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/graph_snapshots_synthetic_tiny_c1_a2_r0.csv \
  --trace test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/generation_trace_synthetic_tiny_c1_a2.csv \
  --selected-features test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/selected_features_synthetic_tiny_c1_a2.csv \
  --run-id 0 \
  --n-nodes 8 \
  --output test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/graph_evolution_ui.html \
  --layout spring
```

The UI supports interactive zoom/pan and SVG export from the browser.

## NeSI manifest workflow

```bash
python run_blur_ga.py make-tasks \
  --preset nesi_full \
  --data-dir ../Dataset/prepared_tabarena \
  --output-root results_nesi \
  --datasets all \
  --classifiers 1 \
  --ga-types 0 1 2 3 4 5 \
  --repeats 10 \
  --outer-folds 3 \
  --inner-folds 3 \
  --tasks-out tasks_tabarena.csv

sbatch --array=0-<N-1> hpc/job_slurm_blur_ga_task.sh tasks_tabarena.csv

python run_blur_ga.py aggregate --results-root results_nesi
```

For array jobs, `make-tasks` automatically switches to nested, race-safe output:

```text
repeat_xx/outer_fold_xx/run_xxx/
```

Then `aggregate` rebuilds root-level summary/vector files after all jobs finish.

## Output files to keep for later analysis

Important root-level files:

- `nested_summary_*.csv`
- `run_summary_*_r*.csv`
- `generation_trace_*.csv` and `generation_trace_*_r*.csv`
- `selected_features_*.csv` and `selected_features_*_r*.csv`
- `graph_snapshots_*_r*.csv`
- `eVIG_*_r*.csv`, `eVIG_edges_*_r*.csv`, `eVIG_coefficients_*_r*.csv`
- legacy vectors: `bfi_*.csv`, `test_score_*.csv`, `time_*.csv`, `gen_*.csv`, `evals_*.csv`, `nedges_*.csv`, `bind_*.csv`

## Verified smoke outputs in this package

This zip includes generated smoke outputs under `test_outputs/`:

- `test_outputs/blur_ga_smoke/synthetic_tiny/blur_ga_stage1_pairwise/`: batch run for `ga_type=2`, including graph snapshots and `graph_evolution_ui.html`.
- `test_outputs/blur_ga_tasks/synthetic_tiny/blur_ga_stage1_pairwise/`: manifest/task-style run plus aggregate output.
- `test_outputs/blur_ga_smoke_emp/`: direct positional run for the legacy empirical-linkage variant, `ga_type=1`.
