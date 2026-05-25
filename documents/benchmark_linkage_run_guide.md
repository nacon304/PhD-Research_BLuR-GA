# Linkage benchmark run guide

Current active methods:

| `ga_type` | Method |
|---:|---|
| 0 | Standard GA |
| 1 | Empirical linkage legacy |
| 2 | Pairwise Lasso linkage |
| 3 | Main-controlled pairwise Lasso linkage |

## Smoke run

```bash
python run_linkage_eval.py run \
  --dataset-root ../Dataset/prepared_linkage_benchmark_mini_debug \
  --output-root ../Results/results_linkage_smoke \
  --datasets ising_spin_glass_grid5x5_posneg_seed0 \
  --ga-types 0 1 2 3 \
  --repeat-id 0 --outer-fold-id 0 \
  --popsize 20 --max-gen 5 \
  --lr-gap-gen 1 --lr-min-samples 5 --lr-auto-min-samples false \
  --save-generation-trace true --save-graph-snapshots true
```

## Full task manifest

```bash
python run_linkage_eval.py make-tasks \
  --dataset-root ../Dataset/prepared_linkage_benchmark \
  --output-root ../Results/results_linkage_full \
  --datasets all \
  --ga-types 1 2 3 \
  --repeats 10 --outer-folds 3 \
  --tasks-out tasks_linkage.csv
```
