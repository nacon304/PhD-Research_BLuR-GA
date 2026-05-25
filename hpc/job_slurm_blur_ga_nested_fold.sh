#!/bin/bash
#SBATCH --job-name=BLuR-GA_NCV
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# Usage:
#   sbatch --array=0-9 hpc/job_slurm_blur_ga_nested_fold.sh zoo 1 1 0 /path/to/out --data-dir /path/to/data [extra args]
# One array task = one repeat/seed for one dataset + one outer fold + one algorithm.

DATASET="$1"
CLASSIFIER="$2"        # 1 = KNN-3, 2 = KNN-5
GA_TYPE="$3"           # 0 = standard GA, 1 = empirical linkage GA, 2 = pairwise Lasso, 3 = main+pairwise Lasso
OUTER_FOLD_ID="$4"     # 0-based outer fold id
OUT_ROOT="$5"
shift 5
EXTRA_ARGS=("$@")

REPEAT_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Pass only the base seed. run_blur_ga_oop.py derives the actual run seed as
# base_seed + 100000 * repeat_id + 1000 * outer_fold_id.  Computing it here
# as well would double-count repeat/fold offsets.
BASE_SEED="${BASE_SEED:-1}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUT_ROOT"

python run_blur_ga.py "$DATASET" "$CLASSIFIER" "$GA_TYPE" \
  --output-dir "$OUT_ROOT" \
  --repeat-id "$REPEAT_ID" \
  --outer-fold-id "$OUTER_FOLD_ID" \
  --seed "$BASE_SEED" \
  --artifact-layout nested \
  --write-aggregate-outputs false \
  "${EXTRA_ARGS[@]}"
