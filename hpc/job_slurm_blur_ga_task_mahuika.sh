#!/bin/bash
#SBATCH --job-name=BLuR_FS
#SBATCH --account=vuw04643
#SBATCH --partition=parallel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# Usage:
#   sbatch --array=0-<N-1>%<limit> hpc/job_slurm_blur_ga_task_mahuika.sh hpc/tasks_fs_uniform_positive_full.csv

TASKS_CSV="$1"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-blur_ga}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

# Load/activate your Python environment. Edit this block if your environment name/path differs.
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/Miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/Miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "WARNING: conda not found; using PYTHON_BIN=$PYTHON_BIN from current environment" >&2
fi

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

$PYTHON_BIN run_blur_ga.py run-task --tasks "$TASKS_CSV" --task-id "$TASK_ID"
