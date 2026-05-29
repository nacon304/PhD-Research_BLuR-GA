#!/bin/bash
#SBATCH --job-name=BLuR_FS
#SBATCH --account=vuw04643
#SBATCH --partition=genoa
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

module purge
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-/nesi/project/vuw04643/nguyennha1/conda/envs/blur_ga}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

$PYTHON_BIN run_blur_ga.py run-task --tasks "$TASKS_CSV" --task-id "$TASK_ID"
