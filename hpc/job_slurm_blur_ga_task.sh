#!/bin/bash
#SBATCH --job-name=BLuR-GA_TASK
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# Usage:
#   python run_blur_ga.py make-tasks ... --tasks-out tasks_tabarena.csv
#   sbatch --array=0-<N-1> hpc/job_slurm_blur_ga_task.sh tasks_tabarena.csv

TASKS_CSV="$1"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python run_blur_ga.py run-task --tasks "$TASKS_CSV" --task-id "$TASK_ID"
