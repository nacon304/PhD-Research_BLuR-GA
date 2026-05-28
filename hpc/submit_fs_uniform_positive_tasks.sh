#!/bin/bash
set -euo pipefail

# Submit the combined FS manifest as a Slurm array.
# Run from the BLuR-GA project root.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TASKS_CSV="${TASKS_CSV:-hpc/tasks_fs_uniform_positive_full.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-../Results/results_fs_uniform_positive_full}"
ARRAY_LIMIT="${ARRAY_LIMIT:-40}"
CONDA_ENV="${CONDA_ENV:-blur_ga}"

cd "$PROJECT_DIR"

if [[ ! -f "$TASKS_CSV" ]]; then
  echo "Task manifest not found: $TASKS_CSV" >&2
  echo "Create it first:" >&2
  echo "  bash hpc/make_fs_uniform_positive_tasks.sh" >&2
  exit 1
fi

N_TASKS=$(( $(wc -l < "$TASKS_CSV") - 1 ))
if [[ "$N_TASKS" -le 0 ]]; then
  echo "No tasks found in $TASKS_CSV" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/slurm_logs"

echo "Submitting $N_TASKS tasks with concurrency limit $ARRAY_LIMIT"
echo "Tasks:   $TASKS_CSV"
echo "Results: $RESULTS_ROOT"

sbatch \
  --array="0-$((N_TASKS - 1))%${ARRAY_LIMIT}" \
  --output="$RESULTS_ROOT/slurm_logs/%x_%A_%a.out" \
  --error="$RESULTS_ROOT/slurm_logs/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONDA_ENV="$CONDA_ENV" \
  hpc/job_slurm_blur_ga_task_mahuika.sh "$TASKS_CSV"
