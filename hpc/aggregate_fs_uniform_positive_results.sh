#!/bin/bash
set -euo pipefail

# Aggregate nested per-task outputs after the Slurm array finishes.
# Run from the BLuR-GA project root.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-../Results/results_fs_uniform_positive_full}"
CONDA_ENV="${CONDA_ENV:-blur_ga}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/Miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/Miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

$PYTHON_BIN run_blur_ga.py aggregate --results-root "$RESULTS_ROOT"

echo "Done. Aggregated files were written inside: $RESULTS_ROOT"
