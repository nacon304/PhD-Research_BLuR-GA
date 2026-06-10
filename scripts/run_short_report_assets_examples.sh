#!/usr/bin/env bash
set -euo pipefail

# Run from the BLuR-GA project root.
# Adjust RESULTS_BASE if your result folders are stored elsewhere.
$RESULTS_BASE = "..\Results"

python scripts/make_short_report_assets.py `
  --results-root "$RESULTS_BASE/results_fs_uniform_positive_small" `
  --comparison-name report1_pairwise_lasso_vs_baseline `
  --baseline-method standard_ga `
  --include-methods standard_ga `
    blur_ga_pairwise_lasso_uniform_positive_binary_nopre `
    blur_ga_pairwise_lasso_uniform_positive_binary_pre

python scripts/make_short_report_assets.py `
  --results-root "$RESULTS_BASE/results_fs_uniform_positive_small_extend" `
  --comparison-name report2_guided_pair_vs_previous `
  --baseline-method standard_ga