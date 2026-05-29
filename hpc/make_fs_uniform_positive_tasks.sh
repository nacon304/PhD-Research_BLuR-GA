#!/bin/bash
set -euo pipefail

# Create one combined task manifest for the FS performance sweep.
# Methods covered:
#   1) type 0 standard GA
#   2) type 1 + uniform positive
#   3) type 2 + uniform positive + binary + no pre-explore
#   4) type 2 + uniform positive + binary + pre-explore
#   5) type 2 + uniform positive + spin + no pre-explore
#   6) type 2 + uniform positive + spin + pre-explore
#
# Run from the BLuR-GA project root on Mahuika.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-../Dataset/prepared_feature_selection}"
RESULTS_ROOT="${RESULTS_ROOT:-../Results/results_fs_uniform_positive_medium_highdim}"
TASKS_OUT="${TASKS_OUT:-hpc/tasks_fs_uniform_positive_medium_highdim.csv}"
DATASET_GROUP="${DATASET_GROUP:-medium_highdim}"
# RESULTS_ROOT="${RESULTS_ROOT:-../Results/results_fs_uniform_positive_full}"
# TASKS_OUT="${TASKS_OUT:-hpc/tasks_fs_uniform_positive_full.csv}"
# DATASET_GROUP="${DATASET_GROUP:-all}"
CLASSIFIERS="${CLASSIFIERS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python.exe}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$TASKS_OUT")" "$RESULTS_ROOT" hpc/_task_parts
rm -f hpc/_task_parts/*.csv "$TASKS_OUT"

COMMON_ARGS=(
  --preset nesi_full
  --data-dir "$DATA_DIR"
  --dataset-group "$DATASET_GROUP"
  --classifiers $CLASSIFIERS
  --repeat 1
  --inner-folds 4
  --outer-folds 4
  --bb-pattern-top-fraction 0.20
  --bb-weight-mode positive
  --bb-search-mode uniform
  --bb-gawll-update-interval 10
  --pre-lr-mutation-multiplier 3.0
  --pre-lr-immigrant-rate 0.10
  --save-generation-trace true
  --save-graph-snapshots false
  --save-linkage-events false
  --artifact-layout nested
  --write-aggregate-outputs false
)

make_part() {
  local name="$1"
  shift
  local part="hpc/_task_parts/${name}.csv"
  echo "[make-tasks] $name -> $part"
  "$PYTHON_BIN" run_blur_ga.py make-tasks "${COMMON_ARGS[@]}" "$@" --tasks-out "$part"
}

# Method-folder names encode type1 BB suffixes and type2 encoding/pre flags,
# so all ablations can safely share the same output root.
make_part standard_ga \
  --output-root "$RESULTS_ROOT" \
  --ga-types 0 \
  --build-building-blocks false \
  --bb-search-mode none \
  --pre-lr-explore false

make_part type1_uniform_positive \
  --output-root "$RESULTS_ROOT" \
  --ga-types 1 \
  --build-building-blocks true \
  --pre-lr-explore false

make_part type2_uniform_positive_binary_nopre \
  --output-root "$RESULTS_ROOT" \
  --ga-types 2 \
  --build-building-blocks true \
  --lr-encoding binary \
  --pre-lr-explore false

make_part type2_uniform_positive_binary_pre \
  --output-root "$RESULTS_ROOT" \
  --ga-types 2 \
  --build-building-blocks true \
  --lr-encoding binary \
  --pre-lr-explore true

# make_part type2_uniform_positive_spin_nopre \
#   --output-root "$RESULTS_ROOT" \
#   --ga-types 2 \
#   --build-building-blocks true \
#   --lr-encoding spin \
#   --pre-lr-explore false

# make_part type2_uniform_positive_spin_pre \
#   --output-root "$RESULTS_ROOT" \
#   --ga-types 2 \
#   --build-building-blocks true \
#   --lr-encoding spin \
#   --pre-lr-explore true

"$PYTHON_BIN" - <<'PY' "$TASKS_OUT" hpc/_task_parts/*.csv
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
parts = [Path(p) for p in sys.argv[2:]]
rows = []
fieldnames = None
for part in parts:
    with part.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row['task_id'] = str(len(rows))
            rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote combined manifest: {out} ({len(rows)} tasks)")
PY

wc -l "$TASKS_OUT"
