#!/bin/bash

# This is a handy script for benchmarking in serial
# be sure to run "chmod +x run_all.sh" so this script will work

# Stop on error
set -e

MODEL="phi4"

# "MediatedCausality_bootstrap_0"
# "MediatedCausalitySmoking_tdist_0"
declare -a TASKS=(
  "SimpleInequalityWithMethod_tdist_0"
  "SimpleInequalityWithMethod_bootstrap_0"
  "SimpleInequality_tdist_0"
  "SimpleInequality_bootstrap_0"
  "MediatedCausalitySmoking_bootstrap_0"
  "MediatedCausalityWithMethod_tdist_0"
  "MediatedCausalityWithMethod_bootstrap_0"
  "MediatedCausality_tdist_0"
)

for TASK in "${TASKS[@]}"; do
  FULL_TASK="${TASK}_${MODEL}_0"
  echo "Running $FULL_TASK..."
  python3 analyze.py "$FULL_TASK" --grade_estimate --verbose
done

echo "All benchmarks completed."