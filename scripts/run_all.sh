#!/bin/bash

# This is a handy script for benchmarking in serial
# be sure to run "chmod +x run_all.sh" so this script will work

# Stop on error
set -e

MODEL="gpt-4o" # Change this to whatever model you want to use

declare -a TASKS=(
  "SimpleInequalityWithMethod_tdist"
  "SimpleInequalityWithMethod_bootstrap"
  "SimpleInequality_tdist"
  "SimpleInequality_bootstrap"
  "MediatedCausality_tdist"
  "MediatedCausality_bootstrap"
  "MediatedCausalitySmoking_tdist"
  "MediatedCausalitySmoking_bootstrap"
  "MediatedCausalityWithMethod_tdist"
  "MediatedCausalityWithMethod_bootstrap"
)

for TASK in "${TASKS[@]}"; do
  echo "Running $TASK with $MODEL..."
  python3 -m tether.core.run "$TASK" "$MODEL" --verbose
done

echo "All benchmarks completed."