#!/bin/bash

# This is a handy script for benchmarking in serial
# be sure to run "chmod +x run_all.sh" so this script will work

# Stop on error
set -e

MODEL="phi4" # "deepseek-r1:32b" # Change this to whatever model you want to use

declare -a TASKS=(
  "MediatedCausality_tdist_0"
  "MediatedCausality_bootstrap_0"
  "MediatedCausalitySmoking_tdist_0"
  "MediatedCausalitySmoking_bootstrap_0"
  "MediatedCausalityWithMethod_tdist_0"
  "MediatedCausalityWithMethod_bootstrap_0"
  "SimpleInequality_tdist_0"
  "SimpleInequality_bootstrap_0"
  "SimpleInequalityWithMethod_tdist_0"
  "SimpleInequalityWithMethod_bootstrap_0"
)

for TASK in "${TASKS[@]}"; do
  echo "Running $TASK with $MODEL..."
  python3 run.py "$TASK" "$MODEL" --verbose
done

echo "All benchmarks completed."