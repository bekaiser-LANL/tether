#!/bin/bash

# Stop on error
set -e

echo "Running script 1..."
python3 run.py SimpleInequality_tdist o3 --verbose

echo "Running script 2..."
python3 run.py SimpleInequalityWithMethod_tdist o3 --verbose

echo "Running script 3..."
python3 run.py SimpleInequalityWithMethod_bootstrap o3 --verbose

echo "All scripts completed."
