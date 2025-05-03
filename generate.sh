#!/bin/bash

#Submit this script with: sbatch filename

#SBATCH --time=8:00:00               # walltime
#SBATCH -N 1                          # number of cluster nodes
#SBATCH --job-name=tether_generate            # job name
#SBATCH --output=JOB_%j.txt           # output file name
#SBATCH --error=ERR_%j.txt            # error File
#SBATCH --mail-user=dorianisp@lanl.gov # email address
#SBATCH --mail-type=ALL               # receive email for job events
#SBATCH --signal=23@60                # send signal to job at [seconds] before end
#SBATCH --qos=standard

echo "about to start application"
hostname
date

source /lustre/scratch5/dmperez/llms/conda_envs/llm/etc/profile.d/conda.sh
conda activate llm
python generate.py SimpleInequality --n_problems=180 --n_numbers=100
