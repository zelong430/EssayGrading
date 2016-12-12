#!/bin/bash
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 1-00:10              # Runtime in D-HH:MM
#SBATCH -p holyseasgpu          # Partition to submit to
#SBATCH --mem=30000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o train.out      # File to which STDOUT will be written
#SBATCH -e train.err      # File to which STDERR will be written

source activate NLP
python train.py
