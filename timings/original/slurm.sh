#!/bin/bash

#SBATCH --job-name=lebwohl-lasher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:12:00
#SBATCH --account=CHEM033484
#SBATCH --partition=teach_cpu

# Load the Python module.
module load languages/python/3.12.3

# Benchmark the script for different lattice sizes.
python ../benchmark.py "python lebwohl_lasher.py 50 25 0.5 0" 5 "original_25.txt"
python ../benchmark.py "python lebwohl_lasher.py 50 50 0.5 0" 5 "original_50.txt"
python ../benchmark.py "python lebwohl_lasher.py 50 75 0.5 0" 5 "original_75.txt"

python ../benchmark.py "python lebwohl_lasher.py 50 100 0.5 0" 5 "original_100.txt"
python ../benchmark.py "python lebwohl_lasher.py 50 150 0.5 0" 5 "original_150.txt"
python ../benchmark.py "python lebwohl_lasher.py 50 200 0.5 0" 5 "original_200.txt"