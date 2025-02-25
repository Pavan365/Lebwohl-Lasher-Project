#!/bin/bash

#SBATCH --job-name=lebwohl-lasher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:15:00
#SBATCH --account=CHEM033484
#SBATCH --partition=teach_cpu

# Load the Python module.
module load languages/python/3.12.3

# Set the number of threads.
export OMP_NUM_THREADS=16

# Benchmark the script for different lattice sizes.
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 25 0.5 0" 5 "parallel_cython_25.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 50 0.5 0" 5 "parallel_cython_50.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 75 0.5 0" 5 "parallel_cython_75.txt"

python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 100 0.5 0" 5 "parallel_cython_100.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 200 0.5 0" 5 "parallel_cython_200.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 300 0.5 0" 5 "parallel_cython_300.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 400 0.5 0" 5 "parallel_cython_400.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 500 0.5 0" 5 "parallel_cython_500.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 600 0.5 0" 5 "parallel_cython_600.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 700 0.5 0" 5 "parallel_cython_700.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 800 0.5 0" 5 "parallel_cython_800.txt"
python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 900 0.5 0" 5 "parallel_cython_900.txt"

python ../benchmark.py "python run_lebwohl_lasher_p_cython.py 50 1000 0.5 0" 5 "parallel_cython_1000.txt"