#!/bin/bash

#SBATCH --job-name=lebwohl-lasher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --account=CHEM033484
#SBATCH --partition=teach_cpu

# Load the Python module.
module load languages/python/3.12.3

# Set the number of threads.
export NUMBA_NUM_THREADS=1

# Benchmark the script for different lattice sizes.
# Perform 1 pre-run so numba can cache the results.

python lebwohl_lasher_p_numba.py 50 25 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 25 0.5 0" 5 "parallel_numba_25.txt"

python lebwohl_lasher_p_numba.py 50 50 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 50 0.5 0" 5 "parallel_numba_50.txt"

python lebwohl_lasher_p_numba.py 50 75 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 75 0.5 0" 5 "parallel_numba_75.txt"


python lebwohl_lasher_p_numba.py 50 100 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 100 0.5 0" 5 "parallel_numba_100.txt"

python lebwohl_lasher_p_numba.py 50 200 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 200 0.5 0" 5 "parallel_numba_200.txt"

python lebwohl_lasher_p_numba.py 50 300 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 300 0.5 0" 5 "parallel_numba_300.txt"

python lebwohl_lasher_p_numba.py 50 400 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 400 0.5 0" 5 "parallel_numba_400.txt"

python lebwohl_lasher_p_numba.py 50 500 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 500 0.5 0" 5 "parallel_numba_500.txt"

python lebwohl_lasher_p_numba.py 50 600 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 600 0.5 0" 5 "parallel_numba_600.txt"

python lebwohl_lasher_p_numba.py 50 700 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 700 0.5 0" 5 "parallel_numba_700.txt"

python lebwohl_lasher_p_numba.py 50 800 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 800 0.5 0" 5 "parallel_numba_800.txt"

python lebwohl_lasher_p_numba.py 50 900 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 900 0.5 0" 5 "parallel_numba_900.txt"


python lebwohl_lasher_p_numba.py 50 1000 0.5 0
python ../benchmark.py "python lebwohl_lasher_p_numba.py 50 1000 0.5 0" 5 "parallel_numba_1000.txt"