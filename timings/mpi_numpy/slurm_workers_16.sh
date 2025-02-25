#!/bin/bash

#SBATCH --job-name=lebwohl-lasher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=17
#SBATCH --time=00:15:00
#SBATCH --account=CHEM033484
#SBATCH --partition=teach_cpu

# Load the Python module.
module load languages/python/3.12.3

# Benchmark the script for different lattice sizes.
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 50 0.5 0" 5 "mpi_numpy_50.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 75 0.5 0" 5 "mpi_numpy_75.txt"

python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 100 0.5 0" 5 "mpi_numpy_100.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 200 0.5 0" 5 "mpi_numpy_200.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 300 0.5 0" 5 "mpi_numpy_300.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 400 0.5 0" 5 "mpi_numpy_400.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 500 0.5 0" 5 "mpi_numpy_500.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 600 0.5 0" 5 "mpi_numpy_600.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 700 0.5 0" 5 "mpi_numpy_700.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 800 0.5 0" 5 "mpi_numpy_800.txt"
python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 900 0.5 0" 5 "mpi_numpy_900.txt"

python ../benchmark.py "mpiexec -n 17 python lebwohl_lasher_mpi_numpy.py 50 1000 0.5 0" 5 "mpi_numpy_1000.txt"