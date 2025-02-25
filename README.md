# Accelerated Computing - Lebwohl Lasher Project

## Overview
<p align="justify">
This is the codebase for the accelerated computing project of accelerating 
Monte Carlo simulations of a 2D Lebwohl Lasher model. This GitHub repo contains 
various versions of the original <code>lebwohl_lasher.py</code> script, which 
has been accelerated using various methods such as <code>NumPy</code>, 
<code>Numba</code>, <code>Cython</code> and <code>MPI</code>. 
</p>

<p align="justify">
The original script implements the Metropolis Monte Carlo method using a pure 
<code>Python</code> approach with minimal use of <code>NumPy</code>. This means 
that slow double-nested for loops which scale as <code>O(n x n)</code> 
(n-squared) are implemented in multiple important functions leading to a slow 
simulation. Therefore, the original script has accelerated using the different 
methods listed before and the performance of each method has been measured.
</p>

> [!NOTE]
> The format of commits for this repo take inspiration from https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13 in an attempt to keep things organised and findable.
> However the use of the correct tags is something to be improved.


## Directories

```./original```
+ ```lebwohl_lasher.py```: Contains the ```NumPy``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.
- ```performance_logs```: Contains the performance logs using ```cProfile``` generated from tests.

```./serial_numba```
+ ```lebwohl_lasher_s_numba.py```: Contains the serial ```Numba``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.
- ```performance_logs```: Contains the performance logs using ```cProfile``` generated from tests.

```./serial_cython```
+ ```lebwohl_lasher_s_cython.pyx```: Contains the serial ```Cython``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.
- ```performance_logs```: Contains the performance logs using ```cProfile``` generated from tests.

```./parallel_numba```
+ ```lebwohl_lasher_p_numba.py```: Contains the parallel ```Numba``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.
- ```performance_logs```: Contains the performance logs using ```cProfile``` generated from tests.

```./parallel_cython```
+ ```lebwohl_lasher_p_cython.pyx```: Contains the parallel ```Cython``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.
- ```performance_logs```: Contains the performance logs using ```cProfile``` generated from tests.

```./mpi_numpy```
+ ```lebwohl_lasher_mpi_numpy.py```: Contains the ```MPI (NumPy Vectorised)``` accelerated script.
+ ```testing.ipynb```: Contains the tests performed to ensure correctness of the updated script.

- ```output_logs```: Contains the output files generated from tests.

```./timings```

+ ```benchmark.py```: Contains the script used for benchmarking the different Lebwohl-Lasher scripts.
+ ```statistics.py```: Contains the script used for calculating average runtimes and standard deviations.

- ```comparison.ipynb```: Contains the main plots and numerical calculations for comparing different versions of the script.


## Files

+ ```./environment.yaml```: Contains the local conda environment used for developing this project.
+ ```./README.md```: This README file.
