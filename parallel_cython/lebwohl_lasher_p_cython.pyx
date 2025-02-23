#cython: language_level=3

# Import required libraries.
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, exp, sin

# Initialise the Cython NumPy API.
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double cell_energy(double[:, :] lattice, int lattice_length, int x_pos, int y_pos) nogil:
    """
    Calculates the reduced energy of a single cell in the square lattice taking 
    into account periodic boundary conditions. Equation 1 in the notes is used 
    to calculate the energy which involves summing over contributions from the 
    neighbouring cells.

    + U_{reduced] = U / ε   where ε is set to 1.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.

    lattice_length : int
      The side length of the square lattice.

    x_pos : int
      The x-position of the cell in the square lattice.

    y_pos : int
      The y-position of the cell in the square lattice.

    Returns
    -------
    energy : float
      The reduced energy of the cell.
    """

    # Store the positions of the neighbouring cells in the x-direction.
    # Take into account wraparound.
    cdef int x_pos_right = (x_pos + 1) % lattice_length
    cdef int x_pos_left = (x_pos - 1) % lattice_length
    
    # Store the positions of the neighbouring cells in the y-direction.
    # Take into account wraparound.
    cdef int y_pos_above = (y_pos + 1) % lattice_length
    cdef int y_pos_below = (y_pos - 1) % lattice_length

    # Create a variable to store the angle.
    # Create a variable to store the energy.
    cdef double angle
    cdef double energy
    
    # Store the angle of the cell.
    cdef double cell_angle = lattice[x_pos, y_pos]

    # Calculate the energy contribution from the cell to the right.
    angle = cell_angle - lattice[x_pos_right, y_pos]
    energy = cos(angle) ** 2

    # Calculate the energy contribution from the cell to the left.
    angle = cell_angle - lattice[x_pos_left, y_pos]
    energy += cos(angle) ** 2

    # Calculate the energy contribution from the cell above.
    angle = cell_angle - lattice[x_pos, y_pos_above]
    energy += cos(angle) ** 2

    # Calculate the energy contribution from the cell below.
    angle = cell_angle - lattice[x_pos, y_pos_below]
    energy += cos(angle) ** 2
    
    return (4 - (3 * energy)) * 0.5


@cython.boundscheck(False)
@cython.wraparound(False)
def total_energy(cnp.ndarray[cnp.double_t, ndim=2] lattice, int lattice_length):
    """
    Calculates the total reduced energy of the lattice.

    + E_{reduced] = E / ε   where ε is set to 1.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in square lattice.

    lattice_length : int
      The side length of the square lattice.
    
    Returns
    -------
    energy : float
      The total reduced energy of the lattice.
    """

    # Define a view to the lattice.
    cdef double[:, :] lattice_view = lattice

    # Create a variable to store the total energy.
    cdef double energy = 0.0

    # Define the iterating variables.
    cdef int i, j

    # Sum over the energy of each cell in the lattice.
    for i in prange(lattice_length, nogil=True):
        for j in range(lattice_length):
            # Calculate the energy of the cell.
            energy += cell_energy(lattice_view, lattice_length, i, j)

    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_order(cnp.ndarray[cnp.double_t, ndim=2] lattice, int lattice_length):
    """
    Calculates the order parameter of the square lattice using the order tensor 
    approach as defined in equation 3 of the notes.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.
      
    lattice_length : int
      The side length of the square lattice.

    Returns
    -------
    float
      The order parameter of the lattice.
    """

    # Create an array to store the order tensor.
    cdef cnp.ndarray[cnp.double_t, ndim=2] order_tensor = np.zeros((3, 3))

    # Define a view to the lattice.
    cdef double[:, :] lattice_view = lattice

    # Define a view to the order tensor.
    cdef double[:, :] order_tensor_view = order_tensor

    # Define variables to store the cosine and sine values of each cell's angle.
    cdef double cos_theta, sin_theta
    
    # Define iterating variables.
    cdef int i, j

    # Define variables to store the sum of order tensor terms in each thread.
    # These will be private to each thread and reduced at the end, avoiding race conditions.
    cdef double sum_00 = 0.0, sum_11 = 0.0, sum_01 = 0.0

    # Loop through each cell in the lattice.
    for i in prange(lattice_length, nogil=True):
      for j in range(lattice_length):
        # Calculate the cosine and sine of the cell's angle.
        cos_theta = cos(lattice_view[i, j])
        sin_theta = sin(lattice_view[i, j])

        # Calculate the diagonal terms.
        sum_00 += cos_theta * cos_theta
        sum_11 += sin_theta * sin_theta

        # Calculate the off-diagonal term.
        sum_01 += cos_theta * sin_theta

    # Calculate the lattice size.
    cdef int lattice_size = lattice_length * lattice_length

    # Calculate the final diagonal terms.
    order_tensor_view[0, 0] = ((3 * sum_00) - lattice_size) / (2 * lattice_size)
    order_tensor_view[1, 1] = ((3 * sum_11) - lattice_size) / (2 * lattice_size)
    order_tensor_view[2, 2] = -(<double> lattice_size) / (2 * lattice_size)

    # Calculate the final off-diagonal terms.
    order_tensor_view[0, 1] = (3 * sum_01) / (2 * lattice_size)
    order_tensor_view[1, 0] = order_tensor_view[0, 1]

    # Calculate the eigenvalues of the order tensor.
    # Use the "np.linalg.eigh" as the order tensor is symmetric.
    cdef cnp.ndarray[cnp.double_t, ndim=1] eigenvalues = np.linalg.eigh(order_tensor)[0]

    return eigenvalues.max()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def monte_carlo_step(cnp.ndarray[cnp.double_t, ndim=2] lattice, int lattice_length, double temperature):
    """
    Performs a Monte Carlo step which attempts to change the orientation of 
    each cell in the lattice once on average. The reduced temperature is used 
    in the calculations. The function returns the acceptance ratio of the Monte 
    Carlo step which represents the fraction of successful cell orientation 
    changes. The aim is to keep the acceptance ratio around 0.5 for an optimal 
    simulation.

    + T_{reduced} = kT / ε

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.

    lattice_length : int
      The side length of the square lattice.

    temperature : float
      The reduced temperature, with a range between 0 and 2.

    Returns
    -------
    float
      The acceptance ratio for the Monte Carlo step.
    """

    # Calculate the standard deviation of the distribution for angle changes.
    cdef float angle_std = temperature + 0.1

    # Create a variable to store the number of accepted to changes.
    cdef int num_accepted = 0

    # Generate the positions in the lattice to visit.
    cdef cnp.ndarray[int, ndim=2] x_positions = np.random.randint(0, high=lattice_length, size=(lattice_length, lattice_length), dtype=np.int32)
    cdef cnp.ndarray[int, ndim=2] y_positions = np.random.randint(0, high=lattice_length, size=(lattice_length, lattice_length), dtype=np.int32)
    
    # Generate the random angles for cell orientations.
    cdef cnp.ndarray[cnp.double_t, ndim=2] angles = np.random.normal(scale=angle_std, size=(lattice_length, lattice_length))

    # Generate random, uniform distributed, numbers for the Monte Carlo test.
    cdef cnp.ndarray[cnp.double_t, ndim=2] mc_test_nums = np.random.uniform(size=(lattice_length, lattice_length))

    # Create memory-views to the arrays.
    cdef int[:, :] x_positions_view = x_positions
    cdef int[:, :] y_positions_view = y_positions
    
    cdef double[:, :] angles_view = angles
    cdef double[:, :] mc_test_nums_view = mc_test_nums

    # Create a memory-view to the lattice.
    cdef double[:, :] lattice_view = lattice

    # Define the iterating variables.
    cdef int i, j

    # Define variables used inside the loop.
    cdef int x_pos, y_pos
    cdef double angle, energy_before, energy_after, boltzmann

    # Attempt to change the orientation of each cell in the lattice.
    for i in range(lattice_length):
        for j in range(lattice_length):
            # Get x and y position of the cell.
            x_pos = x_positions_view[i, j]
            y_pos = y_positions_view[i, j]

            # Get the random angle.
            angle = angles_view[i, j]

            # Calculate the energy of the cell before the orientation change.
            energy_before = cell_energy(lattice_view, lattice_length, x_pos, y_pos)
            
            # Change the orientation of the cell.
            lattice_view[x_pos, y_pos] += angle

            # Calculate the energy of the cell after the orientation change.
            energy_after = cell_energy(lattice_view, lattice_length, x_pos, y_pos)
            
            # If energy after the orientation change is lower, accept the change.
            if energy_after <= energy_before:
                num_accepted += 1
            
            # Otherwise, perform the Monte Carlo test.
            else:
                # Calculate the Boltzmann factor.
                boltzmann = exp(-(energy_after - energy_before) / temperature)

                # If the Boltzmann factor is greater than a random (uniform) number.
                # Accept the orientation change.
                if boltzmann >= mc_test_nums_view[i, j]:
                    num_accepted += 1
                
                # Otherwise, undo the orientation change.
                else:
                    lattice_view[x_pos, y_pos] -= angle

    return (<double> num_accepted) / (lattice_length * lattice_length)