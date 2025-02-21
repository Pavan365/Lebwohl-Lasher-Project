#cython: language_level=3

# Import required libraries.
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin

# Initialise the Cython NumPy API.
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def cell_energy(cnp.ndarray[cnp.double_t, ndim=2] lattice, int lattice_length, int x_pos, int y_pos):
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

    # Define a view to the lattice.
    cdef double[:, :] lattice_view = lattice

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
    cdef double cell_angle = lattice_view[x_pos, y_pos]

    # Calculate the energy contribution from the cell to the right.
    angle = cell_angle - lattice_view[x_pos_right, y_pos]
    energy = cos(angle) ** 2

    # Calculate the energy contribution from the cell to the left.
    angle = cell_angle - lattice_view[x_pos_left, y_pos]
    energy += cos(angle) ** 2

    # Calculate the energy contribution from the cell above.
    angle = cell_angle - lattice_view[x_pos, y_pos_above]
    energy += cos(angle) ** 2

    # Calculate the energy contribution from the cell below.
    angle = cell_angle - lattice_view[x_pos, y_pos_below]
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

    # Create a variable to store the total energy.
    cdef double energy = 0.0

    # Define the iterating variables.
    cdef int i, j

    # Sum over the energy of each cell in the lattice.
    for i in range(lattice_length):
        for j in range(lattice_length):
            # Calculate the energy of the cell.
            energy += cell_energy(lattice, lattice_length, i, j)

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

    # Loop through each cell in the lattice.
    for i in range(lattice_length):
      for j in range(lattice_length):
        # Calculate the cosine and sine of the cell's angle.
        cos_theta = cos(lattice_view[i, j])
        sin_theta = sin(lattice_view[i, j])

        # Calculate the diagonal terms.
        order_tensor_view[0, 0] += cos_theta * cos_theta
        order_tensor_view[1, 1] += sin_theta * sin_theta

        # Calculate the off-diagonal term.
        order_tensor_view[0, 1] += cos_theta * sin_theta

    # Calculate the lattice size.
    cdef int lattice_size = lattice_length * lattice_length

    # Calculate the final diagonal terms.
    order_tensor_view[0, 0] = ((3 * order_tensor_view[0, 0]) - lattice_size) / (2 * lattice_size)
    order_tensor_view[1, 1] = ((3 * order_tensor_view[1, 1]) - lattice_size) / (2 * lattice_size)

    # Calculate the final off-diagonal terms.
    order_tensor_view[0, 1] = (3 * order_tensor_view[0, 1]) / (2 * lattice_size)
    order_tensor_view[1, 0] = order_tensor_view[0, 1]

    # Calculate the eigenvalues of the order tensor.
    # Use the "np.linalg.eigh" as the order tensor is symmetric.
    cdef cnp.ndarray[cnp.double_t, ndim=1] eigenvalues = np.linalg.eigh(order_tensor)[0]

    return eigenvalues.max()