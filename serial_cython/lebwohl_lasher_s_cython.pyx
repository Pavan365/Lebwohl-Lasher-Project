#cython: language_level=3

# Import required libraries.
cimport cython
cimport numpy as cnp
from libc.math cimport cos

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