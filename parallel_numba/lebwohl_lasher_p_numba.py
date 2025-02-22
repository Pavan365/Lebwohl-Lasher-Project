"""
This is a basic Python implementation of a 2D Lebwohl-Lasher model using Monte 
Carlo simulations.

Reference: P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972)

Command Line Usage:
  $ python lebwohl_lasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

Positional Arguments:
  ITERATIONS    number of Monte Carlo steps, where 1-MCS attempts to change 
                each cell in the square lattice once on average (i.e. SIZE*SIZE 
                attempts)

  SIZE          side length of the square lattice

  TEMPERATURE   reduced temperature in the range 0.0 - 2.0
  
  PLOTFLAG      flag for deciding which plot to generate
                  + 0 : no     plot
                  + 1 : energy plot
                  + 2 : angle  plot
                  + 3 : black  plot      

+ The initial configuration of the cells in the lattice is set at random.  
+ The boundaries of the lattice are periodic throughout the simulation. 
+ During time-stepping, an array containing two domains is used; these domains 
  alternate between old and new data.

Original Code: Dr Simon Hanna (2023-10-16)
"""

# Import standard libraries.
import datetime
import sys
import time

# Import 3rd party libraries. 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np


def init_lattice(lattice_length):
    """
    Creates and initialises the square lattice. The lattice will contain cells 
    with random orientations in the range [0, 2π].

    Parameters
    ----------
    lattice_length : int
      The side length of the square lattice to create.

    Returns
    -------
    numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.
    """

    # Generate the square lattice.
    return np.random.random_sample((lattice_length, lattice_length)) * 2.0 * np.pi


def plot_lattice(lattice, lattice_length, plot_flag):
    """
    Generates a plot of the square lattice using matplotlib's quiver plot. The 
    "plot_flag" parameter is used to control the colouring of the plot.

    plot_flag options:
      + 0 : no     plot
      + 1 : energy plot
      + 2 : angles plot
      + 3 : black  plot

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.
    
    lattice_length : int
      The side length of the square lattice.
    
    plot_flag : int
      A flag to control which plot to generate.
    """

    # If "plot_flag" is 0, return.
    if plot_flag == 0:
        return
    
    # Generate the x and y components of the quivers.
    u = np.cos(lattice)
    v = np.sin(lattice)

    # Generate the x and y positions of the quivers.
    x = np.arange(lattice_length)
    y = np.arange(lattice_length)

    # Create an array to store the colour of each cell in the lattice.
    colours = np.zeros((lattice_length, lattice_length))

    # If "plot_flag" is 2, colour the quivers according to the energy of the cells.
    if plot_flag == 1:
        # Set the colour map of the image.
        mpl.rc("image", cmap="rainbow")

        # Calculate the colour of each cell in the lattice.
        for i in range(lattice_length):
            for j in range(lattice_length):
                # Calculate the energy of the cell.
                colours[i,j] = cell_energy(lattice, lattice_length, i, j)
        
        # Normalise the colour map according to the minimum and maximum energy.
        norm = plt.Normalize(colours.min(), colours.max())

    # If "plot_flag" is 2, colour the quivers according to the angle of the cells.
    elif plot_flag == 2: # colour the arrows according to angle
        # Set the colour-map of the image.
        mpl.rc("image", cmap="hsv")

        # Calculate the angle of each cell.
        # Normalise the colour map according to the minimum and maximum angle.
        colours = lattice % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)

    # Otherwise, colour the quivers black.
    else:
        # Set the colour-map of the image.
        mpl.rc('image', cmap='gist_gray')

        # Set the colour of each cell.
        # Normalise the colour map.
        colours = np.zeros_like(lattice)
        norm = plt.Normalize(vmin=0, vmax=1)

    # Set the options for the quiver plot.
    quiver_options = dict(pivot='middle', headlength=0, headwidth=1, scale=(1.1 * lattice_length))

    # Generate the quiver plot.
    fig, ax = plt.subplots()

    ax.quiver(x, y, u, v, colours, norm=norm, **quiver_options)
    ax.set_aspect('equal')

    plt.show()


def save_data(lattice_length, num_steps, temperature, ratio, energy, order, runtime):
    """
    Saves the total energy, order parameter and acceptance ratio of the lattice 
    at each Monte Carlo step in the simulation to a text file. The parameters 
    used to run the simulation are also saved in the header of the file. The 
    filename is generated based off the data and time at execution.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.
    
    lattice_length : float
      The side length of the square lattice.

    num_steps : int
      The number of Monte Carlo steps performed.

    temperature : float
      The reduced temperature, with a range between 0 and 2.

    ratio : numpy.ndarray, float(num_steps)
      The acceptance ratio at each Monte Carlo step.

    energy : numpy.ndarray, float(num_steps)
      The total reduced energy of the lattice at each Monte Carlo step.

    order : numpy.ndarray, float(num_steps)
      The order parameter of the lattice at each Monte Carlo step.

    runtime : float
      The runtime of the Monte Carlo simulation.
    """

    # Create a filename based on the current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"LL-Output-{current_datetime:s}.txt"

    # Create a buffer/stream to the output file.
    file_out = open(filename, "w")

    # Write a header containing the parameters of the simulation.
    print("#=====================================================", file=file_out)
    print(f"# File created:        {current_datetime:s}", file=file_out)
    print(f"# Size of lattice:     {lattice_length:d}x{lattice_length:d}", file=file_out)
    print(f"# Number of MC steps:  {num_steps:d}", file=file_out)
    print(f"# Reduced temperature: {temperature:5.3f}", file=file_out)
    print(f"# Run time (s):        {runtime:8.6f}", file=file_out)
    print("#=====================================================", file=file_out)
    print("# MC step:  Ratio:     Energy:   Order:", file=file_out)
    print("#=====================================================", file=file_out)

    # Write the data from the Monte Carlo simulation.
    for i in range(num_steps + 1):
        print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f}", file=file_out)
    
    # Close the buffer/stream to the output file.
    file_out.close()


@numba.njit(cache=True)
def cell_energy(lattice, lattice_length, x_pos, y_pos):
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

    # Create a variable to store the energy.
    energy = 0.0

    # Store the positions of the neighbouring cells in the x-direction.
    # Take into account wraparound.
    x_pos_right = (x_pos + 1) % lattice_length
    x_pos_left = (x_pos - 1) % lattice_length
    
    # Store the positions of the neighbouring cells in the y-direction.
    # Take into account wraparound.
    y_pos_above = (y_pos + 1) % lattice_length
    y_pos_below = (y_pos - 1) % lattice_length

    # Calculate the energy contribution from the cell to the right.
    angle = lattice[x_pos, y_pos] - lattice[x_pos_right, y_pos]
    energy += np.cos(angle) ** 2

    # Calculate the energy contribution from the cell to the left.
    angle = lattice[x_pos, y_pos] - lattice[x_pos_left, y_pos]
    energy += np.cos(angle) ** 2

    # Calculate the energy contribution from the cell above.
    angle = lattice[x_pos, y_pos] - lattice[x_pos, y_pos_above]
    energy += np.cos(angle) ** 2

    # Calculate the energy contribution from the cell below.
    angle = lattice[x_pos, y_pos] - lattice[x_pos, y_pos_below]
    energy += np.cos(angle) ** 2
    
    return (4 - (3 * energy)) * 0.5


@numba.njit(cache=True, parallel=True)
def total_energy(lattice, lattice_length):
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
    energy = 0.0

    # Sum over the energy of each cell in the lattice.
    for i in numba.prange(lattice_length):
        for j in range(lattice_length):
            # Calculate the energy of the cell.
            energy += cell_energy(lattice, lattice_length, i, j)

    return energy


@numba.njit(cache=True)
def calculate_order(lattice, lattice_length):
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
    order_tensor = np.zeros((3, 3))

    # Generate a 2D unit vector for each cell in the lattice.
    l_ab = np.vstack((np.cos(lattice), np.sin(lattice))).reshape(2, lattice_length, lattice_length)

    # Loop over each dimension.
    for a in range(2):
        for b in range(2):
            # Loop through each cell in the lattice.
            for i in range(lattice_length):
                for j in range(lattice_length):
                    # Calculate the order tensor term.
                    order_tensor[a, b] += 3 * l_ab[a, i, j] * l_ab[b, i, j]

    # Calculate the size of the lattice.
    lattice_size = lattice_length * lattice_length

    # Calculate the diagonal terms.
    order_tensor[0, 0] -= lattice_size
    order_tensor[1, 1] -= lattice_size
    order_tensor[2, 2] = -lattice_size
   
    # Normalise the order tensor.
    order_tensor = order_tensor / (2 * lattice_size)

    # Calculate the eigenvalues of the order tensor.
    # Use the "np.linalg.eigh" as the order tensor is symmetric.
    eigenvalues = np.linalg.eigh(order_tensor)[0]

    return eigenvalues.max()


@numba.njit(cache=True)
def monte_carlo_step(lattice, lattice_length, temperature):
    """
    Performs a Monte Carlo step which attempts to change the orientation of 
    each cell in the lattice once on average. The reduced temperature is used 
    in the calculations. The function returns the acceptance ratio of the Monte 
    Carlo step which represents the fraction of successful cell orientation 
    changes.

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
    angle_std = temperature + 0.1

    # Create a variable to store the number of accepted to changes.
    num_accepted = 0

    # Generate the positions in the lattice to visit.
    x_positions = np.random.randint(0, high=lattice_length, size=(lattice_length, lattice_length))
    y_positions = np.random.randint(0, high=lattice_length, size=(lattice_length, lattice_length))
    
    # Generate the random angles for cell orientations.
    angles = np.random.normal(0, scale=angle_std, size=(lattice_length, lattice_length))

    # Generate random, uniform distributed, numbers for the Monte Carlo test.
    mc_test_nums = np.random.uniform(0, 1.0, size=(lattice_length, lattice_length))

    # Attempt to change the orientation of each cell in the lattice.
    for i in range(lattice_length):
        for j in range(lattice_length):
            # Get x and y position of the cell.
            x_pos = x_positions[i, j]
            y_pos = y_positions[i, j]

            # Get the random angle.
            angle = angles[i, j]

            # Calculate the energy of the cell before the orientation change.
            energy_before = cell_energy(lattice, lattice_length, x_pos, y_pos)
            
            # Change the orientation of the cell.
            lattice[x_pos, y_pos] += angle

            # Calculate the energy of the cell after the orientation change.
            energy_after = cell_energy(lattice, lattice_length, x_pos, y_pos)
            
            # If energy after the orientation change is lower, accept the change.
            if energy_after <= energy_before:
                num_accepted += 1
            
            # Otherwise, perform the Monte Carlo test.
            else:
                # Calculate the Boltzmann factor.
                boltzmann = np.exp(-(energy_after - energy_before) / temperature)

                # If the Boltzmann factor is greater than a random (uniform) number.
                # Accept the orientation change.
                if boltzmann >= mc_test_nums[i, j]:
                    num_accepted += 1
                
                # Otherwise, undo the orientation change.
                else:
                    lattice[x_pos, y_pos] -= angle

    return num_accepted / (lattice_length * lattice_length)


def main(program_name, num_steps, lattice_length, temperature, plot_flag):
    """
    The main function of the program.

    Parameters
    ----------
    program_name : string
      The name of the program.

    num_steps : int
      The number of Monte Carlo steps (MCS) to perform.

    lattice_length : int
      The side length of the square lattice to simulate.
    
    temperature : float
      The reduced temperature, with a range between 0 and 2.

    plot_flag : int
      A flag to control which plot to generate.
    """

    # Create and initialise the lattice.
    lattice = init_lattice(lattice_length)

    # Plot the initial lattice.
    plot_lattice(lattice, lattice_length, plot_flag)

    # Create arrays to store the total energy, order parameter and acceptance ratio.
    energy = np.zeros(num_steps + 1, dtype=np.float64)
    order = np.zeros(num_steps + 1, dtype=np.float64)
    ratio = np.zeros(num_steps + 1, dtype=np.float64)

    # Calculate the initial energy and order of the lattice.
    energy[0] = total_energy(lattice, lattice_length)
    order[0] = calculate_order(lattice, lattice_length)

    # Set the initial acceptance ratio to 0.5, which is the "ideal" value.
    ratio[0] = 0.5 

    # Begin a timer.
    start_time = time.time()

    # Perform a Monte Carlo simulation.
    for i in range(1, num_steps + 1):
        # Perform a Monte Carlo step.
        # Get the acceptance ratio of the Monte Carlo step.
        ratio[i] = monte_carlo_step(lattice, lattice_length, temperature)

        # Calculate the total energy and order parameter of the lattice.
        energy[i] = total_energy(lattice, lattice_length)
        order[i] = calculate_order(lattice, lattice_length)

    # End the timer.
    # Calculate the runtime.
    end_time = time.time()
    runtime = end_time - start_time
    
    # Output the final results.
    print(f"{program_name}: Size: {lattice_length:d}, Steps: {num_steps:d}, T*: {temperature:5.3f}: Order: {order[num_steps - 1]:5.3f}, Time: {runtime:8.6f} s")

    # Generate the output data file.
    save_data(lattice_length, num_steps, temperature, ratio, energy, order, runtime)

    # Plot the final lattice.
    plot_lattice(lattice, lattice_length, plot_flag)


if __name__ == "__main__":
    # If the correct number of command line arguments were passed.
    # Store them and call the main function.
    if int(len(sys.argv)) == 5:
        # Get the command line arguments.
        PROGRAM_NAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOT_FLAG = int(sys.argv[4])

        # Call the main function.
        main(PROGRAM_NAME, ITERATIONS, SIZE, TEMPERATURE, PLOT_FLAG)

    # Otherwise, show an error message.
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
