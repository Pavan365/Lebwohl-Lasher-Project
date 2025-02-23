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

# Import 3rd party libraries. 
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI
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
        # Normalise the colour map according to the minimum and maximum energy.
        colours = lattice_energies(lattice)
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


def lattice_energies(lattice):
    """
    Calculates the reduced energy of all the cells in the lattice.
    
    + E_{reduced] = E / ε   where ε is set to 1.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in square lattice.

    Returns
    -------
    energies : numpy.ndarray, float(lattice_length, lattice_length)
      The array containing the reduced energy of each cell in the lattice.
    """

    # Calculate the energy contributions from the cells to the right.
    angles = lattice - np.roll(lattice, shift=-1, axis=1)
    energies = np.cos(angles) ** 2

    # Calculate the energy contributions from the cells to the left.
    angles = lattice - np.roll(lattice, shift=1, axis=1)
    energies += np.cos(angles) ** 2

    # Calculate the energy contributions from the cells above.
    angles = lattice - np.roll(lattice, shift=-1, axis=0)
    energies += np.cos(angles) ** 2

    # Calculate the energy contributions from the cells below.
    angles = lattice - np.roll(lattice, shift=1, axis=0)
    energies += np.cos(angles) ** 2

    # Calculate the energies.
    energies = (4 - (3 * energies)) * 0.5

    return energies


def calculate_order(lattice):
    """
    Calculates the order parameter of the lattice using the order tensor 
    approach as defined in equation 3 of the notes.

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the lattice.
      
    Returns
    -------
    float
      The order parameter of the lattice.
    """

    # Create an array to store the order tensor.
    order_tensor = np.zeros((3, 3))
    
    # Generate a 2D unit vector for each cell in the lattice.
    l_ab = np.array((np.cos(lattice), np.sin(lattice)))

    # Calculate the size of the lattice.
    lattice_size = lattice.shape[0] * lattice.shape[1]

    # Calculate the x-x and y-y (diagonals) contributions to the order tensor.
    diagonals = 3 * l_ab * l_ab

    order_tensor[0, 0] = np.sum(diagonals[0]) - lattice_size
    order_tensor[1, 1] = np.sum(diagonals[1]) - lattice_size

    # Calculate the off-diagonal contributions to the order tensor.
    order_tensor[0, 1] = order_tensor[1, 0] = np.sum(3 * l_ab[0] * l_ab[1])

    # Calculate the z-z contribution 
    order_tensor[2, 2] = -lattice_size

    # Normalise the order tensor.
    order_tensor = order_tensor / (2 * lattice_size)
    
    # Calculate the eigenvalues of the order tensor.
    # Use the "np.linalg.eigh" as the order tensor is symmetric.
    eigenvalues = np.linalg.eigh(order_tensor)[0]

    return eigenvalues.max()


def mc_step_worker(lattice, temperature, angles, mc_test_nums, mask):
    """
    A worker function for the "monte_carlo_step" function. This function 
    applies the Monte Carlo step on cells in the lattice that are selected 
    using the given mask. The function returns the number of accepted cell 
    orientation changes.
    
    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.

    temperature : float
      The reduced temperature, with a range between 0 and 2.

    angles : numpy.ndarray, float(lattice_length, lattice_length)
      The array containing the angles to make the cell orientation changes. 

    mc_test_nums : numpy.ndarray, float(lattice_length, lattice_length)
      The array containing the numbers to perform the Monte Carlo test with.

    mask : numpy.ndarray, bool(lattice_length, lattice_length)
      The array which selects the cells to perform the Monte Carlo step on. It 
      should be a boolean mask.
    
    Returns
    -------
    int
        The number of accepted orientation changes.
    """

    # Calculate the energy of the cells before the orientation changes.
    energy_before = lattice_energies(lattice)[mask]

    # Perform the orientation changes.
    lattice[mask] += angles[mask]

    # Calculate the energy of the cells after the orientation changes.
    energy_after = lattice_energies(lattice)[mask]

    # Get the cells whose orientation change is accepted from a decrease in energy.
    # This is a boolean mask. 
    energy_accepted_mask = energy_after <= energy_before

    # Calculate the Boltzmann factor of the cells.
    boltzmann = np.exp(-(energy_after - energy_before) / temperature)

    # Get the cells whose orientation change is accepted from the Monte Carlo test.
    # This is a boolean mask.
    boltzmann_accepted_mask = boltzmann >= mc_test_nums[mask]

    # Get the cells whose orientation change is accepted overall. 
    # This is a boolean mask.
    accepted_mask = np.logical_or(energy_accepted_mask, boltzmann_accepted_mask)

    # Revert the cells whose orientation change is not accepted.
    cells_final_state = lattice[mask]
    cells_final_state[~accepted_mask] -= angles[mask][~accepted_mask]

    lattice[mask] = cells_final_state

    return accepted_mask.sum()


def monte_carlo_step(lattice, temperature):
    """
    Performs a Monte Carlo step which attempts to change the orientation of 
    each cell in the lattice. The reduced temperature is used in the 
    calculations. The function returns the number of successful cell 
    orientation changes.

    + T_{reduced} = kT / ε

    Parameters
    ----------
    lattice : numpy.ndarray, float(lattice_length, lattice_length)
      The array representing the cells in the square lattice.

    temperature : float
      The reduced temperature, with a range between 0 and 2.

    Returns
    -------
    int
        The number of accepted orientation changes.
    """

    # Calculate the standard deviation of the distribution for angle changes.
    angle_std = temperature + 0.1

    lattice_rows = lattice.shape[0]
    lattice_columns = lattice.shape[1]
    
    # Generate the random angles for cell orientation changes.
    angles = np.random.normal(scale=angle_std, size=(lattice_rows, lattice_columns))
    
    # Generate random, uniform distributed, numbers for the Monte Carlo test.
    mc_test_nums = np.random.uniform(size=(lattice_rows, lattice_columns))

    # Create a checkerboard mask that selects cells which do not neighbour each other.
    checkerboard_mask = np.indices((lattice_rows, lattice_columns)).sum(axis=0) % 2 == 0

    # Get the mask for the "white" squares.
    # Set the first and last rows to False, as these rows contain the neighbouring rows.
    white_squares = checkerboard_mask.copy()
    white_squares[0] = False
    white_squares[-1] = False

    # Get the mask for the "black" squares.
    # Set the first and last rows to False, as these rows contain the neighbouring rows.
    black_squares = ~checkerboard_mask.copy()
    black_squares[0] = False
    black_squares[-1] = False

    # Perform the Monte Carlo step on the "white" squares/cells.
    num_accepted_one = mc_step_worker(lattice, temperature, angles, mc_test_nums, white_squares) 

    # Perform the Monte Carlo step on the "black" squares/cells.
    num_accepted_two = mc_step_worker(lattice, temperature, angles, mc_test_nums, black_squares)

    return num_accepted_one + num_accepted_two


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

    # Define the MPI communicator.
    comm = MPI.COMM_WORLD

    # Get the task IDs.
    # Get the number of tasks.
    task_id = comm.Get_rank()
    num_tasks = comm.Get_size()

    # Calculate the number of workers.
    num_workers = num_tasks - 1

    # Define the minimum and maximum number of workers.
    # The maximum number of workers is set such that each worker receives at least 5 rows.
    MIN_WORKERS = 1
    MAX_WORKERS = lattice_length // 5

    # Define the master task ID.
    MASTER = 0

    # Define the communication tags.
    BEGIN = 1
    WORKER_ABOVE = 2
    WORKER_BELOW = 3
    DONE = 4

    # Give the master task its work.
    if task_id == MASTER:
        # If the number of workers are invalid, abort.
        if num_workers < MIN_WORKERS or num_workers > MAX_WORKERS:
            # Print an error message.
            print("error: invalid number of workers")
            comm.Abort()
        
        # Create and initialise the lattice.
        master_lattice = init_lattice(lattice_length)

        # Plot the initial lattice.
        plot_lattice(master_lattice, lattice_length, plot_flag)

        # Create an array to store the total energy (summed over workers).
        master_energy = np.zeros(num_steps + 1, dtype=np.float64)

        # Create an array to store the order parameter (averaged over workers).
        master_average_order = np.zeros(num_steps + 1, dtype=np.float64)

        # Create an array to store the acceptance ratio (calculated).
        master_ratio = np.zeros(num_steps + 1, dtype=np.int32)

        # Calculate the number of rows to send to each worker.
        # Calculate the number of leftover rows.
        num_rows = lattice_length // num_workers
        extra_rows = lattice_length % num_workers

        # Set the row-offset.
        row_offset = 0

        # Start a timer.
        start_time = MPI.Wtime()

        # Send each worker its block of the lattice.
        for i in range(1, num_workers + 1):
            # Set the final number of rows to send to the worker.
            worker_rows = num_rows

            # Take into account the leftover rows.
            if i <= extra_rows:
                worker_rows += 1

            # Calculate the IDs of the neighbouring workers.
            # If this is the first worker.
            if i == 1:
                worker_above = num_workers
                worker_below = i + 1
            
            # If this is the last worker.
            elif i == num_workers:
                worker_above = i - 1
                worker_below = 1
            
            # If this worker is neither the first or last.
            else:
                worker_above = i - 1
                worker_below = i + 1

            # Send the starting information to the worker.
            # Send the worker its row-offset and the number of rows.
            comm.send(row_offset, dest=i, tag=BEGIN)
            comm.send(worker_rows, dest=i, tag=BEGIN)

            # Send the worker the workers above and below it.
            comm.send(worker_above, dest=i, tag=BEGIN)
            comm.send(worker_below, dest=i, tag=BEGIN)
            
            # Send the worker its block of the lattice.
            comm.Send(master_lattice[row_offset:(row_offset+worker_rows), :], dest=i, tag=BEGIN)

            # Update the row-offset.
            row_offset += worker_rows

        # Wait for the results from each of the workers.
        for i in range(1, num_tasks):
            # Receive the final information.
            # Receive the row-offset and number of rows of the worker.
            row_offset = comm.recv(source=i, tag=DONE)
            worker_rows = comm.recv(source=i, tag=DONE)

            # Receive the worker's block of the lattice.
            comm.Recv(master_lattice[row_offset:(row_offset+worker_rows), :], source=i, tag=DONE)

        # Sum over worker-local total energy values.
        # Sum over worker-local order parameter values.
        # Sum over worker-local number of accepted cell orientation changes values.
        comm.Reduce(None, master_energy, op=MPI.SUM, root=MASTER)
        comm.Reduce(None, master_average_order, op=MPI.SUM, root=MASTER)
        comm.Reduce(None, master_ratio, op=MPI.SUM, root=MASTER)

        # Calculate the average order parameter.
        master_average_order = master_average_order / num_workers

        # Calculate the acceptance ratio.
        master_ratio = master_ratio.astype(np.float64) / (lattice_length * lattice_length)
        master_ratio[0] = 0.5

        # End the timer.
        end_time = MPI.Wtime()
        runtime = end_time - start_time

        # Plot the final lattice.
        plot_lattice(master_lattice, lattice_length, plot_flag)

        # Generate the output data file.
        save_data(lattice_length, num_steps, temperature, master_ratio, master_energy, master_average_order, runtime)

        # Output the final results.
        print(f"{program_name}: Size: {lattice_length:d}, Steps: {num_steps:d}, T*: {temperature:5.3f}: Order: {master_average_order[-1]:5.3f}, Time: {runtime:8.6f} s")

    # Give the worker tasks their work.
    else:
        # Receive the starting information.
        # Receive the row-offset and the number of rows.
        row_offset = comm.recv(source=MASTER, tag=BEGIN)
        worker_rows = comm.recv(source=MASTER, tag=BEGIN)

        # Receive the workers above and below.
        worker_above = comm.recv(source=MASTER, tag=BEGIN)
        worker_below = comm.recv(source=MASTER, tag=BEGIN)

        # Create a worker-local lattice.
        # Make this with an extra two rows so it can store the neighbouring rows (above and below).
        worker_lattice = np.zeros((worker_rows + 2, lattice_length), dtype=np.float64)

        # Receive the block of the lattice to work on.
        comm.Recv([worker_lattice[1:-1, :], (worker_rows * lattice_length), MPI.DOUBLE], source=MASTER, tag=BEGIN)

        # Open receiving channels first to avoid communication locks.
        # Receive the neighbouring row above from the worker above.
        # Receive the neighbouring row below from the worker below.
        receive_row_above = comm.Irecv(worker_lattice[0, :], source=worker_above, tag=WORKER_ABOVE)
        receive_row_below = comm.Irecv(worker_lattice[-1, :], source=worker_below, tag=WORKER_BELOW)

        # Send the first row to the worker above.
        # Send the final row to the worker below.
        send_row_above = comm.Isend(worker_lattice[1, :], dest=worker_above, tag=WORKER_BELOW)
        send_row_below = comm.Isend(worker_lattice[-2, :], dest=worker_below, tag=WORKER_ABOVE)

        # Wait for all communications to finish.
        MPI.Request.Waitall([receive_row_above, receive_row_below, send_row_above, send_row_below])

        # Create an array to store the worker-local total energy.
        worker_energy = np.zeros(num_steps + 1, dtype=np.float64)

        # Initialise the first value.
        # Exclude the neighbouring rows (above and below).
        worker_energy[0] = np.sum(lattice_energies(worker_lattice)[1:-1])

        # Create an array to store the worker-local order parameter.
        worker_order = np.zeros(num_steps + 1, dtype=np.float64)

        # Initialise the first value.
        # Exclude the neighbouring rows (above and below).
        worker_order[0] = calculate_order(worker_lattice[1:-1])

        # Create an array to store the worker-local number of accepted cell orientation changes.
        worker_accepted = np.zeros(num_steps + 1, dtype=np.int32)

        # Perform Monte Carlo simulation.
        for i in range(1, num_steps + 1):
            # Perform a Monte Carlo step.
            # Get the number of accepted cell orientation changes.
            worker_accepted[i] = monte_carlo_step(worker_lattice, temperature)

            # Calculate the total energy and order parameter of the lattice.
            # Exclude the neighbouring rows (above and below).
            worker_energy[i] = np.sum(lattice_energies(worker_lattice)[1:-1])
            worker_order[i] = calculate_order(worker_lattice[1:-1])

            # Open receiving channels first to avoid communication locks.
            # Receive the neighbouring row above from the worker above.
            # Receive the neighbouring row below from the worker below.
            receive_row_above = comm.Irecv(worker_lattice[0, :], source=worker_above, tag=WORKER_ABOVE)
            receive_row_below = comm.Irecv(worker_lattice[-1, :], source=worker_below, tag=WORKER_BELOW)

            # Send the first row to the worker above.
            # Send the final row to the worker below.
            send_row_above = comm.Isend(worker_lattice[1, :], dest=worker_above, tag=WORKER_BELOW)
            send_row_below = comm.Isend(worker_lattice[-2, :], dest=worker_below, tag=WORKER_ABOVE)

            # Wait for all communications to finish.
            MPI.Request.Waitall([receive_row_above, receive_row_below, send_row_above, send_row_below])

        # Send the final information.
        # Send the row-offset and the number of rows worked on.
        comm.send(row_offset, dest=MASTER, tag=DONE)
        comm.send(worker_rows, dest=MASTER, tag=DONE)

        # Send the final state of the lattice block.
        comm.Send(worker_lattice[1:-1, :], dest=MASTER, tag=DONE)

        # Send the worker-local total energy values.
        # Send the worker-local order parameter values.
        # Send the worker-local number of accepted cell orientation changes values.
        comm.Reduce(worker_energy, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_order, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_accepted, None, op=MPI.SUM, root=MASTER)


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
