# Import standard libraries.
import csv
import glob
import os
import sys

# Import 3rd party libraries.
import numpy as np


def calculate_statistic(folder: str, prefix: str, save_file: str) -> None:
    """
    Calculates the average runtime and standard deviation for each file in a 
    given set of files. The function then saves the results to a CSV file. The 
    input filename format is expected to be "prefix_size.txt".

    Parameters
    ----------
    folder : str
        The folder containing the text files which contain the runtimes.

    prefix : str
        The prefix of the text files which contain the runtimes.

    save_file : str
        The file to save the average and standard deviation of the different 
        runtimes to.
    """
    
    # Create a list to store the results for each file.
    results = []

    # Loop over each file.
    for file in glob.glob(os.path.join(folder, f"{prefix}_*.txt")):
        # Get the lattice size from the filename.
        filename = os.path.basename(file)
        lattice_size = int(filename.rsplit("_", 1)[-1].split(".")[0])
        
        # Read the runtimes.
        runtimes = np.loadtxt(file, skiprows=2)

        # Calculate the average runtime.
        # Calculate the standard deviation.
        runtimes_average = runtimes.mean()
        runtimes_std = runtimes.std()

        # Save the results.
        results.append([lattice_size, runtimes_average, runtimes_std])

    # Convert the results to a NumPy array.
    # Sort the results in ascending lattice size.
    results = np.array(results)
    results = results[np.argsort(results[:, 0])]

    # Create the header row for the CSV file.
    # Save the results.
    header = "Lattice-Size,Runtime-Average,Runtime-STD"
    np.savetxt(save_file, results, delimiter=",", fmt="%d,%.18f,%.18f", header=header, comments="")


if __name__ == "__main__":
    # If an invalid number of command line arguments are passed, show an error message and abort.
    if len(sys.argv) != 4:
        print("usage: python statistics.py <FOLDER> <PREFIX> <SAVE_FILE>")
        sys.exit(1)
    
    # Otherwise, benchmark the given script.
    calculate_statistic(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"success: saved runtime statistics to {sys.argv[3]}")
    sys.exit(0)