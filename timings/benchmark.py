# Import required libraries.
import subprocess
import sys


def benchmark_script(run_command: str, num_runs: int, save_file: str) -> None:
    """
    Benchmarks a script using the given command and save the runtimes to the 
    given file. This function expects that the python script being benchmarked 
    outputs just the runtime to the command line.

    Parameters
    ----------
    run_command : str
        The command used to run the script.

    num_runs : int
        The number of times to run the script.

    save_file : str
        The name of the file to save the runtimes to.
    """

    # Create a list to store the runtimes.
    runtimes = []

    # Split the run command.
    command = run_command.split()

    # Run the script for the given number of times.
    for i in range(num_runs):
        # Get the runtime.
        res = subprocess.run(command, capture_output=True)
        runtime = res.stdout.strip().decode()

        # Store the runtime.
        runtimes.append(runtime) 
    
    # Write the runtimes to the given file.
    with open(save_file, "w") as file:
        # Write the run command used.
        file.write(f"{run_command}\n\n")

        # Write the runtimes.
        for runtime in runtimes:
            file.write(f"{runtime}\n")


if __name__ == "__main__":
    # If an invalid number of command line arguments are passed, show an error message and abort.
    if len(sys.argv) != 4:
        print("usage: python benchmark.py <RUN_COMMAND> <NUM_RUNS> <SAVE_FILE>")
        sys.exit(1)
    
    # Otherwise, benchmark the given script.
    benchmark_script(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    print(f"success: saved runtimes to {sys.argv[3]}")
    sys.exit(0)