import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_script_in_subfolder(script_name, folder_path, cwd):
    destination_script_path = os.path.join(folder_path, script_name)

    try:
        # Copy the script to the subdirectory
        shutil.copy(os.path.join(cwd, script_name), destination_script_path)
        print(f"Copied {script_name} to {folder_path}")

        # Set the PYTHONPATH to include the top directory
        env = os.environ.copy()
        env["PYTHONPATH"] = cwd  # Add the top-level directory to PYTHONPATH

        # Run the Python script using the current Python executable, setting the cwd to the subfolder
        print(f"Running {script_name} in {folder_path}")
        subprocess.run([sys.executable, destination_script_path], check=True, env=env, cwd=folder_path)

    except subprocess.CalledProcessError as e:
        print(f"Error running {destination_script_path}: {e}")

    finally:
        # Optionally, delete the copied script after running it
        if os.path.isfile(destination_script_path):
            os.remove(destination_script_path)
            print(f"Deleted {destination_script_path}")

def run_copied_python_script_in_subfolders(script_name):
    cwd = os.getcwd()  # Get the current working directory

    # Check if the specified script exists in the current directory
    if not os.path.isfile(os.path.join(cwd, script_name)):
        print(f"Error: {script_name} not found in the current directory.")
        return

    # List of subfolders to process
    subfolders = [
        os.path.join(cwd, folder) for folder in os.listdir(cwd)
        if os.path.isdir(os.path.join(cwd, folder))
    ]

    # Use ProcessPoolExecutor to run scripts in parallel
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor for each subfolder
        futures = [
            executor.submit(run_script_in_subfolder, script_name, folder_path, cwd)
            for folder_path in subfolders
        ]

        # Process the results as they complete
        for future in as_completed(futures):
            try:
                future.result()  # Will raise an exception if the script failed
            except Exception as e:
                print(f"Script execution failed: {e}")

if __name__ == "__main__":
    # Specify the name of the Python script in the current directory to copy and run
    script_to_copy = "3Dres_no_dist.py"

    run_copied_python_script_in_subfolders(script_to_copy)
