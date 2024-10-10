import os
import shutil
import subprocess
import sys

def run_copied_python_script_in_subfolders(script_name):
    cwd = os.getcwd()  # Get the current working directory
    python_exec = sys.executable  # Get the current Python executable
    script_path = os.path.join(cwd, script_name)  # Path of the script in the current directory

    # Check if the specified script exists in the current directory
    if not os.path.isfile(script_path):
        print(f"Error: {script_name} not found in the current directory.")
        return

    # Loop through all items in the current working directory
    for folder in os.listdir(cwd):
        folder_path = os.path.join(cwd, folder)

        # Check if the item is a directory (only go one level deep)
        if os.path.isdir(folder_path):
            # Construct the destination path for the script to be copied
            destination_script_path = os.path.join(folder_path, script_name)

            try:
                # Copy the script to the subdirectory
                shutil.copy(script_path, destination_script_path)
                print(f"Copied {script_name} to {folder_path}")

                # Change the working directory to the subfolder
                os.chdir(folder_path)

                # Set the PYTHONPATH to include the top directory
                env = os.environ.copy()
                env["PYTHONPATH"] = cwd  # Add the top-level directory to PYTHONPATH

                # Run the Python script using the current Python executable
                print(f"Running {script_name} in {folder_path}")
                subprocess.run([python_exec, script_name], check=True, env=env)

            except subprocess.CalledProcessError as e:
                print(f"Error running {destination_script_path}: {e}")

            finally:
                # Change back to the original working directory
                os.chdir(cwd)

                # Optionally, delete the copied script after running it
                if os.path.isfile(destination_script_path):
                    os.remove(destination_script_path)
                    print(f"Deleted {destination_script_path}")

if __name__ == "__main__":
    # Specify the name of the Python script in the current directory to copy and run
    script_to_copy = "your_script.py"
    
    run_copied_python_script_in_subfolders(script_to_copy)
