import os
import subprocess
import sys

def run_python_scripts_in_subfolders():
    cwd = os.getcwd()  # Get the current working directory
    python_exec = sys.executable  # Get the current Python executable

    # Loop through all items in the current working directory
    for folder in os.listdir(cwd):
        folder_path = os.path.join(cwd, folder)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Loop through all items in the subfolder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                # Check if the item is a Python file
                if file.endswith(".py"):
                    print(f"Running {file_path} in {folder_path}")

                    try:
                        # Change to the directory where the Python script is located
                        os.chdir(folder_path)

                        # Run the Python script using the current Python executable
                        subprocess.run([python_exec, file_path], check=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running {file_path}: {e}")

                    finally:
                        # Change back to the original working directory
                        os.chdir(cwd)

if __name__ == "__main__":
    run_python_scripts_in_subfolders()
