import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
# The maximum number of FEM simulations to run at the same time.
# Adjust this based on your system's CPU cores and memory.
MAX_JOBS = 100 

# Path to the python executable in your conda environment
PYTHON_EXEC = "/data/1bali/miniforge3/envs/multi_view_3DQA/bin/python"

# The name of your second script (the one that performs FEM analysis)
SCRIPT_TO_RUN = "./generate_mesh_and_simulate.py"  # RENAME THIS to the actual filename

# The top-level directory where your first script saved all its results.
# This script will search recursively through this directory to find the .npy files.
LOG_SEARCH_DIR = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_infer_meshpoints"

val_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/val_rebuttal.txt'
PARENT_DIR = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal"
with open(val_file_path, 'r') as fread:
    lines = [line.strip() for line in fread]
geometry_names =[os.path.basename(line) for line in lines]
MESHES =[f"{PARENT_DIR}/{geometry_name}/renders_pyvista/{geometry_name}.obj" for geometry_name in geometry_names]

FOLDERS = sorted(os.listdir(LOG_SEARCH_DIR)) #[f"{LOG_SEARCH_DIR}/{log_file_name}" for log_file_name in sorted(os.listdir(LOG_SEARCH_DIR))]

FOLDERS = [
    f'{mesh_name}__view-ortho__prompt-geo_mid__nv-5__prompt_type-geo_mid__grid-12__run-2__LLM-qwen3-vl-235b-a22b-instruct.log'
    for mesh_name in MESHES]

def find_jobs(root_dir):
    """Finds all 'refinement_points_*.npy' files to be processed."""
    npy_files = []
    print(f"Searching for *.npy files in {root_dir}...")
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.startswith("refinement_points_") and f.endswith(".npy"):
                npy_files.append(os.path.join(dirpath, f))
    return npy_files

def run_job(refinement_points_path):
    """Executes a single FEM simulation for a given .npy file."""
    
    # --- Create a descriptive log name from the input path ---
    # Example input: .../test_meshes_rebuttal/00200070/refinement_points_ortho_10views_..._5run.npy
    # Example output: logs_fem/00200070__ortho_10views_..._5run.log
    
    log_dir = "./logs_fem_runs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Extract the mesh name (e.g., '00200070') and the core experiment name
    mesh_name = os.path.basename(os.path.dirname(refinement_points_path))
    experiment_name = os.path.basename(refinement_points_path).replace("refinement_points_", "").replace(".npy", "")
    
    log_name = f"{mesh_name}__{experiment_name}.log"
    log_path = os.path.join(log_dir, log_name)
    
    # --- Build and execute the command ---
    cmd = [
        PYTHON_EXEC, "-u", SCRIPT_TO_RUN,
        "--refinement_points_path", refinement_points_path,
    ]
    
    # Execute the command, redirecting all output to the log file
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    # Return True for success (exit code 0), False otherwise
    return result.returncode == 0, log_name


if __name__ == '__main__':
    # 1. Find all the .npy files that need to be processed
    jobs_to_run = find_jobs(SEARCH_DIR)
    
    if not jobs_to_run:
        print("No 'refinement_points_*.npy' files found. Exiting.")
        exit()
        
    total_jobs = len(jobs_to_run)
    print(f"Found {total_jobs} total jobs to run.")
    print(f"Running up to {MAX_JOBS} jobs concurrently...\n")
    
    # 2. Run the jobs in parallel
    start_time = time.time()
    successful_jobs = 0
    failed_jobs = 0

    with ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
        # Submit all jobs to the pool
        future_to_job = {executor.submit(run_job, npy_path): npy_path for npy_path in jobs_to_run}
        
        # Process results as they complete, with a progress bar
        with tqdm(total=total_jobs, desc="Running FEM Simulations", unit="job") as pbar:
            for future in as_completed(future_to_job):
                try:
                    success, log_name = future.result()
                    if success:
                        successful_jobs += 1
                    else:
                        failed_jobs += 1
                        tqdm.write(f"[FAILED] Check log: {log_name}")
                except Exception as exc:
                    failed_jobs += 1
                    # Get the original npy path that caused the error for better debugging
                    failed_npy_path = future_to_job[future]
                    tqdm.write(f"[ERROR] Job for '{os.path.basename(failed_npy_path)}' raised an exception: {exc}")
                
                # Update progress bar stats
                pbar.set_postfix(Success=successful_jobs, Failed=failed_jobs)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\nAll FEM simulations finished in {elapsed_time:.2f} seconds!")
    print(f"Final Count - Success: {successful_jobs} | Failed: {failed_jobs}")

### How to Use

# 1.  **Save the Code:** Save the script above as `run_fem_parallel.py` in the same directory as your FEM analysis script.
# 2.  **Rename Your Script:** Make sure the `SCRIPT_TO_RUN` variable in the parallel script matches the filename of your FEM analysis script. For example, if your script is named `fem_script.py`, change the line to:
#     ```python
#     SCRIPT_TO_RUN = "./fem_script.py" 
#     ```
# 3.  **Run from Terminal:** Execute the parallel script from your terminal:
#     ```bash
#     python run_fem_parallel.py
#     ```
# 4.  **Monitor Progress:** The script will first search for all the `.npy` files and then show a `tqdm` progress bar as it executes the FEM simulations in parallel.
# 5.  **Check Logs:** If any job fails, a message will be printed telling you exactly which log file to check inside the newly created `logs_fem_runs` directory. The logs are named to clearly correspond to the experiment that was run.