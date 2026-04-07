import os
import subprocess
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
MAX_JOBS = 10

PYTHON_EXEC = "/data/1bali/miniforge3/envs/multi_view_3DQA/bin/python"
SCRIPT_TO_RUN = "./infer_baseline.py"  # <-- MODIFIED
PARENT_DIR = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal"

# =========================
# PARAMETER SWEEPS
# =========================

val_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/val_rebuttal.txt'
with open(val_file_path, 'r') as fread:
    lines = [line.strip() for line in fread]
    geometry_names =[os.path.basename(line) for line in lines]
    MESHES =[f"{PARENT_DIR}/{geometry_name}/renders_pyvista/{geometry_name}.obj" for geometry_name in geometry_names]

# --- MODIFIED PARAMETERS FOR THE NEW SCRIPT ---
FEATURE_ANGLES = [20.0, 30.0, 45.0]  # Dihedral angle thresholds to test
CURVATURE_PERCENTILES = [90.0, 95.0, 98.0] # Curvature percentile thresholds to test

def run_job(args):
    """Executes a single run with the given parameters for the combined baseline."""
    # --- MODIFIED: Unpack new arguments ---
    mesh_path, feature_angle, curvature_percentile = args
    
    mesh_name = os.path.basename(mesh_path).replace(".obj", "")
    
    # --- MODIFIED: Log directory name ---
    log_dir = f"./logs_infer_combined_baseline"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- MODIFIED: Log file name ---
    log_name = f"{mesh_name}_baseline_dihedral{feature_angle}deg_curve{curvature_percentile}p.log"
    log_path = os.path.join(log_dir, log_name)
    
    # --- MODIFIED: Build command with new arguments ---
    cmd =[
        PYTHON_EXEC, "-u", SCRIPT_TO_RUN,
        "--mesh_path", mesh_path,
        "--feature_angle", str(feature_angle),
        "--curvature_percentile", str(curvature_percentile)
    ]
    
    # Execute the command and redirect stdout and stderr to the log file
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    # Return True if successful, False if it crashed
    return result.returncode == 0, log_name


if __name__ == '__main__':
    # --- MODIFIED: Generate combinations from new parameter lists ---
    combinations = list(itertools.product(
        MESHES, FEATURE_ANGLES, CURVATURE_PERCENTILES
    ))
    
    total_jobs = len(combinations)
    print(f"Total jobs to run: {total_jobs}")
    print(f"Running up to {MAX_JOBS} jobs concurrently...\n")
    
    start_time = time.time()
    successful_jobs = 0
    failed_jobs = 0

    with ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
        # Submit all jobs to the executor
        future_to_job = {executor.submit(run_job, combo): combo for combo in combinations}
        
        # Track completion as they finish with tqdm
        with tqdm(total=total_jobs, desc="Processing Baseline Jobs", unit="job") as pbar:
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
                    tqdm.write(f"[ERROR] Job generated an exception: {exc}")
                
                # Update the dynamic stats and tick the progress bar forward
                pbar.set_postfix(Success=successful_jobs, Failed=failed_jobs)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\nAll baseline jobs done in {elapsed_time:.2f} seconds!")
    print(f"Final Count - Success: {successful_jobs} | Failed: {failed_jobs}")