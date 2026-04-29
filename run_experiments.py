import os
import subprocess
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
MAX_JOBS = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PYTHON_EXEC = "/data/1bali/miniforge3/envs/multi_view_3DQA/bin/python"
SCRIPT_TO_RUN = "./infer_meshpoints.py"  # Based on your example output
PARENT_DIR = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes" #29.03.2023
val_file_path = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/val.log"
# =========================
# PARAMETER SWEEPS
# =========================

with open(val_file_path, 'r') as fread:
    lines = [line.strip() for line in fread]
geometry_names =sorted([os.path.basename(line) for line in lines])
MESHES =[f"{PARENT_DIR}/{geometry_name}/renders_pyvista/{geometry_name}.obj" for geometry_name in geometry_names]

#/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal/00200037/renders_pyvista_with_meshpoints_ortho_10views_qwen3-vl-235b-a22b-instruct_10grid_geo_maxprompt_3run/view_e0_a120.png
LLM_NAMES =["google/gemini-3-flash-preview", "qwen/qwen3-vl-235b-a22b-instruct", "~anthropic/claude-haiku-4.5", 'openai/gpt-5.4-mini'] # 'openai/gpt-5-mini', "x-ai/grok-4-fast" "anthropic/claude-sonnet-4.5, x-ai/grok-4-fast, openai/gpt-4.1, openai/gpt-5-mini, "qwen/qwen3-vl-235b-a22b-instruct""
GRID_SIZES = [11]
NUM_VIEWS = list(range(1, 11))  # 1 through 10
VIEW_TYPES = ["ortho"]#["ortho", "random"]
RUNS = list(range(1, 6))        # 1 through 5
PROMPT_TYPES = ["geo_max", 'geo_mid', 'geo_none']
LOAD_CASES = ['compression', 'torsion', 'bending'] # 'torsion', 'bending', 'compression'
def run_job(args):
    """Executes a single run with the given parameters."""
    llm_name, prompt_type, grid, load_case, num_views, mesh_path, view_type, run = args
    
    mesh_name = os.path.basename(mesh_path).replace(".obj", "")
    short_llm_name = llm_name.split('/')[-1]
    
    # Create a distinct log directory for the LLM
    log_dir = f"/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes/{mesh_name}/{load_case}_{short_llm_name}_{prompt_type}prompt_{view_type}_{num_views}views_{grid}grid_{run}run"
    os.makedirs(log_dir, exist_ok=True)
    
    # Construct a detailed log file name based on parameters
    log_name = "inference.log"
    log_path = os.path.join(log_dir, log_name)
    
    # Build the command arguments
    cmd =[
        PYTHON_EXEC, "-u", SCRIPT_TO_RUN,
        "--mesh_path", mesh_path,
        "--view_selection_strategy", view_type,
        "--prompt_type", prompt_type,
        "--num_views", str(num_views),
        "--grid_size", str(grid),
        "--LLM_name", llm_name,
        "--run", str(run),
        "--load_case", load_case
    ]
    
    # Execute the command and redirect stdout and stderr to the log file (equivalent to > log.txt 2>&1)
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    # Return True if successful, False if it crashed
    return result.returncode == 0, log_path


if __name__ == '__main__':
    # Generate all possible combinations
    combinations = list(itertools.product(
        LLM_NAMES, PROMPT_TYPES, GRID_SIZES, LOAD_CASES, NUM_VIEWS, MESHES, VIEW_TYPES, RUNS
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
        with tqdm(total=total_jobs, desc="Processing Jobs", unit="job") as pbar:
            for future in as_completed(future_to_job):
                combo = future_to_job[future]

                llm_name, prompt_type, grid, load_case, num_views, mesh_path, view_type, run = combo
                mesh_name = os.path.basename(mesh_path).replace(".obj", "")
                short_llm = llm_name.split("/")[-1]

                case_str = (
                    f"{mesh_name} | {load_case} | {short_llm} | "
                    f"{prompt_type} | {view_type} | nv={num_views} | grid={grid} | run={run}"
                )

                try:
                    success, log_name = future.result()
                    if success:
                        successful_jobs += 1
                    else:
                        failed_jobs += 1
                        tqdm.write(f"[FAILED] {case_str} -> {log_name}")
                except Exception as exc:
                    failed_jobs += 1
                    tqdm.write(f"[ERROR] {case_str} -> {exc}")

                # 👇 show full case in progress bar
                pbar.set_postfix(
                    Success=successful_jobs,
                    Failed=failed_jobs,
                    Case=case_str[:80]  # truncate to avoid breaking tqdm formatting
                )
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\nAll done in {elapsed_time:.2f} seconds!")
    print(f"Final Count - Success: {successful_jobs} | Failed: {failed_jobs}")