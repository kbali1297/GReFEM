import os
import sys

#1. FIX OPENMPI PATH LOSS (Must happen BEFORE importing fem_fenics / dolfinx)
os.environ["OPAL_PREFIX"] = "/data/1bali/miniforge3/envs/multi_view_3DQA" #--- IGNORE ---
# Optional: Prevent OpenMPI from complaining about too many processes
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1" 

import numpy as np
import trimesh 
import gmsh    
import multiprocessing
from functools import partial
from tqdm import tqdm  # <-- IMPORT TQDM
from cad_element_sizes import get_element_sizes, CAD_ELEMENT_SIZES
# Now it is safe to import your custom modules
from utils import *
from fem_fenics import *
from generate_renders import *

def run_experiment(CAD_file_name, experiment_name, test_case_dir):
    """
    Worker function that processes a single CAD model and loading case.
    """
    # Note: Prints inside worker processes might slightly overlap with tqdm, 
    # but they will still show up in the terminal log.
    print(f"--- Starting: {CAD_file_name} | {experiment_name} ---")
    
    cad_file_path = f"{test_case_dir}/{CAD_file_name}"

    loading_case = experiment_name.split('_gemini')[0]  # Assuming experiment_name starts with loading case like "compression", "bending", etc.
    refinement_points_path = f"{cad_file_path}/{experiment_name}/refinement_points_prefilt.npy"
    
    mesh_path = f'{cad_file_path}/renders_pyvista/{CAD_file_name}.obj'
    
    step_file_path = None
    for file in os.listdir(cad_file_path):
        if file.endswith('.step'):
            step_file_path = f'{cad_file_path}/{file}'
            break
            
    if step_file_path is None:
        print(f"Skipping {CAD_file_name}: No .step file found.")
        return None

    surf_mesh = trimesh.load(mesh_path)
    object_center = np.mean(surf_mesh.vertices, axis=0)
    
    ## Compute object volume  
    gmsh.initialize()
    gmsh.model.add("vol")
    gmsh.model.occ.importShapes(step_file_path)
    gmsh.model.occ.synchronize()

    # Get all 3D entities (volumes)
    volumes = gmsh.model.getEntities(dim=3)

    if len(volumes) == 0:
        gmsh.finalize()
        raise RuntimeError(f"No volumes found in STEP file: {step_file_path}")

    first_vol_dim, first_vol_tag = volumes[0]
    total_volume = gmsh.model.occ.getMass(first_vol_dim, first_vol_tag)
    
    if len(volumes) > 1:
        print(f"Notice: Assembly detected in {CAD_file_name} with {len(volumes)} solids. Only considering the first solid (Volume = {total_volume:.2f}).")

    gmsh.finalize()
    
    # Calculate initial sizing
    # if CAD_file_name in['Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back']:
    #     h_max = 1.0 * np.power(total_volume/268, 0.25)
    # elif CAD_file_name in['Electrical_Parts_Servos_SG-90_SG90-1-arm-horn']:
    #     h_max = 0.6 * np.power(total_volume/268, 0.25)
    # elif CAD_file_name in['00200012', '00200016']:
    #     h_max = 2.0 * np.power(total_volume/268, 0.5)
    # elif CAD_file_name in['00210005']:
    #     h_max = 1.0 * np.power(total_volume/268, 0.5)
    # elif CAD_file_name in['Electrical_Parts_Servos_SG-90_Servo-sg90']:
    #     h_max = 0.5 * np.power(total_volume/268, 0.3)
    # else:
    #     h_max = 1.0 * np.power(total_volume/268, 0.3)
        
    # h_min, h_fine = h_max/5, h_max/10
    
    # print(f'Object: {CAD_file_name} | Experiment: {experiment_name} | Vol: {total_volume:.2f} | Initial h_max: {h_max:.2f}')

    # ==============================================================
    # 1. GENERATE FINE MESH FIRST (With Element Limit Feedback Loop)
    # ==============================================================
    fine_msh_path = f'{cad_file_path}/fine_mesh_{loading_case}.msh'
    #max_elements = 1000000
    
    # import meshio # Ensure meshio is imported for counting

    # while True:
    #     output_msh_fine = generate_or_refine_mesh(
    #         step_or_mesh_path=step_file_path,
    #         h_min=h_fine, h_max=h_fine, suffix=None,
    #         verbose=False, out_msh=fine_msh_path
    #     )
        
    #     # Load and count 3D elements
    #     m = meshio.read(output_msh_fine)
    #     num_3d_cells = sum(len(cb.data) for cb in m.cells if cb.type in["tetra", "hexahedron"])
                
    #     if num_3d_cells <= max_elements:
    #         break  # Element count is acceptable, exit loop
    #     else:
    #         # Scale h_fine up based on the inverse cubic volume law, plus 10% safety margin
    #         scale_factor = (num_3d_cells / max_elements) ** (1.0 / 3.0) * 1.10
    #         h_fine *= scale_factor
            
    #         # ** Scale h_max and h_min accordingly to maintain the sizing ratios **
    #         h_max = h_fine * 10.0
    #         h_min = h_max / 5.0

    # ==============================================================
    # 2. GENERATE REFINED AND COARSE MESHES USING VALIDATED SIZES
    # ==============================================================
    refinement_points = np.load(refinement_points_path)
    refined_msh_path = f'{cad_file_path}/{experiment_name}/refined_mesh_prefilt.msh'
    h_min = CAD_ELEMENT_SIZES[CAD_file_name]['h_min']
    h_max = CAD_ELEMENT_SIZES[CAD_file_name]['h_max']
    out_msh_coarse = generate_or_refine_mesh(
        step_or_mesh_path=step_file_path,
        h_min=h_max, h_max=h_max, suffix=None,
        verbose=False, out_msh=f'{cad_file_path}/{experiment_name}/coarse_mesh_{loading_case}.msh'
    )
    output_msh_refined = generate_or_refine_mesh(
        step_or_mesh_path=step_file_path,
        h_min=h_min, h_max=h_max, suffix=None,
        verbose=False, out_msh=refined_msh_path, points_of_interest=refinement_points
    )
    
    render_tet_mesh_views(output_msh_refined, 
                        output_dir=f"{cad_file_path}/{experiment_name}/renders_tet3D_pyvista_refined", 
                        surf_mesh_path=mesh_path,
                        n_azimuth=[60, 300], n_elevation=[-36, 36], add_axes=False)

    coarse_msh_path = f'{cad_file_path}/{experiment_name}/coarse_mesh_{loading_case}.msh'
    
   
    exp_folder = f'{cad_file_path}/{experiment_name}'
    
    ## Simulate load
    # Pass the isolated coarse mesh instead of the shared one
    #coarse_msh_path_exp = f'{cad_file_path}/coarse_mesh_{loading_case}.msh'
    tet3D_mesh_paths =[coarse_msh_path, output_msh_refined]
    
    solved_uh, res_cand = compute_disp_error(
        candidate_mshs=tet3D_mesh_paths, 
        ref_msh=fine_msh_path, outfile=None, 
        solve_reference=False, experiment_name=experiment_name, 
        load_case=loading_case
    )
    
    return res_cand

def worker_wrapper(args, test_case_dir):
    """
    Wraps the worker function so we can return the CAD and Experiment names 
    back to the main thread alongside the result.
    """
    cad_name, exp_name = args
    try:
        res = run_experiment(cad_name, exp_name, test_case_dir)
        return cad_name, exp_name, res, None
    except Exception as e:
        return cad_name, exp_name, None, str(e)


if __name__ == '__main__':
    # 1. FIX THE MPI FORK CRASH
    multiprocessing.set_start_method('spawn', force=True)
    
    test_case_dir = '/data/1bali/GReFEM/test_meshes'
    
    task_pool =[]
    
    # 2. DYNAMIC TASK DISCOVERY
    for cad_file_name in os.listdir(test_case_dir):
        cad_folder_path = os.path.join(test_case_dir, cad_file_name)
        if not os.path.isdir(cad_folder_path):
            continue
            
        for fname in os.listdir(cad_folder_path):
            if 'gemini' in fname and fname.startswith(('compression', 'bending', 'torsion', 'bending_compression', 'torsion_compression')) and '1run' in fname and '11' in fname:
                if 'ortho' in fname:
                    task_pool.append((cad_file_name, fname))
                elif 'random' in fname:
                    task_pool.append((cad_file_name, fname))

    log_file_paths = ['/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_1.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_2.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_3.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_4.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_5.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_6.log',
                      '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_7.log',]
    ## Task discovery for ERROR sims
    # log_file_paths = ['/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026.log',
    #                   '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_2.log',
    #                   '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_3.log',
    #                   '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_4.log',]
    
    succesful_taskpool = []
    for log_file_path in log_file_paths:
        with open(log_file_path, 'r') as log_file:
            log_lines = log_file.readlines()
            for line in log_lines:
                if '[SUCCESS]' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        cad_name = parts[0].split(']')[-1].strip()
                        exp_name = parts[1].strip()
                        succesful_taskpool.append((cad_name, exp_name))

    # Filter out successful tasks from the main task pool
    task_pool = [task for task in task_pool if task not in succesful_taskpool] 

    # for task in tqdm(task_pool):
    #     #if task[0] in done_cads: continue
    #     #if not 'geo_max' in task[1] or not 'ortho' in task[1] or not 'gemini' in task[1]: continue
    #     #compression_gemini-3-flash-preview_geo_noneprompt_ortho_4views_11grid_3run
    #     #if task[0] not in ['Mechanical_Parts_Mountings_SHF08_SHF08'] or task[1] not in ['compression_gemini-3-flash-preview_geo_noneprompt_ortho_4views_11grid_3run']: continue # want to debug this specific case first
    #     print(f"\nProcessing CAD: {task[0]} | Experiment: {task[1]}")
    #     res_cand = run_experiment(task[0], task[1], test_case_dir)        
    #     #done_cads.append(task[0])
    #     #res_cands[task[0]] = res_cand

    #print(res_cands)
    print(f"Total combinations to process after filtering: {len(task_pool)}")
    #bending_claude-haiku-4.5_geo_maxprompt_ortho_1views_11grid_3run
    # unique_cads = set(cad for cad, _ in task_pool)
    # unique_loading_cases = set(exp.split('_gemini')[0] for _, exp in task_pool)
    # unique_prompts = set(exp.split('preview_')[1].split('prompt')[0] for _, exp in task_pool)
    # unique_view_selection = set('ortho' if 'ortho' in exp else 'random' for _, exp in task_pool)
    # unqique_num_views = set(exp.split('views')[0].split('_')[-1].replace('views', '') for _, exp in task_pool)
    num_processes = min(100, len(task_pool)) 
    res_cands = {}
    
    if num_processes > 0:
        print(f"Starting pool with {num_processes} processes...\n")

        bound_worker = partial(worker_wrapper, test_case_dir=test_case_dir)

        # 4. RUN IN PARALLEL AND COLLECT RESULTS USING TQDM
        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap_unordered yields results as soon as they finish, perfect for a progress bar!
            iterator = pool.imap_unordered(bound_worker, task_pool)
            
            # Wrap the iterator with tqdm
            with tqdm(total=len(task_pool), desc="Processing meshes", unit="task") as pbar:
                for cad_name, exp_name, res, error in iterator:
                    
                    # Log case-specific info safely using tqdm.write()
                    # (This prevents text from breaking the visual progress bar)
                    if error:
                        tqdm.write(f"[ERROR] {cad_name} | {exp_name} | Failed with: {error}")
                    else:
                        tqdm.write(f"[SUCCESS] {cad_name} | {exp_name} | Result: {res}")
                        res_cands[f'{cad_name}_{exp_name}'] = res
                        
                    # Update the dynamic postfix to show what just finished
                    #short_exp = exp_name.split('_')[0] # e.g., 'compression'
                    pbar.set_postfix(last_cad=cad_name, last_exp=exp_name)
                    pbar.update(1)
            
        print("\nAll processing complete! Final Results Summary:")
        for cad_exp, res in res_cands.items():
            print(f"{cad_exp}: {res}")
    else:
        print("No tasks met the filtering criteria. Check your folder names and conditions.")
# if __name__ == '__main__':
#     # Example of running a single experiment without multiprocessing for debugging
#     test_case_dir = './test_meshes'
#     #CAD_file_name = 'Electrical_Parts_Servos_SG-90_Servo-sg90'
#     for cad_file_name in os.listdir(test_case_dir):
#         cad_folder_path = os.path.join(test_case_dir, cad_file_name)
#         if not os.path.isdir(cad_folder_path):
#             continue
            
#         for fname in os.listdir(cad_folder_path):
#             if 'gemini' in fname and fname.startswith(('compression', 'bending', 'torsion', 'bending_compression', 'torsion_compression')) and '1run' in fname and '11' in fname:
#                 if 'ortho' in fname:
#                     task_pool.append((cad_file_name, fname))
#                 elif 'random' in fname:
#                     task_pool.append((cad_file_name, fname))

#     ## Task discovery for ERROR sims
#     log_file_paths = ['/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026.log',
#                       '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_2.log',
#                       '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_3.log',
#                       '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_3.05.2026_4.log',]
#     succesful_taskpool = []
#     for log_file_path in log_file_paths:
#         with open(log_file_path, 'r') as log_file:
#             log_lines = log_file.readlines()
#             for line in log_lines:
#                 if '[SUCCESS]' in line:
#                     parts = line.split('|')
#                     if len(parts) >= 3:
#                         cad_name = parts[0].split(']')[-1].strip()
#                         exp_name = parts[1].strip()
#                         succesful_taskpool.append((cad_name, exp_name))

#     # Filter out successful tasks from the main task pool
#     task_pool = [task for task in task_pool if task not in succesful_taskpool] 

                
#     done_cads = [] 
#     res_cands = {}           
#     for task in tqdm(task_pool):
#         #if task[0] in done_cads: continue
#         #if not 'geo_max' in task[1] or not 'ortho' in task[1] or not 'gemini' in task[1]: continue
#         #compression_gemini-3-flash-preview_geo_noneprompt_ortho_4views_11grid_3run
#         #if task[0] not in ['Mechanical_Parts_Mountings_SHF08_SHF08'] or task[1] not in ['compression_gemini-3-flash-preview_geo_noneprompt_ortho_4views_11grid_3run']: continue # want to debug this specific case first
#         print(f"\nProcessing CAD: {task[0]} | Experiment: {task[1]}")
#         res_cand = run_experiment(task[0], task[1], test_case_dir)        
#         done_cads.append(task[0])
#         res_cands[task[0]] = res_cand

#     print(res_cands)