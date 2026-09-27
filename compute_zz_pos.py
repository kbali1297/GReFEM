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

# Now it is safe to import your custom modules
from utils import *
from fem_fenics import *
from generate_renders import *
from tqdm import tqdm
import time

def run_experiment(CAD_file_name, load_case, test_case_dir):
    """
    Worker function that processes a single CAD model and loading case.
    """
    #try:
    print(f"--- Starting: {CAD_file_name} | {load_case} ---")
    
    cad_file_path = f"{test_case_dir}/{CAD_file_name}"

    mesh_path = f'{cad_file_path}/renders_pyvista/{CAD_file_name}.obj'
    
    step_file_path = None
    for file in os.listdir(cad_file_path):
        if file.endswith('.step'):
            step_file_path = f'{cad_file_path}/{file}'
            break
            
    if step_file_path is None:
        print(f"Skipping {CAD_file_name}: No .step file found.")
        return

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
    if CAD_file_name in ['Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back']:
        h_max = 1.0 * np.power(total_volume/268, 0.25)
    elif CAD_file_name in['Electrical_Parts_Servos_SG-90_SG90-1-arm-horn']:
        h_max = 0.6 * np.power(total_volume/268, 0.25)
    elif CAD_file_name in ['00200012', '00200016']:
        h_max = 2.0 * np.power(total_volume/268, 0.5)
    elif CAD_file_name in ['00210005']:
        h_max = 1.0 * np.power(total_volume/268, 0.5)
    elif CAD_file_name in['Electrical_Parts_Servos_SG-90_Servo-sg90']:
        h_max = 0.5 * np.power(total_volume/268, 0.3)
    else:
        h_max = 1.0 * np.power(total_volume/268, 0.3)
        
    h_min, h_fine = h_max/5, h_max/10
    
    print(f'Object: {CAD_file_name} | Load Case: {load_case} | Vol: {total_volume:.2f} | Initial h_max: {h_max:.2f}')

    # ==============================================================
    # 1. GENERATE FINE MESH FIRST (With Element Limit Feedback Loop)
    # ==============================================================
    fine_msh_path = f'{cad_file_path}/fine_mesh_{load_case}.msh'
    max_elements = 2000000
    
    import meshio # Ensure meshio is imported for counting

    while True:
        output_msh_fine = generate_or_refine_mesh(
            step_or_mesh_path=step_file_path,
            h_min=h_fine, h_max=h_fine, suffix=None,
            verbose=False, out_msh=fine_msh_path
        )
        
        # Load and count 3D elements
        m = meshio.read(output_msh_fine)
        num_3d_cells = sum(len(cb.data) for cb in m.cells if cb.type in["tetra", "hexahedron"])
                
        print(f"   -> [Check] Fine mesh has {num_3d_cells} elements (h_fine={h_fine:.4f})")
        
        if num_3d_cells <= max_elements:
            output_msh_coarse = generate_or_refine_mesh(
            step_or_mesh_path=step_file_path,
            h_min=h_min, h_max=h_max, suffix=None,
            verbose=False, out_msh=f'{cad_file_path}/coarse_mesh_{load_case}.msh'
            )
            break  # Element count is acceptable, exit loop
        else:
            # Scale h_fine up based on the inverse cubic volume law, plus 10% safety margin
            scale_factor = (num_3d_cells / max_elements) ** (1.0 / 3.0) * 1.10
            h_fine *= scale_factor
            
            # ** Scale h_max and h_min accordingly to maintain the sizing ratios **
            h_max = h_fine * 10.0
            h_min = h_max / 5.0
            
            print(f"   -> Exceeded {max_elements}! Scaled h_fine to {h_fine:.4f} (New h_max={h_max:.4f}). Remeshing fine mesh...")

        
    print(f"   -> Final validated sizes for {CAD_file_name} | {load_case}: h_max={h_max:.4f}, h_min={h_min:.4f}, h_fine={h_fine:.4f}")

      # TEMP EXIT TO TEST MESHING ONLY

    # solved_uh, res_cand = compute_disp_error(
    #     candidate_mshs=[], 
    #     ref_msh=output_msh_fine, outfile=None, 
    #     solve_reference=True, experiment_name=None, 
    #     load_case=load_case
    # )
    
    # zz_pos_file_path = return_zz_field(output_msh_fine, solved_uh['ref'])
    # render_mesh_with_ground_truth(mesh_path, zz_pos_file_path, output_dir=f'{cad_file_path}/renders_zz_pos_{load_case}', loading_case=load_case, n_azimuth=[60,300], n_elevation=[-36, 36], top_percentile=99)

    print(f'+++ Finished: {CAD_file_name} | {load_case} +++')
    # except Exception as e:
    #     print(f"!!! Error processing {CAD_file_name} with {load_case}: {e} !!!")

def worker_wrapper(args, test_case_dir):
    # args is a tuple of (CAD_file_name, experiment_name)
    return run_experiment(args[0], args[1], test_case_dir)

if __name__ == '__main__':
    # 1. FIX THE MPI FORK CRASH
    # Force Python to spawn fresh processes instead of forking the parent's memory
    multiprocessing.set_start_method('spawn', force=True)
    
    test_case_dir = '/data/1bali/GReFEM/test_meshes'
    
    # Optional: If you still want to restrict to specific CADs from your txt file, keep this.
    # Otherwise, you can just rely on os.listdir(test_case_dir) as done below.
    with open('/data/1bali/GReFEM/val_7.04.2026.txt', 'r') as fread:
        valid_cads = set(os.path.basename(line.strip()) for line in fread.readlines())
    
    task_pool =[]
    
    # 2. DYNAMIC TASK DISCOVERY (from your bottom main)
    for cad_file_name in os.listdir(test_case_dir):
        # if cad_file_name not in valid_cads:
        #     continue  # Skip if not in your validation list
            
        cad_folder_path = os.path.join(test_case_dir, cad_file_name)
        if not os.path.isdir(cad_folder_path):
            continue
            

        for load_case in ['compression', 'bending', 'torsion', 'bending_compression', 'torsion_compression']:
            #if not cad_file_name.startswith('005') and load_case in ['compression', 'bending','torsion']: continue  # Skip hidden files/folders
            task_pool.append((cad_file_name, load_case))
                    
    for cad, exp in task_pool:
        #print(f"Queued Task: {cad} | {exp}")
        if cad not in ['00200090']: continue
        run_experiment(cad, exp, test_case_dir)  # Run sequentially for debugging

    # Warning: 100 processes doing 3D meshing + FEniCS matrix solving might crash your RAM!
    # If it freezes, lower this to 20 or 40.
    num_processes = min(100, len(task_pool)) 
    if num_processes > 0:
        print(f"Starting pool with {num_processes} processes...")

        bound_worker = partial(worker_wrapper, test_case_dir=test_case_dir)

        # 4. RUN IN PARALLEL AND COLLECT RESULTS
        with multiprocessing.Pool(processes=num_processes) as pool:
            # pool.map returns a list of results in the exact order of final_tasks
            results_list = []

            with multiprocessing.Pool(processes=num_processes) as pool:
                for (cad, exp), result in zip(
                    task_pool,
                    tqdm(
                        pool.imap_unordered(bound_worker, task_pool),
                        total=len(task_pool),
                        desc="Processing CAD tasks",
                    ),
                ):
                   
                    results_list.append(result)
                    tqdm.write(f"✔ Finished: {cad} | {exp}")