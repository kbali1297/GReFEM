import os
import sys

# 1. FIX OPENMPI PATH LOSS (Must happen BEFORE importing fem_fenics / dolfinx)
os.environ["OPAL_PREFIX"] = "/data/1bali/GReFEM_env"
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

def run_experiment(CAD_file_name, loading_case, test_case_dir):
    """
    Worker function that processes a single CAD model and loading case.
    """
    try:
        print(f"--- Starting: {CAD_file_name} | {loading_case} ---")
        # Replaced argparse: Construct the log file string directly
        log_file_name = f'{CAD_file_name}__view-ortho__prompt-geo_max__nv-10__prompt_type-geo_max__grid-10__run-1__LLM-gemini-3-flash-preview'
        
        cad_filename = log_file_name.split('__')[0]
        view = log_file_name.split('view-')[1].split('__')[0]
        prompt = log_file_name.split('prompt-')[1].split('__')[0]
        nv = log_file_name.split('nv-')[1].split('__')[0]
        grid = log_file_name.split('grid-')[1].split('__')[0]
        run = log_file_name.split('run-')[1].split('__')[0]
        LLM = log_file_name.split('LLM-')[1].split('.')[0]

        cad_file_path = f"{test_case_dir}/{cad_filename}"
        experiment_name = f"{view}_{nv}views_{LLM}_{grid}grid_{prompt}prompt_{run}run"
        refinement_points_path = f"{cad_file_path}/refinement_points_{experiment_name}.npy"
        
        cad_file_name_base = os.path.basename(cad_file_path)
        mesh_path = f'{cad_file_path}/renders_pyvista/{cad_file_name_base}.obj'
        
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
        object_radius = np.linalg.norm(surf_mesh.vertices - object_center, axis=1).max()
        
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
        
        # Calculate sizing
        path_check = refinement_points_path.split('/')[-2]
        if path_check in ['Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back']:
            h_max = 1.0 * np.power(total_volume/268, 0.25)
        elif path_check in ['Electrical_Parts_Servos_SG-90_SG90-1-arm-horn']:
            h_max = 0.6 * np.power(total_volume/268, 0.25)
        elif path_check in ['00200012', '00200016']:
            h_max = 2.0 * np.power(total_volume/268, 0.5)
        elif path_check in ['00210005']:
            h_max = 1.0 * np.power(total_volume/268, 0.5)
        else:
            h_max = 1.0 * np.power(total_volume/268, 0.3)
            
        h_min, h_fine = h_max/5, h_max/10
        
        print(f'Object: {CAD_file_name} | Load: {loading_case} | Vol: {total_volume:.2f} | h_max: {h_max:.2f}')

        # IMPORTANT: Appended loading_case to mesh files to prevent process race-conditions writing to the same file!
        coarse_mesh_target = f'{cad_file_path}/coarse_mesh_{loading_case}.msh'
        output_msh_coarse = generate_or_refine_mesh(
            step_or_mesh_path=step_file_path, 
            h_min=h_min, h_max=h_max, suffix=None,
            verbose=False, out_msh=coarse_mesh_target
        )

        fine_mesh_target = f'{cad_file_path}/fine_mesh_{loading_case}.msh'
        output_msh_fine = generate_or_refine_mesh(
            step_or_mesh_path=step_file_path,
            h_min=h_fine, h_max=h_fine, suffix=None,
            verbose=False, out_msh=fine_mesh_target
        )

        ## Simulate load
        tet3D_mesh_paths = [output_msh_coarse]
        solved_uh, res_cand = compute_disp_error(
            candidate_mshs=tet3D_mesh_paths, 
            ref_msh=output_msh_fine, outfile=None, 
            solve_reference=True, experiment_name=experiment_name, 
            loading_case=loading_case
        )
        
        zz_pos_file_path = return_zz_field(output_msh_fine, solved_uh['ref'])
        render_pos_views(zz_pos_file_path, f'{cad_file_path}/renders_zz_pos_{loading_case}', 12, 9)

        print(f'+++ Finished: {CAD_file_name} | {loading_case} +++')
    
    except Exception as e:
        print(f"!!! Error processing {CAD_file_name} with {loading_case}: {e} !!!")

def worker_wrapper(args, test_case_dir):
    return run_experiment(args[0], args[1], test_case_dir)

if __name__ == '__main__':
    # 2. FIX THE MPI FORK CRASH
    # Force Python to spawn fresh processes instead of forking the parent's memory
    multiprocessing.set_start_method('spawn', force=True)
    
    test_case_dir = '/data/1bali/GReFEM/test_meshes_7.04.2026'
    
    with open('/data/1bali/GReFEM/val_7.04.2026.txt', 'r') as fread:
        lines = [line.strip() for line in fread.readlines()]
    
    CAD_file_names = sorted([os.path.basename(line) for line in lines]) 
    
    # NOTE: "shear" is not defined in your earlier FEniCS solvers dict unless you added it!
    # Valid ones from the previous code were: "compression", "bending", "torsion"
    loading_cases = ["compression", "bending", "torsion", "shear"] 
    
    tasks = []
    for CAD_file_name in CAD_file_names:
        for loading_case in loading_cases:
            tasks.append((CAD_file_name, loading_case))

    print(f"Total combinations to process: {len(tasks)}")

    # Warning: 80 processes doing 3D meshing + FEniCS matrix solving might crash your RAM!
    # If it freezes, lower this to 20 or 40.
    num_processes = 80 
    print(f"Starting pool with {num_processes} processes...")

    bound_worker = partial(worker_wrapper, test_case_dir=test_case_dir)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(bound_worker, tasks)
        
    print("All processing complete!")