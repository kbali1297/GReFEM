import os
import numpy as np
from utils import *
from fem_fenics import *
from generate_renders import *
import argparse

if __name__ == '__main__':


    test_case_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal_28.03.2026'
    
    with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/val_rebuttal_28.03.2026.txt', 'r') as fread:
        lines = [line.strip() for line in fread.readlines()]
    CAD_file_names = sorted([os.path.basename(line) for line in lines]) 
    for CAD_file_name in CAD_file_names:
        if CAD_file_name.startswith('002'): continue #skip some of the already processed ones
        #CAD_file_name = '00200005'
        # if CAD_file_name in ['00200005', '00200002', '00200008', 
        #                      '00200022','00200030', '00200035', 
        #                      '00200036', '00200037', '00200039', 
        #                      '00200050', '00200060', '00200069',
        #                      '00200070', '00200076', '00200081',
        #                     '00200089', '00200090','00210000',
        #                       '00210005', '00210018','00210021', 
        #                      '00210058','00210067' ]: continue #already processed
        parser = argparse.ArgumentParser(description="Run multiple experiments to infer orthographic views and identify stress concentration areas using LLM.")
        parser.add_argument('--log_file_name', type=str, default=f'{CAD_file_name}__view-ortho__prompt-geo_max__nv-10__prompt_type-geo_max__grid-10__run-1__LLM-gemini-3-flash-preview', help="Path to the 3D mesh file (.obj format).")
        
        args = parser.parse_args()
        refinement_points_path_list = []
        
        #test_case_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal'
        cad_filename = args.log_file_name.split('__')[0]
        view = args.log_file_name.split('view-')[1].split('__')[0]
        prompt = args.log_file_name.split('prompt-')[1].split('__')[0]
        nv = args.log_file_name.split('nv-')[1].split('__')[0]
        grid = args.log_file_name.split('grid-')[1].split('__')[0]
        run = args.log_file_name.split('run-')[1].split('__')[0]
        LLM =  args.log_file_name.split('LLM-')[1].split('.')[0]

        cad_file_path = f"{test_case_dir}/{cad_filename}"
        experiment_name = f"{view}_{nv}views_{LLM}_{grid}grid_{prompt}prompt_{run}run"
        refinement_points_path = f"{cad_file_path}/refinement_points_{experiment_name}.npy"
        cad_file_name = os.path.basename(cad_file_path)
        mesh_path = f'{cad_file_path}/renders_pyvista/{cad_file_name}.obj'
        for file in os.listdir(cad_file_path):
            if file.endswith('.step'):
                step_file_path = f'{cad_file_path}/{file}'
        
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
            raise RuntimeError("No volumes found in STEP file")

        total_volume = 0.0
        # for dim, tag in volumes:
        #     vol = gmsh.model.occ.getMass(dim, tag)
        #     total_volume += vol

        first_vol_dim, first_vol_tag = volumes[0]
        total_volume = gmsh.model.occ.getMass(first_vol_dim, first_vol_tag)
        
        if len(volumes) > 1:
            print(f"Notice: Assembly detected with {len(volumes)} solids. Only considering the first solid (Volume = {total_volume:.2f}).")

        gmsh.finalize()
        #print("Volume =", total_volume)
        if refinement_points_path.split('/')[-2] in ['Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back']:
            h_max = 1.0 * np.power(total_volume/268, 0.25) #h_rel = 1.0 for object_volume = 268 units, shorter sizes since surface mesh is not properly closed
        elif refinement_points_path.split('/')[-2] in ['Electrical_Parts_Servos_SG-90_SG90-1-arm-horn']:
            h_max = 0.6 * np.power(total_volume/268, 0.25) #h_rel = 1.0 for object_volume = 268 units, shorter sizes since surface mesh is not properly closed
        elif refinement_points_path.split('/')[-2] in ['00200012']:
            h_max = 2.0 * np.power(total_volume/268, 0.5) #h_rel = 1.0 for object_volume = 268 units, shorter sizes since surface mesh is not properly closed
        elif refinement_points_path.split('/')[-2] in ['00210005']:
            h_max = 1.0 * np.power(total_volume/268, 0.5) #h_rel = 1.0 for object_volume = 268 units, shorter sizes since surface mesh is not properly closed
        elif refinement_points_path.split('/')[-2] in ['00200016']:
            h_max = 2.0 * np.power(total_volume/268, 0.5) #h_rel = 1.0 for object_volume = 268 units, shorter sizes since surface mesh is not properly closed
        else:
            h_max = 1.0 * np.power(total_volume/268, 0.3) #h_rel = 1.0 for object_volume = 268 units
        h_min, h_fine = h_max/5, h_max/10
        
        print(f'Object Name: {CAD_file_name}, Object volume: {total_volume}, h_max: {h_max}, h_min: {h_min}, h_fine: {h_fine}')
        continue
        refinement_points = np.load(refinement_points_path)
        output_msh_refined = generate_or_refine_mesh(step_or_mesh_path=step_file_path, 
                                                    points_of_interest=refinement_points, 
                                                    h_min=h_min,h_max=h_max,suffix=None,#f'refined_{experiment_name}',
                                                    verbose=False, out_msh=f'{cad_file_path}/refined_mesh/{experiment_name}_refined.msh')# f'{step_file_path.replace(".step", "")}~refined.msh'
        render_tet_mesh_views(output_msh_refined, 
                            output_dir=f"{cad_file_path}/renders_tet3D_pyvista_refined_{experiment_name}", 
                            surf_mesh_path=mesh_path,
                                n_azimuth=12, n_elevation=9, add_axes=False)

        output_msh_coarse = f'{cad_file_path}/coarse_mesh.msh' #generate_or_refine_mesh(step_or_mesh_path=step_file_path, 
                                                    # h_min=h_min,h_max=h_max,suffix=None,#'coarse',
                                                    # verbose=False, out_msh=f'{cad_file_path}/coarse_mesh.msh') #f'{step_file_path.replace(".step", "")}~coarse.msh' #

        output_msh_coarse = generate_or_refine_mesh(step_or_mesh_path=step_file_path, 
                                                    h_min=h_min,h_max=h_max,suffix=None,#'coarse',
                                                    verbose=False, out_msh=f'{cad_file_path}/coarse_mesh.msh') #f'{step_file_path.replace(".step", "")}~coarse.msh' #

        # render_tet_mesh_views(output_msh_coarse, output_dir=f"{cad_file_path}/renders_tet3D_pyvista_coarse", surf_mesh_path=mesh_path,
        #                    n_azimuth=12, n_elevation=9)
        
        output_msh_fine = f'{cad_file_path}/fine_mesh.msh' #generate_or_refine_mesh(step_or_mesh_path=step_file_path,
                                                # h_min=h_fine,h_max=h_fine,suffix=None,#'fine',
                                                # verbose=False, out_msh=f'{cad_file_path}/fine_mesh.msh')#f'{step_file_path.replace(".step", "")}~fine.msh'#

        output_msh_fine = generate_or_refine_mesh(step_or_mesh_path=step_file_path,
                                                h_min=h_fine,h_max=h_fine,suffix=None,#'fine',
                                                verbose=False, out_msh=f'{cad_file_path}/fine_mesh.msh')#f'{step_file_path.replace(".step", "")}~fine.msh'#

        
        # render_tet_mesh_views(output_msh_fine, output_dir=f"{cad_file_path}/renders_tet3D_pyvista_fine", surf_mesh_path=mesh_path,
        #                                 n_azimuth=12, n_elevation=9)
        
        ## Simulate load on the two meshes, See accuracy of the displacement field and energy field
        tet3D_mesh_paths = [output_msh_coarse, output_msh_refined]
        
        solved_uh, res_cand = compute_disp_error(candidate_mshs=tet3D_mesh_paths, ref_msh=output_msh_fine, outfile=None, solve_reference=True, experiment_name=experiment_name)
        # f'{cad_file_path}/mesh_results_fenics/{experiment_name}.log'
        print('Simulation results on candidate meshes:')
        ## Also compute indicator field on a uniformly fine field and compare if the locations match to where the meshing is done
        zz_pos_file_path = return_zz_field(output_msh_fine, solved_uh['ref'])
        
        render_pos_views(zz_pos_file_path, f'{cad_file_path}/renders_zz_pos', 12, 9)

        print('Lets See!')
