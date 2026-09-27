api_key = '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'
import requests
import base64
from utils import *
import trimesh
from generate_renders import render_mesh_views
import numpy as np
import pyvista as pv
import cv2
from infer import infer_NN
import argparse
from tqdm import tqdm
import os
import sys

if __name__ == '__main__':
    
    parent_dir = '/data/1bali/GReFEM/test_meshes' 
    CAD_file_name = '00520044'
    load_case = 'compression'
    num_views = 6
    prompt_type = 'geo_max'
    grid_size = 11
    run=1
    llm_name = "google/gemini-3-flash-preview" # "google/gemini-flash-latest" --- IGNORE ---
    parser = argparse.ArgumentParser(description="Run multiple experiments to infer orthographic views and identify stress concentration areas using LLM.")
    parser.add_argument('--mesh_path', type=str, default=f'{parent_dir}/{CAD_file_name}/renders_pyvista/{CAD_file_name}.obj')
    parser.add_argument('--model_ckpt', type=str, default='./model_saves/ortho_view_selector_40.pth')
    parser.add_argument('--prompt_type', type=str, default=prompt_type) # 'geo_max', 'geo_mid', 'geo_none'
    parser.add_argument('--grid_size', type=int, default=grid_size)
    parser.add_argument('--LLM_name', type=str, default=llm_name)
    parser.add_argument('--view_selection_strategy', type=str, default="ortho")
    parser.add_argument('--num_views', type=int, default=num_views)
    parser.add_argument('--num_perseptive_views', type=int, default=2)
    parser.add_argument('--load_case', type=str, default=load_case) #'torsion', 'bending', 'compression'
    parser.add_argument('--edge_angle_thresh', type=float, default=30)
    parser.add_argument('--run', type=int, default=run)
    args = parser.parse_args()

    print(args)
    parent_dir = '/'.join(args.mesh_path.split('/')[:-3])
    model_name = args.LLM_name.split('/')[-1]
    experiment_name = f'{args.load_case}_{model_name}_{args.prompt_type}prompt_{args.view_selection_strategy}_{args.num_views}views_{args.grid_size}grid_{args.run}run'

    gridx, gridy = args.grid_size, args.grid_size  
    mesh_file_path = args.mesh_path
    cad_object = os.path.basename(mesh_file_path).replace(".obj", "") 
    
    model_ckpt = args.model_ckpt
    
    view_dir = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial'
    all_loaded_view_paths =[f'{parent_dir}/{cad_object}/renders_pyvista_mesh_{args.load_case}/{view_name}' for view_name in os.listdir(view_dir) if not view_name.startswith('view_e-90') and not view_name.startswith('view_e90')]
    perspective_views = np.random.choice(all_loaded_view_paths, size=2, replace=False).tolist()

    all_view_paths = [f'{view_dir}/{view_name}' for view_name in sorted(os.listdir(view_dir))]
    infer_views_paths =[]
    if args.view_selection_strategy == 'ortho':
        with open(f'{parent_dir}/{cad_object}/pred_ortho_views2.log', 'r') as fread:
            # Read lines, strip whitespace
            raw_lines =[line.strip() for line in fread.readlines() if line.strip()][:args.num_views]
            
            for line in raw_lines:
                # Extract JUST the image name (e.g., 'view_e0_a0.png') from the old absolute path
                view_name = os.path.basename(line)
                
                # Reconstruct the correct path using the CURRENT parent_dir
                current_correct_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}'
                
                # Make sure the file actually exists before we try to process it
                if not os.path.exists(current_correct_path):
                    raise FileNotFoundError(f"Image missing or network drive timeout: {current_correct_path}")
                
                infer_views_paths.append(current_correct_path)
    elif args.view_selection_strategy == 'random':
        remaining_views = list(set(all_view_paths) - set(perspective_views))
        infer_views_paths = np.random.choice(remaining_views, size=args.num_views, replace=False).tolist()

    #infer_views_paths.extend([f'{view_dir}/{view_name}' for view_name in os.listdir(view_dir) if view_name.startswith('view_e-90_a0') or view_name.startswith('view_e90_a0')])
    
    output_grid_dir = f'{parent_dir}/{cad_object}/{experiment_name}'
    os.makedirs(output_grid_dir, exist_ok=True)
    
    gridded_views =[]
    gridx, gridy = args.grid_size, args.grid_size
    for img_path in tqdm(infer_views_paths, desc="Overlaying Grids", file=sys.stdout):
        gridded_path = overlay_grid(img_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
                                    font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
                                    line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    arrow_cell=None, output_dir=output_grid_dir)
        gridded_views.append(gridded_path)

    print(f'Top Views Gridded: {gridded_views}')

    # --- Prompt Construction ---
    prompt = get_prompt_1(args.load_case, prompt_type=args.prompt_type, perspective_views=perspective_views, gridded_views=gridded_views)  # You would define this function to return your actual prompt text

    messages =[]
    inference_imgs =[]
    img_paths = perspective_views + gridded_views
    for img_path in img_paths:
        with open(img_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            inference_imgs.append(encoded_img)

    content =[{"type": "text", "text": prompt}]
    content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
    messages.append({"role": "user", "content": content})
    
    # --- Protected API Call ---
    print(f"Prompt Stage 1 Region Selection: Waiting for OpenRouter API Response ({args.LLM_name})...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": args.LLM_name, "temperature": 0.0, "messages": messages},
            timeout=60  # Prevent infinite hanging
        )
        response.raise_for_status() # Catches 429, 502, etc.
        response_data = response.json()
        
        if 'choices' not in response_data:
            print(f"API Error Response format: {response_data}")
            sys.exit(1)
            
        output = response_data['choices'][0]['message']['content']
        print("Response Received:", output)
        
    except Exception as e:
        print(f"API Request Failed: {e}")
        sys.exit(1) # Exit cleanly to inform parent script

    pred_cells_lists = parse_geom_feature_cells(output, prompt_type=args.prompt_type, infer_views=infer_views_paths, feature_categories=['I.C.E', 'E.C/P.C', 'T.H', 'F'])
    
    # output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}_{args.load_case}'
    # os.makedirs(output_2d_dir, exist_ok=True)
    
    pixel_coords_ICE, pixel_coords_EC_PC, pixel_coords_TH, pixel_coords_F = [], [], [], []
    output_img_paths = set()  # Use a set to avoid duplicates
    
    ## Define point density for point prediction and projection, will use the same for the geometric heuristic
    mesh = trimesh.load(mesh_file_path, force='mesh')
    bbox_min, bbox_max = mesh.bounds  # shape (2, 3)
    point_density = np.linalg.norm(bbox_max - bbox_min) / (66.667 * 2) # Adjust divisor for more/less density
    
    for i, view_path in enumerate(gridded_views):
        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view_path)}'
        
        #out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8
        up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,1)
        img = cv2.imread(view_path)
        if img is None:
            print(f"Failed to load image: {view_path}")
            continue
        
        s_step, p_offset = calculate_dynamic_sampling_params(
        point_density=point_density, 
        img_shape=(1000, 1000), 
        cam_pos=cam_pos, 
        F_pos=F_pos, 
        parallel_scale=parallel_scale, 
        orthographic=True)

        img_numpy = np.array(img)
        geom = extract_all_contours_geometry(img_numpy, sampling_step=s_step, pixel_offset=p_offset, 
                    cam_pos=cam_pos, F_pos=F_pos, up_cam_vec=up_cam_vec, img_dims = (1000,1000), parallel_scale=parallel_scale, mesh=mesh)
        ec_pc_candidates, th_candidates =[], []
        for contour in geom:
            # insert sampling distance into each
            ec_pc_bool, ec_pc_points = detect_extrusion_boundary_or_protruding_boundary_vectorized(geom=contour, consensus_ratio=0.8)  # You would fill in the actual mesh and camera parameters here
            if ec_pc_bool:
                ec_pc_candidates.append(contour)
            elif ec_pc_bool==False and len(ec_pc_points['p_on']) > 0:
                print(f"Found {len(ec_pc_points['p_on'])} EC/PC candidate points that are close to the contour but didn't meet the consensus ratio. These could be borderline cases worth visualizing.")
                ec_pc_candidates.append(ec_pc_points)  # Optionally include these borderline cases for visualization
            elif detect_through_hole(geom=contour):  # You would fill in the actual mesh and camera parameters here
                th_candidates.append(contour)
        
        # Plot ec_pc candidates for visual debugging
        # debug_img = img.copy()
        # for candidate in ec_pc_candidates:
        #     for pt in candidate['p_on']:
        #         cv2.circle(debug_img, tuple(pt.astype(int)), radius=3, color=(255, 0, 0), thickness=-1)
        # debug_out_path = f'{output_2d_dir}/debug_ec_pc_{os.path.basename(view_path)}'
        # cv2.imwrite(debug_out_path, debug_img)
        # print(f"Saved EC/PC candidate debug image to {debug_out_path}") 

        # #Plot th candidates for visual debugging
        # debug_img_th = img.copy()
        # for candidate in th_candidates:
        #     for pt in candidate['p_on']:
        #         cv2.circle(debug_img_th, tuple(pt.astype(int)), radius=3, color=(0, 0, 255), thickness=-1)
        # debug_out_path_th = f'{output_2d_dir}/debug_th_{os.path.basename(view_path)}'
        # cv2.imwrite(debug_out_path_th, debug_img_th)
        # print(f"Saved TH candidate debug image to {debug_out_path_th}")

         #Plot f candidates for visual debugging
        # debug_img_f = img.copy()
        # for candidate in f_candidates:
        #     for pt in candidate['p_on']:
        #         cv2.circle(debug_img_f, tuple(pt.astype(int)), radius=3, color=(0, 0, 255), thickness=-1)
        # debug_out_path_f = f'{output_2d_dir}/debug_f_{os.path.basename(view_path)}'
        # cv2.imwrite(debug_out_path_f, debug_img_f)
        # print(f"Saved F candidate debug image to {debug_out_path_f}")

        # Here you would call analyze_cell_patch_vectorized() for each cell patch in the image
        pred_cells = pred_cells_lists[i]

        pixels_coords_view =[]    
        for category, patch_nums in pred_cells.items():
            
            
            if category == 'I.C.E':
                #print(f"Analyzing Internal Concave Edges in {view_path} for patches {patch_nums}")
                pixel_candidates = []
                for patch_num in patch_nums:
                    ## convert image to numpy array and extract the patch corresponding to patch_num
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    candidates = edge_detection_contour(view_path, patch_num, gridx, gridy, angle_thresh=args.edge_angle_thresh)
                    pixel_candidates.extend(candidates)
                
                pixel_coords_ICE.append(pixel_candidates)

                # Debug visualization for ICE candidates
                # mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 255, 0), 
                #                                  spot_positions=pixel_candidates, output_path=out_path)
    
            elif category == 'E.C/P.C':
                #print(f"Analyzing Extruded Contour / Portruding Contour in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for extrusion boundaries / protruding boundaries
                pixel_candidates = []
                for patch_num in patch_nums:
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    for candidate in ec_pc_candidates:
                        
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixel_candidates.extend(candidate['p_on'])
                
                # Debug visualization for EC/PC candidates
                # mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                #                                  spot_positions=pixel_candidates, output_path=out_path)
                pixel_coords_EC_PC.append(pixel_candidates)

            elif category == 'T.H':
                #print(f"Analyzing Through Holes / Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for through holes / fillets
                pixel_candidates = []
                for patch_num in patch_nums:
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    for candidate in th_candidates:
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixel_candidates.extend(candidate['p_on'])
                
                
                pixel_coords_TH.append(pixel_candidates)
                # Debug visualization for TH candidates
                # mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 0, 255), 
                #                                  spot_positions=pixel_candidates, output_path=out_path)
            
            elif category == 'F':
                #print(f"Analyzing Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for fillets
            
                # pixel_candidates, concave_bool = detect_parts_of_contour(geom, patch_nums, gridx, gridy, img_numpy.shape)
                
                # # output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                # #                                 spot_positions=pixel_candidates, output_path=out_path))
                # if not concave_bool: continue
                # pixel_coords_F.append(pixel_candidates)
                pixel_candidates = []
                concave_parts_img = extract_concave_parts(geom)

                concave_contours = [part['concave_pixels'] for part in concave_parts_img]
                
                for concave_contour in concave_contours:
                    for patch_num in patch_nums:
                        patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                        if is_contour_in_patch(concave_contour, patch_num, gridx, gridy, img_numpy.shape):
                            pixel_candidates.extend(concave_contour)
                pixel_coords_F.append(pixel_candidates)
                # Debug visualization for F candidates
                # mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                #                                  spot_positions=pixel_candidates, output_path=out_path)

                print(f"Finished processing {category} for {view_path}. Found {len(pixel_candidates)} candidate pixels.")

    ## Next Steps: Project all pixel_coords_view back to 3D using pixel_to_mesh.
    ## Continued projections to be done only for I.C.E, F and T.H for now, since E.C/P.C are visible already in the view.
    ## Need to then check the filteration prompt for all the loading cases
    ## Need to train orthoviews to select orthographic views on even more cad data (ideally as much as possible!! Train on as much data as possible and add positional embeddings to thhe views after arranging them in that order)

    # --- 3D Raycasting (VECTORIZED) ---
    mark_points_ICE, mark_points_F, mark_points_TH =[], [], []
    for view, point_list_ICE, point_list_F, point_list_TH in tqdm(zip(gridded_views, pixel_coords_ICE, pixel_coords_F, pixel_coords_TH), total=len(gridded_views), desc="Mapping ICE to 3D Mesh", file=sys.stdout):

        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'

        #out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8

        for edge_category, point_list in zip(['I.C.E', 'F', 'T.H'], [point_list_ICE, point_list_F, point_list_TH]):
            
            edge_list = point_list
            if len(edge_list) == 0: continue

            u_arr =[px[0] for px in edge_list]
            v_arr =[px[1] for px in edge_list]

            up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,-1)
            all_hits = pixel_to_mesh(cam_pos=cam_pos, F_pos=F_pos, u=u_arr, v=v_arr, 
                                    up_cam_vec=np.array(up_cam_vec), parallel_scale=parallel_scale,
                                    img_H=1000, img_W=1000, mesh=mesh, orthographic=True, return_all_hits=True)

            mark_points = []
            for p_world in all_hits:
                if p_world is not None and p_world.shape[0] >= 2:
                    for j in range(0, p_world.shape[0] - 1, 2):
                        if j>0: break
                        p_enter = p_world[j]
                        p_exit = p_world[j+1]
                        edge_points_on_mesh = p_enter + (p_exit - p_enter) * np.linspace(0, 1, max(2, round(np.linalg.norm(p_exit - p_enter)/point_density)))[:, None]
                        mark_points.extend([p_enter] + list(edge_points_on_mesh) + [p_exit])
                        
            if edge_category == 'I.C.E':
                mark_points_ICE.extend(mark_points)
            elif edge_category == 'F':
                mark_points_F.extend(mark_points)
            elif edge_category == 'T.H':
                mark_points_TH.extend(mark_points)
    
    mark_points_EC_PC =[]
    for view, point_list_EC_PC in tqdm(zip(gridded_views, pixel_coords_EC_PC), total=len(gridded_views), desc="Mapping EC/PC to 3D Mesh", file=sys.stdout):

        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'

        #out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8
        
        if len(point_list_EC_PC) == 0: continue
        
        u_arr =[px[0] for px in point_list_EC_PC]
        v_arr =[px[1] for px in point_list_EC_PC]

        up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,-1)
        all_hits = pixel_to_mesh(cam_pos=cam_pos, F_pos=F_pos, u=u_arr, v=v_arr, 
                                up_cam_vec=np.array(up_cam_vec), parallel_scale=parallel_scale,
                                img_H=1000, img_W=1000, mesh=mesh, orthographic=True, return_all_hits=True)

        for p_world in all_hits:
            if p_world is not None and p_world.shape[0] >= 2:
                mark_points_EC_PC.append(p_world[0]) # Only take the first intersection point for EC/PC since they are visible in the view and we want to avoid noise from deeper ray intersections. For ICE, F and TH we will consider all ray intersections since they may not be visible in the view and we want to capture potential hidden features.


    ## After raycasting time to filter out cells
    # --- Render Output ---+ mark_points_TH
    points_3d = filter_points_by_density_fast(mark_points_ICE + mark_points_EC_PC + mark_points_TH + mark_points_F, point_density=point_density)  # + mark_points_F Adjust threshold as needed
    
    np.save(f'{parent_dir}/{cad_object}/{experiment_name}/refinement_points_prefilt.npy', np.array(points_3d))

    if len(np.array(points_3d)) == 0:
        print(f'No points detected for refinement for {os.path.basename(mesh_file_path)} with experiment {experiment_name}')
        exit(0)
    render_mesh_views(mesh_file_path, output_dir=f'{parent_dir}/{cad_object}/{experiment_name}/meshpoints_prefilt', n_azimuth=[60,300], n_elevation=[-36, 36],
                    orthographic=False, points_3d=points_3d, verbose=True, add_axes=False, opacity=0.7)

    ## ---- Filter views based on visibility of detected features ----
    filteration_views = []
    ## Right now doing it only for fillets snoice they are open to select 
    points_3d_F = filter_points_by_density_fast(np.array(mark_points_F), point_density=point_density)
            
    for view in gridded_views:
        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'
        
        el_angle, az_angle = extract_el_az_from_viewpath(view_path)
        ortho_angle_pairs = get_perpendicular_views(el_angle, az_angle)
        ## Select views with good visibility of the detected features (will be used for subsequent filtering prompts to the LLM)
        valid_views = {}
        for el_az in ortho_angle_pairs:
            el_angle_ortho, az_angle_ortho = el_az
            #if not (el_angle_ortho == 0 and az_angle_ortho == 0): continue
            cam_pos, F_pos, view_radius = return_cam_position(el_angle_ortho, az_angle_ortho, mesh_path=mesh_file_path)
            
            ## Check visibility
            visible, avg_distance_to_visible = check_points_visibility(mesh, cam_pos, points_3d_F, point_density)
            num_visible = np.sum(visible)
            if num_visible > 0.1 * len(points_3d_F):  # If more than 10% of the detected fillet points are visible in this orthogonal view, we consider it a good candidate for the next prompt
                if (el_angle_ortho, az_angle_ortho) not in valid_views.keys():
                    valid_views[(el_angle_ortho, az_angle_ortho)] = avg_distance_to_visible
                else:
                    # If the view is already in valid_views, we can update the average distance if this new one is better (lower)
                    if avg_distance_to_visible < valid_views[(el_angle_ortho, az_angle_ortho)]:
                        valid_views[(el_angle_ortho, az_angle_ortho)] = avg_distance_to_visible
        
    ## Filter valid views based on angle between views ad avg_dist (lower is better)
    valid_views_sorted = sorted(valid_views.items(), key=lambda x: x[1], reverse=True)  # Sort by average distance to visible points in descending order (lower distance is better)
    ## Non max suppression based on angle difference (keep views that are at least 15 degrees apart)
    if len(valid_views_sorted) > 0:
        filteration_views = [valid_views_sorted[0][0]]  # Start with the best view
    else:
        filteration_views = []
        print("No valid orthogonal views found with sufficient visibility of detected features.")
        print(f'No points detected for final refinement for {os.path.basename(mesh_file_path)} with experiment {experiment_name}')
        exit(0)
    
    for view, avg_dist in valid_views_sorted[1:]:
        if angle_btw_views(view, filteration_views[-1]) >= 90:  # If the angle between this view and the last selected view is greater than or equal to 90 degrees
            filteration_views.append(view)
    
    filteration_view_imgs = []
    for el_az in filteration_views:
        el_angle, az_angle = el_az
        ## Synthesize load and marked point visualizations
        render_mesh_views_with_load(mesh_file_path, output_dir_prefix=f'{parent_dir}/{cad_object}/{experiment_name}/meshpoints_with_load',
                                    n_azimuth=[az_angle], n_elevation=[el_angle], orthographic=False, 
                                    points_3d=points_3d, verbose=True, add_axes=False, loading_type=args.load_case)
        filteration_view_imgs.append(f'{parent_dir}/{cad_object}/{experiment_name}/meshpoints_with_load/view_e{el_angle}_a{az_angle}.png')
    

    # --- Prompt Construction ---
    prompt = get_prompt_2(loading_case=args.load_case, prompt_type=args.prompt_type, perspective_views=perspective_views, gridded_views=filteration_view_imgs)  # You would define this function to return your actual prompt text

    messages =[]
    inference_imgs =[]
    gridded_views = []
    for view_path in filteration_view_imgs:
        gridded_path = overlay_grid(view_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
                                    font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
                                    line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    arrow_cell=None, output_dir=output_grid_dir)
        gridded_views.append(gridded_path)
    
    img_paths = perspective_views + gridded_views
    for img_path in img_paths:
        with open(img_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            inference_imgs.append(encoded_img)

    content =[{"type": "text", "text": prompt}]
    content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
    messages.append({"role": "user", "content": content})

    # --- Protected API Call ---
    print(f"Prompt Stage 2 Region Filteration: Waiting for OpenRouter API Response ({args.LLM_name})...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": args.LLM_name, "temperature": 0.0, "messages": messages},
            timeout=60  # Prevent infinite hanging
        )
        response.raise_for_status() # Catches 429, 502, etc.
        response_data = response.json()
        
        if 'choices' not in response_data:
            print(f"API Error Response format: {response_data}")
            sys.exit(1)
            
        output = response_data['choices'][0]['message']['content']
        print("Response Received:", output)
        
    except Exception as e:
        print(f"API Request Failed: {e}")
        sys.exit(1) # Exit cleanly to inform parent script

    pred_cells_lists = parse_geom_feature_cells(output, prompt_type=args.prompt_type, infer_views=filteration_view_imgs, feature_categories=['Cells'])
    points_to_remove =[]

    for view_path, pred_cells in zip(filteration_view_imgs, pred_cells_lists):
        img = cv2.imread(view_path)
        if img is None:
            print(f"Failed to load image for final visualization: {view_path}")
            continue
        
        ## Remove the points located in the pred_cells in these views
        el,az = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(el, az, mesh_path=mesh_file_path)
        visible_idxs, _ = check_points_visibility(mesh, cam_pos, np.array(points_3d), point_density)    
        visible_points = np.array(points_3d)[visible_idxs]
        visible_pts_pixel_coords = space_to_pixel(visible_points, cam_pos, F_pos, up_cam_vec=np.array([0,1,0]) if 'view_e90' not in view_path and 'view_e-90' not in view_path else np.array([0,0,-1])
                                                  , parallel_scale=parallel_scale, img_H=1000, img_W=1000, orthographic=True)
        for pt, pixel in zip(visible_points, visible_pts_pixel_coords):
            if is_pixel_in_pred_cells(pixel, pred_cells['Cells'], gridx, gridy, img.shape):
                points_to_remove.append(pt)

    ## Filter off points_to_remove from points_3d and re-render the view with the remaining points for final visualization
    points_to_keep = remove_points_by_proximity(points_3d, points_to_remove, point_density)
    
    render_mesh_views(mesh_file_path, output_dir=f'{parent_dir}/{cad_object}/{experiment_name}/meshpoints_final',n_azimuth=[60, 300],n_elevation=[36, -36], 
                orthographic=False, points_3d=points_to_keep, verbose=True, add_axes=False, opacity=0.7)
    
    refinement_points = np.array(points_to_keep)
    np.save(f'{parent_dir}/{cad_object}/{experiment_name}/refinement_points_final.npy', refinement_points)

    if len(refinement_points) == 0:
        print(f'No points detected for refinement for {os.path.basename(mesh_file_path)} with experiment {experiment_name}')
    
    print('Lets See!!')  