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
    
    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal_29.03.2026'
    CAD_file_name = '00210076'
    
    parser = argparse.ArgumentParser(description="Run multiple experiments to infer orthographic views and identify stress concentration areas using LLM.")
    parser.add_argument('--mesh_path', type=str, default=f'{parent_dir}/{CAD_file_name}/renders_pyvista/{CAD_file_name}.obj')
    parser.add_argument('--model_ckpt', type=str, default='/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/model_saves_19.01.2026/ortho_view_selector_40.pth')
    parser.add_argument('--prompt_type', type=str, default='geo_max')
    parser.add_argument('--grid_size', type=int, default=11)
    parser.add_argument('--LLM_name', type=str, default="google/gemini-3-flash-preview")
    parser.add_argument('--view_selection_strategy', type=str, default="ortho")
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--num_perseptive_views', type=int, default=2)
    parser.add_argument('--min_arc_ratio', type=float, default=0.2)
    parser.add_argument('--cons_cell_lookup', type=int, default=0)
    parser.add_argument('--num_points_ray', type=int, default=15)
    parser.add_argument('--edge_angle_thresh', type=float, default=20)
    parser.add_argument('--run', type=int, default=100)
    args = parser.parse_args()

    print(args)
    parent_dir = '/'.join(args.mesh_path.split('/')[:-3])
    model_name = args.LLM_name.split('/')[-1]
    experiment_name = f'{args.view_selection_strategy}_{args.num_views}views_{model_name}_{args.grid_size}grid_{args.prompt_type}prompt_{args.run}run'

    gridx, gridy = args.grid_size, args.grid_size  
    mesh_path = args.mesh_path
    cad_object = os.path.basename(mesh_path).replace(".obj", "") 
    
    model_ckpt = args.model_ckpt
    
    view_dir = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial'
    all_view_paths =[f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}' for view_name in os.listdir(view_dir)]
    perspective_views = np.random.choice(all_view_paths, size=args.num_perseptive_views, replace=False).tolist()

    infer_views_paths =[]
    if args.view_selection_strategy == 'ortho':
        with open(f'{parent_dir}/{cad_object}/pred_ortho_views2.txt', 'r') as fread:
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

    infer_views =[os.path.basename(view_path) for view_path in infer_views_paths]
    
    output_grid_dir = f'{parent_dir}/{cad_object}/{experiment_name}/renders_pyvista_mesh_initial'
    os.makedirs(output_grid_dir, exist_ok=True)
    
    gridded_views =[]
    for img_path in tqdm(infer_views_paths, desc="Overlaying Grids", file=sys.stdout):
        gridded_path = overlay_grid(img_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
                                    font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
                                    line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    arrow_cell=None, output_dir=output_grid_dir)
        gridded_views.append(gridded_path)

    print(f'Top Views Gridded: {gridded_views}')
    
    # --- Prompt Construction ---
    if args.prompt_type == 'geo_max':
        prompt = (f"You are given multiple images of a single CAD part.\n\n"
                f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
                f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
                f"Each orthographic view has a visible grid with numbered cells.\n\n"
                f"A force acts on the object in the downward direction on its top surface.\n"
                f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
                f"Task:\n"
                f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
                f"1) Internal corner edges (concave edges that fold into the object).\n"
                f"2) Circular features such as holes, fillets, or arches.\n\n"
                f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
                f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
                f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
                f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
                f"Output format (repeat for each gridded image):\n\n"
                f"***I.C.E: c1, c2, c3 ***\n"
                f"***C.H: c4, c5 ***\n\n"
                f"Rules:\n"
                f"- List only cell numbers.\n"
                f"- If no cells apply, write: NONE.\n"
                f"- DO NOT Add explanations \n")
    elif args.prompt_type == 'geo_mid':
        prompt = (f"You are given multiple images of a single CAD part.\n\n"
                f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
                f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
                f"Each orthographic view has a visible grid with numbered cells.\n\n"
                f"A force acts on the object in the downward direction on its top surface.\n"
                f"Task:\n"
                f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
                f"1) Internal corner edges.\n"
                f"2) Circular features such as holes, fillets, or arches.\n\n"
                f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
                f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
                f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
                f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
                f"Output format (repeat for each gridded image):\n\n"
                f"***I.C.E: c1, c2, c3 ***\n"
                f"***C.H: c4, c5 ***\n\n"
                f"Rules:\n"
                f"- List only cell numbers.\n"
                f"- If no cells apply, write: NONE.\n"
                f"- DO NOT Add explanations \n")
    else:
        prompt = (f"You are given multiple images of a single CAD part.\n\n"
                f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
                f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
                f"Each orthographic view has a visible grid with numbered cells.\n\n"
                f"A force acts on the object in the downward direction on its top surface.\n"
                f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
                f"Task:\n"
                f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
                f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
                f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
                f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
                f"Output format (repeat for each gridded image):\n\n"
                f"***Cells: c1, c2, c3 ***\n"
                f"Rules:\n"
                f"- List only cell numbers.\n"
                f"- If no cells apply, write: NONE.\n"
                f"- DO NOT Add explanations \n")

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
    print(f"Waiting for OpenRouter API Response ({args.LLM_name})...")
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

    # --- Parse LLM Output ---
    pred_cells = {'ICE': [], 'CH':[]}
    if args.prompt_type in['geo_max', 'geo_mid']:
        for feature_label in['***I.C.E: ', '***C.H: ']:
            cell_strs =[out.split('***')[0] for out in output.split(feature_label)[1:]]
            for cell_str in cell_strs:
                cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
                pred_cells['ICE' if feature_label == '***I.C.E: ' else 'CH'].append(cell_list)
    else:
        cell_strs =[out.split('***')[0] for out in output.split('***Cells')[1:]]
        for cell_str in cell_strs:
            cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
            pred_cells['ICE'].append(cell_list)
            pred_cells['CH'].append(cell_list)

    print("Identified cell numbers:", pred_cells['ICE'], pred_cells['CH'])
    
    cons_cell_nums = {'ICE':[set() for _ in range(len(infer_views))], 'CH':[set() for _ in range(len(infer_views))]}
    for feature_label in['ICE', 'CH']:
        for list_idx, cell_list in enumerate(pred_cells[feature_label]):
            if list_idx >= len(infer_views):
                print(f"Warning: LLM returned extra predictions for {feature_label}. Ignoring extras.")
                break
            for cell_num in cell_list:
                for look_x in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
                    for look_y in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
                        neighbor_cell_num = cell_num + look_x + look_y * gridx
                        if 1 <= neighbor_cell_num <= gridx * gridy:
                            cons_cell_nums[feature_label][list_idx].add(neighbor_cell_num)
    
    # --- 2D Edge Detection and Grid Matching ---
    output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}'
    os.makedirs(output_2d_dir, exist_ok=True)

    pixel_coords_ICE =[]
    output_img_paths =[]
    
    for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['ICE'])), 
                                            total=len(infer_views), desc="2D Edge Detection (ICE)", file=sys.stdout):
        view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
        pixels_coords_view =[]
        for cell_num in cell_num_list:
            candidates = edge_detection_contour(view_img_path, cell_num, gridx, gridy, angle_thresh=args.edge_angle_thresh)
            pixels_coords_view.extend(candidates)
        
        out_path = f'{output_2d_dir}/{view_img}'
        output_img_paths.append(mark_spots_in_image(view_img_path, spot_radius=4, spot_color=(0, 255, 0), 
                                                    spot_positions=pixels_coords_view, output_path=out_path))
        pixel_coords_ICE.append(pixels_coords_view)
        
    pixel_coords_CH =[]
    for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['CH'])), 
                                            total=len(infer_views), desc="2D Circle Detection (CH)", file=sys.stdout):
        view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
        pixels_coords_view = {'circles':[], 'pts':[]}
        circles = detect_all_circles_cv2(view_img_path, min_arc_ratio=args.min_arc_ratio)
        
        for cell_num in tqdm(cell_num_list, desc="Matching cells to circles", leave=True, file=sys.stdout):
            for circle in circles:
                if circle_in_img_patch(circle, view_img_path, cell_num, gridx, gridy):
                    pixels_coords_view = add_circle_pts(circle, pixels_coords_view)    
            
        pixel_coords_CH.append(pixels_coords_view)
        output_img_paths[i] = mark_spots_in_image(output_img_paths[i], spot_radius=4, spot_color=(255, 0, 0), 
                                                spot_positions=pixels_coords_view['pts'], output_path=f'{output_2d_dir}/{view_img}')

    # --- 3D Raycasting (VECTORIZED) ---
    mesh = trimesh.load(mesh_path)
    img_height, img_width = 1000, 1000

    mark_points_ICE =[]
    for view, edge_list in tqdm(zip(infer_views, pixel_coords_ICE), total=len(infer_views), desc="Mapping ICE to 3D Mesh", file=sys.stdout):
        if len(edge_list) == 0: continue
        camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
        u_arr =[px[0] for px in edge_list]
        v_arr =[px[1] for px in edge_list]
        
        all_hits = pixel_to_mesh(cam_pos=camera_pos, F_pos=object_center, u=u_arr, v=v_arr, 
                                up_cam_vec=np.array([0,1,0]), parallel_scale=0.8*view_radius,
                                img_H=img_height, img_W=img_width, fov_in_degrees=30, 
                                mesh=mesh, orthographic=True, return_all_hits=True)
        
        for p_world in all_hits:
            if p_world is not None and p_world.shape[0] >= 2:
                for j in range(0, p_world.shape[0] - 1, 2):
                    if j>0: break
                    p_enter = p_world[j]
                    p_exit = p_world[j+1]
                    edge_points_on_mesh = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray+1)[:-1, None]
                    mark_points_ICE.extend(edge_points_on_mesh)

    mark_points_CH =[]
    for view, CH_view_list in tqdm(zip(infer_views, pixel_coords_CH), total=len(infer_views), desc="Mapping CH circles to 3D Mesh", file=sys.stdout):
        if len(CH_view_list['circles']) == 0: continue
        camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
        ray_dir = object_center - camera_pos
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        for circle in tqdm(CH_view_list['circles'], desc="Raycasting circles", leave=True, file=sys.stdout):
            
            center_hits_list = pixel_to_mesh(
                cam_pos=camera_pos, F_pos=object_center, 
                u=[circle['xc']], v=[circle['yc']], 
                up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
                img_H=img_height, img_W=img_width, fov_in_degrees=30, 
                mesh=mesh, orthographic=True, return_all_hits=True
            )
            
            center_hits = center_hits_list[0] if center_hits_list and center_hits_list[0] is not None else None
            
            # --- Conditional logic for small holes ---
            is_small_hole = (4 <= circle['r'] <= 15)
            # If it's a small hole AND it's a blind hole (center_hits is not None), skip it
            if is_small_hole and (center_hits is not None and len(center_hits) > 1):
                continue

            depth_C0 = None
            if center_hits is not None and center_hits.shape[0] >= 2:
                depth_C0 = np.dot(center_hits[0] - camera_pos, ray_dir)

            angles = np.linspace(0, 2 * np.pi, int(12 * max(circle['r'] // 10, 1)))
            
            # --- Inner Probe Rays for Noise Filtering ---
            r_rim = circle['r'] + 4
            r_inner = max(0, circle['r'] - 4)
            
            u_rim = circle['xc'] + r_rim * np.cos(angles)
            v_rim = circle['yc'] + r_rim * np.sin(angles)
            
            u_inner = circle['xc'] + r_inner * np.cos(angles)
            v_inner = circle['yc'] + r_inner * np.sin(angles)
            
            u_combined = np.concatenate([u_rim, u_inner])
            v_combined = np.concatenate([v_rim, v_inner])
            
            all_hits_combined = pixel_to_mesh(
                cam_pos=camera_pos, F_pos=object_center, u=u_combined, v=v_combined, 
                up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
                img_H=img_height, img_W=img_width, fov_in_degrees=30, 
                mesh=mesh, orthographic=True, return_all_hits=True
            )
            
            num_angles = len(angles)
            
            for j_idx in range(num_angles):
                hit_intersections = all_hits_combined[j_idx]
                inner_intersections = all_hits_combined[j_idx + num_angles]
                
                if hit_intersections is not None and hit_intersections.shape[0] >= 2:
                    
                    for j in range(0, hit_intersections.shape[0] - 1, 2):
                        if j > 0: break
                        
                        p_enter = hit_intersections[j]
                        p_exit = hit_intersections[j+1]
                        depth_R0 = np.dot(p_enter - camera_pos, ray_dir)
                        
                        # --- ROBUST NOISE REJECTION LOGIC ---
                        is_true_hole, is_true_boss = False, False
                        depth_threshold = max(1e-3, 0.002 * view_radius)
                        
                        # 1. Test with the inner probe ray first
                        if inner_intersections is not None:
                            if inner_intersections.shape[0] == 1:
                                is_true_hole = True # Inner probe passed through -> True Hole
                            elif inner_intersections.shape[0] >= 2:
                                depth_inner = np.dot(inner_intersections[0] - camera_pos, ray_dir)
                                if depth_inner > depth_R0 + depth_threshold:
                                    is_true_hole = True # Inner probe hit deeper -> True Blind Hole
                                if depth_inner < depth_R0 - depth_threshold:
                                    is_true_boss = True 

                        # 2. Fallback to center ray (for tiny holes where inner probe is same as center)
                        if not is_true_hole and center_hits is not None and is_small_hole:
                            if center_hits.shape[0] == 1:
                                is_true_hole = True # Center ray passed through -> True Hole
                            elif center_hits.shape[0] >= 2 and depth_C0 is not None:
                                if depth_C0 > depth_R0 + depth_threshold:
                                    is_true_hole = True # Center hit deeper -> True Blind Hole
                                    #p_exit = p_enter

                        # --- APPLY PROJECTION OR DISCARD ---
                        if is_true_hole:
                            # It's a real hole. Project points normally.
                            if depth_C0 is not None: # Adjust depth for blind holes
                                depth_R1 = np.dot(p_exit - camera_pos, ray_dir)
                                if depth_R0 - 1e-3 <= depth_C0 <= depth_R1 + 1e-3:
                                    p_exit = p_enter + ray_dir * (depth_C0 - depth_R0)
                            
                            edge_points = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray + 1)[:-1, None]
                            mark_points_CH.extend(edge_points)
                        elif is_true_boss:
                            # It's a boss. Project points onto the outer surface (p_enter).
                            mark_points_CH.append(p_enter)
                        else:
                            # NOISE DETECTED: Annular circle or flat artifact. DISCARD.
                            continue

    # --- Render Output ---
    refinement_points = np.array(mark_points_CH + mark_points_ICE)
    np.save(f'{parent_dir}/{cad_object}/refinement_points_{experiment_name}.npy', refinement_points)
    
    if len(refinement_points) == 0:
        print(f'No points detected for refinement for {os.path.basename(mesh_path)} with experiment {experiment_name}')
    else:
        print("Rendering Output Views with Meshpoints...")
        # render_mesh_views(mesh_path, output_dir=f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints_{experiment_name}', 
        #             n_azimuth=12, n_elevation=3, orthographic=False, 
        #             points_3d=mark_points_CH + mark_points_ICE, verbose=True, add_axes=False)

    print(f'Pipeline completed successfully.')




# api_key = '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'
# import requests
# import base64
# from utils import *
# import trimesh
# from generate_renders import render_mesh_views
# import numpy as np
# import pyvista as pv
# import cv2
# from infer import infer_NN
# import argparse
# from tqdm import tqdm
# import os
# import sys

# if __name__ == '__main__':
    
#     parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal'
#     CAD_file_name = '00210076'
#     parser = argparse.ArgumentParser(description="Run multiple experiments to infer orthographic views and identify stress concentration areas using LLM.")
#     parser.add_argument('--mesh_path', type=str, default=f'{parent_dir}/{CAD_file_name}/renders_pyvista/{CAD_file_name}.obj')
#     parser.add_argument('--model_ckpt', type=str, default='/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/model_saves_19.01.2026/ortho_view_selector_40.pth')
#     parser.add_argument('--prompt_type', type=str, default='geo_max')
#     parser.add_argument('--grid_size', type=int, default=11)
#     parser.add_argument('--LLM_name', type=str, default="google/gemini-3-flash-preview")
#     parser.add_argument('--view_selection_strategy', type=str, default="ortho")
#     parser.add_argument('--num_views', type=int, default=5)
#     parser.add_argument('--num_perseptive_views', type=int, default=2)
#     parser.add_argument('--min_arc_ratio', type=float, default=0.2)
#     parser.add_argument('--cons_cell_lookup', type=int, default=0)
#     parser.add_argument('--num_points_ray', type=int, default=15)
#     parser.add_argument('--edge_angle_thresh', type=float, default=20)
#     parser.add_argument('--run', type=int, default=100)
#     args = parser.parse_args()

#     print(args)
#     parent_dir = '/'.join(args.mesh_path.split('/')[:-3])
#     model_name = args.LLM_name.split('/')[-1]
#     experiment_name = f'{args.view_selection_strategy}_{args.num_views}views_{model_name}_{args.grid_size}grid_{args.prompt_type}prompt_{args.run}run'

#     gridx, gridy = args.grid_size, args.grid_size  
#     mesh_path = args.mesh_path
#     cad_object = os.path.basename(mesh_path).replace(".obj", "") 
    
#     model_ckpt = args.model_ckpt
    
#     view_dir = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial'
#     all_view_paths =[f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}' for view_name in os.listdir(view_dir)]
#     perspective_views = np.random.choice(all_view_paths, size=args.num_perseptive_views, replace=False).tolist()

#     infer_views_paths =[]
#     if args.view_selection_strategy == 'ortho':
#         with open(f'{parent_dir}/{cad_object}/pred_ortho_views2.txt', 'r') as fread:
#             # Read lines, strip whitespace
#             raw_lines =[line.strip() for line in fread.readlines() if line.strip()][:args.num_views]
            
#             for line in raw_lines:
#                 # Extract JUST the image name (e.g., 'view_e0_a0.png') from the old absolute path
#                 view_name = os.path.basename(line)
                
#                 # Reconstruct the correct path using the CURRENT parent_dir
#                 current_correct_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}'
                
#                 # Make sure the file actually exists before we try to process it
#                 if not os.path.exists(current_correct_path):
#                     raise FileNotFoundError(f"Image missing or network drive timeout: {current_correct_path}")
                
#                 infer_views_paths.append(current_correct_path)

#     elif args.view_selection_strategy == 'random':
#         remaining_views = list(set(all_view_paths) - set(perspective_views))
#         infer_views_paths = np.random.choice(remaining_views, size=args.num_views, replace=False).tolist()

#     infer_views =[os.path.basename(view_path) for view_path in infer_views_paths]
    
#     output_grid_dir = f'{parent_dir}/{cad_object}/{experiment_name}/renders_pyvista_mesh_initial'
#     os.makedirs(output_grid_dir, exist_ok=True)
    
#     gridded_views =[]
#     for img_path in tqdm(infer_views_paths, desc="Overlaying Grids", file=sys.stdout):
#         gridded_path = overlay_grid(img_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
#                                     font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
#                                     line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
#                                     arrow_cell=None, output_dir=output_grid_dir)
#         gridded_views.append(gridded_path)

#     print(f'Top Views Gridded: {gridded_views}')
    
#     # --- Prompt Construction ---
#     if args.prompt_type == 'geo_max':
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"1) Internal corner edges (concave edges that fold into the object).\n"
#                   f"2) Circular features such as holes, fillets, or arches.\n\n"
#                   f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***I.C.E: c1, c2, c3 ***\n"
#                   f"***C.H: c4, c5 ***\n\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")
#     elif args.prompt_type == 'geo_mid':
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"1) Internal corner edges.\n"
#                   f"2) Circular features such as holes, fillets, or arches.\n\n"
#                   f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***I.C.E: c1, c2, c3 ***\n"
#                   f"***C.H: c4, c5 ***\n\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")
#     else:
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***Cells: c1, c2, c3 ***\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")

#     messages =[]
#     inference_imgs =[]
#     img_paths = perspective_views + gridded_views
#     for img_path in img_paths:
#         with open(img_path, "rb") as img_file:
#             encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
#             inference_imgs.append(encoded_img)

#     content =[{"type": "text", "text": prompt}]
#     content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
#     messages.append({"role": "user", "content": content})
    
#     # --- Protected API Call ---
#     print(f"Waiting for OpenRouter API Response ({args.LLM_name})...")
#     try:
#         response = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#             json={"model": args.LLM_name, "temperature": 0.0, "messages": messages},
#             timeout=60  # Prevent infinite hanging
#         )
#         response.raise_for_status() # Catches 429, 502, etc.
#         response_data = response.json()
        
#         if 'choices' not in response_data:
#             print(f"API Error Response format: {response_data}")
#             sys.exit(1)
            
#         output = response_data['choices'][0]['message']['content']
#         print("Response Received:", output)
        
#     except Exception as e:
#         print(f"API Request Failed: {e}")
#         sys.exit(1) # Exit cleanly to inform parent script

#     # --- Parse LLM Output ---
#     pred_cells = {'ICE': [], 'CH':[]}
#     if args.prompt_type in['geo_max', 'geo_mid']:
#         for feature_label in['***I.C.E: ', '***C.H: ']:
#             cell_strs =[out.split('***')[0] for out in output.split(feature_label)[1:]]
#             for cell_str in cell_strs:
#                 cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
#                 pred_cells['ICE' if feature_label == '***I.C.E: ' else 'CH'].append(cell_list)
#     else:
#         cell_strs =[out.split('***')[0] for out in output.split('***Cells')[1:]]
#         for cell_str in cell_strs:
#             cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
#             pred_cells['ICE'].append(cell_list)
#             pred_cells['CH'].append(cell_list)

#     print("Identified cell numbers:", pred_cells['ICE'], pred_cells['CH'])
    
#     cons_cell_nums = {'ICE':[set() for _ in range(len(infer_views))], 'CH':[set() for _ in range(len(infer_views))]}
#     for feature_label in['ICE', 'CH']:
#         for list_idx, cell_list in enumerate(pred_cells[feature_label]):
#             if list_idx >= len(infer_views):
#                 print(f"Warning: LLM returned extra predictions for {feature_label}. Ignoring extras.")
#                 break
#             for cell_num in cell_list:
#                 for look_x in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
#                     for look_y in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
#                         neighbor_cell_num = cell_num + look_x + look_y * gridx
#                         if 1 <= neighbor_cell_num <= gridx * gridy:
#                             cons_cell_nums[feature_label][list_idx].add(neighbor_cell_num)
    
#     # --- 2D Edge Detection and Grid Matching ---
#     output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}'
#     os.makedirs(output_2d_dir, exist_ok=True)

#     pixel_coords_ICE =[]
#     output_img_paths =[]
    
#     for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['ICE'])), 
#                                              total=len(infer_views), desc="2D Edge Detection (ICE)", file=sys.stdout):
#         view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
#         pixels_coords_view =[]
#         for cell_num in cell_num_list:
#             candidates = edge_detection_contour(view_img_path, cell_num, gridx, gridy, angle_thresh=args.edge_angle_thresh)
#             pixels_coords_view.extend(candidates)
        
#         out_path = f'{output_2d_dir}/{view_img}'
#         output_img_paths.append(mark_spots_in_image(view_img_path, spot_radius=4, spot_color=(0, 255, 0), 
#                                                     spot_positions=pixels_coords_view, output_path=out_path))
#         pixel_coords_ICE.append(pixels_coords_view)
        
#     pixel_coords_CH =[]
#     for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['CH'])), 
#                                              total=len(infer_views), desc="2D Circle Detection (CH)", file=sys.stdout):
#         view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
#         pixels_coords_view = {'circles':[], 'pts':[]}
#         circles = detect_all_circles_cv2(view_img_path, min_arc_ratio=args.min_arc_ratio)
        
#         for cell_num in tqdm(cell_num_list, desc="Matching cells to circles", leave=True, file=sys.stdout):
#             for circle in circles:
#                 if circle_in_img_patch(circle, view_img_path, cell_num, gridx, gridy):
#                     pixels_coords_view = add_circle_pts(circle, pixels_coords_view)    
            
#         pixel_coords_CH.append(pixels_coords_view)
#         output_img_paths[i] = mark_spots_in_image(output_img_paths[i], spot_radius=4, spot_color=(255, 0, 0), 
#                                                   spot_positions=pixels_coords_view['pts'], output_path=f'{output_2d_dir}/{view_img}')

#     # --- 3D Raycasting (VECTORIZED) ---
#     mesh = trimesh.load(mesh_path)
#     img_height, img_width = 1000, 1000

#     mark_points_ICE =[]
#     for view, edge_list in tqdm(zip(infer_views, pixel_coords_ICE), total=len(infer_views), desc="Mapping ICE to 3D Mesh", file=sys.stdout):
#         if len(edge_list) == 0: continue
#         camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
#         u_arr =[px[0] for px in edge_list]
#         v_arr =[px[1] for px in edge_list]
        
#         all_hits = pixel_to_mesh(cam_pos=camera_pos, F_pos=object_center, u=u_arr, v=v_arr, 
#                                  up_cam_vec=np.array([0,1,0]), parallel_scale=0.8*view_radius,
#                                  img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                                  mesh=mesh, orthographic=True, return_all_hits=True)
        
#         for p_world in all_hits:
#             if p_world is not None and p_world.shape[0] >= 2:
#                 for j in range(0, p_world.shape[0] - 1, 2):
#                     if j>0: break
#                     p_enter = p_world[j]
#                     p_exit = p_world[j+1]
#                     edge_points_on_mesh = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray+1)[:-1, None]
#                     mark_points_ICE.extend(edge_points_on_mesh)

#     mark_points_CH =[]
#     for view, CH_view_list in tqdm(zip(infer_views, pixel_coords_CH), total=len(infer_views), desc="Mapping CH circles to 3D Mesh", file=sys.stdout):
#         if len(CH_view_list['circles']) == 0: continue
#         camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
#         ray_dir = object_center - camera_pos
#         ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
#         for circle in tqdm(CH_view_list['circles'], desc="Raycasting circles", leave=True, file=sys.stdout):
            
#             center_hits_list = pixel_to_mesh(
#                 cam_pos=camera_pos, F_pos=object_center, 
#                 u=[circle['xc']], v=[circle['yc']], 
#                 up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
#                 img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                 mesh=mesh, orthographic=True, return_all_hits=True
#             )
            
#             center_hits = center_hits_list[0] if center_hits_list and center_hits_list[0] is not None else None
            
#             # --- Conditional logic for small holes ---
#             is_small_hole = (4 <= circle['r'] <= 15)
#             # If it's a small hole AND it's a blind hole (center_hits is not None), skip it
#             if is_small_hole and (center_hits is not None and len(center_hits) > 1):
#                 continue

#             depth_C0 = None
#             if center_hits is not None and center_hits.shape[0] > 0:
#                 depth_C0 = np.dot(center_hits[0] - camera_pos, ray_dir)

#             angles = np.linspace(0, 2 * np.pi, int(12 * max(circle['r'] // 10, 1))) # More points for larger circles to ensure good coverage
            
#             # --- Inner Probe Rays for Noise Filtering ---
#             # Define the rim (expanded by 4 pixels to sit just outside the edge)
#             r_rim = circle['r'] + 4
#             # Define an inner probe (8 pixels inward from the rim) to test for a void
#             r_inner = max(0, circle['r'] - 4) 
            
#             u_rim = circle['xc'] + r_rim * np.cos(angles)
#             v_rim = circle['yc'] + r_rim * np.sin(angles)
            
#             u_inner = circle['xc'] + r_inner * np.cos(angles)
#             v_inner = circle['yc'] + r_inner * np.sin(angles)
            
#             # Combine rim and inner points to vectorize the raycast in one go
#             u_combined = np.concatenate([u_rim, u_inner])
#             v_combined = np.concatenate([v_rim, v_inner])
            
#             all_hits_combined = pixel_to_mesh(
#                 cam_pos=camera_pos, F_pos=object_center, 
#                 u=u_combined, v=v_combined, 
#                 up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
#                 img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                 mesh=mesh, orthographic=True, return_all_hits=True
#             )
            
#             num_angles = len(angles)
            
#             for j_idx in range(num_angles):
#                 # Retrieve the hits for the rim and its corresponding inner probe
#                 hit_intersections = all_hits_combined[j_idx]
#                 inner_intersections = all_hits_combined[j_idx + num_angles]
                
#                 if hit_intersections is not None and hit_intersections.shape[0] >= 2:
                    
#                     for j in range(0, hit_intersections.shape[0] - 1, 2):
#                         if j > 0: break 
#                         p_enter = hit_intersections[j]
#                         p_exit = hit_intersections[j+1]
                        
#                         depth_R0 = np.dot(p_enter - camera_pos, ray_dir)
                        
#                         # --- NOISE REJECTION LOGIC ---
#                         is_true_hole = False
#                         depth_threshold = max(1e-3, 0.002 * view_radius) # Buffer for slight slopes
                        
#                         # 1. Did the inner probe find a void?
#                         if inner_intersections is None or inner_intersections.shape[0] == 0:
#                             # Inner probe went cleanly through -> True Through-Hole
#                             is_true_hole = True
#                         else:
#                             depth_inner = np.dot(inner_intersections[0] - camera_pos, ray_dir)
#                             if depth_inner > depth_R0 + depth_threshold:
#                                 # Inner probe fell into a blind cavity -> True Blind Hole
#                                 is_true_hole = True
                        
#                         # 2. Fallback for tiny holes where inner probe = center probe
#                         if not is_true_hole and depth_C0 is not None:
#                             if depth_C0 > depth_R0 + depth_threshold:
#                                 is_true_hole = True
                        
#                         # --- APPLY PROJECTION DISTANCE ---
#                         if is_true_hole:
#                             # It is a real hole. Apply standard logic to stop at the blind floor if needed.
#                             if depth_C0 is not None:
#                                 depth_R1 = np.dot(p_exit - camera_pos, ray_dir)
#                                 if depth_R0 - 1e-3 <= depth_C0 <= depth_R1 + 1e-3:
#                                     p_exit = p_enter + ray_dir * (depth_C0 - depth_R0)
#                         else:
#                             # NOISE DETECTED: This is a massive annular circle or a flat artifact.
#                             # It is NOT a real boundary. Project only 1/5th (20%) of the depth.
#                             p_exit = p_enter + (p_exit - p_enter) * 0.2
#                         # ----------------------------------

#                         edge_points = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray+1)[:-1, None]
#                         mark_points_CH.extend(edge_points)

#     # --- Render Output ---
#     refinement_points = np.array(mark_points_CH + mark_points_ICE)
#     np.save(f'{parent_dir}/{cad_object}/refinement_points_{experiment_name}.npy', refinement_points)
    
#     if len(refinement_points) == 0:
#         print(f'No points detected for refinement for {os.path.basename(mesh_path)} with experiment {experiment_name}')
#     else:
#         print("Rendering Output Views with Meshpoints...")
#         render_mesh_views(mesh_path, output_dir=f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints_{experiment_name}', 
#                       n_azimuth=12, n_elevation=3, orthographic=False, 
#                       points_3d=mark_points_CH + mark_points_ICE, verbose=True, add_axes=False)

#     print(f'Pipeline completed successfully.')




# api_key = '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'
# import requests
# import base64
# from utils import *
# import trimesh
# from generate_renders import render_mesh_views
# import numpy as np
# import pyvista as pv
# import cv2
# from infer import infer_NN
# import argparse
# from tqdm import tqdm
# import os
# import sys

# if __name__ == '__main__':
    
#     parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal'
#     CAD_file_name = '00230017'
#     parser = argparse.ArgumentParser(description="Run multiple experiments to infer orthographic views and identify stress concentration areas using LLM.")
#     parser.add_argument('--mesh_path', type=str, default=f'{parent_dir}/{CAD_file_name}/renders_pyvista/{CAD_file_name}.obj')
#     parser.add_argument('--model_ckpt', type=str, default='/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/model_saves_19.01.2026/ortho_view_selector_40.pth')
#     parser.add_argument('--prompt_type', type=str, default='geo_max')
#     parser.add_argument('--grid_size', type=int, default=11)
#     parser.add_argument('--LLM_name', type=str, default="google/gemini-3-flash-preview")
#     parser.add_argument('--view_selection_strategy', type=str, default="ortho")
#     parser.add_argument('--num_views', type=int, default=10)
#     parser.add_argument('--num_perseptive_views', type=int, default=2)
#     parser.add_argument('--min_arc_ratio', type=float, default=0.2)
#     parser.add_argument('--cons_cell_lookup', type=int, default=0)
#     parser.add_argument('--num_points_ray', type=int, default=15)
#     parser.add_argument('--edge_angle_thresh', type=float, default=20)
#     parser.add_argument('--run', type=int, default=100)
#     args = parser.parse_args()

#     print(args)
#     parent_dir = '/'.join(args.mesh_path.split('/')[:-3])
#     model_name = args.LLM_name.split('/')[-1]
#     experiment_name = f'{args.view_selection_strategy}_{args.num_views}views_{model_name}_{args.grid_size}grid_{args.prompt_type}prompt_{args.run}run'

#     gridx, gridy = args.grid_size, args.grid_size  
#     mesh_path = args.mesh_path
#     cad_object = os.path.basename(mesh_path).replace(".obj", "") 
    
#     model_ckpt = args.model_ckpt
    
#     view_dir = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial'
#     all_view_paths =[f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}' for view_name in os.listdir(view_dir)]
#     perspective_views = np.random.choice(all_view_paths, size=args.num_perseptive_views, replace=False).tolist()

#     infer_views_paths =[]
#     if args.view_selection_strategy == 'ortho':
#         with open(f'{parent_dir}/{cad_object}/pred_ortho_views2.txt', 'r') as fread:
#             infer_views_paths =[line.strip() for line in fread.readlines()][:args.num_views]
#             # Read lines, strip whitespace
#             raw_lines =[line.strip() for line in fread.readlines() if line.strip()][:args.num_views]
            
#             for line in raw_lines:
#                 # 1. FIX THE PATH MISMATCH:
#                 # Extract JUST the image name (e.g., 'view_e0_a0.png') from the old absolute path
#                 view_name = os.path.basename(line)
                
#                 # Reconstruct the correct path using the CURRENT parent_dir
#                 current_correct_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}'
                
#                 # 2. FIX THE SILENT OPENCV FAILURE:
#                 # Make sure the file actually exists before we try to process it
#                 if not os.path.exists(current_correct_path):
#                     raise FileNotFoundError(f"Image missing or network drive timeout: {current_correct_path}")
                
#                 infer_views_paths.append(current_correct_path)

#     elif args.view_selection_strategy == 'random':
#         remaining_views = list(set(all_view_paths) - set(perspective_views))
#         infer_views_paths = np.random.choice(remaining_views, size=args.num_views, replace=False).tolist()

#     infer_views =[os.path.basename(view_path) for view_path in infer_views_paths]
    
#     output_grid_dir = f'{parent_dir}/{cad_object}/{experiment_name}/renders_pyvista_mesh_initial'
#     os.makedirs(output_grid_dir, exist_ok=True)
    
#     gridded_views =[]
#     for img_path in tqdm(infer_views_paths, desc="Overlaying Grids", file=sys.stdout):
#         gridded_path = overlay_grid(img_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
#                                     font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
#                                     line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
#                                     arrow_cell=None, output_dir=output_grid_dir)
#         gridded_views.append(gridded_path)

#     print(f'Top Views Gridded: {gridded_views}')
    
#     # --- Prompt Construction ---
#     if args.prompt_type == 'geo_max':
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"1) Internal corner edges (concave edges that fold into the object).\n"
#                   f"2) Circular features such as holes, fillets, or arches.\n\n"
#                   f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***I.C.E: c1, c2, c3 ***\n"
#                   f"***C.H: c4, c5 ***\n\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")
#     elif args.prompt_type == 'geo_mid':
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"1) Internal corner edges.\n"
#                   f"2) Circular features such as holes, fillets, or arches.\n\n"
#                   f"I.C.E: Internal Corner Edge, C.H: Circular Hole (or fillets/arches)\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***I.C.E: c1, c2, c3 ***\n"
#                   f"***C.H: c4, c5 ***\n\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")
#     else:
#         prompt = (f"You are given multiple images of a single CAD part.\n\n"
#                   f"The first {len(perspective_views)} images show general 3D views of the object (not gridded).\n"
#                   f"The remaining {len(gridded_views)} images show orthographic views of the same object.\n"
#                   f"Each orthographic view has a visible grid with numbered cells.\n\n"
#                   f"A force acts on the object in the downward direction on its top surface.\n"
#                   f"Neglect the features too far away from the vertical axis of loading passing through the center of the object as they are not stress critical\n"
#                   f"Task:\n"
#                   f"For EACH gridded orthographic image, identify grid cells that contain stress critical:\n"
#                   f"Use the non-gridded {len(perspective_views)} 3D views ONLY to understand the overall shape.\n"
#                   f"Base all cell predictions ONLY on what is visible in the corresponding gridded orthographic image.\n"
#                   f"Do NOT guess or infer cells that are not clearly visible in that image.\n\n"
#                   f"Output format (repeat for each gridded image):\n\n"
#                   f"***Cells: c1, c2, c3 ***\n"
#                   f"Rules:\n"
#                   f"- List only cell numbers.\n"
#                   f"- If no cells apply, write: NONE.\n"
#                   f"- DO NOT Add explanations \n")

#     messages =[]
#     inference_imgs =[]
#     img_paths = perspective_views + gridded_views
#     for img_path in img_paths:
#         with open(img_path, "rb") as img_file:
#             encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
#             inference_imgs.append(encoded_img)

#     content =[{"type": "text", "text": prompt}]
#     content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
#     messages.append({"role": "user", "content": content})
    
#     print(f"Waiting for OpenRouter API Response ({args.LLM_name})...")
#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",
#         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#         json={"model": args.LLM_name, "temperature": 0.0, "messages": messages}
#     )

#     output = response.json()['choices'][0]['message']['content']
#     print("Response Received:", output)

#     # --- Parse LLM Output ---
#     pred_cells = {'ICE': [], 'CH':[]}
#     if args.prompt_type in ['geo_max', 'geo_mid']:
#         for feature_label in['***I.C.E: ', '***C.H: ']:
#             cell_strs =[out.split('***')[0] for out in output.split(feature_label)[1:]]
#             for cell_str in cell_strs:
#                 cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
#                 pred_cells['ICE' if feature_label == '***I.C.E: ' else 'CH'].append(cell_list)
#     else:
#         cell_strs = [out.split('***')[0] for out in output.split('***Cells')[1:]]
#         for cell_str in cell_strs:
#             cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
#             pred_cells['ICE'].append(cell_list)
#             pred_cells['CH'].append(cell_list)

#     print("Identified cell numbers:", pred_cells['ICE'], pred_cells['CH'])
    
#     cons_cell_nums = {'ICE':[set() for _ in range(len(infer_views))], 'CH':[set() for _ in range(len(infer_views))]}
#     for feature_label in['ICE', 'CH']:
#         for list_idx, cell_list in enumerate(pred_cells[feature_label]):
#             if list_idx >= len(infer_views):
#                 print(f"Warning: LLM returned extra predictions for {feature_label}. Ignoring extras.")
#                 break
#             for cell_num in cell_list:
#                 for look_x in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
#                     for look_y in range(-args.cons_cell_lookup, args.cons_cell_lookup +1):
#                         neighbor_cell_num = cell_num + look_x + look_y * gridx
#                         if 1 <= neighbor_cell_num <= gridx * gridy:
#                             cons_cell_nums[feature_label][list_idx].add(neighbor_cell_num)
    
#     # --- 2D Edge Detection and Grid Matching ---
#     output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}'
#     os.makedirs(output_2d_dir, exist_ok=True)

#     pixel_coords_ICE =[]
#     output_img_paths =[]
    
#     for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['ICE'])), 
#                                              total=len(infer_views), desc="2D Edge Detection (ICE)", file=sys.stdout):
#         view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
#         pixels_coords_view =[]
#         for cell_num in cell_num_list:
#             candidates = edge_detection_contour(view_img_path, cell_num, gridx, gridy, angle_thresh=args.edge_angle_thresh)
#             pixels_coords_view.extend(candidates)
        
#         out_path = f'{output_2d_dir}/{view_img}'
#         output_img_paths.append(mark_spots_in_image(view_img_path, spot_radius=4, spot_color=(0, 255, 0), 
#                                                     spot_positions=pixels_coords_view, output_path=out_path))
#         pixel_coords_ICE.append(pixels_coords_view)
        
#     pixel_coords_CH =[]
#     for i, (view_img, cell_num_list) in tqdm(enumerate(zip(infer_views, cons_cell_nums['CH'])), 
#                                              total=len(infer_views), desc="2D Circle Detection (CH)", file=sys.stdout):
#         view_img_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_img}'
#         pixels_coords_view = {'circles':[], 'pts':[]}
#         circles = detect_all_circles_cv2(view_img_path, min_arc_ratio=args.min_arc_ratio)
        
#         for cell_num in tqdm(cell_num_list, desc="Matching cells to circles", leave=True, file=sys.stdout):
#             for circle in circles:
#                 if circle_in_img_patch(circle, view_img_path, cell_num, gridx, gridy):
#                     pixels_coords_view = add_circle_pts(circle, pixels_coords_view)    
            
#         pixel_coords_CH.append(pixels_coords_view)
#         output_img_paths[i] = mark_spots_in_image(output_img_paths[i], spot_radius=4, spot_color=(255, 0, 0), 
#                                                   spot_positions=pixels_coords_view['pts'], output_path=f'{output_2d_dir}/{view_img}')

#     # --- 3D Raycasting (VECTORIZED) ---
#     mesh = trimesh.load(mesh_path)
#     img_height, img_width = 1000, 1000

#     mark_points_ICE =[]
#     for view, edge_list in tqdm(zip(infer_views, pixel_coords_ICE), total=len(infer_views), desc="Mapping ICE to 3D Mesh", file=sys.stdout):
#         if len(edge_list) == 0: continue
#         camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
#         u_arr =[px[0] for px in edge_list]
#         v_arr = [px[1] for px in edge_list]
        
#         all_hits = pixel_to_mesh(cam_pos=camera_pos, F_pos=object_center, u=u_arr, v=v_arr, 
#                                  up_cam_vec=np.array([0,1,0]), parallel_scale=0.8*view_radius,
#                                  img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                                  mesh=mesh, orthographic=True, return_all_hits=True)
        
#         for p_world in all_hits:
#             if p_world is not None and p_world.shape[0] >= 2:
#                 for j in range(0, p_world.shape[0] - 1, 2):
#                     if j>0: break
#                     p_enter = p_world[j]
#                     p_exit = p_world[j+1]
#                     edge_points_on_mesh = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray+1)[:-1, None]
#                     mark_points_ICE.extend(edge_points_on_mesh)

#     mark_points_CH =[]
#     for view, CH_view_list in tqdm(zip(infer_views, pixel_coords_CH), total=len(infer_views), desc="Mapping CH circles to 3D Mesh", file=sys.stdout):
#         if len(CH_view_list['circles']) == 0: continue
#         camera_pos, object_center, view_radius = return_cam_position(view, mesh_path)
        
#         ray_dir = object_center - camera_pos
#         ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
#         for circle in tqdm(CH_view_list['circles'], desc="Raycasting circles", leave=True, file=sys.stdout):
            
#             center_hits_list = pixel_to_mesh(
#                 cam_pos=camera_pos, F_pos=object_center, 
#                 u=[circle['xc']], v=[circle['yc']], 
#                 up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
#                 img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                 mesh=mesh, orthographic=True, return_all_hits=True
#             )
            
#             center_hits = center_hits_list[0] if center_hits_list and center_hits_list[0] is not None else None
            
#             # --- START OF CHANGE: Conditional logic for small holes ---
#             is_small_hole = (4 <= circle['r'] <= 15)
            
#             # If it's a small hole AND it's a blind hole (center_hits is not None), then skip it.
#             if is_small_hole and (len(center_hits) >1):
#                 continue # Skip this circle and move to the next one
#             # --- END OF CHANGE ---

#             depth_C0 = None
#             if center_hits is not None and center_hits.shape[0] > 0:
#                 depth_C0 = np.dot(center_hits[0] - camera_pos, ray_dir)

#             angles = np.linspace(0, 2 * np.pi, int(12 * max(circle['r'] // 50, 1)))
#             circle['r'] += 4
#             u_base = circle['xc'] + circle['r'] * np.cos(angles)
#             v_base = circle['yc'] + circle['r'] * np.sin(angles)
            
#             all_hits = pixel_to_mesh(
#                 cam_pos=camera_pos, F_pos=object_center, 
#                 u=u_base, v=v_base, 
#                 up_cam_vec=np.array([0, 1, 0]), parallel_scale=0.8 * view_radius,
#                 img_H=img_height, img_W=img_width, fov_in_degrees=30, 
#                 mesh=mesh, orthographic=True, return_all_hits=True
#             )
            
#             for hit_intersections in all_hits:
#                 if hit_intersections is not None and hit_intersections.shape[0] >= 2:
                    
#                     for j in range(0, hit_intersections.shape[0] - 1, 2):
#                         if j>0: break 
#                         p_enter = hit_intersections[j]
#                         p_exit = hit_intersections[j+1]
                        
#                         if j == 0 and depth_C0 is not None:
#                             depth_R0 = np.dot(p_enter - camera_pos, ray_dir)
#                             depth_R1 = np.dot(p_exit - camera_pos, ray_dir)
                            
#                             if depth_R0 - 1e-3 <= depth_C0 <= depth_R1 + 1e-3:
#                                 p_exit = p_enter + ray_dir * (depth_C0 - depth_R0)


#                         edge_points = p_enter + (p_exit - p_enter) * np.linspace(0, 1, num=args.num_points_ray+1)[:-1, None]
#                         mark_points_CH.extend(edge_points)

#     # --- Render Output ---
#     refinement_points = np.array(mark_points_CH + mark_points_ICE)
#     np.save(f'{parent_dir}/{cad_object}/refinement_points_{experiment_name}.npy', refinement_points)
    
#     if len(refinement_points) == 0:
#         print(f'No points detected for refinement for {os.path.basename(mesh_path)} with experiment {experiment_name}')
#     else:
#         print("Rendering Output Views with Meshpoints...")
#         render_mesh_views(mesh_path, output_dir=f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints_{experiment_name}', 
#                       n_azimuth=12, n_elevation=3, orthographic=False, 
#                       points_3d=mark_points_CH + mark_points_ICE, verbose=True, add_axes=False)

#     print(f'Pipeline completed successfully.')