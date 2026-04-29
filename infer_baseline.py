import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from generate_renders import render_mesh_views
from tqdm import tqdm
from utils import filter_points_by_density_fast

def calculate_cloud_density(points, mesh_bbox_diagonal):
    """
    Calculates the Average Nearest Neighbor distance of the LLM point cloud.
    This dictates the target spatial resolution (d_NN) for the baseline.
    """
    if len(points) < 2:
        return mesh_bbox_diagonal * 0.01 
        
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    mean_nn_dist = np.mean(distances[:, 1])
    
    safe_spacing = max(mean_nn_dist, mesh_bbox_diagonal * 0.001)
    return safe_spacing

def upsample_concave_edges(mesh, feature_angle=30.0, target_spacing=1.0):
    """
    Finds sharp, concave (internal) edges (Angles > feature_angle).
    """
    # Merge duplicated vertices so trimesh recognizes sharp edges as connected
    mesh.merge_vertices()
    mesh.fix_normals()
    
    adj_pairs = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    
    edge_points =[]
    
    for i in tqdm(range(len(adj_pairs)), desc="Upsampling Sharp Concave Edges", leave=False):
        face1_idx, face2_idx = adj_pairs[i]
        
        normal1 = mesh.face_normals[face1_idx]
        normal2 = mesh.face_normals[face2_idx]
        
        angle_rad = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # 1. Check for SHARP edges
        if angle_deg > feature_angle:
            centroid1 = mesh.triangles_center[face1_idx]
            centroid2 = mesh.triangles_center[face2_idx]
            vec_1_to_2 = centroid2 - centroid1
            dot_product = np.dot(vec_1_to_2, normal1)
            
            # 2. Check if CONCAVE (bends inward)
            if dot_product > 0:
                p1 = mesh.vertices[adj_edges[i][0]]
                p2 = mesh.vertices[adj_edges[i][1]]
                
                dist = np.linalg.norm(p2 - p1)
                num_pts = max(2, int(np.ceil(dist / target_spacing)))
                
                alphas = np.linspace(0, 1, num_pts)
                sampled_points_on_edge = p1 + alphas[:, None] * (p2 - p1)
                edge_points.append(sampled_points_on_edge)

    if not edge_points:
        return np.empty((0, 3))
        
    return np.vstack(edge_points)

def upsample_concave_surfaces(mesh, feature_angle=30.0, target_spacing=1.0):
    """
    Finds smooth, concave surfaces like holes and fillets (Angles between 1° and feature_angle).
    Replaces the brittle PyVista curvature approach with robust geometry.
    """
    mesh.merge_vertices()
    mesh.fix_normals()
    
    adj_pairs = mesh.face_adjacency
    
    concave_face_indices = set()
    
    # 1. Identify smooth concave faces
    for i in tqdm(range(len(adj_pairs)), desc="Identifying Smooth Concave Surfaces", leave=False):
        face1_idx, face2_idx = adj_pairs[i]
        
        normal1 = mesh.face_normals[face1_idx]
        normal2 = mesh.face_normals[face2_idx]
        
        angle_rad = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Check for SMOOTH curves (ignore flat planes < 1.0°, and sharp edges > feature_angle)
        if 1.0 < angle_deg <= feature_angle:
            centroid1 = mesh.triangles_center[face1_idx]
            centroid2 = mesh.triangles_center[face2_idx]
            vec_1_to_2 = centroid2 - centroid1
            dot_product = np.dot(vec_1_to_2, normal1)
            
            # Check if CONCAVE (bends inward like a hole or fillet)
            if dot_product > 0:
                concave_face_indices.add(face1_idx)
                concave_face_indices.add(face2_idx)
                
    if not concave_face_indices:
        return np.empty((0, 3))
        
    concave_faces = list(concave_face_indices)
    areas = mesh.area_faces[concave_faces]
    total_area = np.sum(areas)
    
    # 2. Determine how many points to sample based on the LLM's area density (spacing^2)
    target_area_per_point = target_spacing ** 2
    num_samples = int(np.ceil(total_area / target_area_per_point))
    
    if num_samples == 0:
        return np.empty((0, 3))
        
    print(f"    -> Distributing {num_samples} points across concave holes/fillets (Area: {total_area:.2f})")
    
    # --- 3. CUSTOM BARYCENTRIC SAMPLING (VERSION-AGNOSTIC) ---
    
    # Pick faces randomly, but weight the probability by the face's area
    probabilities = areas / total_area
    chosen_face_indices = np.random.choice(concave_faces, size=num_samples, p=probabilities)
    
    # Generate random barycentric coordinates for the chosen triangles
    r1 = np.random.rand(num_samples)
    r2 = np.random.rand(num_samples)
    sqrt_r1 = np.sqrt(r1)
    
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v
    
    # Get the 3D vertices of the randomly chosen faces
    face_vertices = mesh.vertices[mesh.faces[chosen_face_indices]]
    
    # Compute the final 3D coordinates using the barycentric weights
    sampled_points = (u[:, None] * face_vertices[:, 0, :] + 
                      v[:, None] * face_vertices[:, 1, :] + 
                      w[:, None] * face_vertices[:, 2, :])
    
    return sampled_points

    
if __name__ == '__main__':
    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_rebuttal_baseline'
    total_llm_points = 0
    total_baseline_points = 0
    for CAD_file_name in sorted(os.listdir(parent_dir)):
        
        llm_points_path = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes/{CAD_file_name}/compression_gemini-3-flash-preview_geo_maxprompt_ortho_5views_11grid_1run/refinement_points_prefilt.npy'
        parser = argparse.ArgumentParser(description="Comprehensive Classical Baseline for Stress Concentration.")
    
        parser.add_argument('--mesh_path', type=str, default=f'{parent_dir}/{CAD_file_name}/renders_pyvista/{CAD_file_name}.obj')
        parser.add_argument('--llm_points_path', type=str, default=llm_points_path, help="Path to the .npy file generated by your LLM method.")
        
        parser.add_argument('--feature_angle', type=float, default=30.0, help="Angle threshold (degrees) separating sharp edges from smooth curves.")
        args = parser.parse_args()

        print(f"Running Dense Geometric Baseline for: {os.path.basename(args.mesh_path)}")
        
        cad_object = os.path.basename(args.mesh_path).replace(".obj", "") 
        experiment_name = f'baseline_DENSE_dihedral{args.feature_angle}deg'
        
        # --- 1. Load Meshes and Calculate Target Density ---
        print("Loading meshes...")
        mesh_trimesh = trimesh.load_mesh(args.mesh_path)
        
        print("Loading LLM predictions to determine sampling density...")
        llm_points = np.load(args.llm_points_path)
        bbox_diag = np.linalg.norm(mesh_trimesh.bounding_box.extents)
        
        #target_spacing = calculate_cloud_density(llm_points, mesh_bbox_diagonal=bbox_diag)
        #target_spacing = 10 * target_spacing # Loosen the density requirement to get more points for a stronger baseline. Adjust as needed.
        mesh = trimesh.load(args.mesh_path, force='mesh')
        bbox_min, bbox_max = mesh.bounds  # shape (2, 3)
        target_spacing = np.linalg.norm(bbox_max - bbox_min) / (66.667 * 2) # Adjust divisor for more/less density
    
        print(f"Target Point Spacing derived from LLM: {target_spacing:.5f} units")

        # --- 2. Compute Both Dense Heuristics ---
        print(f"\n[1/2] Extracting and upsampling SHARP Concave Edges (> {args.feature_angle}°)...")
        dihedral_points = upsample_concave_edges(mesh_trimesh, feature_angle=args.feature_angle, target_spacing=target_spacing)
        dihedral_points = filter_points_by_density_fast(dihedral_points, target_spacing * 0.75) # Optional post-filtering to ensure minimum spacing
        print(f"      -> Generated {dihedral_points.shape[0]} points on sharp concave edges.")
        
        print(f"\n[2/2] Extracting and upsampling SMOOTH Concave Surfaces (Holes/Fillets)...")
        curvature_points = upsample_concave_surfaces(mesh_trimesh, feature_angle=args.feature_angle, target_spacing=target_spacing)
        curvature_points = filter_points_by_density_fast(curvature_points, target_spacing * 0.75) # Optional post-filtering to ensure minimum spacing
        print(f"      -> Generated {curvature_points.shape[0]} points on smooth concave surfaces.")

        # --- 3. Combine and De-duplicate the results ---
        if dihedral_points.shape[0] > 0 and curvature_points.shape[0] > 0:
            combined_points = np.vstack((dihedral_points, curvature_points))
        elif dihedral_points.shape[0] > 0:
            combined_points = dihedral_points
        else: 
            combined_points = curvature_points
            
        if combined_points.shape[0] == 0:
            print("\nWarning: No critical points found with the given parameters.")
            final_refinement_points = np.empty((0, 3))
        else:
            # Rounding to 4 decimals removes practically identical overlapping points
            final_refinement_points = np.unique(np.round(combined_points, decimals=4), axis=0)
        
        print(f"\n=========================================")
        print(f"Total LLM Nodes Provided:        {len(llm_points)}")
        print(f"Total Dense Baseline Nodes Gen.: {len(final_refinement_points)}")
        print(f"=========================================")

        total_llm_points += len(llm_points)
        total_baseline_points += len(final_refinement_points)
        # --- 4. Save Output ---
        output_dir = f'{parent_dir}/{cad_object}'
        os.makedirs(output_dir, exist_ok=True)
        
        npy_save_path = f'{output_dir}/refinement_points_{experiment_name}.npy'
        np.save(npy_save_path, final_refinement_points)
        print(f"Saved refinement points to: {npy_save_path}")

        # --- 5. Render Views for Visual Comparison ---
        if len(final_refinement_points) > 0:
            pass
            # print("Rendering output views with marked mesh points...")
            render_mesh_views(args.mesh_path, 
                            output_dir=f'{output_dir}/renders_pyvista_with_meshpoints_{experiment_name}', 
                            n_azimuth=[60], n_elevation=[-36,36], orthographic=False, 
                            points_3d=final_refinement_points.tolist(),
                            verbose=True, add_axes=False, opacity=0.7)

        print('\nCombined dense baseline pipeline completed successfully.')

    print('Average LLM points per mesh:', total_llm_points / (len(os.listdir(parent_dir))-2))
    print('Average Baseline points per mesh:', total_baseline_points / (len(os.listdir(parent_dir))-2))