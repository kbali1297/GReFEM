import os
import numpy as np
import pickle
import re
from scipy.spatial import cKDTree
from tqdm import tqdm

def read_pos_points(pos_file, remove_top_bottom_dist, top_percentile=99.9):
    """Read SP points from a .pos file and keep the top percentile by value."""
    pattern = re.compile(r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};")
    points, values = [],[]
    y_max, y_min = -np.inf, np.inf
    
    with open(pos_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                x, y, z, val = map(float, match.groups())
                points.append([x, y, z])
                y_max = max(y_max, y)
                y_min = min(y_min, y)
                values.append(val)
    
    filtered_points, filtered_values = [],[]
    for pt_idx, point in enumerate(points):
        if point[1] > y_max - remove_top_bottom_dist or point[1] < y_min + remove_top_bottom_dist:
            continue
        filtered_points.append(point)
        filtered_values.append(values[pt_idx])

    points = np.asarray(filtered_points)
    values = np.asarray(filtered_values)

    if len(values) == 0:
        return np.empty((0, 3)), np.empty((0,))

    cutoff = np.percentile(values, top_percentile)
    mask = values >= cutoff
    return points[mask], values[mask]

def precision_recall_f1(stress_pts, refine_pts, r, tree_stress=None, tree_refine=None):
    """Compute precision, recall, and F1 score."""
    if len(stress_pts) == 0 or len(refine_pts) == 0:
        return 0.0, 0.0, 0.0

    if tree_stress is None:
        tree_stress = cKDTree(stress_pts)
    if tree_refine is None:
        tree_refine = cKDTree(refine_pts)

    dist_s_to_r, _ = tree_refine.query(stress_pts, k=1)
    recall = np.mean(dist_s_to_r <= r)

    dist_r_to_s, _ = tree_stress.query(refine_pts, k=1)
    precision = np.mean(dist_r_to_s <= r)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if np.isnan(precision) or np.isnan(recall):
        precision, recall, f1 = 0.0, 0.0, 0.0
    return precision, recall, f1

if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION
    # ==========================================
    baseline_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal_baseline'
    gt_meshes_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal_26.03.2026'
    
    # Needs to match the feature_angle used when you ran the baseline script
    feature_angle = 30.0 
    baseline_exp_name = f'baseline_DENSE_dihedral{feature_angle}deg'

    r_thresh = {
        'Electrical_Parts_Servos_SG-90_Servo-sg90':       4*0.528,
        'Mechanical_Parts_Mountings_SC8UU_SC8UU':         4*0.340,
        'Mechanical_Parts_Mountings_SHF08_SHF08':         4*0.503,
        'Electrical_Parts_Servos_SG-90_SG90-1-arm-horn':  4*0.120,
        'Electrical_Parts_Servos_SG-90_SG90-4-arms-horn': 4*0.226,
        'Mechanical_Parts_Mountings_SK08_SK08-SK08':      4*0.563,
        '00200002': 4*0.772, '00200005': 4*1.047, '00200008': 4*0.629,
        '00200022': 4*0.709, '00200030': 4*1.99,  '00200037': 4*0.572,
        '00200039': 4*1.491, '00200050': 4*0.735, '00200069': 4*0.446,
        '00200070': 4*0.311, '00200076': 4*0.420, '00200089': 4*1.763,
        '00200090': 4*1.351, '00210005': 4*2.144, '00210018': 4*1.732,
        '00210021': 4*0.949, '00210058': 4*0.857, '00210070': 4*0.653,
        '00210076': 4*0.133, '00210090': 4*0.560, '00210097': 4*0.786,
        '00220004': 4*0.856, '00220071': 4*0.875, '00220074': 4*1.342,
        '00230003': 4*0.645, '00230017': 4*0.7228,
    }

    # ==========================================
    # 1. LOAD GT POINTS (Cached)
    # ==========================================
    zz_points_dict_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/zz_points_dicts/zz_points_dict_rebuttal_28.03.2026.pkl'
    zz_tree_dict_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/zz_points_dicts/zz_points_tree_dict_rebuttal_28.03.2026.pkl'

    if os.path.exists(zz_points_dict_path) and os.path.exists(zz_tree_dict_path):
        print("Loading cached GT Ground Truth dicts...")
        with open(zz_points_dict_path, 'rb') as f:
            zz_points_dict = pickle.load(f)
        with open(zz_tree_dict_path, 'rb') as f:
            zz_points_tree_dict = pickle.load(f)
    else:
        print("Computing GT points for all meshes (cache missing)...")
        zz_points_dict, zz_points_tree_dict = {}, {}
        os.makedirs(os.path.dirname(zz_points_dict_path), exist_ok=True)
        
        for mesh_name in tqdm(r_thresh.keys(), desc="Loading .pos GT files"):
            
            zz_pos_path = f'{gt_meshes_dir}/{mesh_name}/fine_mesh_zz.pos'
            if os.path.exists(zz_pos_path):
                zz_points, _ = read_pos_points(zz_pos_path, remove_top_bottom_dist=r_thresh[mesh_name])
                zz_points_dict[mesh_name] = zz_points
                zz_points_tree_dict[mesh_name] = cKDTree(zz_points)
        
        with open(zz_points_dict_path, 'wb') as f: pickle.dump(zz_points_dict, f)
        with open(zz_tree_dict_path, 'wb') as f: pickle.dump(zz_points_tree_dict, f)

    # ==========================================
    # 2. EVALUATE BASELINE METRICS
    # ==========================================
    metrics = {'precision':[], 'recall': [], 'f1':[]}
    missing_files = 0

    print(f"\nEvaluating Baseline: {baseline_exp_name}")
    print("-" * 50)

    for mesh_name, thresh in r_thresh.items():
        if mesh_name not in zz_points_dict:
            continue # Skip if no GT available
        if mesh_name in ['00210058', '00200005']: continue  # Skip these meshes due to incomplete data
    
        npy_path = f"{baseline_dir}/{mesh_name}/refinement_points_{baseline_exp_name}.npy"
        
        if not os.path.exists(npy_path):
            missing_files += 1
            continue

        # Load baseline predicted points
        try:
            baseline_pts = np.load(npy_path)
            
            p, r, f1 = precision_recall_f1(
                stress_pts=zz_points_dict[mesh_name], 
                refine_pts=baseline_pts, 
                r=thresh, 
                tree_stress=zz_points_tree_dict[mesh_name]
            )
            
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['f1'].append(f1)
            
        except Exception as e:
            print(f"Error evaluating {mesh_name}: {e}")

    # ==========================================
    # 3. PRINT AGGREGATED RESULTS
    # ==========================================
    if not metrics['f1']:
        print("\nERROR: No baseline outputs were found or successfully evaluated.")
        if missing_files > 0:
            print(f"Could not find .npy files for {missing_files} meshes. Check your feature_angle parameter.")
    else:
        avg_p = np.mean(metrics['precision'])
        avg_r = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        
        print(f"Evaluated {len(metrics['f1'])} meshes.")
        if missing_files > 0:
            print(f"Skipped {missing_files} meshes (missing files).")
        
        print("\n=== CLASSICAL BASELINE RESULTS ===")
        print(f"Mean Precision@r : {avg_p:.4f}")
        print(f"Mean Recall@r    : {avg_r:.4f}  (Coverage)")
        print(f"Mean F1@r        : {avg_f1:.4f}")
        print("==================================\n")