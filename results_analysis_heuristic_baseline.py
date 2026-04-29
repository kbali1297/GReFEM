import os
import numpy as np
import pickle
import re
from scipy.spatial import cKDTree
from tqdm import tqdm
# Assuming results_analysis_rebuttal contains the updated function returning 7 values
from results_analysis_rebuttal import precision_recall_f1, read_pos_points

if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION
    # ==========================================
    baseline_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_rebuttal_baseline'
    
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
    zz_points_dict_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl'
    zz_tree_dict_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl'

    with open(zz_points_dict_path, 'rb') as f:
        zz_points_dict = pickle.load(f)
    with open(zz_tree_dict_path, 'rb') as f:
        zz_points_tree_dict = pickle.load(f)

    # ==========================================
    # 2. EVALUATE BASELINE METRICS
    # ==========================================
    metrics = {'precision':[], 'recall': [], 'f1':[]}
    
    # Micro-average accumulators
    total_matched_stress = 0
    total_matched_refine = 0
    total_stress_pts = 0
    total_refine_pts = 0
    
    missing_files = 0

    print(f"\nEvaluating Baseline: {baseline_exp_name}")
    print("-" * 50)

    for mesh_name, thresh in r_thresh.items():
        npy_path = f"{baseline_dir}/{mesh_name}/refinement_points_{baseline_exp_name}.npy"
            
        if not os.path.exists(npy_path):
            missing_files += 1
            continue

        # Load baseline predicted points once per mesh (baseline is geometry-based, not load-case-based)
        try:
            baseline_pts = np.load(npy_path)

            for load_case in ['compression', 'torsion', 'bending']:
                gt_key = f'{mesh_name}_{load_case}'
                if gt_key not in zz_points_dict:
                    continue

                # UPDATED: Capture 7 return values
                p, r, f1, m_stress, m_refine, n_stress, n_refine = precision_recall_f1(
                    stress_pts=zz_points_dict[gt_key], 
                    refine_pts=baseline_pts, 
                    r=thresh, 
                    tree_stress=zz_points_tree_dict[gt_key]
                )

                # Store for Macro-average
                metrics['precision'].append(p)
                metrics['recall'].append(r)
                metrics['f1'].append(f1)

                # Store for Micro-average
                total_matched_stress += m_stress
                total_matched_refine += m_refine
                total_stress_pts += n_stress
                total_refine_pts += n_refine

        except Exception as e:
            print(f"Error evaluating {mesh_name}: {e}")

    # ==========================================
    # 3. PRINT AGGREGATED RESULTS
    # ==========================================
    if not metrics['f1']:
        print("\nERROR: No baseline outputs were found or successfully evaluated.")
    else:
        # Macro Calculations
        avg_p = np.mean(metrics['precision'])
        avg_r = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        
        # Micro Calculations
        micro_p = total_matched_refine / total_refine_pts if total_refine_pts > 0 else 0
        micro_r = total_matched_stress / total_stress_pts if total_stress_pts > 0 else 0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

        print(f"Evaluated {len(metrics['f1'])} mesh-load case combinations.")
        if missing_files > 0:
            print(f"Skipped {missing_files} meshes (missing files).")
        
        print("\n=== CLASSICAL BASELINE RESULTS ===")
        print(f"{'Metric':<15} | {'Macro (Mean)':<12} | {'Micro (Weighted)':<12}")
        print("-" * 45)
        print(f"{'Precision@r':<15} | {avg_p:<12.4f} | {micro_p:<12.4f}")
        print(f"{'Recall@r':<15} | {avg_r:<12.4f} | {micro_r:<12.4f}")
        print(f"{'F1@r':<15} | {avg_f1:<12.4f} | {micro_f1:<12.4f}")
        print("==================================\n")
        
        print(f"Total points analyzed:")
        print(f"  Refine Pts (Pred): {total_refine_pts}")
        print(f"  Stress Pts (GT):   {total_stress_pts}")