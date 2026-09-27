import os
import numpy as np
import pickle
import re
from scipy.spatial import cKDTree
from tqdm import tqdm
import json
# Assuming results_analysis_rebuttal contains the updated function returning 7 values
from results_analysis_rebuttal import precision_recall_f1, read_pos_points

if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION
    # ==========================================
    baseline_dir = '/data/1bali/GReFEM/test_meshes_rebuttal_baseline'
    feature_angle = 30.0 
    baseline_exp_name = f'baseline_DENSE_dihedral{feature_angle}deg'

    element_size_path = os.path.join(os.path.dirname(__file__), 'cad_element_sizes.json')
    with open(element_size_path, 'r') as f:
        cad_element_sizes = json.load(f)
    r_thresh = {mesh_name: 4.0 * vals['h_min'] for mesh_name, vals in cad_element_sizes.items()}

    # ==========================================
    # 1. LOAD GT POINTS (Cached)
    # ==========================================
    zz_points_dict_path = '/data/1bali/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl'
    zz_tree_dict_path = '/data/1bali/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl'

    with open(zz_points_dict_path, 'rb') as f:
        zz_points_dict = pickle.load(f)
    with open(zz_tree_dict_path, 'rb') as f:
        zz_points_tree_dict = pickle.load(f)

    remove_list =[
        '00520044', '00530042', '00530061', '00210058', '00210018', 
        '00200037', '00200039', '00200050', '00210005', '00230017'
    ]

    # ==========================================
    # 2. PASS 1: GLOBAL POINT COUNTING & MASKING
    # ==========================================
    print("Pass 1: Counting global points...")
    mesh_point_counts = {}
    total_global_pts = 0
    valid_meshes =[]

    for mesh_name, thresh in r_thresh.items():
        if mesh_name in remove_list: continue
        npy_path = f"{baseline_dir}/{mesh_name}/refinement_points_{baseline_exp_name}.npy"
        
        if os.path.exists(npy_path):
            pts = np.load(npy_path)
            mesh_point_counts[mesh_name] = len(pts)
            total_global_pts += len(pts)
            valid_meshes.append(mesh_name)

    print(f"Total unique baseline points across all meshes: {total_global_pts}")
    
    # --- Create the Global Uniform Mask ---
    # NOTE: If you want the FINAL printout of "Refine Pts (Pred)" to say 22,000, 
    # keep in mind that your script evaluates every point up to 5 times (once per load case).
    # If that is the case, set `target_points = 22000 // 5`. 
    # If you mean 22,000 UNIQUE geometric points, leave it as 22000.
    target_points = 370851 // 5  # Adjust this if you want to sample down to a specific number of points globally
    
    if total_global_pts > target_points:
        global_mask = np.zeros(total_global_pts, dtype=bool)
        global_mask[:target_points] = True
        np.random.seed(42)  # Ensures the same 22k points are chosen every run
        np.random.shuffle(global_mask)
        print(f"Sampling down to {target_points} points globally...")
    else:
        global_mask = np.ones(total_global_pts, dtype=bool)
        print(f"Dataset has fewer than {target_points} points. Keeping all.")

    # ==========================================
    # 3. PASS 2: EVALUATE BASELINE METRICS
    # ==========================================
    metrics = {'precision':[], 'recall': [], 'f1':[]}
    
    total_matched_stress = 0
    total_matched_refine = 0
    total_stress_pts = 0
    total_refine_pts = 0

    # Per-load-case raw counts for micro-aggregation by load case.
    per_loadcase_counts = {}
    
    print(f"\nEvaluating Baseline: {baseline_exp_name}")
    print("-" * 50)

    current_global_idx = 0

    for mesh_name in valid_meshes:
        npy_path = f"{baseline_dir}/{mesh_name}/refinement_points_{baseline_exp_name}.npy"
        thresh = r_thresh[mesh_name]

        try:
            baseline_pts = np.load(npy_path)

            # --- Apply exact uniform global sampling for this specific mesh ---
            num_pts = mesh_point_counts[mesh_name]
            mesh_mask = global_mask[current_global_idx : current_global_idx + num_pts]
            current_global_idx += num_pts
            
            baseline_pts = baseline_pts[mesh_mask]
            # ------------------------------------------------------------------

            for load_case in['compression', 'torsion', 'bending', 'torsion_compression', 'bending_compression']:
                gt_key = f'{mesh_name}_{load_case}'
                if gt_key not in zz_points_dict:
                    continue

                p, r, f1, m_stress, m_refine, n_stress, n_refine = precision_recall_f1(
                    stress_pts=zz_points_dict[gt_key], 
                    refine_pts=baseline_pts, 
                    r=thresh, 
                    tree_stress=zz_points_tree_dict[gt_key]
                )

                metrics['precision'].append(p)
                metrics['recall'].append(r)
                metrics['f1'].append(f1)

                total_matched_stress += m_stress
                total_matched_refine += m_refine
                total_stress_pts += n_stress
                total_refine_pts += n_refine

                lc = per_loadcase_counts.setdefault(
                    load_case,
                    {'matched_refine': 0, 'num_refine': 0,
                     'matched_stress': 0, 'num_stress': 0,
                     'num_refinement_points': []}
                )
                lc['matched_refine'] += m_refine
                lc['num_refine'] += n_refine
                lc['matched_stress'] += m_stress
                lc['num_stress'] += n_stress
                lc['num_refinement_points'].append(len(baseline_pts))

        except Exception as e:
            print(f"Error evaluating {mesh_name}: {e}")

    # ==========================================
    # 4. PRINT AGGREGATED RESULTS
    # ==========================================
    if not metrics['f1']:
        print("\nERROR: No baseline outputs were found or successfully evaluated.")
    else:
        avg_p = np.mean(metrics['precision'])
        avg_r = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        
        micro_p = total_matched_refine / total_refine_pts if total_refine_pts > 0 else 0
        micro_r = total_matched_stress / total_stress_pts if total_stress_pts > 0 else 0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

        print(f"Evaluated {len(metrics['f1'])} mesh-load case combinations.")
        
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

        # ==========================================
        # 5. PER-LOAD-CASE MICRO METRICS (5 rows)
        # ==========================================
        import csv
        out_dir = '/data/1bali/GReFEM/logs_plots'
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f'heuristic_baseline_micro_PRF1_by_loadcase_{baseline_exp_name}.csv')

        rows = []
        load_order = ['bending', 'bending_compression', 'compression', 'torsion', 'torsion_compression']
        for lc_name in load_order:
            if lc_name not in per_loadcase_counts:
                continue
            c = per_loadcase_counts[lc_name]
            mp = c['matched_refine'] / c['num_refine'] if c['num_refine'] > 0 else float('nan')
            mr = c['matched_stress'] / c['num_stress'] if c['num_stress'] > 0 else float('nan')
            mf1 = (2 * mp * mr) / (mp + mr) if (mp + mr) > 0 else float('nan')
            avg_refp_k = float(np.mean(c['num_refinement_points'])) / 1000.0
            rows.append({
                'load_case': lc_name,
                'micro_precision': round(mp, 4),
                'micro_recall': round(mr, 4),
                'micro_F1': round(mf1, 4),
                'matched_refine': c['matched_refine'],
                'num_refine': c['num_refine'],
                'matched_stress': c['matched_stress'],
                'num_stress': c['num_stress'],
                'Avg Refinement points (k)': round(avg_refp_k, 2),
            })

        with open(out_csv, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out_csv}")
        print(f"\n=== HEURISTIC BASELINE PER-LOAD-CASE (micro) ===")
        header = f"{'load_case':<22} {'P':>8} {'R':>8} {'F1':>8} {'AvgRefPts(k)':>14}"
        print(header)
        print('-' * len(header))
        for row in rows:
            print(f"{row['load_case']:<22} {row['micro_precision']:>8.4f} "
                  f"{row['micro_recall']:>8.4f} {row['micro_F1']:>8.4f} "
                  f"{row['Avg Refinement points (k)']:>14.2f}")