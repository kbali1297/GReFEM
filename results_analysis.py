import os
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import List, Dict, Any
from scipy.spatial import cKDTree
from tqdm import tqdm

def plot_aggregated_metric_icml(
    df,
    metric_col,
    agg_fn="mean",
    xlabel="Number of views used for inference",
    ylabel="",
    title="",
    results_dir=".",
    filename="plot.pdf",
    fmt="pdf",
    ylim=None,
):
    # Fix filename format
    filename_fmt = filename.split('.')[-1].lower()
    filename = filename.replace(f'.{filename_fmt}', f'.{fmt}')

    # Aggregate data
    agg = (
        df
        .groupby(["num_views_inference", "view_type"])[metric_col]
        .agg(agg_fn)
        .reset_index()
    )

    pivot = (
        agg
        .pivot(
            index="num_views_inference",
            columns="view_type",
            values=metric_col
        )
        .sort_index()
    )

    x = pivot.index.to_numpy()
    avg_results = {}  # Changed from auc_results

    if "ortho" in pivot.columns:
        y_ortho = pivot["ortho"].to_numpy()
        avg_results["ortho"] = np.mean(y_ortho)  # Simple mean instead of trapz

    if "random" in pivot.columns:
        y_random = pivot["random"].to_numpy()
        avg_results["random"] = np.mean(y_random)  # Simple mean instead of trapz

    # ICML style setup
    plt.style.use('default')
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.figure(figsize=(5.0, 2.7))

    # Plot lines
    if "ortho" in pivot.columns:
        plt.plot(
            pivot.index,
            pivot["ortho"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="GReFEM Views",
            color='blue'
        )
    if "random" in pivot.columns:
        plt.plot(
            pivot.index,
            pivot["random"],
            marker="s",
            linewidth=1.5,
            markersize=3,
            label="Random Views",
            color='red'
        )

    # Stronger labels
    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=10, pad=6, fontweight='bold')

    plt.legend(fontsize=9, framealpha=0.9)

    # Horizontal grid
    plt.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.5)

    # Ticks
    ax = plt.gca()
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(plt.MaxNLocator(prune=None))

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout(pad=0.4)

    # Save
    if fmt == "png":
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'png'}
    else:
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'pdf'}

    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path, **save_kwargs)
    plt.close()

    print(f"Saved: {full_path} ({fmt.upper()})")
    print(f"Average {metric_col}: ortho={avg_results.get('ortho', 'N/A'):.3f}, random={avg_results.get('random', 'N/A'):.3f}")
    return avg_results  # Returns simple averages

def parse_responses_and_totals(log_path: str) -> Dict[str, Any]:
    """
    Parse individual ***I.C.E*** and ***C.H*** responses and compute totals.
    Ignores summary arrays at the end of the file.
    """

    with open(log_path, "r") as f:
        text = f.read()

    # Ignore any post-summary content
    text = text.split("Identified cell numbers")[0]

    block_pattern = re.compile(
        r"\*\*\*I\.C\.E:\s*(.*?)\*\*\*\s*\n\s*\*\*\*C\.H:\s*(.*?)\*\*\*",
        re.DOTALL
    )

    def parse_cells(s: str) -> List[int]:
        s = s.strip()
        if s == "NONE":
            return []
        return [int(x.strip()) for x in s.split(",")]

    per_view = []
    ice_all = []
    ch_all = []

    for ice_raw, ch_raw in block_pattern.findall(text):
        ice_cells = parse_cells(ice_raw)
        ch_cells = parse_cells(ch_raw)

        per_view.append({
            "ICE": ice_cells,
            "CH": ch_cells
        })

        ice_all.extend(ice_cells)
        ch_all.extend(ch_cells)

    return {
        "per_view": per_view,

        # raw totals (with repetition across views)
        "ICE_total": len(ice_all),
        "CH_total": len(ch_all),
        "total_cells": len(ice_all) + len(ch_all),

        # unique cell counts
        "ICE_unique": sorted(set(ice_all)),
        "CH_unique": sorted(set(ch_all)),
        "total_unique_cells": sorted(set(ice_all + ch_all)),
        "num_unique_cells": len(set(ice_all + ch_all)),
    }

# def read_pos_points(pos_file, value_threshold):
#     """
#     Read SP points from a .pos file and filter by value threshold.

#     Returns:
#         points: (M, 3) numpy array
#         values: (M,) numpy array
#     """
#     pattern = re.compile(
#         r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};"
#     )

#     points = []
#     values = []

#     all_points = []
#     with open(pos_file, "r") as f:
#         for line in f:
#             match = pattern.search(line)
#             if match:
#                 x, y, z, val = map(float, match.groups())
#                 all_points.append([x, y, z])
#                 if val > value_threshold:
#                     points.append([x, y, z])
#                     values.append(val)

#     return np.array(points), np.array(values)

def read_pos_points(pos_file, top_percentile=99.9):
    """
    Read SP points from a .pos file and keep the top percentile by value.

    Args:
        pos_file (str): path to .pos file
        top_percentile (float): percentile cutoff (e.g. 95 keeps top 5%)

    Returns:
        points: (M, 3) numpy array
        values: (M,) numpy array
    """
    pattern = re.compile(
        r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};"
    )

    points = []
    values = []

    with open(pos_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                x, y, z, val = map(float, match.groups())
                points.append([x, y, z])
                values.append(val)

    points = np.asarray(points)
    values = np.asarray(values)

    if len(values) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # Percentile cutoff
    cutoff = np.percentile(values, top_percentile)

    mask = values >= cutoff

    return points[mask], values[mask]

def chamfer_distance(A, B, tree_A=None, tree_B=None):
    """
    Compute symmetric Chamfer distance between two point clouds.
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("One of the point sets is empty.")

    if tree_A is None:
        tree_A = cKDTree(A)
    if tree_B is None: 
        tree_B = cKDTree(B)

    dist_A_to_B, _ = tree_B.query(A, k=1)
    dist_B_to_A, _ = tree_A.query(B, k=1)

    return dist_A_to_B.mean() + dist_B_to_A.mean()

def chamfer_distance_one_sided(A, B, tree_B=None):
    """
    Compute one-sided Chamfer distance from A to B.
    Measures how well B covers A.
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("One of the point sets is empty.")

    if tree_B is None:
        tree_B = cKDTree(B)

    dist_A_to_B, _ = tree_B.query(A, k=1)
    return dist_A_to_B.mean()

def precision_recall_f1(stress_pts, refine_pts, r, tree_stress=None, tree_refine=None):
    """
    Compute precision, recall, and F1 score between stress points and refinement points.

    Args:
        stress_pts: (Ns, 3) array of high-stress points (ground truth)
        refine_pts: (Nr, 3) array of refinement points (prediction)
        r: distance threshold (e.g. mesh size h)

    Returns:
        precision, recall, f1
    """
    if len(stress_pts) == 0 or len(refine_pts) == 0:
        return 0.0, 0.0, 0.0

    if tree_stress is None:
        tree_stress = cKDTree(stress_pts)
    if tree_refine is None:
        tree_refine = cKDTree(refine_pts)

    # Recall: stress → refine
    dist_s_to_r, _ = tree_refine.query(stress_pts, k=1)
    recall = np.mean(dist_s_to_r <= r)

    # Precision: refine → stress
    dist_r_to_s, _ = tree_stress.query(refine_pts, k=1)
    precision = np.mean(dist_r_to_s <= r)

    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

if __name__ == '__main__':
    ## Maintain a dict of experiment_names and their respective L2 and energy error stats
    parent_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes'
    samples = []
    LLM_MODELS = ['qwen3-vl-235b-a22b-instruct', 'gpt_5_mini', 'gemini_3_flash', 'grok_4_fast', 'gpt-4.1', 'claude-sonnet-4.5']
    
    jobs_to_rerun = []
    plot_save_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_mesh_simulation_plots_combined'  #
    os.makedirs(plot_save_dir, exist_ok=True)

    ## Compute pos points for all different meshes
    zz_points_dict, zz_points_tree_dict = {}, {}
    radius_dict = {}
    ## Define r_thresh = 4 * h_min_geom
    r_thresh = {
        'Electrical_Parts_Servos_SG-90_Servo-sg90':                     4*0.5278,
        'Mechanical_Parts_Mountings_SC8UU_SC8UU':                       4*1.116,
        'Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back':   4*0.971,
        'Electrical_Parts_Servos_SG-90_SG90-1-arm-horn':                4*0.120,
        'Mechanical_Parts_Mountings_SK08_SK08':                         4*0.563
    }
    for mesh_name in tqdm(os.listdir(parent_dir), total=len(os.listdir(parent_dir))):
        zz_pos_path = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/{mesh_name}/fine_mesh_zz.pos'
        zz_points, _ = read_pos_points(zz_pos_path)
        zz_points_dict[mesh_name] = zz_points
        ## Compute tree
        zz_points_tree_dict[mesh_name] = cKDTree(zz_points)
        center = zz_points.mean(axis=0)
        radius_dict[mesh_name] = np.linalg.norm(zz_points - center, axis=1).max()

    for llm_model in LLM_MODELS[:1]:
        results_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_mesh_simulation_{llm_model}'  #
        infer_mesh_points_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_{llm_model}'  #
        
        total_cells_predicted = 0
        #total_num_views_neighbour0 = 55 * 5 * 5 * 3 #total number of views X num_runs X num_objects X num_grids
        for experiment_name in tqdm(os.listdir(results_dir), total=len(os.listdir(results_dir))):
            experiment_path = f'{results_dir}/{experiment_name}'
            if experiment_name.endswith('.png') or experiment_name.endswith('.jpg'):
                continue
            if 'ortho' in experiment_name:
                mesh_name = experiment_name.split('_ortho')[0]
                view_type = 'ortho'
            else:
                mesh_name = experiment_name.split('_random')[0]
                view_type = 'random'
            
            #if mesh_name == 'Electrical_Parts_Servos_SG-90_SG90-1-arm-horn': continue  # Skip this mesh due to incomplete data

            num_views_inference = experiment_name.split(f'_{view_type}_')[1].split('views_')[0]
            llm_model = experiment_name.split(f'views_')[1].split('_')[0]
            grid_size = experiment_name.split('grid_')[0].split('_')[-1]
            num_neighbors = experiment_name.split('grid_')[1].split('neighbours')[0]
            run_num = experiment_name.split('neighbours_')[1].split('run')[0]

            refinement_points_npy = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes/{mesh_name}/refinement_points_{view_type}_{num_views_inference}views_{llm_model}_{grid_size}grid_{num_neighbors}neighbours_{run_num}run.npy'
            num_refinement_points = np.nan
            chamfer_dist = np.nan
            precision, recall, f1 = np.nan, np.nan, np.nan
            try:
                refinement_points = np.load(refinement_points_npy)
                num_refinement_points = refinement_points.shape[0]
                ## Compute chamfer distance between fine_mesh_zz_pos file and refinement points to compute if it matches
                #chamfer_dist_normalized = chamfer_distance(pos_points_dict[mesh_name], refinement_points, tree_A=pos_points_tree_dict[mesh_name])/radius_dict[mesh_name]
                ## Compute precision and recall of refinement points with GT zz stress val points
                precision, recall, f1 = precision_recall_f1(zz_points_dict[mesh_name], refinement_points, r=r_thresh[mesh_name], tree_stress=zz_points_tree_dict[mesh_name])
            except: pass
                #print(f'{refinement_points_npy} does not exist')
                

            #'Electrical_Parts_Servos_SG-90_Servo-sg90__view-ortho__nv-1__cons-0__grid-10__run-1__LLM-claude-sonnet-4.5.log'
            infer_mesh_points_exp_name = f'{mesh_name}__view-{view_type}__nv-{num_views_inference}__cons-{num_neighbors}__grid-{grid_size}__run-{run_num}__LLM-{llm_model}.log'
            cells_predicted = parse_responses_and_totals(f'{infer_mesh_points_dir}/{infer_mesh_points_exp_name}')
            num_cells_predicted = cells_predicted['total_cells']
            total_cells_predicted += num_cells_predicted
            # try:
            #     cells_predicted = parse_responses_and_totals(f'{infer_mesh_points_dir}/{infer_mesh_points_exp_name}')
            #     num_cells_predicted = cells_predicted['total_cells']
            # except: 
            #     jobs_to_rerun.append(f'{infer_mesh_points_dir}/{infer_mesh_points_exp_name}')
            #     cells_predicted = np.nan
            #     num_cells_predicted = np.nan
            
            
            l2_err_coarse = np.nan
            rel_l2_err_coarse = np.nan
            energy_err_coarse = np.nan
            rel_energy_err_coarse = np.nan
            l2_err_refined = np.nan
            rel_l2_err_refined = np.nan
            energy_err_refined = np.nan
            rel_energy_err_refined = np.nan
            l2_ref = np.nan
            energy_ref = np.nan

            coarse_read, refined_read = 0, 0
            with open(experiment_path, 'r') as f_exp:
                lines = f_exp.readlines()
                for line in lines:
                    ## Want to read a line like this:
                    # ortho_4views_gemini-3-flash-preview_10grid_0neighbours_1run_refined.msh: {'L2_err': 1.6607874938378016, 'L2_ref': 2.770242314340069, 'rel_L2': 0.5995098281622474, 'energy_err': 3493979.4591371156, 'energy_ref': 469257.7616347239, 'rel_energy': 7.445757416915935}
                    if line.startswith('coarse_mesh'):
                        coarse_dict_str = line.split('mesh.msh:')[1].strip()
                        coarse_dict = eval(coarse_dict_str)
                        l2_err_coarse = coarse_dict['L2_err']
                        rel_l2_err_coarse = coarse_dict['rel_L2']
                        energy_err_coarse = coarse_dict['energy_err']
                        rel_energy_err_coarse = coarse_dict['rel_energy']
                        coarse_read = 1

                    if line.startswith('ortho') or line.startswith('random') and coarse_read==1:
                        refined_dict_str = line.split('refined.msh:')[1].strip()
                        refined_dict = eval(refined_dict_str)
                        l2_err_refined = refined_dict['L2_err']
                        rel_l2_err_refined = refined_dict['rel_L2']
                        energy_err_refined = refined_dict['energy_err']
                        rel_energy_err_refined = refined_dict['rel_energy']
                        l2_ref = refined_dict['L2_ref']
                        energy_ref = refined_dict['energy_ref']
                        refined_read = 1
                
            if coarse_read==0 or refined_read==0:
                #print(f"Incomplete data for experiment: {experiment_name}")
                precision, recall, f1 = np.nan, np.nan, np.nan
                        
            samples.append({
            'llm_model': llm_model,
            'mesh_name': mesh_name,
            'view_type': view_type,
            'num_views_inference': int(num_views_inference),
            'llm_model': llm_model,
            'grid_size': grid_size,
            'num_neighbors': num_neighbors,
            'run': run_num,
            'num_cells_predicted': num_cells_predicted,
            'num_refinement_points': num_refinement_points,
            'l2_err_refined': l2_err_refined,
            'rel_l2_err_refined': rel_l2_err_refined,
            'energy_err_refined': energy_err_refined,
            'rel_energy_err_refined': rel_energy_err_refined,
            'l2_err_coarse': coarse_dict['L2_err'],
            'rel_l2_err_coarse': coarse_dict['rel_L2'],
            'energy_err_coarse': coarse_dict['energy_err'],
            'rel_energy_err_coarse': coarse_dict['rel_energy'],
            'l2_ref': refined_dict['L2_ref'],
            'energy_ref': refined_dict['energy_ref'],
            'rel_l2_err_normalized_refined': rel_l2_err_refined / coarse_dict['rel_L2'],
            'rel_energy_err_normalized_refined': rel_energy_err_refined / coarse_dict['rel_energy'],
            #'CD_normalized': chamfer_dist_normalized
            'precision':precision,
            'recall':recall,
            'F1':f1
            })
        

    df = pd.DataFrame(samples)

    ## Plot average relative L2 error across meshes comparison for ortho and random for increasing number of views and neighbors 0, grid_size 10
    num_neighbors = '0'
    #grid_size = '10'
    top_num_views = 10
    df_filt = df[(df['num_neighbors']==num_neighbors) & (df['num_views_inference']<=top_num_views)] #& (df['grid_size']==grid_size)]

    Mean_L2_norm = plot_aggregated_metric_icml(
        df=df_filt,
        metric_col="rel_l2_err_normalized_refined",
        agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="$\delta_{L^2,norm}$",
        #title=f"Average Normalized Relative L2 Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"avg_normalized_rel_L2_error_vs_num_views.png",
        #ylim=(0.7, 1.15)
    )

    Mean_energy_norm = plot_aggregated_metric_icml(
        df=df_filt,
        metric_col="rel_energy_err_normalized_refined",
        agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="$\delta_{E,norm}$",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"avg_normalized_rel_energy_error_vs_num_views.png",
        #ylim=(0.7, 1.15)
    )

    # AUC_CD_norm = plot_aggregated_metric_icml2(
    #     df=df_filt,
    #     metric_col="CD_normalized",
    #     agg_fn="mean",
    #     xlabel="k = No. of Grid inference views",
    #     ylabel="$CD_{norm}$",
    #     #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"CD_normalized_vs_num_views.png",
    # )

    Mean_precision = plot_aggregated_metric_icml(
        df=df_filt,
        metric_col="precision",
        agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="Precision@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"Precision@r_vs_num_views_{llm_model}.png",
    )
    Mean_recall = plot_aggregated_metric_icml(
        df=df_filt,
        metric_col="recall",
        agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="Coverage@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"Recall@r_vs_num_views_{llm_model}.png",
    )
    Mean_F1 = plot_aggregated_metric_icml(
        df=df_filt,
        metric_col="F1",
        agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="F1@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"F1@r_vs_num_views_{llm_model}.png",
    )

    print(f'{llm_model} Mean_energy_norm: {Mean_energy_norm}')
    print(f'{llm_model} Mean_L2_norm: {Mean_L2_norm}')
    #print(f'AUC CD norm: {AUC_CD_norm}')
    print(f'{llm_model} Mean_precision: {Mean_precision}')
    print(f'{llm_model} Mean_recall: {Mean_recall}')
    print(f'{llm_model} Mean_F1: {Mean_F1}')

    #total_views = top_num_views * (top_num_views+1)/2 * 5 * 5 * 3
    #df_filt_ortho = df_filt[df['view_type']=="ortho"]
    #print(f'Average number of cells per view:{df_filt_ortho["num_cells_predicted"].sum()/total_views}')

    ## First average CD across views, and then average it for top views
    # Make sure CD is numeric
    # df_filt_ortho["CD"] = pd.to_numeric(df_filt_ortho["CD"], errors="coerce")

    # # --- Step 1: average CD across runs (per view configuration) ---
    # cd_per_view = (
    #     df_filt_ortho
    #     .groupby([
    #         "mesh_name",
    #         "llm_model",
    #         "grid_size",
    #         "num_views_inference"
    #     ])["CD"]
    #     .mean()   # NaNs automatically ignored
    #     .reset_index(name="CD_mean_per_view")
    # )

    # # --- Step 2: global average across top-k views ---
    # global_cd_avg = cd_per_view["CD_mean_per_view"].mean()

    # print(
    #     f"Global average Chamfer Distance "
    #     f"(ortho, top {top_num_views} views): {global_cd_avg:.6e}"
    # )









    #  plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_l2_err_refined",
    #     agg_fn="mean",
    #     ylabel="Average Relative L2 Error (Refined Mesh)",
    #     title=f"Average Relative L2 Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_rel_L2_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

    # ## Plot average relative energy error across meshes comparison for ortho and random for increasing number of views and neighbors 0, grid_size 10
    # plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_energy_err_refined",
    #     agg_fn="mean",
    #     ylabel="Average Relative Energy Error (Refined Mesh)",
    #     title=f"Average Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_rel_energy_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

    # plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="energy_err_refined",
    #     agg_fn="mean",
    #     ylabel="Average Energy Error (Refined Mesh)",
    #     title=f"Average Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_energy_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

    # plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="l2_err_refined",
    #     agg_fn="mean",
    #     ylabel="Average L2 Error (Refined Mesh)",
    #     title=f"Average L2 Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_l2_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

    # plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_l2_err_normalized_refined",
    #     agg_fn="mean",
    #     ylabel="Average Normalized Relative L2 Error (Refined Mesh)",
    #     title=f"Average Normalized Relative L2 Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_normalized_rel_L2_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

    # plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_energy_err_normalized_refined",
    #     agg_fn="mean",
    #     ylabel="Average Normalized Relative Energy Error (Refined Mesh)",
    #     title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_normalized_rel_energy_error_vs_num_views_{num_neighbors}neighbors_{grid_size}grid.png",
    # )

