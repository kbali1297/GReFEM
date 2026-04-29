import os
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import List, Dict, Any
from scipy.spatial import cKDTree
from tqdm import tqdm
import pickle


def print_variance_stats(df, metric_col, group_by_cols):
    """
    Computes and prints the mean and standard deviation of a metric grouped by specific columns.
    Useful for reporting stability across runs, grid sizes, or prompts.
    """
    stats = df.groupby(group_by_cols)[metric_col].agg(['mean', 'std']).reset_index()
    print(f"\n--- Variance Analysis for {metric_col} grouped by {group_by_cols} ---")
    print(stats.to_string(index=False))
    return stats

def plot_aggregated_metric_flexible(
    df,
    metric_col,
    x_col="num_views_inference",
    hue_col="view_type",  # The column used to separate the lines (e.g., view_type, prompt_type, llm_model)
    xlabel="Number of views used for inference",
    ylabel="",
    title="",
    results_dir=".",
    filename="plot.pdf",
    fmt="pdf",
    ylim=None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    filename_fmt = filename.split('.')[-1].lower()
    filename = filename.replace(f'.{filename_fmt}', f'.{fmt}')

    # --- DYNAMIC AGGREGATION ---
    # Group by the specified x-axis and hue to get mean and std across all other variables (like runs, meshes, etc.)
    agg = df.groupby([x_col, hue_col])[metric_col].agg(['mean', 'std']).reset_index()

    plt.style.use('default')
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.figure(figsize=(5.0, 2.7))
    ax = plt.gca()

    hues = agg[hue_col].unique()
    
    # Generate colors dynamically if there are many hues, or define a static map
    colors = plt.cm.tab10(np.linspace(0, 1, len(hues)))
    markers =['o', 's', '^', 'D', 'v', 'p']

    for idx, hue_val in enumerate(hues):
        subset = agg[agg[hue_col] == hue_val].sort_values(x_col)
        
        x = subset[x_col]
        y_mean = subset['mean']
        y_std = subset['std']
        
        ax.plot(
            x, y_mean,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            markersize=3,
            label=f"{hue_col}: {hue_val}",
            color=colors[idx]
        )
        
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            color=colors[idx],
            alpha=0.15,
            linewidth=0
        )

    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=10, pad=6, fontweight='bold')
    plt.legend(fontsize=9, framealpha=0.9)
    plt.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    
    # Only force integer ticks if the x_axis is inherently numeric/integer
    if pd.api.types.is_numeric_dtype(df[x_col]):
        ax.xaxis.set_major_locator(plt.MaxNLocator(prune=None, integer=True))

    if ylim is not None: plt.ylim(ylim)
    plt.tight_layout(pad=0.4)

    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close()
    print(f"Saved: {full_path}")

    return agg

def plot_micro_aggregated_metric(
    df, x_col="num_views_inference", hue_col="view_type",
    xlabel="Number of views", ylabel="", title="", results_dir=".", filename="plot.pdf"
):
    import os
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.0, 2.7))
    ax = plt.gca()
    
    hues = df[hue_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(hues)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, hue_val in enumerate(hues):
        subset = df[df[hue_col] == hue_val]
        
        # Group by x_col and calculate sums for the micro-metric
        grouped = subset.groupby(x_col)[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum().reset_index()
        
        # Calculate the Micro Metrics per X-axis step
        grouped['micro_precision'] = grouped['matched_refine'] / grouped['num_refine']
        grouped['micro_recall'] = grouped['matched_stress'] / grouped['num_stress']
        grouped['micro_F1'] = 2 * (grouped['micro_precision'] * grouped['micro_recall']) / (grouped['micro_precision'] + grouped['micro_recall'])
        
        # Choose which metric to plot based on ylabel (as a quick hack)
        if "Precision" in ylabel: y_vals = grouped['micro_precision']
        elif "Recall" in ylabel or "Coverage" in ylabel: y_vals = grouped['micro_recall']
        else: y_vals = grouped['micro_F1']

        ax.plot(
            grouped[x_col], y_vals,
            marker=markers[idx % len(markers)],
            linewidth=1.5, markersize=3,
            label=f"{hue_col}: {hue_val}", color=colors[idx]
        )

    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=10, pad=6, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(prune=None, integer=True))
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_stability(df, metric="F1"):
    """
    Calculates specific variance metrics for the ICML rebuttal.
    """
    print(f"\n--- STABILITY ANALYSIS FOR {metric.upper()} ---")
    
    # 1. RUN STABILITY (Variance across the 5 runs for the same setup)
    # Group by everything EXCEPT 'run'
    run_vars = df.groupby(['mesh_name', 'num_views_inference', 'grid_size', 'prompt_type', 'llm_model'])[metric].std()
    print(f"Mean Std Dev across Runs (Repeatability): {run_vars.mean():.4f}")

    # 2. PROMPT STABILITY (Variance across different prompt types)
    prompt_vars = df.groupby(['mesh_name', 'num_views_inference', 'grid_size', 'run', 'llm_model'])[metric].std()
    print(f"Mean Std Dev across Prompt Types (Robustness): {prompt_vars.mean():.4f}")

    # 3. GRID STABILITY (Variance across grid resolutions 10, 11, 12)
    grid_vars = df.groupby(['mesh_name', 'num_views_inference', 'prompt_type', 'run', 'llm_model'])[metric].std()
    print(f"Mean Std Dev across Grid Sizes (Resolution sensitivity): {grid_vars.mean():.4f}")

    # 4. MODEL STABILITY (Variance across different LLMs)
    if df['llm_model'].nunique() > 1:
        model_vars = df.groupby(['mesh_name', 'num_views_inference', 'grid_size', 'prompt_type', 'run'])[metric].std()
        print(f"Mean Std Dev across LLM Models (Model sensitivity): {model_vars.mean():.4f}")


def read_pos_points(pos_file, remove_top_bottom_dist, top_percentile=99.9):
    """
    Read SP points from a .pos file and keep the top percentile by value.

    Args:
        pos_file (str): path to .pos file
        remove_top_bottom_dist (float): distance to remove from top and bottom of the object along y loading -axis, as they are on average conventionally high stress but not necessarily the most informative for refinement
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
    
    filtered_points, filtered_values = [], []
    for pt_idx, point in enumerate(points):
        if point[1] > y_max - remove_top_bottom_dist or point[1] < y_min + remove_top_bottom_dist:
            continue
        filtered_points.append(point)
        filtered_values.append(values[pt_idx])

    points = np.asarray(filtered_points)
    values = np.asarray(filtered_values)

    if len(values) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # Percentile cutoff
    cutoff = np.percentile(values, top_percentile)

    mask = values >= cutoff

    return points[mask], values[mask]

def precision_recall_f1(stress_pts, refine_pts, r, tree_stress=None, tree_refine=None):
    """
    Compute precision, recall, and F1 score, AND return raw point counts for micro-averaging.
    """
    num_stress = len(stress_pts)
    num_refine = len(refine_pts)

    if num_stress == 0 or num_refine == 0:
        return 0.0, 0.0, 0.0, 0, 0, num_stress, num_refine

    if tree_stress is None:
        tree_stress = cKDTree(stress_pts)
    if tree_refine is None:
        tree_refine = cKDTree(refine_pts)

    # Recall: stress → refine (How many GT stress points were successfully covered by predicted points?)
    dist_s_to_r, _ = tree_refine.query(stress_pts, k=1)
    matched_stress = np.sum(dist_s_to_r <= r)
    recall = matched_stress / num_stress

    # Precision: refine → stress (How many predicted points actually hit a GT stress area?)
    dist_r_to_s, _ = tree_stress.query(refine_pts, k=1)
    matched_refine = np.sum(dist_r_to_s <= r)
    precision = matched_refine / num_refine

    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, matched_stress, matched_refine, num_stress, num_refine

if __name__ == '__main__':

    df_cache_path = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/df_neurips_2026.csv"
    parent_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes'  #
    plot_save_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/logs_plots'  #
    infer_mesh_points_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_infer_meshpoints_30.03.2026'  #for gemini refer to the logs in 29.03.2026 folder please, for all other models the former contains the log files
        
    os.makedirs(plot_save_dir, exist_ok=True)

    if os.path.exists(df_cache_path):
        print(f"Loading cached DataFrame from {df_cache_path}")
        df = pd.read_csv(df_cache_path)
    else:
        ## Maintain a dict of experiment_names and their respective L2 and energy error stats
        samples = []
        LLM_models = ['gemini-3-flash-preview']
        #LLM_MODELS = ['qwen3-vl-235b-a22b-instruct', 'gpt_5_mini', 'gemini_3_flash', 'grok_4_fast', 'gpt-4.1', 'claude-sonnet-4.5']
        #LLM_MODELS = ['qwen3-vl-235b-a22b-instruct', 'gemini-3-flash', 'grok-4-fast']
        jobs_to_rerun = []
    
        ## Define r_thresh = 4 * h_min_geom
        r_thresh = {
            'Electrical_Parts_Servos_SG-90_Servo-sg90':                     4*0.528,
            'Mechanical_Parts_Mountings_SC8UU_SC8UU':                       4*0.340,
            'Mechanical_Parts_Mountings_SHF08_SHF08':                       4*0.503,
            'Electrical_Parts_Servos_SG-90_SG90-1-arm-horn':                4*0.120,
            'Electrical_Parts_Servos_SG-90_SG90-4-arms-horn':               4*0.226,
            'Mechanical_Parts_Mountings_SK08_SK08-SK08':                    4*0.563,
            '00200002':                                                     4*0.772,
            '00200005':                                                     4*1.047,
            '00200008':                                                     4*0.629,
            '00200022':                                                     4*0.709,
            '00200030':                                                     4*1.99,
            '00200037':                                                     4*0.572,
            '00200039':                                                     4*1.491,
            '00200050':                                                     4*0.735,
            '00200069':                                                     4*0.446,
            '00200070':                                                     4*0.311,
            '00200076':                                                     4*0.420,
            '00200089':                                                     4*1.763,
            '00200090':                                                     4*1.351,
            '00210005':                                                     4*2.144,
            '00210018':                                                     4*1.732,
            '00210021':                                                     4*0.949,
            '00210058':                                                     4*0.857,
            '00210070':                                                     4*0.653,
            '00210076':                                                     4*0.133,
            '00210090':                                                     4*0.560,
            '00210097':                                                     4*0.786,
            '00220004':                                                     4*0.856,
            '00220071':                                                     4*0.875,
            '00220074':                                                     4*1.342,
            '00230003':                                                     4*0.645,
            '00230017':                                                     4*0.7228,
        }

        load_cases = ['compression', 'torsion', 'bending']
        if os.path.exists('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts'):
            with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl', 'rb') as f:
                zz_points_dict = pickle.load(f)
            with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl', 'rb') as f:
                zz_points_tree_dict = pickle.load(f)
        else:
            print('Computing zz points for all meshes...')
            ## Compute pos points for all different meshes
            zz_points_dict, zz_points_tree_dict = {}, {}
            radius_dict = {}
            for mesh_name in tqdm(os.listdir(parent_dir), total=len(os.listdir(parent_dir))):
                for load_case in load_cases:
                    zz_pos_path = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes/{mesh_name}/fine_mesh_{load_case}_zz.pos'
                    zz_points, _ = read_pos_points(zz_pos_path, remove_top_bottom_dist=r_thresh[mesh_name])
                    zz_points_dict[f'{mesh_name}_{load_case}'] = zz_points
                    ## Compute tree
                    zz_points_tree_dict[f'{mesh_name}_{load_case}'] = cKDTree(zz_points)
                    #center = zz_points.mean(axis=0)
                    #radius_dict[f'{mesh_name}_{load_case}'] = np.linalg.norm(zz_points - center, axis=1).max()
            os.makedirs('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts', exist_ok=True)
            with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl', 'wb') as f:
                pickle.dump(zz_points_dict, f)
            with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl', 'wb') as f:
                pickle.dump(zz_points_tree_dict, f)

        #for llm_model in LLM_MODELS:
        #results_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_mesh_simulation_{llm_model}'  #
        
        total_cells_predicted = 0
        #total_num_views_neighbour0 = 55 * 5 * 5 * 3 #total number of views X num_runs X num_objects X num_grids
        ## Populate experiments list
        
        ## Run cases
        LLM_NAMES =["google/gemini-3-flash-preview"]#, "qwen/qwen3-vl-235b-a22b-instruct", "~anthropic/claude-haiku-4.5", 'openai/gpt-5.4-mini'] # 'openai/gpt-5-mini', "x-ai/grok-4-fast" "anthropic/claude-sonnet-4.5, x-ai/grok-4-fast, openai/gpt-4.1, openai/gpt-5-mini, "qwen/qwen3-vl-235b-a22b-instruct""
        GRID_SIZES = [11]
        NUM_VIEWS = [7] #list(range(1, 11))  # 1 through 10
        VIEW_TYPES = ["ortho"]#["ortho", "random"]
        RUNS = [1,2,3,4,5]#list(range(1, 6))        # 1 through 5
        PROMPT_TYPES = ["geo_max"]#['geo_max', 'geo_mid', 'geo_none']
        LOAD_CASES = ['compression', 'torsion', 'bending'] # 'torsion', 'bending', 'compression'
        experiments_name_list = []
        experiments_tuple_list = []
        for llm_model in LLM_NAMES:
            for num_views in NUM_VIEWS:
                for view_type in VIEW_TYPES:
                    for prompt_type in PROMPT_TYPES:    
                        for grid_size in GRID_SIZES:
                            for run in RUNS:
                                for load_case in LOAD_CASES:
                                    #bending_gemini-3-flash-preview_geo_maxprompt_ortho_5views_11grid_1run
                                    experiment_name = f'{load_case}_{os.path.basename(llm_model)}_{prompt_type}prompt_{view_type}_{num_views}views_{grid_size}grid_{run}run'
                                    experiments_name_list.append(experiment_name)
                                    experiments_tuple_list.append((load_case, os.path.basename(llm_model), prompt_type, view_type, num_views, grid_size, run))
                                    

        for mesh_name in tqdm(sorted(os.listdir(parent_dir)), total=len(os.listdir(parent_dir))):
            
            #if mesh_name in ['00210058', '00200005']: continue  # Skip these meshes due to incomplete data
            for exp_name, exp_tuple in zip(experiments_name_list, experiments_tuple_list):

                load_case, llm_model, prompt_type, view_type, num_views_inference, grid_size, run_num = exp_tuple
                #if llm_model not in LLM_models: continue
                exp_dir = f'{parent_dir}/{mesh_name}/{exp_name}'
                refinement_points_npy = f'{exp_dir}/refinement_points_final.npy' if os.path.exists(f'{exp_dir}/refinement_points_final.npy') else f'{exp_dir}/refinement_points_prefilt.npy'
                num_refinement_points = np.nan
                precision, recall, f1 = np.nan, np.nan, np.nan
                #try:
                refinement_points = np.load(refinement_points_npy )
                num_refinement_points = refinement_points.shape[0]
                # Compute precision and recall of refinement points with GT zz stress val points
                precision, recall, f1, matched_stress, matched_refine, num_stress, num_refine = precision_recall_f1(
                    zz_points_dict[f'{mesh_name}_{load_case}'], 
                    refinement_points, 
                    r=r_thresh[mesh_name], 
                    tree_stress=zz_points_tree_dict[f'{mesh_name}_{load_case}']
                )#except: pass
                    #print(f'{refinement_points_npy} does not exist')

                # cells_predicted = parse_responses_and_totals(f'{infer_mesh_points_dir}/{experiment_name}')
                # num_cells_predicted = cells_predicted['total_cells']
                # total_cells_predicted += num_cells_predicted
                # # try:
                # #     cells_predicted = parse_responses_and_totals(f'{infer_mesh_points_dir}/{infer_mesh_points_exp_name}')
                # #     num_cells_predicted = cells_predicted['total_cells']
                # # except: 
                # #     jobs_to_rerun.append(f'{infer_mesh_points_dir}/{infer_mesh_points_exp_name}')
                # #     cells_predicted = np.nan
                # #     num_cells_predicted = np.nan


                # l2_err_coarse = np.nan
                # rel_l2_err_coarse = np.nan
                # energy_err_coarse = np.nan
                # rel_energy_err_coarse = np.nan
                # l2_err_refined = np.nan
                # rel_l2_err_refined = np.nan
                # energy_err_refined = np.nan
                # rel_energy_err_refined = np.nan
                # l2_ref = np.nan
                # energy_ref = np.nan

                # coarse_read, refined_read = 0, 0
                # with open(experiment_path, 'r') as f_exp:
                #     lines = f_exp.readlines()
                #     for line in lines:
                #         ## Want to read a line like this:
                #         # ortho_4views_gemini-3-flash-preview_10grid_0neighbours_1run_refined.msh: {'L2_err': 1.6607874938378016, 'L2_ref': 2.770242314340069, 'rel_L2': 0.5995098281622474, 'energy_err': 3493979.4591371156, 'energy_ref': 469257.7616347239, 'rel_energy': 7.445757416915935}
                #         if line.startswith('coarse_mesh'):
                #             coarse_dict_str = line.split('mesh.msh:')[1].strip()
                #             coarse_dict = eval(coarse_dict_str)
                #             l2_err_coarse = coarse_dict['L2_err']
                #             rel_l2_err_coarse = coarse_dict['rel_L2']
                #             energy_err_coarse = coarse_dict['energy_err']
                #             rel_energy_err_coarse = coarse_dict['rel_energy']
                #             coarse_read = 1

                #         if line.startswith('ortho') or line.startswith('random') and coarse_read==1:
                #             refined_dict_str = line.split('refined.msh:')[1].strip()
                #             refined_dict = eval(refined_dict_str)
                #             l2_err_refined = refined_dict['L2_err']
                #             rel_l2_err_refined = refined_dict['rel_L2']
                #             energy_err_refined = refined_dict['energy_err']
                #             rel_energy_err_refined = refined_dict['rel_energy']
                #             l2_ref = refined_dict['L2_ref']
                #             energy_ref = refined_dict['energy_ref']
                #             refined_read = 1

                # if coarse_read==0 or refined_read==0:
                #     #print(f"Incomplete data for experiment: {experiment_name}")
                #     precision, recall, f1 = np.nan, np.nan, np.nan
                # print('Examine PRecision, Recall, F1 for experiment: ', experiment_name)            
                samples.append({
                    'llm_model': llm_model,
                    'mesh_name': mesh_name,
                    'view_type': view_type,
                    'num_views_inference': int(num_views_inference),
                    'grid_size': grid_size,
                    'prompt_type': prompt_type,
                    'run': run_num,
                    'num_refinement_points': num_refinement_points,
                    'precision': precision, # Macro metrics per-mesh
                    'recall': recall,
                    'F1': f1,
                    # Raw counts for Micro-averaging
                    'matched_stress': matched_stress,
                    'matched_refine': matched_refine,
                    'num_stress': num_stress,
                    'num_refine': num_refine 
                })
            

        df = pd.DataFrame(samples)
        df.to_csv(df_cache_path, index=False)
        print(f"Saved DataFrame to {df_cache_path}")
    ## Plot average relative L2 error across meshes comparison for ortho and random for increasing number of views and neighbors 0, grid_size 10
    #grid_size = '10'
    top_num_views = 7
    prompt_type = 'geo_max'
    llm_model = 'gemini-3-flash-preview'
    #llm_model = 'grok-4-fast'
    #llm_model = 'qwen3-vl-235b-a22b-instruct'
    df_filt = df[(df['prompt_type']==prompt_type) & (df['num_views_inference']==top_num_views) & (df['llm_model']==llm_model)] #& (df['grid_size']==grid_size)]

    # Mean_L2_norm = plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_l2_err_normalized_refined",
    #     agg_fn="mean",
    #     xlabel="k = No. of Grid inference views",
    #     ylabel="$\delta_{L^2,norm}$",
    #     #title=f"Average Normalized Relative L2 Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_normalized_rel_L2_error_vs_num_views.png",
    #     #ylim=(0.7, 1.15)
    # )

    # Mean_energy_norm = plot_aggregated_metric_icml(
    #     df=df_filt,
    #     metric_col="rel_energy_err_normalized_refined",
    #     agg_fn="mean",
    #     xlabel="k = No. of Grid inference views",
    #     ylabel="$\delta_{E,norm}$",
    #     #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
    #     results_dir=plot_save_dir,
    #     filename=f"avg_normalized_rel_energy_error_vs_num_views.png",
    #     #ylim=(0.7, 1.15)
    # )

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

    Mean_precision = plot_aggregated_metric_flexible(
        df=df_filt,
        metric_col="precision",
        #agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="Precision@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"Precision@r_vs_num_views_{llm_model}_{prompt_type}.png",
    )
    Mean_recall = plot_aggregated_metric_flexible(
        df=df_filt,
        metric_col="recall",
        #agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="Coverage@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"Recall@r_vs_num_views_{llm_model}_{prompt_type}.png",
    )
    Mean_F1 = plot_aggregated_metric_flexible(
        df=df_filt,
        metric_col="F1",
        #agg_fn="mean",
        xlabel="k = No. of Grid inference views",
        ylabel="F1@r",
        #title=f"Average Normalized Relative Energy Error vs Number of Views ({num_neighbors} Neighbors {grid_size} Grid)",
        results_dir=plot_save_dir,
        filename=f"F1@r_vs_num_views_{llm_model}_{prompt_type}.png",
    )

    # print(f'{llm_model} Mean_energy_norm: {Mean_energy_norm}')
    # print(f'{llm_model} Mean_L2_norm: {Mean_L2_norm}')
    #print(f'AUC CD norm: {AUC_CD_norm}')
    total_matched_refine = df_filt['matched_refine'].sum()
    total_refine_pts = df_filt['num_refine'].sum()
    
    total_matched_stress = df_filt['matched_stress'].sum()
    total_stress_pts = df_filt['num_stress'].sum()

    micro_precision = total_matched_refine / total_refine_pts if total_refine_pts > 0 else 0
    micro_recall = total_matched_stress / total_stress_pts if total_stress_pts > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"\n--- {llm_model} | Prompt: {prompt_type} ---")
    print("MACRO-AVERAGE (Treats all meshes equally, good for geometry robustness):")
    print(f"  Mean Precision: {df_filt['precision'].mean():.4f}")
    print(f"  Mean Recall:    {df_filt['recall'].mean():.4f}")
    print(f"  Mean F1:        {df_filt['F1'].mean():.4f}")

    print("\nMICRO-AVERAGE (Weighted by volume of points, good for global point-wise accuracy):")
    print(f"  Micro Precision: {micro_precision:.4f} ({total_matched_refine}/{total_refine_pts} pts)")
    print(f"  Micro Recall:    {micro_recall:.4f} ({total_matched_stress}/{total_stress_pts} pts)")
    print(f"  Micro F1:        {micro_f1:.4f}")

    print('Lets See')



