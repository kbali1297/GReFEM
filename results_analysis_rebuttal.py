import os
import json
import numpy as np
import ast
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
    show_std=True,
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
        
        if show_std:
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


def plot_preaggregated_metric(
    df,
    metric_col,
    x_col="num_views_inference",
    hue_col="view_type",
    xlabel="Number of views used for inference",
    ylabel="",
    title="",
    results_dir=".",
    filename="plot.pdf",
    fmt="pdf",
    ylim=(0.0, 1.0),
    annotate_hue=None,
    annotate_decimals=6,
    annotate_prefix=None,
):
    """Plot pre-aggregated values directly without any additional averaging."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    filename_fmt = filename.split('.')[-1].lower()
    filename = filename.replace(f'.{filename_fmt}', f'.{fmt}')

    plt.style.use('default')
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.figure(figsize=(5.4, 3.0))
    ax = plt.gca()

    hues = sorted(df[hue_col].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(hues)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'X', '*']

    for idx, hue_val in enumerate(hues):
        subset = df[df[hue_col] == hue_val].sort_values(x_col)
        if subset.empty:
            continue

        x = subset[x_col].astype(float).values
        y = subset[metric_col].astype(float).values

        ax.plot(
            x,
            y,
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=4,
            label=str(hue_val),
            color=colors[idx],
        )

        # Annotate requested curve values to avoid visual ambiguity.
        if annotate_hue is not None and str(hue_val) == str(annotate_hue):
            for xv, yv in zip(x, y):
                if np.isfinite(yv):
                    label_core = f"{yv:.{annotate_decimals}f}"
                    label = f"{annotate_prefix}@{int(xv)}={label_core}" if annotate_prefix else label_core
                    ax.annotate(
                        label,
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=colors[idx], linewidth=0.8),
                    )

    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=10, pad=6, fontweight='bold')
    plt.legend(fontsize=8, framealpha=0.9)
    plt.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(prune=None, integer=True))
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout(pad=0.4)

    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close()
    print(f"Saved: {full_path}")

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

def get_total_cells(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Find all text between triple asterisks *** ... ***
    blocks = re.findall(r'\*\*\*(.*?)\*\*\*', content)
    
    # 2. Count all numbers found inside those blocks
    all_numbers = re.findall(r'\d+', " ".join(blocks))
    
    return len(all_numbers)


def parse_rel_error_ratios_from_logs(log_paths: List[str]) -> Dict[Any, Dict[str, float]]:
    """
    Parse [SUCCESS] lines from simulation logs and compute refined/coarse ratios:
      rel_L2_ratio = refined.rel_L2 / coarse.rel_L2
      rel_E_ratio  = refined.rel_energy / coarse.rel_energy
    Returns a dict keyed by (mesh_name, experiment_name).
    """
    ratios_by_case = {}
    success_pattern = re.compile(r"\[SUCCESS\]\s+(.+?)\s+\|\s+(.+?)\s+\|\s+Result:\s+(\{.*\})")

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Warning: log file not found, skipping {log_path}")
            continue

        with open(log_path, 'r') as f:
            for line in f:
                match = success_pattern.search(line)
                if not match:
                    continue

                mesh_name = match.group(1).strip()
                experiment_name = match.group(2).strip()
                result_str = match.group(3).strip()

                try:
                    result_dict = ast.literal_eval(result_str)
                    coarse = result_dict.get('coarse', {})
                    refined = result_dict.get('refined', {})

                    coarse_rel_l2 = coarse.get('rel_L2', np.nan)
                    refined_rel_l2 = refined.get('rel_L2', np.nan)
                    coarse_rel_e = coarse.get('rel_energy', np.nan)
                    refined_rel_e = refined.get('rel_energy', np.nan)

                    rel_l2_ratio = np.nan
                    rel_e_ratio = np.nan
                    if np.isfinite(coarse_rel_l2) and coarse_rel_l2 != 0 and np.isfinite(refined_rel_l2):
                        rel_l2_ratio = refined_rel_l2 / coarse_rel_l2
                    if np.isfinite(coarse_rel_e) and coarse_rel_e != 0 and np.isfinite(refined_rel_e):
                        rel_e_ratio = refined_rel_e / coarse_rel_e

                    ratios_by_case[(mesh_name, experiment_name)] = {
                        'rel_L2': rel_l2_ratio,
                        'rel_E': rel_e_ratio,
                    }
                except Exception:
                    continue

    return ratios_by_case


def build_rel_ratio_df_from_logs(log_paths: List[str]) -> pd.DataFrame:
    """
    Build a lightweight dataframe of relative-error ratios directly from [SUCCESS] log lines.
    This is used for plotting when cached CSV does not include load_case.
    """
    ratios_by_case = parse_rel_error_ratios_from_logs(log_paths)
    rows = []
    exp_pattern = re.compile(
        r"^([^_]+)_(.+?)_(geo_max|geo_mid|geo_none)prompt_(ortho|random)_(\d+)views_(\d+)grid_(\d+)run$"
    )

    for (mesh_name, experiment_name), ratio_vals in ratios_by_case.items():
        match = exp_pattern.match(experiment_name)
        if not match:
            continue

        load_case, llm_model, prompt_type, view_type, num_views, grid_size, run_num = match.groups()
        rows.append({
            'mesh_name': mesh_name,
            'load_case': load_case,
            'llm_model': llm_model,
            'prompt_type': prompt_type,
            'view_type': view_type,
            'num_views_inference': int(num_views),
            'grid_size': int(grid_size),
            'run': int(run_num),
            'rel_L2': ratio_vals.get('rel_L2', np.nan),
            'rel_E': ratio_vals.get('rel_E', np.nan),
        })

    return pd.DataFrame(rows)

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

def read_pos_points(pos_file, loading_case, top_percentile=99, remove_boundary_frac=0.02):
    """
    Read SP points from a .pos file, filter out boundary/load concentrations based on 
    the loading case, and keep the top percentile of the remaining interior values.

    Args:
        pos_file (str): path to .pos file
        loading_case (str): "compression", "bending", "torsion", "shear", etc.
        top_percentile (float): percentile cutoff (e.g. 99 keeps top 1%)
        remove_boundary_frac (float): Fraction of bounding box to trim near faces.

    Returns:
        points: (M, 3) numpy array
        values: (M,) numpy array
    """
    pattern = re.compile(r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};")

    points_list = []
    values_list = []

    # Initialize bounding box coordinates
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    if not os.path.exists(pos_file):
        print(f"Warning: {pos_file} not found.")
        return np.empty((0, 3)), np.empty((0,))

    with open(pos_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                x, y, z, val = map(float, match.groups())
                points_list.append([x, y, z])
                values_list.append(val)
                
                # Update bounding box
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

    all_points = np.asarray(points_list)
    all_values = np.asarray(values_list)

    if len(all_values) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # Calculate absolute distance thresholds
    x_dist = (x_max - x_min) * remove_boundary_frac
    y_dist = (y_max - y_min) * remove_boundary_frac
    
    # Initialize mask for interior points
    interior_mask = np.ones(len(all_points), dtype=bool)

    # Apply Case-Specific Boundary Filtering
    if loading_case in ["compression", "bending", "torsion", "shear"]:
        # Remove top and bottom concentrations along the Y-loading axis
        interior_mask &= (all_points[:, 1] < (y_max - y_dist)) & (all_points[:, 1] > (y_min + y_dist))
    
    elif loading_case in ["bending_compression", "torsion_compression"]:
        # Remove side boundaries (X) and the top face (Y)
        interior_mask &= (all_points[:, 0] < (x_max - x_dist)) & (all_points[:, 0] > (x_min + x_dist))
        interior_mask &= (all_points[:, 1] < (y_max - y_dist))
    
    # Filter points/values to the interior
    interior_points = all_points[interior_mask]
    interior_values = all_values[interior_mask]

    if len(interior_values) == 0:
        return np.empty((0, 3)), np.empty((0,))

    # Calculate percentile cutoff based ONLY on interior stress/indicator values
    cutoff = np.percentile(interior_values, top_percentile)
    
    # Final mask: inside the interior AND above the percentile cutoff
    final_mask = interior_values >= cutoff

    return interior_points[final_mask], interior_values[final_mask]

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

    df_cache_path = "/data/1bali/GReFEM/df_neurips_2026.csv"
    parent_dir = f'/data/1bali/GReFEM/test_meshes'  #
    plot_save_dir = f'/data/1bali/GReFEM/logs_plots'  #
    infer_mesh_points_dir = f'/data/1bali/GReFEM/logs_large_infer_meshpoints_30.03.2026'  #for gemini refer to the logs in 29.03.2026 folder please, for all other models the former contains the log files
        
    os.makedirs(plot_save_dir, exist_ok=True)

    rebuild_cache = True
    if os.path.exists(df_cache_path):
        print(f"Loading cached DataFrame from {df_cache_path}")
        df = pd.read_csv(df_cache_path, low_memory=False, dtype={'mesh_name': str})
        required_cols = {'rel_L2', 'rel_E'}
        if not required_cols.issubset(df.columns):
            print("Cached DataFrame is missing rel_L2/rel_E columns. Recomputing cache...")
        else:
            gemini_cached = df[df.get('llm_model', pd.Series(dtype=str)) == 'gemini-3-flash-preview']
            gemini_views = set(gemini_cached.get('view_type', pd.Series(dtype=str)).dropna().unique().tolist())
            if {'ortho', 'random'}.issubset(gemini_views):
                rebuild_cache = False
            else:
                print("Cached DataFrame is missing Gemini ortho/random paired coverage. Recomputing cache...")

    if rebuild_cache:
        ## Maintain a dict of experiment_names and their respective L2 and energy error stats
        samples = []
        #LLM_models = ["google/gemini-3-flash-preview", "qwen/qwen3-vl-235b-a22b-instruct", "anthropic/claude-haiku-4.5", 'openai/gpt-5.4-mini']
        #LLM_MODELS = ['qwen3-vl-235b-a22b-instruct', 'gpt_5_mini', 'gemini_3_flash', 'grok_4_fast', 'gpt-4.1', 'claude-sonnet-4.5']
        #LLM_MODELS = ['qwen3-vl-235b-a22b-instruct', 'gemini-3-flash', 'grok-4-fast']
        jobs_to_rerun = []
    
        ## Define r_thresh = 4 * h_min_geom from cad_element_sizes.json
        element_size_path = os.path.join(os.path.dirname(__file__), 'cad_element_sizes.json')
        with open(element_size_path, 'r') as f:
            cad_element_sizes = json.load(f)
        r_thresh = {mesh_name: 4.0 * vals['h_min'] for mesh_name, vals in cad_element_sizes.items()}

        load_cases = ['compression', 'torsion', 'bending', 'torsion_compression', 'bending_compression']
        if os.path.exists('/data/1bali/GReFEM/zz_points_dicts'):
            with open('/data/1bali/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl', 'rb') as f:
                zz_points_dict = pickle.load(f)
            with open('/data/1bali/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl', 'rb') as f:
                zz_points_tree_dict = pickle.load(f)
        else:
            print('Computing zz points for all meshes...')
            ## Compute pos points for all different meshes
            zz_points_dict, zz_points_tree_dict = {}, {}
            radius_dict = {}
            for mesh_name in tqdm(os.listdir(parent_dir), total=len(os.listdir(parent_dir))):
                for load_case in load_cases:
                    zz_pos_path = f'/data/1bali/GReFEM/test_meshes/{mesh_name}/fine_mesh_{load_case}_zz.pos'
                    zz_points, _ = read_pos_points(zz_pos_path, load_case, top_percentile=99)
                    zz_points_dict[f'{mesh_name}_{load_case}'] = zz_points
                    ## Compute tree
                    zz_points_tree_dict[f'{mesh_name}_{load_case}'] = cKDTree(zz_points)
                    #center = zz_points.mean(axis=0)
                    #radius_dict[f'{mesh_name}_{load_case}'] = np.linalg.norm(zz_points - center, axis=1).max()
            os.makedirs('/data/1bali/GReFEM/zz_points_dicts', exist_ok=True)
            with open('/data/1bali/GReFEM/zz_points_dicts/zz_points_dict_neurips.pkl', 'wb') as f:
                pickle.dump(zz_points_dict, f)
            with open('/data/1bali/GReFEM/zz_points_dicts/zz_points_tree_dict_neurips.pkl', 'wb') as f:
                pickle.dump(zz_points_tree_dict, f)

        #for llm_model in LLM_MODELS:
        #results_dir = f'/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/logs_large_mesh_simulation_{llm_model}'  #
        
        total_cells_predicted = 0
        #total_num_views_neighbour0 = 55 * 5 * 5 * 3 #total number of views X num_runs X num_objects X num_grids
        ## Populate experiments list
        
        ## Run cases
        LLM_NAMES =["google/gemini-3-flash-preview", "qwen/qwen3-vl-235b-a22b-instruct", "anthropic/claude-haiku-4.5", 'openai/gpt-5.4-mini']#, "qwen/qwen3-vl-235b-a22b-instruct", "anthropic/claude-haiku-4.5", 'openai/gpt-5.4-mini'] # 'openai/gpt-5-mini', "x-ai/grok-4-fast" "anthropic/claude-sonnet-4.5, x-ai/grok-4-fast, openai/gpt-4.1, openai/gpt-5-mini, "qwen/qwen3-vl-235b-a22b-instruct""
        GRID_SIZES = [11]
        NUM_VIEWS = list(range(1, 11))  # 1 through 10
        VIEW_TYPES = ["ortho", "random"]
        RUNS = [1]#,2,3,4,5]#list(range(1, 6))        # 1 through 5
        PROMPT_TYPES = ['geo_max', 'geo_mid', 'geo_none']
        LOAD_CASES = ['compression', 'torsion', 'bending', 'torsion_compression', 'bending_compression'] # 'torsion', 'bending', 'compression'
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

        base_log = '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026.log'
        indexed_logs = [f'/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_{i}.log' for i in range(1, 9)]
        rel_error_ratios = parse_rel_error_ratios_from_logs([base_log] + indexed_logs)
                                    

        for mesh_name in tqdm(sorted(os.listdir(parent_dir)), total=len(os.listdir(parent_dir))):
            
            #if mesh_name in ['00210058', '00200005']: continue  # Skip these meshes due to incomplete data
            for exp_name, exp_tuple in zip(experiments_name_list, experiments_tuple_list):

                load_case, llm_model, prompt_type, view_type, num_views_inference, grid_size, run_num = exp_tuple
                #if llm_model not in LLM_models: continue
                exp_dir = f'{parent_dir}/{mesh_name}/{exp_name}'
                refinement_points_npy = f'{exp_dir}/refinement_points_final.npy' if os.path.exists(f'{exp_dir}/refinement_points_final.npy') else f'{exp_dir}/refinement_points_prefilt.npy'
                num_refinement_points = np.nan
                precision, recall, f1 = np.nan, np.nan, np.nan
                matched_stress, matched_refine, num_stress, num_refine = np.nan, np.nan, np.nan, np.nan
                try:
                    refinement_points = np.load(refinement_points_npy )
                    num_refinement_points = refinement_points.shape[0]
                    # Compute precision and recall of refinement points with GT zz stress val points
                    precision, recall, f1, matched_stress, matched_refine, num_stress, num_refine = precision_recall_f1(
                        zz_points_dict[f'{mesh_name}_{load_case}'], 
                        refinement_points, 
                        r=r_thresh[mesh_name], 
                        tree_stress=zz_points_tree_dict[f'{mesh_name}_{load_case}']
                    )
                except: pass
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
                inference_log_path = f'/data/1bali/GReFEM/test_meshes/{mesh_name}/{exp_name}/inference.log'
                total_cells = np.nan
                if os.path.exists(inference_log_path):
                    total_cells = get_total_cells(inference_log_path)
                rel_ratios = rel_error_ratios.get((mesh_name, exp_name), {})
                samples.append({
                    'llm_model': llm_model,
                    'mesh_name': mesh_name,
                    'load_case': load_case,
                    'view_type': view_type,
                    'num_views_inference': int(num_views_inference),
                    'grid_size': grid_size,
                    'prompt_type': prompt_type,
                    'run': run_num,
                    'num_refinement_points': num_refinement_points,
                    'precision': precision, # Macro metrics per-mesh
                    'recall': recall,
                    'F1': f1,
                    'rel_L2': rel_ratios.get('rel_L2', np.nan),
                    'rel_E': rel_ratios.get('rel_E', np.nan),
                    'total_cells': total_cells/num_views_inference if np.isfinite(total_cells) else np.nan,
                    # Raw counts for Micro-averaging
                    'matched_stress': matched_stress,
                    'matched_refine': matched_refine,
                    'num_stress': num_stress,
                    'num_refine': num_refine 
                })
            

        df = pd.DataFrame(samples)
        df.to_csv(df_cache_path, index=False)
        print(f"Saved DataFrame to {df_cache_path}")
    ## Plot paired ortho/random curves for Gemini only.
    ## Average over all parameters except num_views_inference and view_type,
    ## while dropping any parameter-key where either ortho or random has missing values.
    ## Use uniaxial load cases only, and only prompt type geo_mid.
    gemini_model = 'gemini-3-flash-preview'
    prompt_type_filter = 'geo_mid'
    uniaxial_cases = ['bending', 'compression', 'torsion']
    remove_list = [
        '00520044',  
        '00530042', 
        '00530061', 
        '00210058', 
        '00210018', 
        #'00220004', 
        #'00230003', 
        '00200037', 
        '00200039', 
        '00200050', 
        '00210005', 
        '00230017'
    ]
    mesh_names = [cad_name for cad_name in sorted(os.listdir(parent_dir)) if not cad_name in remove_list]
    if 'load_case' in df.columns:
        gemini_df = df[
            (df['mesh_name'].isin(mesh_names))
            & (df['llm_model'] == gemini_model)
            & (df['load_case'].isin(uniaxial_cases))
            & (df['prompt_type'] == prompt_type_filter)
            & (df['num_views_inference'].between(1, 10))
            & (df['view_type'].isin(['ortho', 'random']))
        ].copy()
    else:
        base_log = '/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026.log'
        indexed_logs = [f'/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_4.05.2026_{i}.log' for i in range(1, 9)]
        rel_log_df = build_rel_ratio_df_from_logs([base_log] + indexed_logs)
        gemini_df = rel_log_df[
            (rel_log_df['mesh_name'].isin(mesh_names))
            & (rel_log_df['llm_model'] == gemini_model)
            & (rel_log_df['load_case'].isin(uniaxial_cases))
            & (rel_log_df['prompt_type'] == prompt_type_filter)
            & (rel_log_df['num_views_inference'].between(1, 10))
            & (rel_log_df['view_type'].isin(['ortho', 'random']))
        ].copy()

    def build_paired_metric_df(df_in, metric_col):
        base_cols = ['mesh_name', 'num_views_inference', 'grid_size', 'prompt_type', 'run']
        work = df_in[base_cols + ['view_type', metric_col]].copy()

        # Strict rule requested: if either side has any missing value under a key,
        # drop both ortho and random for that key before averaging.
        grouped = work.groupby(base_cols)
        has_both_views = grouped['view_type'].nunique() == 2
        has_any_nan = grouped[metric_col].apply(lambda s: s.isna().any())
        valid_keys = (has_both_views & (~has_any_nan)).reset_index(name='is_valid')
        valid_keys = valid_keys[valid_keys['is_valid']].drop(columns=['is_valid'])

        if valid_keys.empty:
            return pd.DataFrame(columns=base_cols + ['view_type', metric_col]), 0, len(has_both_views)

        merged = work.merge(valid_keys, on=base_cols, how='inner')
        side_mean = merged.groupby(base_cols + ['view_type'], as_index=False)[metric_col].mean()

        total_keys = len(has_both_views)
        kept_keys = len(valid_keys)
        return side_mean, kept_keys, total_keys

    paired_rel_l2_df, kept_l2, total_l2 = build_paired_metric_df(gemini_df, 'rel_L2')
    paired_rel_e_df, kept_e, total_e = build_paired_metric_df(gemini_df, 'rel_E')

    if not paired_rel_l2_df.empty:
        plot_aggregated_metric_flexible(
            df=paired_rel_l2_df,
            metric_col='rel_L2',
            x_col='num_views_inference',
            hue_col='view_type',
            xlabel='k = No. of Grid inference views',
            ylabel='Rel_L2 (refined/coarse)',
            title='Gemini: Mean Rel_L2 vs Number of Views',
            results_dir=plot_save_dir,
            filename='rel_L2_vs_num_views_gemini_paired.png',
            show_std=False,
        )
    else:
        print('No valid paired data found for rel_L2 (Gemini, ortho+random).')

    if not paired_rel_e_df.empty:
        plot_aggregated_metric_flexible(
            df=paired_rel_e_df,
            metric_col='rel_E',
            x_col='num_views_inference',
            hue_col='view_type',
            xlabel='k = No. of Grid inference views',
            ylabel='Rel_E (refined/coarse)',
            title='Gemini: Mean Rel_E vs Number of Views',
            results_dir=plot_save_dir,
            filename='rel_E_vs_num_views_gemini_paired.png',
            show_std=False,
        )
    else:
        print('No valid paired data found for rel_E (Gemini, ortho+random).')

    print(f"Paired rel_L2 keys kept: {kept_l2}/{total_l2}")
    print(f"Paired rel_E keys kept: {kept_e}/{total_e}")

    # ------------------------------------------------------------
    # Micro-averaged F1 vs Number of Views for all models.
    # We pool raw counts first, then compute precision/recall/F1.
    # ------------------------------------------------------------
    micro_src_df = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['prompt_type'] == 'geo_max')
        & (df['num_views_inference'].between(1, 10))
        & (df['view_type'].isin(['ortho', 'random']))
        & (df['matched_refine'].notna())
        & (df['num_refine'].notna())
        & (df['matched_stress'].notna())
        & (df['num_stress'].notna())
    ].copy()

    if not micro_src_df.empty:
        micro_grouped = micro_src_df.groupby(
            ['num_views_inference', 'llm_model', 'view_type'],
            as_index=False
        )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()

        micro_grouped['micro_precision'] = np.where(
            micro_grouped['num_refine'] > 0,
            micro_grouped['matched_refine'] / micro_grouped['num_refine'],
            np.nan,
        )
        micro_grouped['micro_recall'] = np.where(
            micro_grouped['num_stress'] > 0,
            micro_grouped['matched_stress'] / micro_grouped['num_stress'],
            np.nan,
        )
        micro_grouped['micro_F1'] = np.where(
            (micro_grouped['micro_precision'] + micro_grouped['micro_recall']) > 0,
            2 * (micro_grouped['micro_precision'] * micro_grouped['micro_recall']) / (micro_grouped['micro_precision'] + micro_grouped['micro_recall']),
            np.nan,
        )

        # Ortho-only: one micro-F1 curve per model.
        micro_ortho_df = micro_grouped[
            (micro_grouped['view_type'] == 'ortho') & (micro_grouped['micro_F1'].notna())
        ].copy()
        if not micro_ortho_df.empty:
            plot_preaggregated_metric(
                df=micro_ortho_df,
                metric_col='micro_F1',
                x_col='num_views_inference',
                hue_col='llm_model',
                xlabel='k = No. of Grid inference views',
                ylabel='Micro F1@r',
                title='All Models: Micro F1 vs Number of Views (Ortho)',
                results_dir=plot_save_dir,
                filename='micro_F1_vs_num_views_all_models_ortho.pdf',
                annotate_hue='qwen3-vl-235b-a22b-instruct',
                annotate_decimals=6,
            )
        else:
            print('No valid micro-F1 data found for ortho view type.')

        # Random-only: one micro-F1 curve per model.
        micro_random_df = micro_grouped[
            (micro_grouped['view_type'] == 'random') & (micro_grouped['micro_F1'].notna())
        ].copy()
        if not micro_random_df.empty:
            plot_preaggregated_metric(
                df=micro_random_df,
                metric_col='micro_F1',
                x_col='num_views_inference',
                hue_col='llm_model',
                xlabel='k = No. of Grid inference views',
                ylabel='Micro F1@r',
                title='All Models: Micro F1 vs Number of Views (Random)',
                results_dir=plot_save_dir,
                filename='micro_F1_vs_num_views_all_models_random.pdf',
                annotate_hue='qwen3-vl-235b-a22b-instruct',
                annotate_decimals=6,
            )
        else:
            print('No valid micro-F1 data found for random view type.')

        # Combined (model + view type) plot for direct side-by-side comparison.
        micro_plot_df = micro_grouped[micro_grouped['micro_F1'].notna()].copy()
        micro_plot_df['model_view'] = micro_plot_df['llm_model'].astype(str) + ' | ' + micro_plot_df['view_type'].astype(str)
        plot_preaggregated_metric(
            df=micro_plot_df,
            metric_col='micro_F1',
            x_col='num_views_inference',
            hue_col='model_view',
            xlabel='k = No. of Grid inference views',
            ylabel='Micro F1@r',
            title='All Models: Micro F1 vs Number of Views (Ortho + Random)',
            results_dir=plot_save_dir,
            filename='micro_F1_vs_num_views_all_models_ortho_random.pdf',
            annotate_hue='qwen3-vl-235b-a22b-instruct | ortho',
            annotate_decimals=6,
        )

        # Dedicated Qwen-only micro-F1 artifact to avoid confusion in multi-model figures.
        qwen_micro_df = micro_grouped[
            (micro_grouped['llm_model'] == 'qwen3-vl-235b-a22b-instruct')
            & (micro_grouped['micro_F1'].notna())
        ].copy()
        if not qwen_micro_df.empty:
            qwen_micro_df['view_curve'] = 'Qwen | ' + qwen_micro_df['view_type'].astype(str)
            plot_preaggregated_metric(
                df=qwen_micro_df,
                metric_col='micro_F1',
                x_col='num_views_inference',
                hue_col='view_curve',
                xlabel='k = No. of Grid inference views',
                ylabel='Micro F1@r',
                title='Qwen: Micro F1 vs Number of Views (geo_max)',
                results_dir=plot_save_dir,
                filename='micro_F1_vs_num_views_qwen_geo_max.pdf',
                annotate_hue='Qwen | ortho',
                annotate_decimals=6,
            )
            qwen_micro_csv_path = os.path.join(plot_save_dir, 'micro_F1_qwen_geo_max_table.csv')
            qwen_micro_df.to_csv(qwen_micro_csv_path, index=False)
            print(f"Saved: {qwen_micro_csv_path}")

        # One curve per model: pool ortho + random raw counts together.
        micro_per_model = micro_src_df.groupby(
            ['num_views_inference', 'llm_model'], as_index=False
        )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()

        micro_per_model['micro_precision'] = np.where(
            micro_per_model['num_refine'] > 0,
            micro_per_model['matched_refine'] / micro_per_model['num_refine'],
            np.nan,
        )
        micro_per_model['micro_recall'] = np.where(
            micro_per_model['num_stress'] > 0,
            micro_per_model['matched_stress'] / micro_per_model['num_stress'],
            np.nan,
        )
        micro_per_model['micro_F1'] = np.where(
            (micro_per_model['micro_precision'] + micro_per_model['micro_recall']) > 0,
            2 * (micro_per_model['micro_precision'] * micro_per_model['micro_recall'])
            / (micro_per_model['micro_precision'] + micro_per_model['micro_recall']),
            np.nan,
        )

        micro_per_model_plot = micro_per_model[micro_per_model['micro_F1'].notna()].copy()
        if not micro_per_model_plot.empty:
            plot_preaggregated_metric(
                df=micro_per_model_plot,
                metric_col='micro_F1',
                x_col='num_views_inference',
                hue_col='llm_model',
                xlabel='k = No. of Grid inference views',
                ylabel='Micro F1@r',
                title='All Models: Micro F1 vs Number of Views (geo_max, ortho+random pooled)',
                results_dir=plot_save_dir,
                filename='micro_F1_vs_num_views_all_models_geo_max.pdf',
                ylim=None,
            )
            micro_per_model_csv = os.path.join(plot_save_dir, 'micro_F1_all_models_geo_max_table.csv')
            micro_per_model_plot.to_csv(micro_per_model_csv, index=False)
            print(f"Saved: {micro_per_model_csv}")
        else:
            print('No valid micro-F1 data found for per-model (pooled) plot.')

        print(f"Micro-F1 source rows used (all view types): {len(micro_src_df)}")
        print(f"Micro-F1 grouped rows (all view types): {len(micro_grouped)}")
        print(f"Micro-F1 grouped rows (ortho): {len(micro_ortho_df)}")
        print(f"Micro-F1 grouped rows (random): {len(micro_random_df)}")
        print(f"Micro-F1 per-model pooled rows: {len(micro_per_model_plot)}")
    else:
        print('No valid source rows available for micro-F1 model-vs-views plotting.')

    # ------------------------------------------------------------
    # Micro-averaged F1 vs Number of Views for all models, geo_mid, ortho only.
    # All other parameters (mesh, load_case, grid_size, run) are averaged via
    # raw-count pooling. F1 = 2PR/(P+R) on the pooled counts.
    # ------------------------------------------------------------
    micro_src_geo_mid_df = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['prompt_type'] == 'geo_mid')
        & (df['num_views_inference'].between(1, 10))
        & (df['view_type'] == 'ortho')
        & (df['matched_refine'].notna())
        & (df['num_refine'].notna())
        & (df['matched_stress'].notna())
        & (df['num_stress'].notna())
    ].copy()

    if not micro_src_geo_mid_df.empty:
        micro_per_model_geo_mid = micro_src_geo_mid_df.groupby(
            ['num_views_inference', 'llm_model'], as_index=False
        )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()

        micro_per_model_geo_mid['micro_precision'] = np.where(
            micro_per_model_geo_mid['num_refine'] > 0,
            micro_per_model_geo_mid['matched_refine'] / micro_per_model_geo_mid['num_refine'],
            np.nan,
        )
        micro_per_model_geo_mid['micro_recall'] = np.where(
            micro_per_model_geo_mid['num_stress'] > 0,
            micro_per_model_geo_mid['matched_stress'] / micro_per_model_geo_mid['num_stress'],
            np.nan,
        )
        micro_per_model_geo_mid['micro_F1'] = np.where(
            (micro_per_model_geo_mid['micro_precision'] + micro_per_model_geo_mid['micro_recall']) > 0,
            2 * (micro_per_model_geo_mid['micro_precision'] * micro_per_model_geo_mid['micro_recall'])
            / (micro_per_model_geo_mid['micro_precision'] + micro_per_model_geo_mid['micro_recall']),
            np.nan,
        )

        micro_per_model_geo_mid_plot = micro_per_model_geo_mid[micro_per_model_geo_mid['micro_F1'].notna()].copy()
        if not micro_per_model_geo_mid_plot.empty:
            plot_preaggregated_metric(
                df=micro_per_model_geo_mid_plot,
                metric_col='micro_F1',
                x_col='num_views_inference',
                hue_col='llm_model',
                xlabel='k = No. of Grid inference views',
                ylabel='Micro F1@r',
                title='All Models: Micro F1 vs Number of Views (geo_mid, ortho)',
                results_dir=plot_save_dir,
                filename='micro_F1_vs_num_views_all_models_geo_mid.pdf',
                ylim=None,
            )
            geo_mid_csv = os.path.join(plot_save_dir, 'micro_F1_all_models_geo_mid_table.csv')
            micro_per_model_geo_mid_plot.to_csv(geo_mid_csv, index=False)
            print(f"Saved: {geo_mid_csv}")

            # Sanity print: Gemini @ k=5
            row = micro_per_model_geo_mid_plot[
                (micro_per_model_geo_mid_plot['llm_model'] == 'gemini-3-flash-preview')
                & (micro_per_model_geo_mid_plot['num_views_inference'] == 5)
            ]
            if not row.empty:
                r0 = row.iloc[0]
                print(
                    f"[geo_mid/ortho] Gemini @ k=5: P={r0['micro_precision']:.6f} "
                    f"R={r0['micro_recall']:.6f} F1={r0['micro_F1']:.6f}"
                )
        else:
            print('No valid micro-F1 data found for per-model geo_mid plot.')
    else:
        print('No valid source rows available for geo_mid micro-F1 plotting.')

    # ------------------------------------------------------------
    # Combined geo_max + geo_mid, ortho only, all models on one plot.
    # ------------------------------------------------------------
    micro_src_combined_df = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['prompt_type'].isin(['geo_max', 'geo_mid']))
        & (df['num_views_inference'].between(1, 10))
        & (df['view_type'] == 'ortho')
        & (df['matched_refine'].notna())
        & (df['num_refine'].notna())
        & (df['matched_stress'].notna())
        & (df['num_stress'].notna())
    ].copy()

    if not micro_src_combined_df.empty:
        micro_combined = micro_src_combined_df.groupby(
            ['num_views_inference', 'llm_model', 'prompt_type'], as_index=False
        )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()
        micro_combined['micro_precision'] = np.where(
            micro_combined['num_refine'] > 0,
            micro_combined['matched_refine'] / micro_combined['num_refine'], np.nan,
        )
        micro_combined['micro_recall'] = np.where(
            micro_combined['num_stress'] > 0,
            micro_combined['matched_stress'] / micro_combined['num_stress'], np.nan,
        )
        micro_combined['micro_F1'] = np.where(
            (micro_combined['micro_precision'] + micro_combined['micro_recall']) > 0,
            2 * micro_combined['micro_precision'] * micro_combined['micro_recall']
            / (micro_combined['micro_precision'] + micro_combined['micro_recall']),
            np.nan,
        )
        micro_combined_plot = micro_combined[micro_combined['micro_F1'].notna()].copy()
        prompt_label_map = {'geo_max': 'Explicit+', 'geo_mid': 'Explicit'}
        micro_combined_plot['prompt_label'] = micro_combined_plot['prompt_type'].map(prompt_label_map).fillna(micro_combined_plot['prompt_type'])
        micro_combined_plot['model_prompt'] = (
            micro_combined_plot['llm_model'].astype(str) + ' | ' + micro_combined_plot['prompt_label'].astype(str)
        )

        # Custom plot: no title, ylabel "F1@r", smaller legend, dashed baseline.
        plt.style.use('default')
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plt.rcParams['mathtext.fontset'] = 'stix'

        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        hues = sorted(micro_combined_plot['model_prompt'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(hues)))
        markers = ['o', 's', '^', 'D', 'v', 'p', 'X', '*']
        for idx, hue_val in enumerate(hues):
            sub = micro_combined_plot[micro_combined_plot['model_prompt'] == hue_val].sort_values('num_views_inference')
            ax.plot(
                sub['num_views_inference'].astype(float).values,
                sub['micro_F1'].astype(float).values,
                marker=markers[idx % len(markers)],
                linewidth=1.8,
                markersize=4,
                label=str(hue_val),
                color=colors[idx],
            )

        baseline_val = 0.5362
        ax.axhline(baseline_val, color='black', linestyle='--', linewidth=1.2,
                   label=f'Mesh-baseline heuristic ({baseline_val:.4f})')

        ax.set_xlabel('k = No. of Grid inference views', fontsize=10, fontweight='bold')
        ax.set_ylabel('F1@r', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, framealpha=0.9)
        ax.grid(True, which='major', axis='y', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(prune=None, integer=True))
        plt.tight_layout(pad=0.4)
        out_path = os.path.join(plot_save_dir, 'micro_F1_vs_num_views_all_models_geo_max_geo_mid.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"Saved: {out_path}")

        combined_csv = os.path.join(plot_save_dir, 'micro_F1_all_models_geo_max_geo_mid_table.csv')
        micro_combined_plot.to_csv(combined_csv, index=False)
        print(f"Saved: {combined_csv}")

    # ------------------------------------------------------------
    # Gemini only: ortho vs random for geo_mid across num_views.
    # ------------------------------------------------------------
    gemini_geo_mid_src = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['llm_model'] == 'gemini-3-flash-preview')
        & (df['prompt_type'] == 'geo_mid')
        & (df['num_views_inference'].between(1, 10))
        & (df['view_type'].isin(['ortho', 'random']))
        & (df['matched_refine'].notna())
        & (df['num_refine'].notna())
        & (df['matched_stress'].notna())
        & (df['num_stress'].notna())
    ].copy()

    if not gemini_geo_mid_src.empty:
        gem = gemini_geo_mid_src.groupby(
            ['num_views_inference', 'view_type'], as_index=False
        )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()
        gem['micro_precision'] = np.where(
            gem['num_refine'] > 0, gem['matched_refine'] / gem['num_refine'], np.nan,
        )
        gem['micro_recall'] = np.where(
            gem['num_stress'] > 0, gem['matched_stress'] / gem['num_stress'], np.nan,
        )
        gem['micro_F1'] = np.where(
            (gem['micro_precision'] + gem['micro_recall']) > 0,
            2 * gem['micro_precision'] * gem['micro_recall']
            / (gem['micro_precision'] + gem['micro_recall']),
            np.nan,
        )
        gem_plot = gem[gem['micro_F1'].notna()].copy()
        plot_preaggregated_metric(
            df=gem_plot,
            metric_col='micro_F1',
            x_col='num_views_inference',
            hue_col='view_type',
            xlabel='k = No. of Grid inference views',
            ylabel='F1@r',
            title='',
            results_dir=plot_save_dir,
            filename='micro_F1_vs_num_views_gemini_geo_mid_ortho_vs_random.pdf',
            ylim=None,
        )
        gem_csv = os.path.join(plot_save_dir, 'micro_F1_gemini_geo_mid_ortho_vs_random_table.csv')
        gem_plot.to_csv(gem_csv, index=False)
        print(f"Saved: {gem_csv}")

    # ------------------------------------------------------------
    # Std-dev tables of micro precision / recall / F1.
    # Each row's precision/recall/F1 is already a per-mesh micro metric
    # (computed from raw matched/total counts for that mesh+load_case+run+grid).
    # We provide:
    #   1) Full grouping table (std taken across mesh + run + grid replicates).
    #   2) Marginal tables: std taken across one varied dimension at a time.
    # ------------------------------------------------------------
    metric_cols = ['precision', 'recall', 'F1']
    base = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['view_type'].isin(['ortho', 'random']))
        & (df['num_views_inference'].between(1, 10))
        & (df['precision'].notna())
        & (df['recall'].notna())
        & (df['F1'].notna())
    ].copy()

    # 1) Comprehensive table: std across mesh+run+grid for each fixed combo
    full_keys = ['llm_model', 'load_case', 'prompt_type', 'num_views_inference', 'view_type']
    full_stats = base.groupby(full_keys)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
    full_stats.columns = ['_'.join([c for c in col if c]) for col in full_stats.columns.values]
    full_csv = os.path.join(plot_save_dir, 'metric_std_full_table.csv')
    full_stats.to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")

    # 2) Marginal std tables (vary one dim at a time, hold others fixed)
    all_dims = ['llm_model', 'load_case', 'prompt_type', 'num_views_inference', 'view_type']
    for varied in all_dims:
        held = [d for d in all_dims if d != varied]
        # First aggregate to one value per (held + varied) combo (pool mesh/run/grid via mean)
        per_combo = base.groupby(held + [varied])[metric_cols].mean().reset_index()
        # Then std across the varied dimension while holding held constant
        marg = per_combo.groupby(held)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
        marg.columns = ['_'.join([c for c in col if c]) for col in marg.columns.values]
        marg_csv = os.path.join(plot_save_dir, f'metric_std_marginal_over_{varied}.csv')
        marg.to_csv(marg_csv, index=False)
        print(f"Saved: {marg_csv}")

    # 3) Compressed sensitivity summary: ONE ROW PER DIMENSION.
    # For each dim, std is taken across that dim's values within each fixed
    # combination of all OTHER dims (mesh+run+grid first averaged), then those
    # per-config stds are averaged. This isolates how much each metric varies
    # when the dimension is changed, all else equal.
    sensitivity_rows = []
    for varied in all_dims:
        held = [d for d in all_dims if d != varied]
        per_combo = base.groupby(held + [varied])[metric_cols].mean().reset_index()
        # Need at least 2 values of `varied` per held-combo for std to be defined
        per_combo_std = per_combo.groupby(held)[metric_cols].std()
        sensitivity_rows.append({
            'dimension': varied,
            'n_values': base[varied].nunique(),
            'n_held_configs': len(per_combo_std),
            'precision_std': per_combo_std['precision'].mean(),
            'recall_std': per_combo_std['recall'].mean(),
            'F1_std': per_combo_std['F1'].mean(),
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_csv = os.path.join(plot_save_dir, 'metric_sensitivity_by_dimension.csv')
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    print(f"Saved: {sensitivity_csv}")
    print(sensitivity_df.to_string(index=False))

    # ------------------------------------------------------------
    # Micro-averaged P/R/F1 per (llm_model, load_case).
    # All other dims (mesh, num_views, view_type, prompt_type, grid, run)
    # are pooled by summing raw counts.
    # ------------------------------------------------------------
    micro_lc_src = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['view_type'] == 'ortho')
        & (df['num_views_inference'].between(1, 10))
        & (df['matched_refine'].notna())
        & (df['num_refine'].notna())
        & (df['matched_stress'].notna())
        & (df['num_stress'].notna())
    ].copy()

    micro_by_model_loadcase = micro_lc_src.groupby(
        ['llm_model', 'load_case'], as_index=False
    )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()

    micro_by_model_loadcase['micro_precision'] = np.where(
        micro_by_model_loadcase['num_refine'] > 0,
        micro_by_model_loadcase['matched_refine'] / micro_by_model_loadcase['num_refine'],
        np.nan,
    )
    micro_by_model_loadcase['micro_recall'] = np.where(
        micro_by_model_loadcase['num_stress'] > 0,
        micro_by_model_loadcase['matched_stress'] / micro_by_model_loadcase['num_stress'],
        np.nan,
    )
    micro_by_model_loadcase['micro_F1'] = np.where(
        (micro_by_model_loadcase['micro_precision'] + micro_by_model_loadcase['micro_recall']) > 0,
        2 * micro_by_model_loadcase['micro_precision'] * micro_by_model_loadcase['micro_recall']
        / (micro_by_model_loadcase['micro_precision'] + micro_by_model_loadcase['micro_recall']),
        np.nan,
    )

    micro_lc_csv = os.path.join(plot_save_dir, 'micro_PRF1_by_model_loadcase.csv')
    micro_by_model_loadcase.to_csv(micro_lc_csv, index=False)
    print(f"Saved: {micro_lc_csv}")
    print(micro_by_model_loadcase[
        ['llm_model', 'load_case', 'micro_precision', 'micro_recall', 'micro_F1']
    ].to_string(index=False))

    # ------------------------------------------------------------
    # Summary table per (llm_model, load_case): 20 rows, ortho only.
    # Step 1: micro-aggregate raw counts per (model, load_case, num_views)
    #         pooling prompt_type / mesh / grid / run.
    # Step 2: derive per-k micro P/R/F1, then take mean across k=1..10
    #         and pick the value at k=5 (@top5).
    # Also report avg refinement points per inference (in thousands) and
    # avg cells predicted per view-response.
    # ------------------------------------------------------------
    per_k = micro_lc_src.groupby(
        ['llm_model', 'load_case', 'num_views_inference'], as_index=False
    )[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()
    per_k['micro_precision'] = np.where(per_k['num_refine'] > 0,
                                        per_k['matched_refine'] / per_k['num_refine'], np.nan)
    per_k['micro_recall'] = np.where(per_k['num_stress'] > 0,
                                     per_k['matched_stress'] / per_k['num_stress'], np.nan)
    per_k['micro_F1'] = np.where(
        (per_k['micro_precision'] + per_k['micro_recall']) > 0,
        2 * per_k['micro_precision'] * per_k['micro_recall']
        / (per_k['micro_precision'] + per_k['micro_recall']), np.nan,
    )

    means = per_k.groupby(['llm_model', 'load_case'], as_index=False)[
        ['micro_precision', 'micro_recall', 'micro_F1']
    ].mean().rename(columns={
        'micro_precision': 'Mean Precision',
        'micro_recall': 'Mean Recall',
        'micro_F1': 'Mean F1',
    })

    at5 = per_k[per_k['num_views_inference'] == 5][
        ['llm_model', 'load_case', 'micro_precision', 'micro_recall', 'micro_F1']
    ].rename(columns={
        'micro_precision': 'Precision@top5',
        'micro_recall': 'Recall@top5',
        'micro_F1': 'F1@top5',
    })

    # Avg refinement points per inference (mean per row, in thousands).
    refp = micro_lc_src.groupby(['llm_model', 'load_case'], as_index=False)[
        'num_refinement_points'
    ].mean()
    refp['Avg Refinement points (k)'] = (refp['num_refinement_points'] / 1000.0).round(2)
    refp = refp.drop(columns=['num_refinement_points'])

    # Avg cells predicted per view-response (column already stored as
    # total_cells / num_views_inference at cache build time).
    cells_src = df[
        (df['mesh_name'].isin(mesh_names))
        & (df['view_type'] == 'ortho')
        & (df['num_views_inference'].between(1, 10))
        & (df['total_cells'].notna())
    ]
    cells = cells_src.groupby(['llm_model', 'load_case'], as_index=False)[
        'total_cells'
    ].mean().rename(columns={
        'total_cells': 'Avg. Cells Predicted Per view response',
    })

    summary = means.merge(at5, on=['llm_model', 'load_case']) \
                   .merge(refp, on=['llm_model', 'load_case'], how='left') \
                   .merge(cells, on=['llm_model', 'load_case'], how='left')
    col_order = [
        'llm_model', 'load_case',
        'Mean Precision', 'Precision@top5',
        'Mean Recall', 'Recall@top5',
        'Mean F1', 'F1@top5',
        'Avg Refinement points (k)',
        'Avg. Cells Predicted Per view response',
    ]
    summary = summary[col_order].sort_values(['llm_model', 'load_case']).reset_index(drop=True)
    for c in ['Mean Precision', 'Precision@top5', 'Mean Recall', 'Recall@top5',
              'Mean F1', 'F1@top5', 'Avg. Cells Predicted Per view response']:
        summary[c] = summary[c].round(4)

    summary_csv = os.path.join(plot_save_dir, 'micro_PRF1_summary_model_loadcase.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    print(summary.to_string(index=False))



