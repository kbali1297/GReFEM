"""
generate_mesh_and_simulate_parallel_heuristic.py

Parallel mesh generation + FEM simulation for the GEOMETRIC HEURISTIC
baseline (concave-feature / dihedral-based refinement points).

Per CAD it picks the heuristic refinement points from
    test_meshes_rebuttal_baseline/<cad>/refinement_points_<exp>.npy
where <exp> defaults to "baseline_DENSE_dihedral30.0deg".

Reference (.step), fine mesh (.msh) and surface mesh (.obj) are read
from the original test_meshes/<cad>/ tree; refined / coarse meshes are
written into test_meshes_rebuttal_baseline/<cad>/<exp>_<loading_case>/.

Mirrors the structure / parallelism of generate_mesh_and_simulate_parallel.py.
"""

import os
import sys

# 1. FIX OPENMPI PATH LOSS (must happen BEFORE importing fem_fenics / dolfinx)
os.environ["OPAL_PREFIX"] = "/data/1bali/miniforge3/envs/multi_view_3DQA"
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

import numpy as np
import trimesh
import gmsh
import multiprocessing
from functools import partial
from tqdm import tqdm

from cad_element_sizes import CAD_ELEMENT_SIZES
from utils import *
from fem_fenics import *
from generate_renders import *


# --------------------------- CONFIG ---------------------------
ORIG_DIR = "/data/1bali/GReFEM/test_meshes"
BASELINE_DIR = "/data/1bali/GReFEM/test_meshes_rebuttal_baseline"
EXP_NAME = "baseline_DENSE_dihedral30.0deg"
LOADING_CASES = ["compression", "torsion", "bending",
                 "torsion_compression", "bending_compression"]


def run_experiment(cad_name, loading_case):
    """Run mesh refine + FEM for one (CAD, loading_case) under heuristic baseline."""
    print(f"--- Starting: {cad_name} | {loading_case} ---")

    orig_cad_dir = os.path.join(ORIG_DIR, cad_name)
    base_cad_dir = os.path.join(BASELINE_DIR, cad_name)

    refinement_points_path = os.path.join(
        base_cad_dir, f"refinement_points_{EXP_NAME}.npy"
    )
    if not os.path.exists(refinement_points_path):
        raise FileNotFoundError(refinement_points_path)

    mesh_path = os.path.join(orig_cad_dir, "renders_pyvista", f"{cad_name}.obj")
    fine_msh_path = os.path.join(orig_cad_dir, f"fine_mesh_{loading_case}.msh")
    if not os.path.exists(fine_msh_path):
        raise FileNotFoundError(fine_msh_path)

    # locate STEP file
    step_file_path = None
    for f in os.listdir(orig_cad_dir):
        if f.endswith(".step"):
            step_file_path = os.path.join(orig_cad_dir, f)
            break
    if step_file_path is None:
        raise FileNotFoundError(f"No .step file in {orig_cad_dir}")

    surf_mesh = trimesh.load(mesh_path)
    _ = np.mean(surf_mesh.vertices, axis=0)

    # Volume probe (mirrors original script behaviour)
    gmsh.initialize()
    gmsh.model.add("vol")
    gmsh.model.occ.importShapes(step_file_path)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    if len(volumes) == 0:
        gmsh.finalize()
        raise RuntimeError(f"No volumes in STEP file: {step_file_path}")
    first_vol_dim, first_vol_tag = volumes[0]
    total_volume = gmsh.model.occ.getMass(first_vol_dim, first_vol_tag)
    if len(volumes) > 1:
        print(f"Notice: {cad_name} assembly with {len(volumes)} solids."
              f" Using first solid volume = {total_volume:.2f}")
    gmsh.finalize()

    # Output dir for this experiment
    exp_folder = os.path.join(base_cad_dir, f"{EXP_NAME}_{loading_case}")
    os.makedirs(exp_folder, exist_ok=True)

    refinement_points = np.load(refinement_points_path)

    h_min = CAD_ELEMENT_SIZES[cad_name]["h_min"]
    h_max = CAD_ELEMENT_SIZES[cad_name]["h_max"]

    coarse_msh_path = os.path.join(exp_folder, f"coarse_mesh_{loading_case}.msh")
    refined_msh_path = os.path.join(exp_folder, "refined_mesh_prefilt.msh")

    out_msh_coarse = generate_or_refine_mesh(
        step_or_mesh_path=step_file_path,
        h_min=h_max, h_max=h_max, suffix=None,
        verbose=False, out_msh=coarse_msh_path,
    )
    output_msh_refined = generate_or_refine_mesh(
        step_or_mesh_path=step_file_path,
        h_min=h_min, h_max=h_max, suffix=None,
        verbose=False, out_msh=refined_msh_path,
        points_of_interest=refinement_points,
    )

    render_tet_mesh_views(
        output_msh_refined,
        output_dir=os.path.join(exp_folder, "renders_tet3D_pyvista_refined"),
        surf_mesh_path=mesh_path,
        n_azimuth=[60, 300], n_elevation=[-36, 36], add_axes=False,
    )

    tet3D_mesh_paths = [out_msh_coarse, output_msh_refined]
    experiment_name_tag = f"{loading_case}_{EXP_NAME}"
    solved_uh, res_cand = compute_disp_error(
        candidate_mshs=tet3D_mesh_paths,
        ref_msh=fine_msh_path,
        outfile=None,
        solve_reference=False,
        experiment_name=experiment_name_tag,
        load_case=loading_case,
    )
    return res_cand


def worker_wrapper(args):
    cad_name, loading_case = args
    try:
        res = run_experiment(cad_name, loading_case)
        return cad_name, loading_case, res, None
    except Exception as e:
        return cad_name, loading_case, None, str(e)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # 2. DYNAMIC TASK DISCOVERY
    task_pool = []
    for cad_name in sorted(os.listdir(BASELINE_DIR)):
        cad_dir = os.path.join(BASELINE_DIR, cad_name)
        if not os.path.isdir(cad_dir):
            continue
        npy_path = os.path.join(cad_dir, f"refinement_points_{EXP_NAME}.npy")
        if not os.path.exists(npy_path):
            continue
        if cad_name not in CAD_ELEMENT_SIZES:
            print(f"Skip {cad_name}: no entry in CAD_ELEMENT_SIZES")
            continue
        for lc in LOADING_CASES:
            fine_msh = os.path.join(ORIG_DIR, cad_name, f"fine_mesh_{lc}.msh")
            if not os.path.exists(fine_msh):
                continue
            task_pool.append((cad_name, lc))

    # 3. RESUME: skip tasks already SUCCESS-logged
    log_files = [
        f"/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_heuristic.log",
    ]
    log_files += [
        f"/data/1bali/GReFEM/generate_mesh_and_simulate_parallel_heuristic_{i}.log"
        for i in range(1, 9)
    ]
    successful = set()
    for lp in log_files:
        if not os.path.exists(lp):
            continue
        with open(lp, "r") as fh:
            for line in fh:
                if "[SUCCESS]" not in line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    cad = parts[0].split("]")[-1].strip()
                    lc = parts[1].strip()
                    successful.add((cad, lc))
    task_pool = [t for t in task_pool if t not in successful]

    print(f"Total combinations to process after filtering: {len(task_pool)}")
    if not task_pool:
        print("Nothing to do.")
        sys.exit(0)

    num_processes = min(100, len(task_pool))
    print(f"Starting pool with {num_processes} processes...\n")

    res_cands = {}
    with multiprocessing.Pool(processes=num_processes) as pool:
        iterator = pool.imap_unordered(worker_wrapper, task_pool)
        with tqdm(total=len(task_pool), desc="Heuristic sims", unit="task") as pbar:
            for cad_name, loading_case, res, error in iterator:
                if error:
                    tqdm.write(f"[ERROR] {cad_name} | {loading_case} | Failed with: {error}")
                else:
                    tqdm.write(f"[SUCCESS] {cad_name} | {loading_case} | Result: {res}")
                    res_cands[f"{cad_name}_{loading_case}"] = res
                pbar.set_postfix(last_cad=cad_name, last_lc=loading_case)
                pbar.update(1)

    print("\nAll processing complete! Final Results Summary:")
    for k, v in res_cands.items():
        print(f"{k}: {v}")
