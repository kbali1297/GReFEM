#!/usr/bin/env python3
"""
compute_local_error_tables.py

Standalone driver for the rebuttal tables: loops over existing experiment
outputs and computes interpolation-free scalar QoI errors (global + local
Omega_crit) for every candidate mesh against the already-solved fine
reference solution. Each QoI (strain energy, von Mises percentiles in
Omega_crit, max displacement) is evaluated natively on its own mesh; only
scalars are compared, so no cross-mesh field transfer is involved.

- Reuses existing solutions: if <candidate>_sol.xdmf exists it is NOT re-solved.
  The reference (fine_mesh_{load}_sol.xdmf) is never re-solved.
- Refined meshes are regenerated on the fly from the stored
  refinement_points_prefilt.npy in each experiment dir (cheap gmsh call,
  same h_max/h_min sizing formula as generate_mesh_and_simulate.py) and
  cached under <object>/refined_mesh/.
- Omega_crit is defined by the top-percentile ZZ points from
  fine_mesh_{load}_zz.pos with radius 4*h_min (r_thresh convention from
  results_analysis.py).
- Appends one row per (object, load_case, candidate) to the output CSV
  incrementally, skipping rows already present (safe to re-run / resume).

Usage:
    python compute_local_error_tables.py \
        --test_dir /data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes \
        --load_cases bending compression torsion \
        --experiment_filter '*_1run' \
        --out_csv local_error_table.csv
"""
import os
import csv
import glob
import time
import fnmatch
import argparse
import functools
import traceback

import numpy as np
import meshio

from fem_fenics import (
    compute_qoi_scalars, read_pos_points_top, _SOLVERS,
)
from utils import generate_or_refine_mesh

print = functools.partial(print, flush=True)  # unbuffered logging under nohup


def _fmt_secs(s):
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


import contextlib
import signal


@contextlib.contextmanager
def _solve_time_limit():
    """Abort a single candidate solve+QoI after LOCAL_SOLVE_TIMEOUT_S seconds
    (env; 0/unset disables). Uses SIGALRM, so it fires between Python ops."""
    budget = int(os.environ.get("LOCAL_SOLVE_TIMEOUT_S", "0"))
    if budget <= 0:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError(f"solve timed out after {budget}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(budget)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

LOAD_CASES = {"bending", "compression", "torsion", "shear"}

CSV_FIELDS = [
    "object", "load_case", "candidate", "candidate_msh",
    "n_dofs", "n_cells",
    "energy_total", "energy_crit", "vm_p99_crit", "vm_max_crit", "max_disp",
    "rel_energy_err", "rel_energy_crit_err",
    "rel_vm_p99_err", "rel_vm_max_err", "rel_max_disp_err",
    "n_crit_cells", "crit_radius", "n_crit_points",
    "t_candidate_s",
    "status",
]

QOI_PAIRS = [  # (qoi key, relative-error column)
    ("energy_total", "rel_energy_err"),
    ("energy_crit", "rel_energy_crit_err"),
    ("vm_p99_crit", "rel_vm_p99_err"),
    ("vm_max_crit", "rel_vm_max_err"),
    ("max_disp", "rel_max_disp_err"),
]


def ensure_solved(msh_path, load_case, E, nu):
    """Return path to <msh>_sol.xdmf, solving only if it does not exist."""
    sol_xdmf = msh_path.replace(".msh", "_sol.xdmf")
    if os.path.exists(sol_xdmf) and os.path.exists(sol_xdmf.replace(".xdmf", ".h5")):
        return sol_xdmf
    solver_fn = _SOLVERS[load_case]
    u_xdmf, *_ = solver_fn(msh_path, E=E, nu=nu)
    return u_xdmf


def compute_h_sizes(obj_dir):
    """h_max/h_min from STEP volume, same formula as generate_mesh_and_simulate."""
    import gmsh
    step_files = glob.glob(os.path.join(obj_dir, "*.step"))
    if not step_files:
        raise RuntimeError(f"No .step file in {obj_dir}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("vol")
    gmsh.model.occ.importShapes(step_files[0])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
        gmsh.finalize()
        raise RuntimeError(f"No volumes in {step_files[0]}")
    total_volume = gmsh.model.occ.getMass(*volumes[0])
    gmsh.finalize()

    obj = os.path.basename(obj_dir.rstrip("/"))
    if obj == "Generic_objects_Scale_Models_Cement_mixer_truck_cabin_back":
        h_max = 1.0 * np.power(total_volume / 268, 0.25)
    elif obj == "Electrical_Parts_Servos_SG-90_SG90-1-arm-horn":
        h_max = 0.6 * np.power(total_volume / 268, 0.25)
    elif obj in ("00200012", "00200016"):
        h_max = 2.0 * np.power(total_volume / 268, 0.5)
    elif obj == "00210005":
        h_max = 1.0 * np.power(total_volume / 268, 0.5)
    else:
        h_max = 1.0 * np.power(total_volume / 268, 0.3)
    return h_max, h_max / 5, step_files[0]  # h_max, h_min, step path


def _label_matches_load(label, load_case):
    """True if label is a single-load experiment for load_case.
    Excludes combined-load dirs like 'bending_compression_...'."""
    if not label.startswith(load_case + "_"):
        return False
    rest = label[len(load_case) + 1:]
    return rest.split("_")[0] not in LOAD_CASES


def count_tets(msh_path):
    """Tet count of a mesh file; 0 if unreadable/truncated (failed gmsh run)."""
    try:
        m = meshio.read(msh_path)
    except (Exception, SystemExit):
        # old meshio sys.exit()s on unreadable/truncated files
        return 0
    return sum(len(b.data) for b in m.cells if b.type == "tetra")


def generate_budget_matched_mesh(step_path, refine_pts, h_min, h_max, out_msh,
                                 target_cells=None, tol=0.10, max_iter=8,
                                 h_floor=None):
    """Generate a refined mesh, calibrating h_min so the tet count matches
    target_cells within tol (n_cells ~ h^-3 in refined regions). h never
    drops below h_floor (e.g. the fine-reference element size): refining
    below the reference resolution makes error-vs-reference meaningless
    near stress singularities (vm overshoots the reference peak)."""
    def clamp(h):
        h = min(h, h_max)
        if h_floor is not None:
            h = max(h, h_floor)
        return h

    h = clamp(h_min)
    n = None
    best = (None, float("inf"), h)  # (n, rel gap, h)
    for it in range(max_iter):
        generate_or_refine_mesh(step_or_mesh_path=step_path,
                                points_of_interest=refine_pts,
                                h_min=h, h_max=h_max, suffix=None,
                                verbose=False, out_msh=out_msh)
        n = count_tets(out_msh)
        if target_cells is None:
            return out_msh, n, h
        gap = abs(n - target_cells) / target_cells
        if gap < best[1]:
            best = (n, gap, h)
        if gap <= tol:
            return out_msh, n, h
        h_new = clamp(h * (n / target_cells) ** (1.0 / 2.0))
        if abs(h_new - h) / h < 1e-3:
            break  # clamped at floor/ceiling; cannot improve further
        h = h_new
    # regenerate at the best h seen if the last attempt was not the best
    if best[2] != h and best[0] is not None:
        generate_or_refine_mesh(step_or_mesh_path=step_path,
                                points_of_interest=refine_pts,
                                h_min=best[2], h_max=h_max, suffix=None,
                                verbose=False, out_msh=out_msh)
        n, h = best[0], best[2]
    return out_msh, n, h


def discover_candidates(obj_dir, load_case, experiment_filter,
                        prefer_refine_pts=False):
    """Return {label: spec} of candidate meshes for one object/load.

    spec is either {"msh": path} for an existing mesh, or
    {"refine_pts": npy_path, "msh": target_path} for a refined mesh that
    must be (re)generated from stored refinement points. With
    prefer_refine_pts=True, persisted default-sizing meshes are ignored so
    every refined candidate carries its anchor points (needed for strict
    budget matching).
    """
    cands = {}
    coarse = os.path.join(obj_dir, f"coarse_mesh_{load_case}.msh")
    if os.path.exists(coarse):
        cands["coarse"] = {"msh": coarse}
    # already-persisted refined meshes
    if not prefer_refine_pts:
        for p in sorted(glob.glob(os.path.join(obj_dir, "refined_mesh", "*_refined.msh"))):
            label = os.path.basename(p).replace("_refined.msh", "")
            if _label_matches_load(label, load_case) and fnmatch.fnmatch(label, experiment_filter):
                cands[label] = {"msh": p}
    # experiment dirs with stored refinement points -> regenerate mesh
    for d in sorted(glob.glob(os.path.join(obj_dir, f"{load_case}_*"))):
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d)
        if (label in cands or not _label_matches_load(label, load_case)
                or not fnmatch.fnmatch(label, experiment_filter)):
            continue
        npy = os.path.join(d, "refinement_points_prefilt.npy")
        if os.path.exists(npy):
            cands[label] = {
                "refine_pts": npy,
                "msh": os.path.join(obj_dir, "refined_mesh", f"{label}_refined.msh"),
            }
    return cands


def load_done_keys(out_csv, done_glob=None):
    done = set()
    paths = [out_csv]
    if done_glob:
        for g in done_glob.split():
            paths += glob.glob(g)
    for p in set(paths):
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((row["object"], row["load_case"], row["candidate"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--load_cases", nargs="+",
                    default=["bending", "compression", "torsion"])
    ap.add_argument("--objects", nargs="*", default=None,
                    help="Optional subset of object dir names.")
    ap.add_argument("--experiment_filter", default="*_1run",
                    help="fnmatch pattern applied to experiment dir names "
                         "(default '*_1run'; use '*' for everything).")
    ap.add_argument("--out_csv", default="local_error_table.csv")
    ap.add_argument("--done_glob", default=None,
                    help="Optional glob of additional CSVs whose ok rows "
                         "count as already done (global resume across "
                         "re-sharded runs).")
    ap.add_argument("--baseline_dir", default=None,
                    help="Dir with per-object refinement_points_{baseline_name}.npy "
                         "for the geometric heuristic baseline "
                         "(e.g. test_meshes_rebuttal_baseline).")
    ap.add_argument("--baseline_name", nargs="+",
                    default=["baseline_DENSE_dihedral30.0deg"],
                    help="One or more baseline point-set names. A '{lc}' "
                         "placeholder is replaced by the load case, e.g. "
                         "mech_dihedral30.0deg_{lc} or mlgnn_{lc}.")
    ap.add_argument("--budget_tol", type=float, default=0.10,
                    help="Relative tolerance for matching baseline tet count "
                         "to the mean GReFEM refined tet count.")
    ap.add_argument("--h_scale", type=float, default=1.0,
                    help="Scale factor applied to h_min/h_max of ALL "
                         "generated candidate meshes (>1 = coarser bases). "
                         "Widens the gap to the fine reference so budget "
                         "calibration no longer saturates at h_floor. "
                         "crit_radius stays at 4*unscaled h_min; scaled "
                         "meshes are written to *_hs{scale}_refined.msh.")
    ap.add_argument("--ref_prefix", default="fine_mesh",
                    help="Basename prefix of the fine reference files "
                         "(e.g. fine2_mesh for the extra-fine references "
                         "from solve_fine_reference.py). Budget-matched "
                         "candidates are tagged with the prefix when it "
                         "differs from the default.")
    ap.add_argument("--E", type=float, default=210e9)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--zz_top_percentile", type=float, default=99.9)
    ap.add_argument("--with_oracle", action="store_true",
                    help="Add a ZZ-oracle candidate: refinement anchors = "
                         "top-percentile ZZ points of the fine reference "
                         "(solver-informed upper bound), budget-matched.")
    ap.add_argument("--oracle_top_n", type=int, default=0,
                    help="If >0, oracle uses the N highest-valued ZZ points "
                         "as anchors. If 0, uses the median GReFEM anchor "
                         "count for a matched-anchor comparison.")
    ap.add_argument("--oracle_percentile", type=float, default=0.0,
                    help="If >0, oracle uses ALL (trimmed) ZZ points above "
                         "this percentile as anchors (dense oracle, label "
                         "zz_oracle_dense) instead of top-N. Spread anchors "
                         "let the budget calibration actually reach the "
                         "target instead of saturating at h_floor.")
    ap.add_argument("--oracle_min_spacing", type=float, default=0.0,
                    help="If >0, oracle anchors are picked greedily in "
                         "descending ZZ value with a minimum spacing of "
                         "this factor * h_min: any point within that "
                         "radius of an already-selected anchor is treated "
                         "as part of the same cluster and skipped. "
                         "Selection walks down the ranking until the "
                         "GReFEM anchor count is reached (label "
                         "zz_oracle_spread).")
    ap.add_argument("--match_budget", action="store_true",
                    help="Match the heuristic baseline's tet count to the "
                         "budget-reference GReFEM experiment (see "
                         "--budget_ref_pattern). GReFEM meshes keep default "
                         "sizing; the oracle is unconstrained.")
    ap.add_argument("--budget_ref_pattern", default="*_geo_maxprompt_ortho_5views_*",
                    help="fnmatch pattern selecting the GReFEM experiment "
                         "whose tet count defines the heuristic's budget "
                         "target (default: 5-views ortho maxprompt).")
    ap.add_argument("--with_uniform", action="store_true",
                    help="Add a 'uniform' candidate: budget-matched UNIFORM "
                         "mesh (no adaptive refinement), built beforehand by "
                         "make_uniform_meshes.py. Isolates the benefit of "
                         "adaptive placement from simply using more elements.")
    ap.add_argument("--with_coarse_zz", action="store_true",
                    help="Add a zz_coarse candidate: anchors = top ZZ points "
                         "of the COARSE solve (practical solve->estimate->"
                         "refine baseline). Requires coarse_mesh_{lc}_zz.pos.")
    ap.add_argument("--only_subsample", action="store_true",
                    help="With --subsample_baseline: evaluate ONLY the "
                         "anchor-count-matched _sub variant of each baseline, "
                         "skipping the full dense-anchor variant.")
    ap.add_argument("--subsample_baseline", action="store_true",
                    help="Add an extra heuristic candidate whose anchors are "
                         "uniformly subsampled to the median GReFEM anchor "
                         "count (label suffix _sub), so the budget matcher "
                         "does not have to dilute h_min over the full dense "
                         "anchor set.")
    args = ap.parse_args()

    objects = args.objects or sorted(
        d for d in os.listdir(args.test_dir)
        if os.path.isdir(os.path.join(args.test_dir, d)))

    done = load_done_keys(args.out_csv, args.done_glob)
    write_header = not os.path.exists(args.out_csv)
    fout = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    t_start = time.time()
    obj_times = []
    n_total = len(objects)

    for obj_i, obj in enumerate(objects):
        t_obj = time.time()
        print(f"\n{'='*70}\n[{obj_i+1}/{n_total}] OBJECT {obj}  "
              f"(elapsed {_fmt_secs(time.time()-t_start)})\n{'='*70}")
        obj_dir = os.path.join(args.test_dir, obj)
        for load_case in args.load_cases:
            ref_xdmf = os.path.join(obj_dir, f"{args.ref_prefix}_{load_case}_sol.xdmf")
            zz_pos = os.path.join(obj_dir, f"{args.ref_prefix}_{load_case}_zz.pos")
            if not (os.path.exists(ref_xdmf) and os.path.exists(zz_pos)):
                print(f"[skip] {obj}/{load_case}: missing reference solution or zz.pos")
                continue

            try:
                h_max, h_min, step_path = compute_h_sizes(obj_dir)
            except Exception as e:
                print(f"[skip] {obj}/{load_case}: h size computation failed: {e}")
                continue
            crit_radius = 4 * h_min  # r_thresh convention from results_analysis
            if args.h_scale != 1.0:
                # coarser candidate bases; QoI region + h_floor unchanged
                h_min *= args.h_scale
                h_max *= args.h_scale
            hs_tag = (f"_hs{args.h_scale:g}" if args.h_scale != 1.0 else "")
            # budget-matched meshes (heuristic/oracle) depend on h_floor of
            # the reference; GReFEM default-sizing meshes do not.
            bm_tag = hs_tag + (f"_{args.ref_prefix}"
                               if args.ref_prefix != "fine_mesh" else "")
            crit_points, _ = read_pos_points_top(zz_pos, args.zz_top_percentile,
                                                 load_case=load_case)
            if len(crit_points) == 0:
                print(f"[skip] {obj}/{load_case}: no zz points above percentile")
                continue

            candidates = discover_candidates(obj_dir, load_case,
                                             args.experiment_filter,
                                             prefer_refine_pts=args.match_budget)
            if hs_tag:
                for s in candidates.values():
                    if "refine_pts" in s:
                        s["msh"] = s["msh"].replace(
                            "_refined.msh", f"{hs_tag}_refined.msh")

            # --- reference QoIs (computed early: also yields h_floor, the
            #     fine-reference mean element size; candidates must never
            #     refine below it or vm errors overshoot near singularities) ---
            try:
                ref_qoi = compute_qoi_scalars(
                    ref_xdmf, E=args.E, nu=args.nu,
                    crit_points=crit_points, crit_radius=crit_radius)
            except Exception as e:
                print(f"[skip] {obj}/{load_case}: reference QoI failed: {e}")
                traceback.print_exc()
                continue
            # regular-tet edge from mean cell volume: V = a^3/(6*sqrt(2))
            h_floor = (8.485 * ref_qoi["volume_total"]
                       / ref_qoi["n_cells"]) ** (1.0 / 3.0)
            print(f"[sizes] {obj}/{load_case}: h_min {h_min:.3f} "
                  f"h_max {h_max:.3f} h_floor(ref) {h_floor:.3f}")

            # --- generate all GReFEM refined meshes first (needed for budget) ---
            for label, spec in candidates.items():
                if "refine_pts" not in spec:
                    continue
                # truncated/failed meshes from earlier crashes: regenerate
                if os.path.exists(spec["msh"]) and count_tets(spec["msh"]) == 0:
                    print(f"[warn] {obj}/{load_case}/{label}: existing mesh "
                          f"invalid, regenerating")
                    os.remove(spec["msh"])
                if not os.path.exists(spec["msh"]):
                    try:
                        os.makedirs(os.path.dirname(spec["msh"]), exist_ok=True)
                        generate_or_refine_mesh(
                            step_or_mesh_path=step_path,
                            points_of_interest=np.load(spec["refine_pts"]),
                            h_min=h_min, h_max=h_max, suffix=None,
                            verbose=False, out_msh=spec["msh"])
                    except Exception as e:
                        print(f"[fail] mesh gen {obj}/{load_case}/{label}: {e}")
                    if (os.path.exists(spec["msh"])
                            and count_tets(spec["msh"]) == 0):
                        print(f"[fail] {obj}/{load_case}/{label}: mesh gen "
                              f"produced no tets, removing")
                        os.remove(spec["msh"])

            # --- heuristic budget target: tet count of the budget-reference
            #     GReFEM experiment (default: 5-views ortho maxprompt).
            #     GReFEM meshes themselves keep their default sizing. ---
            budget_target = None
            if args.match_budget:
                ref_labels = [l for l, s in candidates.items()
                              if l != "coarse" and os.path.exists(s["msh"])
                              and fnmatch.fnmatch(l, args.budget_ref_pattern)]
                if ref_labels:
                    ref_cells = [count_tets(candidates[l]["msh"])
                                 for l in ref_labels]
                    ref_cells = [c for c in ref_cells if c > 0]
                    budget_target = (float(np.mean(ref_cells))
                                     if ref_cells else None)
                    print(f"[budget] {obj}/{load_case}: heuristic target "
                          f"{budget_target and int(budget_target)} tets "
                          f"(from {ref_labels})")
                else:
                    # fallback: mean over all GReFEM refined meshes
                    cells = [count_tets(s["msh"]) for l, s in candidates.items()
                             if l != "coarse" and os.path.exists(s["msh"])]
                    cells = [c for c in cells if c > 0]
                    budget_target = float(np.mean(cells)) if cells else None
                    print(f"[budget] {obj}/{load_case}: budget-ref pattern "
                          f"matched nothing; fallback target "
                          f"{budget_target and int(budget_target)} tets")

            # --- baseline candidates (blind heuristic / mechanics-aware
            #     heuristic / ML surrogate) with matched meshing budget ---
            if args.baseline_dir:
                for bl_name in args.baseline_name:
                    bl_name_lc = bl_name.replace("{lc}", load_case)
                    bl_npy = os.path.join(
                        args.baseline_dir, obj,
                        f"refinement_points_{bl_name_lc}.npy")
                    if not os.path.exists(bl_npy):
                        print(f"[skip] {obj}: no baseline npy {bl_npy}")
                        continue
                    if budget_target is not None:
                        target = budget_target
                    else:
                        grefem_cells = [count_tets(s["msh"]) for l, s in candidates.items()
                                        if l != "coarse" and os.path.exists(s["msh"])]
                        target = float(np.mean(grefem_cells)) if grefem_cells else None
                    # ML baselines keep their own label; heuristics keep the
                    # legacy 'heuristic_' prefix (aggregation relies on it)
                    if bl_name_lc.startswith("ml"):
                        bl_label = bl_name_lc
                    else:
                        bl_label = f"heuristic_{bl_name_lc}"
                    # per-load-case names already carry the load case; strip
                    # the duplicate from the mesh filename is not needed --
                    # keep the generic pattern for resumability
                    bl_msh = os.path.join(obj_dir, "refined_mesh",
                                          f"{load_case}_{bl_label}{bm_tag}_refined.msh")
                    bl_pts = np.load(bl_npy)
                    if len(bl_pts) == 0:
                        print(f"[skip] {obj}/{load_case}/{bl_label}: "
                              f"empty anchor set")
                        continue
                    bl_variants = ([] if args.only_subsample
                                   else [(bl_label, bl_msh, bl_pts)])
                    if args.subsample_baseline:
                        anchor_counts = [len(np.load(s["refine_pts"]))
                                         for s in candidates.values()
                                         if "refine_pts" in s]
                        anchor_counts = [c for c in anchor_counts if c > 0]
                        n_sub = (int(np.median(anchor_counts))
                                 if anchor_counts else 300)
                        if n_sub < len(bl_pts):
                            rng = np.random.default_rng(0)
                            sub_pts = bl_pts[rng.choice(
                                len(bl_pts), size=n_sub, replace=False)]
                        else:
                            sub_pts = bl_pts
                        sub_label = f"{bl_label}_sub"
                        sub_msh = os.path.join(
                            obj_dir, "refined_mesh",
                            f"{load_case}_{sub_label}{bm_tag}_refined.msh")
                        print(f"[budget] {obj}/{load_case}: {sub_label} "
                              f"anchors {len(sub_pts)} (from {len(bl_pts)})")
                        bl_variants.append((sub_label, sub_msh, sub_pts))
                    for v_label, v_msh, v_pts in bl_variants:
                        if not os.path.exists(v_msh):
                            try:
                                os.makedirs(os.path.dirname(v_msh),
                                            exist_ok=True)
                                _, n_bl, h_bl = generate_budget_matched_mesh(
                                    step_path, v_pts, h_min, h_max, v_msh,
                                    target_cells=target, tol=args.budget_tol,
                                    h_floor=h_floor)
                                print(f"[budget] {obj}/{load_case}: {v_label} "
                                      f"{n_bl} tets "
                                      f"(target {target and int(target)}, "
                                      f"h_min {h_bl:.3f} "
                                      f"vs default {h_min:.3f})")
                            except Exception as e:
                                print(f"[fail] baseline mesh gen "
                                      f"{obj}/{load_case}/{v_label}: {e}")
                                continue
                        if os.path.exists(v_msh):
                            candidates[v_label] = {"msh": v_msh}

            # --- ZZ-oracle candidate (solver-informed upper bound).
            #     Oracle: top-N ZZ anchors; when --match_budget is set the
            #     mesh budget is calibrated to the same target as the
            #     heuristic (the budget-reference GReFEM mesh), otherwise
            #     default sizing. ---
            if args.with_oracle:
                if args.oracle_min_spacing > 0:
                    or_label = "zz_oracle_spread"
                    or_fname = (f"{load_case}_{or_label}"
                                f"{args.oracle_min_spacing:g}"
                                f"{bm_tag}_refined.msh")
                elif args.oracle_percentile > 0:
                    or_label = "zz_oracle_dense"
                    or_fname = f"{load_case}_{or_label}{bm_tag}_refined.msh"
                else:
                    or_label = "zz_oracle"
                    or_fname = f"{load_case}_{or_label}{bm_tag}_refined.msh"
                or_msh = os.path.join(obj_dir, "refined_mesh", or_fname)
                if not os.path.exists(or_msh):
                    try:
                        os.makedirs(os.path.dirname(or_msh), exist_ok=True)
                        all_pts, all_vals = read_pos_points_top(
                            zz_pos, 0.0, load_case=load_case)
                        # Drop anchors inside the mesher's boundary-exclusion
                        # band (generate_or_refine_mesh ignores POIs within
                        # 6*h_min of the y-extremes), so every selected anchor
                        # is actually used for refinement.
                        y_lo, y_hi = all_pts[:, 1].min(), all_pts[:, 1].max()
                        excl = 6.0 * h_min
                        keep = ((all_pts[:, 1] > y_lo + excl)
                                & (all_pts[:, 1] < y_hi - excl))
                        n_dropped = int((~keep).sum())
                        all_pts, all_vals = all_pts[keep], all_vals[keep]
                        if n_dropped:
                            print(f"[oracle] {obj}/{load_case}: dropped "
                                  f"{n_dropped} boundary-band ZZ points, "
                                  f"{len(all_pts)} remain")
                        if args.oracle_min_spacing > 0:
                            # Greedy value-ordered selection with minimum
                            # spacing: clusters within the radius collapse
                            # to their highest-valued point; walk down the
                            # ranking until the GReFEM anchor count is hit.
                            from scipy.spatial import cKDTree
                            anchor_counts = [
                                len(np.load(s["refine_pts"]))
                                for s in candidates.values()
                                if "refine_pts" in s]
                            anchor_counts = [c for c in anchor_counts if c > 0]
                            n_target = (int(np.median(anchor_counts))
                                        if anchor_counts else 300)
                            radius = args.oracle_min_spacing * h_min
                            order = np.argsort(all_vals)[::-1]
                            pts_sorted = all_pts[order]
                            tree = cKDTree(pts_sorted)
                            suppressed = np.zeros(len(pts_sorted), dtype=bool)
                            sel = []
                            for i in range(len(pts_sorted)):
                                if suppressed[i]:
                                    continue
                                sel.append(i)
                                if len(sel) >= n_target:
                                    break
                                suppressed[tree.query_ball_point(
                                    pts_sorted[i], radius)] = True
                            oracle_pts = pts_sorted[sel]
                            print(f"[oracle] {obj}/{load_case}: spread "
                                  f"selection {len(oracle_pts)}/{n_target} "
                                  f"anchors (radius {radius:.3f})")
                        elif args.oracle_percentile > 0:
                            thr = np.percentile(all_vals,
                                                args.oracle_percentile)
                            oracle_pts = all_pts[all_vals >= thr]
                        else:
                            n_anchor = args.oracle_top_n
                            if n_anchor <= 0:
                                anchor_counts = [
                                    len(np.load(s["refine_pts"]))
                                    for s in candidates.values()
                                    if "refine_pts" in s]
                                anchor_counts = [c for c in anchor_counts
                                                 if c > 0]
                                n_anchor = (int(np.median(anchor_counts))
                                            if anchor_counts else 300)
                            order = np.argsort(all_vals)[::-1][:n_anchor]
                            oracle_pts = all_pts[order]
                        if budget_target is not None:
                            print(f"[oracle] {obj}/{load_case}: "
                                  f"{len(oracle_pts)} anchors, budget target "
                                  f"{int(budget_target)} tets")
                            _, n_or, h_or = generate_budget_matched_mesh(
                                step_path, oracle_pts, h_min, h_max, or_msh,
                                target_cells=budget_target,
                                tol=args.budget_tol, h_floor=h_floor)
                            print(f"[oracle] {obj}/{load_case}: {n_or} tets "
                                  f"(h_min {h_or:.3f} vs default {h_min:.3f})")
                        else:
                            print(f"[oracle] {obj}/{load_case}: "
                                  f"{len(oracle_pts)} anchors, default sizing "
                                  f"(h_min {h_min:.3f})")
                            generate_or_refine_mesh(
                                step_or_mesh_path=step_path,
                                points_of_interest=oracle_pts,
                                h_min=h_min, h_max=h_max, suffix=None,
                                verbose=False, out_msh=or_msh)
                            print(f"[oracle] {obj}/{load_case}: "
                                  f"{count_tets(or_msh)} tets")
                    except Exception as e:
                        print(f"[fail] oracle mesh gen {obj}/{load_case}: {e}")
                        or_msh = None
                if or_msh and os.path.exists(or_msh):
                    candidates[or_label] = {"msh": or_msh}

            # --- uniform-refinement control: same element budget spent
            #     uniformly instead of adaptively (built by
            #     make_uniform_meshes.py). Answers "is adaptive placement
            #     better than simply using a finer uniform mesh?" ---
            if args.with_uniform:
                un_msh = os.path.join(obj_dir, "refined_mesh",
                                      f"{load_case}_uniform{bm_tag}_refined.msh")
                if os.path.exists(un_msh) and count_tets(un_msh) > 0:
                    candidates["uniform"] = {"msh": un_msh}
                else:
                    print(f"[skip] {obj}/{load_case}: no uniform mesh "
                          f"({os.path.basename(un_msh)})")

            # --- practical solver-informed baseline: ZZ recovered from the
            #     COARSE solve (one cheap solve, no reference), anchors and
            #     budget matching identical to the oracle. This is classical
            #     solve -> estimate -> refine, unlike zz_oracle which reads
            #     the fine reference. Needs coarse_mesh_{lc}_zz.pos
            #     (see make_coarse_zz.py). ---
            if args.with_coarse_zz:
                cz_pos = os.path.join(obj_dir, f"coarse_mesh_{load_case}_zz.pos")
                cz_msh = os.path.join(obj_dir, "refined_mesh",
                                      f"{load_case}_zz_coarse{bm_tag}_refined.msh")
                if not os.path.exists(cz_pos):
                    print(f"[skip] {obj}/{load_case}: no coarse zz.pos")
                elif not os.path.exists(cz_msh):
                    try:
                        os.makedirs(os.path.dirname(cz_msh), exist_ok=True)
                        cz_pts, cz_vals = read_pos_points_top(
                            cz_pos, 0.0, load_case=load_case)
                        # Load-application band: tied to the OBJECT's scale
                        # (crit_radius = 4*unscaled h_min), not to the
                        # candidate sizing -- otherwise the band grows with
                        # --h_scale and can swallow the whole part. Fall back
                        # to the unfiltered set if nothing survives (the
                        # mesher ignores in-band POIs anyway).
                        y_lo, y_hi = cz_pts[:, 1].min(), cz_pts[:, 1].max()
                        excl = 1.5 * crit_radius
                        keep = ((cz_pts[:, 1] > y_lo + excl)
                                & (cz_pts[:, 1] < y_hi - excl))
                        if keep.sum() >= 10:
                            cz_pts, cz_vals = cz_pts[keep], cz_vals[keep]
                        else:
                            print(f"[coarse_zz] {obj}/{load_case}: band filter "
                                  f"left {int(keep.sum())} anchors, using "
                                  f"unfiltered set")
                        anchor_counts = [len(np.load(s["refine_pts"]))
                                         for s in candidates.values()
                                         if "refine_pts" in s]
                        anchor_counts = [c for c in anchor_counts if c > 0]
                        n_anchor = (int(np.median(anchor_counts))
                                    if anchor_counts else 300)
                        order = np.argsort(cz_vals)[::-1][:n_anchor]
                        cz_sel = cz_pts[order]
                        if len(cz_sel) == 0:
                            raise RuntimeError("no coarse-ZZ anchors survive")
                        if budget_target is not None:
                            _, n_cz, h_cz = generate_budget_matched_mesh(
                                step_path, cz_sel, h_min, h_max, cz_msh,
                                target_cells=budget_target,
                                tol=args.budget_tol, h_floor=h_floor)
                            print(f"[coarse_zz] {obj}/{load_case}: "
                                  f"{len(cz_sel)} anchors, {n_cz} tets "
                                  f"(h_min {h_cz:.3f})")
                        else:
                            generate_or_refine_mesh(
                                step_or_mesh_path=step_path,
                                points_of_interest=cz_sel,
                                h_min=h_min, h_max=h_max, suffix=None,
                                verbose=False, out_msh=cz_msh)
                    except (Exception, SystemExit) as e:
                        print(f"[fail] coarse_zz mesh gen "
                              f"{obj}/{load_case}: {e}")
                if os.path.exists(cz_msh):
                    candidates["zz_coarse"] = {"msh": cz_msh}

            for label, spec in candidates.items():
                key = (obj, load_case, label)
                if key in done:
                    continue
                msh_path = spec["msh"]
                if not os.path.exists(msh_path):
                    continue
                if count_tets(msh_path) == 0:
                    print(f"[skip] {obj}/{load_case}/{label}: mesh unreadable "
                          f"or empty (gmsh HXT failure)")
                    row = {"object": obj, "load_case": load_case,
                           "candidate": label, "candidate_msh": msh_path,
                           "status": "fail_mesh"}
                    writer.writerow(row)
                    fout.flush()
                    continue
                t_cand = time.time()
                row = {"object": obj, "load_case": load_case,
                       "candidate": label, "candidate_msh": msh_path,
                       "crit_radius": crit_radius,
                       "n_crit_points": len(crit_points)}
                try:
                    with _solve_time_limit():
                        cand_xdmf = ensure_solved(msh_path, load_case,
                                                  args.E, args.nu)
                        res = compute_qoi_scalars(
                            cand_xdmf, E=args.E, nu=args.nu,
                            crit_points=crit_points, crit_radius=crit_radius)
                    row.update({k: res.get(k) for k in CSV_FIELDS if k in res})
                    for qoi, rel_col in QOI_PAIRS:
                        c, r = res.get(qoi), ref_qoi.get(qoi)
                        row[rel_col] = (abs(c - r) / (abs(r) + 1e-30)
                                        if c == c and r == r else float("nan"))
                    # flag refined meshes that degenerated to ~coarse size
                    # (empty anchors or gmsh uniform-fallback)
                    if (label != "coarse" and "coarse" in candidates
                            and os.path.exists(candidates["coarse"]["msh"])):
                        n_coarse = count_tets(candidates["coarse"]["msh"])
                        if res["n_cells"] <= 1.05 * n_coarse:
                            row["status"] = "ok_norefine"
                    row.setdefault("status", None)
                    if not row["status"]:
                        row["status"] = "ok"
                    row["t_candidate_s"] = round(time.time() - t_cand, 1)
                    print(f"[ok] {obj}/{load_case}/{label}: "
                          f"rel_energy_crit_err={row['rel_energy_crit_err']:.4g} "
                          f"rel_vm_p99_err={row['rel_vm_p99_err']:.4g} "
                          f"n_dofs={res.get('n_dofs')} "
                          f"({_fmt_secs(time.time()-t_cand)})")
                except (Exception, SystemExit) as e:
                    row["status"] = f"error: {e}"
                    print(f"[fail] {obj}/{load_case}/{label}: {e}")
                    traceback.print_exc()
                writer.writerow(row)
                fout.flush()

        # ---- per-object progress summary + ETA + running aggregate ----
        obj_times.append(time.time() - t_obj)
        n_done = obj_i + 1
        avg = float(np.mean(obj_times))
        eta = avg * (n_total - n_done)
        print(f"\n[progress] {n_done}/{n_total} objects done | "
              f"this object: {_fmt_secs(obj_times[-1])} | "
              f"avg/object: {_fmt_secs(avg)} | "
              f"ETA: {_fmt_secs(eta)} "
              f"(~finish {time.strftime('%H:%M', time.localtime(time.time()+eta))})")
        try:
            import pandas as pd
            dfp = pd.read_csv(args.out_csv)
            dfp = dfp[dfp["status"] == "ok"].copy()
            dfp["method"] = dfp["candidate"].where(
                dfp["candidate"].isin(["coarse", "zz_oracle"])
                | dfp["candidate"].str.startswith(("heuristic_", "mlgnn")),
                dfp["candidate"].str.replace(r"_\d+run$", "", regex=True))
            ragg = dfp.groupby("method")[
                ["rel_energy_crit_err", "rel_vm_p99_err",
                 "rel_vm_max_err", "n_dofs"]].agg(["mean", "count"]).round(4)
            print("[running aggregate over finished objects]")
            print(ragg.to_string())
        except Exception as e:
            print(f"(running aggregate skipped: {e})")

    fout.close()
    print(f"\nDone. Results in {args.out_csv}")

    # --- Aggregated rebuttal table (mean over objects, per candidate type) ---
    try:
        import pandas as pd
        df = pd.read_csv(args.out_csv)
        df = df[df["status"] == "ok"].copy()  # excludes ok_norefine fallbacks
        # collapse candidate names: coarse stays; refined grouped by method string
        df["method"] = df["candidate"].where(
            df["candidate"] == "coarse",
            df["candidate"].str.replace(r"_\d+run$", "", regex=True))
        agg = df.groupby(["load_case", "method"])[
            ["rel_energy_err", "rel_energy_crit_err", "rel_vm_p99_err",
             "rel_vm_max_err", "rel_max_disp_err", "n_dofs"]].mean().round(4)
        print("\n=== Aggregated (mean over objects/runs) ===")
        print(agg.to_string())
        agg.to_csv(args.out_csv.replace(".csv", "_aggregated.csv"))
    except Exception as e:
        print(f"(aggregation skipped: {e})")


if __name__ == "__main__":
    main()
