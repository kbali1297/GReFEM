#!/usr/bin/env python
"""Uniform-refinement control at matched budget ("uniform" candidate).

The coarse mesh is a single point in the sweep, so it cannot answer "is
adaptive refinement better than simply spending the same elements uniformly?".
This script builds, per (object, load_case, h_scale), a UNIFORM mesh (no
anchors) whose element count is calibrated to the same budget target as the
refined candidates -- i.e. the target used by compute_local_error_tables.py:
the element count of the GReFEM mesh at that sizing level.

Writes <obj>/refined_mesh/{lc}_uniform{tag}_refined.msh so the evaluator can
pick it up as an ordinary pre-existing candidate mesh.

Usage:
  python make_uniform_meshes.py --test_dir test_meshes_dense --objects 00000027 \
      --load_cases bending --h_scale 1.0 --ref_prefix fine2_mesh
"""
import argparse
import functools
import fnmatch
import glob
import os

import numpy as np

from compute_local_error_tables import (compute_h_sizes, count_tets,
                                        _label_matches_load)
from utils import generate_or_refine_mesh

print = functools.partial(print, flush=True)


def uniform_at_budget(step_path, h_start, out_msh, target_cells,
                      tol=0.10, max_iter=8, h_floor=None):
    """Uniform mesh (no POIs) with characteristic length calibrated so the tet
    count matches target_cells. n ~ h^-3, damped the same way as the refined
    budget matcher (exponent 1/2) for stability."""
    h = h_start
    best = (None, float("inf"), h)
    for _ in range(max_iter):
        if h_floor is not None:
            h = max(h, h_floor)
        generate_or_refine_mesh(step_or_mesh_path=step_path,
                                points_of_interest=[], h_min=h, h_max=h,
                                suffix=None, verbose=False, out_msh=out_msh)
        n = count_tets(out_msh)
        if n == 0:
            raise RuntimeError("uniform mesh generation produced no tets")
        gap = abs(n - target_cells) / target_cells
        if gap < best[1]:
            best = (n, gap, h)
        if gap <= tol:
            return n, h
        h_new = h * (n / target_cells) ** (1.0 / 2.0)
        if h_floor is not None:
            h_new = max(h_new, h_floor)
        if abs(h_new - h) / h < 1e-3:
            break
        h = h_new
    if best[0] is not None and best[2] != h:
        generate_or_refine_mesh(step_or_mesh_path=step_path,
                                points_of_interest=[], h_min=best[2],
                                h_max=best[2], suffix=None, verbose=False,
                                out_msh=out_msh)
        return best[0], best[2]
    return count_tets(out_msh), h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--load_cases", nargs="+", required=True)
    ap.add_argument("--h_scale", type=float, default=1.0)
    ap.add_argument("--ref_prefix", default="fine2_mesh")
    ap.add_argument("--budget_ref_pattern",
                    default="*_geo_maxprompt_ortho_5views_*")
    ap.add_argument("--budget_tol", type=float, default=0.10)
    ap.add_argument("--simple_scale", action="store_true",
                    help="Generate the uniform mesh by simply scaling the base "
                         "sizing (h_min*hs, h_max*hs) with no anchors -- i.e. "
                         "the coarse mesh regenerated at this sizing level, "
                         "exactly the rule the sweep applies to every other "
                         "candidate. One gmsh call, no budget iteration.")
    args = ap.parse_args()

    hs_tag = f"_hs{args.h_scale:g}" if args.h_scale != 1.0 else ""
    bm_tag = hs_tag + (f"_{args.ref_prefix}"
                       if args.ref_prefix != "fine_mesh" else "")

    for obj in args.objects:
        od = os.path.join(args.test_dir, obj)
        try:
            h_max, h_min, step_path = compute_h_sizes(od)
        except Exception as e:
            print(f"[skip] {obj}: h sizes failed: {e}")
            continue
        if args.h_scale != 1.0:
            h_min *= args.h_scale
            h_max *= args.h_scale

        for lc in args.load_cases:
            out_msh = os.path.join(od, "refined_mesh",
                                   f"{lc}_uniform{bm_tag}_refined.msh")
            if os.path.exists(out_msh) and count_tets(out_msh) > 0:
                print(f"[skip] {obj}/{lc}: exists")
                continue
            if args.simple_scale:
                # Coarse/uniform mesh at this sizing level: no anchors, just
                # the volume-scaled base sizing multiplied by h_scale --
                # identical rule to every other candidate in the sweep.
                try:
                    os.makedirs(os.path.dirname(out_msh), exist_ok=True)
                    # h_min = h_max: the convention the paper's coarse meshes
                    # use (generate_mesh_and_simulate_parallel.py passes
                    # h_min=h_max, h_max=h_max). Passing h_min=h_max/5 instead
                    # lets gmsh refine at curvature and yields ~2.4x more
                    # elements, i.e. it would NOT be the coarse mesh at this
                    # sizing level.
                    generate_or_refine_mesh(step_or_mesh_path=step_path,
                                            points_of_interest=[],
                                            h_min=h_max, h_max=h_max,
                                            suffix=None, verbose=False,
                                            out_msh=out_msh)
                    n = count_tets(out_msh)
                    if n == 0:
                        raise RuntimeError("no tets")
                    print(f"[ok] {obj}/{lc} hs{args.h_scale:g}: uniform {n} "
                          f"tets (h = h_max = {h_max:.4f})")
                except Exception as e:
                    print(f"[fail] {obj}/{lc}: {e}")
                    if os.path.exists(out_msh):
                        os.remove(out_msh)
                continue

            # Budget target = GReFEM mesh tet count at THIS sizing level.
            # Must match the evaluator's own convention: only *_refined.msh
            # (not _sol/_stress/_strain), only this load case (so `torsion`
            # never picks up `torsion_compression`), and only this h_scale.
            cands = []
            for p in glob.glob(os.path.join(od, "refined_mesh",
                                            "*_refined.msh")):
                base = os.path.basename(p)
                label = base[:-len("_refined.msh")]
                if hs_tag:
                    if not label.endswith(hs_tag):
                        continue
                    label = label[:-len(hs_tag)]
                elif "_hs" in label:
                    continue
                if not _label_matches_load(label, lc):
                    continue
                if not fnmatch.fnmatch(label, args.budget_ref_pattern):
                    continue
                cands.append(p)
            counts = [count_tets(p) for p in cands]
            counts = [c for c in counts if c > 0]
            if not counts:
                print(f"[skip] {obj}/{lc}: no GReFEM mesh to match "
                      f"(pattern {args.budget_ref_pattern})")
                continue
            target = float(np.mean(counts))
            try:
                os.makedirs(os.path.dirname(out_msh), exist_ok=True)
                n, h = uniform_at_budget(step_path, h_min, out_msh, target,
                                         tol=args.budget_tol)
                print(f"[ok] {obj}/{lc} hs{args.h_scale:g}: uniform {n} tets "
                      f"(target {int(target)}, h {h:.4f})")
            except Exception as e:
                print(f"[fail] {obj}/{lc}: {e}")
                if os.path.exists(out_msh):
                    os.remove(out_msh)


if __name__ == "__main__":
    main()
