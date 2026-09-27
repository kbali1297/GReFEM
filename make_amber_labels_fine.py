#!/usr/bin/env python
"""Build fine-reference AMBER expert meshes (v2 training labels).

Replaces make_amber_labels.py, whose experts came from the coarse-capped
fine_mesh ZZ point clouds (ml_baseline/labels) with a FIXED top-1563 anchor
count. Here the supervision reproduces the evaluation table's zz_oracle
recipe exactly, per (object, load_case):

    zz field   = fine2_mesh_{lc}_zz.pos          (fine-reference ZZ, the same
                                                  field the table oracle uses)
    boundary   = drop anchors within 6*h_min of the y-extremes
                 (generate_or_refine_mesh ignores POIs there)
    n_anchor   = len(refinement_points_prefilt.npy) of the GReFEM
                 gemini-3-flash 5views run for this unit (per-unit count,
                 same source the table uses for oracle budget matching)
    mesh       = generate_or_refine_mesh(step, top-N anchors, h_min, h_max)
                 with compute_h_sizes' volume^0.3 law

Units with fewer than --min_anchors prefilt points are skipped.

Input:  units file (lines "object src_dir") = non-cohort objects with full
        fine2 refs, e.g. /tmp/fine2_non_cohort.txt
Output: <out_dir>/<object>_<lc>.msh   (resumable; skipped if valid)

Usage (sharded):
    python make_amber_labels_fine.py --shard 0 --nshards 100 \
        --units_file amber_fine_units.txt --out_dir ml_baseline/amber_experts_fine
"""
import argparse
import functools
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_local_error_tables import (  # noqa: E402
    compute_h_sizes, count_tets)
from fem_fenics import read_pos_points_top  # noqa: E402
from utils import generate_or_refine_mesh  # noqa: E402

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
VAR = "gemini-3-flash-preview_geo_maxprompt_ortho_5views_11grid_1run"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units_file", default="amber_fine_units.txt",
                    help="lines: '<object> <src_dir>'")
    ap.add_argument("--out_dir", default="ml_baseline/amber_experts_fine")
    ap.add_argument("--min_anchors", type=int, default=50,
                    help="skip units whose prefilt count is below this")
    ap.add_argument("--load_cases", nargs="+", default=LCS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    units = [l.split() for l in open(args.units_file) if l.strip()]
    units = [u for i, u in enumerate(units) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(units)} objects")

    n_ok = n_skip = n_fail = 0
    for ci, (obj, src) in enumerate(units):
        obj_dir = os.path.join(src, obj)
        try:
            h_max, h_min, step_path = compute_h_sizes(obj_dir)
        except Exception as e:
            print(f"[fail] {obj}: h sizes: {e}")
            n_fail += len(args.load_cases)
            continue

        for lc in args.load_cases:
            out = os.path.join(args.out_dir, f"{obj}_{lc}.msh")
            if os.path.exists(out) and count_tets(out) > 0:
                n_skip += 1
                continue
            zz_pos = os.path.join(obj_dir, f"fine2_mesh_{lc}_zz.pos")
            npy = os.path.join(obj_dir, f"{lc}_{VAR}",
                               "refinement_points_prefilt.npy")
            if not os.path.exists(zz_pos) or not os.path.exists(npy):
                print(f"[skip] {obj}/{lc}: missing "
                      f"{'zz_pos' if not os.path.exists(zz_pos) else 'prefilt'}")
                continue
            try:
                n_anchor = int(len(np.load(npy)))
                if n_anchor < args.min_anchors:
                    print(f"[skip] {obj}/{lc}: only {n_anchor} prefilt "
                          f"anchors (< {args.min_anchors})")
                    continue
                all_pts, all_vals = read_pos_points_top(
                    zz_pos, 0.0, load_case=lc)
                # same boundary-band exclusion as the table oracle
                y_lo, y_hi = all_pts[:, 1].min(), all_pts[:, 1].max()
                excl = 6.0 * h_min
                keep = ((all_pts[:, 1] > y_lo + excl)
                        & (all_pts[:, 1] < y_hi - excl))
                all_pts, all_vals = all_pts[keep], all_vals[keep]
                if len(all_pts) == 0:
                    print(f"[skip] {obj}/{lc}: no ZZ points after "
                          f"boundary-band exclusion")
                    continue
                order = np.argsort(all_vals)[::-1][:n_anchor]
                anchors = all_pts[order]
                generate_or_refine_mesh(
                    step_or_mesh_path=step_path, points_of_interest=anchors,
                    h_min=h_min, h_max=h_max, suffix=None,
                    verbose=False, out_msh=out)
                nt = count_tets(out)
                if nt == 0:
                    if os.path.exists(out):
                        os.remove(out)
                    raise RuntimeError("mesh gen produced no tets")
                n_ok += 1
                print(f"[ok] {obj}/{lc}: {nt} tets from {len(anchors)} "
                      f"anchors (h_min {h_min:.3f} h_max {h_max:.3f})")
            except Exception as e:
                n_fail += 1
                print(f"[fail] {obj}/{lc}: {type(e).__name__}: {e}")
                traceback.print_exc()
        if (ci + 1) % 5 == 0:
            print(f"[progress] {ci+1}/{len(units)} objects | ok {n_ok} "
                  f"skip {n_skip} fail {n_fail}")

    print(f"\nDone. ok {n_ok}, skipped {n_skip}, failed {n_fail}")


if __name__ == "__main__":
    main()
