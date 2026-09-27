#!/usr/bin/env python
"""Load-conditioning gain: does a method's anchors MOVE with the load case?

Reviewers argue that encoding physics in the prompt means the MLLM is only
following instructions, and that a geometric heuristic is a sufficient proxy.
This script isolates the property a load-blind method cannot have by
construction: for the same geometry, are the anchors predicted for load case
L actually better aligned with the ZZ hotspots of L than the anchors predicted
for a different load case L'?

For each (object, target load case L) we compute, against the ZZ ground truth
of L:
    matched     = precision/recall/F1 of the anchors predicted for L
    mismatched  = mean over L' != L of the same metrics for anchors of L'
    gain        = matched - mismatched

A load-blind method (the DENSE dihedral heuristic) uses identical anchors for
every load case, so its gain is exactly 0 -- not by measurement noise but by
construction. Load-conditioned methods (GReFEM prompts, the mechanics-aware
heuristic, the ML surrogate, AMBER with load one-hot) can score above 0.

Ground truth, match radius and anchor conventions are identical to
loc_precision_recall.py (top-0.1% ZZ of the fine2 reference, r = crit_radius
= 4*h_min, prefilt anchor sets).

Usage:
    python loc_load_discrimination.py --out loc_load_discrimination.csv
"""
import argparse
import functools
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from loc_precision_recall import read_pos_points_top, ZZ_TOP

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
GREFEM_VARIANTS = {"grefem_max": "maxprompt", "grefem_mid": "midprompt",
                   "grefem_none": "noneprompt"}


def load_anchor_sets(obj_dir, lc):
    """All available anchor sets for one (object, load case)."""
    out = {}
    for name, variant in GREFEM_VARIANTS.items():
        g = glob.glob(os.path.join(
            obj_dir, f"{lc}_gemini*_geo_{variant}_ortho_5views_11grid_1run",
            "refinement_points_prefilt.npy"))
        if g:
            pts = np.load(g[0])
            if len(pts):
                out[name] = pts
    # load-blind geometric heuristic: identical for every load case
    p = os.path.join(obj_dir,
                     "refinement_points_baseline_DENSE_dihedral30.0deg.npy")
    if os.path.exists(p):
        pts = np.load(p)
        if len(pts):
            out["heuristic"] = pts
    # mechanics-aware heuristic and ML surrogate: per load case
    for name, tmpl in [("heuristic_mech", "mech_dihedral30.0deg_{lc}"),
                       ("ml_gnn", "mlgnn_{lc}")]:
        p = os.path.join(obj_dir,
                         f"refinement_points_{tmpl.format(lc=lc)}.npy")
        if os.path.exists(p):
            pts = np.load(p)
            if len(pts):
                out[name] = pts
    # AMBER anchors are a mesh, not a point set, so it is not included here;
    # its load conditioning shows up in the downstream error table instead.
    return out


def prf(anchors, gt_tree, gt, r):
    d_a2g, _ = gt_tree.query(anchors)
    d_g2a, _ = cKDTree(anchors).query(gt)
    prec = float(np.mean(d_a2g <= r))
    rec = float(np.mean(d_g2a <= r))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def unit(args):
    obj, lc, src, r = args
    base = "test_meshes_extra" if src == "extra" else "test_meshes"
    d = os.path.join(base, obj)
    try:
        gt, _ = read_pos_points_top(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), ZZ_TOP, load_case=lc)
        if len(gt) == 0:
            return None
        gt_tree = cKDTree(gt)

        matched = load_anchor_sets(d, lc)
        # anchor sets predicted for the OTHER load cases, same geometry
        others = {lc2: load_anchor_sets(d, lc2) for lc2 in LCS if lc2 != lc}

        rows = []
        for m, pts in matched.items():
            p_m, r_m, f_m = prf(pts, gt_tree, gt, r)
            mis = []
            for lc2, sets in others.items():
                if m in sets:
                    mis.append(prf(sets[m], gt_tree, gt, r))
            if not mis:
                continue
            p_x = float(np.mean([x[0] for x in mis]))
            r_x = float(np.mean([x[1] for x in mis]))
            f_x = float(np.mean([x[2] for x in mis]))
            rows.append(dict(
                object=obj, load_case=lc, method=m, r=r,
                n_anchors=len(pts), n_gt=len(gt), n_mismatch=len(mis),
                precision_matched=p_m, precision_mismatched=p_x,
                recall_matched=r_m, recall_mismatched=r_x,
                f1_matched=f_m, f1_mismatched=f_x,
                gain_precision=p_m - p_x, gain_recall=r_m - r_x,
                gain_f1=f_m - f_x))
        return rows
    except Exception as e:
        print(f"[fail] {obj}/{lc}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units_csv", default="combined_200obj_table.csv")
    ap.add_argument("--out", default="loc_load_discrimination.csv")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.units_csv, dtype={"object": str})
    units = (df[df["method"] == "grefem_max"]
             .drop_duplicates(["object", "load_case"])
             [["object", "load_case", "src", "crit_radius"]])
    tasks = [tuple(t) for t in units.itertuples(index=False)]
    print(f"units: {len(tasks)}")

    out = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(unit, tasks, chunksize=4)):
            if res:
                out.extend(res)
            if (i + 1) % 200 == 0:
                print(f"done {i+1}/{len(tasks)}")

    res = pd.DataFrame(out)
    res.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(res)} rows)\n")

    from scipy.stats import wilcoxon
    print(f"{'method':16s} {'n':>5s} {'prec_match':>10s} {'prec_mis':>9s} "
          f"{'gain_prec':>10s} {'gain_f1':>8s} {'p(gain>0)':>10s}")
    for m, g in res.groupby("method"):
        gain = g["gain_precision"].to_numpy()
        if np.allclose(gain, 0):
            p = float("nan")
        else:
            try:
                _, p = wilcoxon(gain, alternative="greater")
            except ValueError:
                p = float("nan")
        print(f"{m:16s} {len(g):5d} {g['precision_matched'].mean():10.4f} "
              f"{g['precision_mismatched'].mean():9.4f} "
              f"{gain.mean():+10.4f} {g['gain_f1'].mean():+8.4f} {p:10.2e}")
    print("\n(a load-blind method has gain identically 0 by construction)")


if __name__ == "__main__":
    main()
