"""Micro-averaged localization P/R/F1 on the final 156-object set.

Differences vs loc_prf1_156.py (macro):
  * GT per unit = top-N fine2 ZZ points by value (after boundary trim),
    with N = n_eq = min(#grefem anchors, #blind-heuristic anchors),
    i.e. GT size matched to the anchor budget (paper's micro protocol).
  * Anchor sets grefem_eq / heuristic_eq are both subsampled to n_eq.
  * Micro averaging: pool point-level hits over ALL units:
        precision = sum(matched anchors) / sum(anchors)
        recall    = sum(matched GT)      / sum(GT)
    F1 from micro precision/recall. mech is also scored against the
    same GT set (its own anchor count, already budget-matched).

Output: loc_prf1_156_micro.csv (per-unit counts) + micro summary.
"""
import glob
import os
import re as _re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from multiprocessing import Pool

BASES = ["test_meshes_dense", "test_meshes", "test_meshes_extra"]
LOADS = ["compression", "bending", "torsion",
         "bending_compression", "torsion_compression"]


def read_pos_points_all(pos_file, load_case=None, remove_boundary_frac=0.02):
    pattern = _re.compile(r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};")
    points, values = [], []
    with open(pos_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                x, y, z, val = map(float, m.groups())
                points.append([x, y, z])
                values.append(val)
    points, values = np.asarray(points), np.asarray(values)
    if len(values) == 0:
        return np.empty((0, 3)), np.empty((0,))
    if load_case is not None:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        x_dist = (x_max - x_min) * remove_boundary_frac
        y_dist = (y_max - y_min) * remove_boundary_frac
        mask = np.ones(len(points), dtype=bool)
        if load_case in ("compression", "bending", "torsion", "shear"):
            mask &= (points[:, 1] < y_max - y_dist) & (points[:, 1] > y_min + y_dist)
        elif load_case in ("bending_compression", "torsion_compression"):
            mask &= (points[:, 0] < x_max - x_dist) & (points[:, 0] > x_min + x_dist)
            mask &= points[:, 1] < y_max - y_dist
        points, values = points[mask], values[mask]
    return points, values


def find_base(obj, lc):
    for b in BASES:
        if os.path.exists(os.path.join(b, obj, f"fine2_mesh_{lc}_zz.pos")):
            return os.path.join(b, obj)
    return None


def unit_counts(args):
    obj, lc, r = args
    d = find_base(obj, lc)
    if d is None:
        return None
    try:
        g = glob.glob(os.path.join(
            d, f"{lc}_gemini*_geo_maxprompt_ortho_5views_11grid_1run",
            "refinement_points_prefilt.npy"))
        bl_p = os.path.join(d, "refinement_points_baseline_DENSE_dihedral30.0deg.npy")
        if not g or not os.path.exists(bl_p):
            return None
        gre = np.load(g[0])
        bl = np.load(bl_p)
        if len(gre) == 0 or len(bl) == 0:
            return None
        n_eq = min(len(gre), len(bl))

        pts, vals = read_pos_points_all(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), load_case=lc)
        if len(vals) == 0:
            return None
        k = min(n_eq, len(vals))
        idx = np.argpartition(vals, -k)[-k:]
        gt = pts[idx]
        gt_tree = cKDTree(gt)

        anchors = {}
        rng_g = np.random.default_rng(1)
        rng_h = np.random.default_rng(2)
        anchors["grefem_eq"] = (gre if len(gre) == n_eq else
                                gre[rng_g.choice(len(gre), size=n_eq,
                                                 replace=False)])
        anchors["heuristic_eq"] = (bl if len(bl) == n_eq else
                                   bl[rng_h.choice(len(bl), size=n_eq,
                                                   replace=False)])
        mech_p = os.path.join(d, f"refinement_points_mech_dihedral30.0deg_{lc}.npy")
        if os.path.exists(mech_p):
            mech = np.load(mech_p)
            if len(mech):
                anchors["mech"] = mech

        rows = []
        for m, a in anchors.items():
            d_a2g, _ = gt_tree.query(a)
            d_g2a, _ = cKDTree(a).query(gt)
            rows.append(dict(object=obj, load_case=lc, method=m, r=r,
                             n_anchors=len(a),
                             matched_anchors=int(np.sum(d_a2g <= r)),
                             n_gt=len(gt),
                             matched_gt=int(np.sum(d_g2a <= r))))
        return rows
    except Exception as e:
        print(f"[fail] {obj}/{lc}: {e}", flush=True)
        return None


def crit_radius_map():
    rmap = {}
    for p in glob.glob("uni_shards/hs1.0/u_*.csv"):
        name = os.path.basename(p)
        if name.endswith("_aggregated.csv"):
            continue
        m = _re.match(r"u_(\d+)_([a-z_]+)\.csv", name)
        if not m:
            continue
        try:
            df = pd.read_csv(p, dtype={"object": str})
        except Exception:
            continue
        if "crit_radius" in df and len(df):
            cr = df["crit_radius"].dropna()
            if len(cr):
                rmap[(m.group(1), m.group(2))] = float(cr.iloc[0])
    comb = pd.read_csv("combined_200obj_table.csv", dtype={"object": str})
    for (o, lc), grp in comb.groupby(["object", "load_case"]):
        cr = grp["crit_radius"].dropna()
        if len(cr) and (o, lc) not in rmap:
            rmap[(o, lc)] = float(cr.iloc[0])
    return rmap


def micro(df):
    out = {}
    for m, g in df.groupby("method"):
        p = g.matched_anchors.sum() / g.n_anchors.sum()
        r = g.matched_gt.sum() / g.n_gt.sum()
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        out[m] = dict(precision=p, recall=r, f1=f1,
                      anchors=g.n_anchors.sum(), gt=g.n_gt.sum(),
                      units=len(g))
    return pd.DataFrame(out).T.round(3)


def main():
    objs = []
    for f in ["final_dense_objects.txt", "final_orig_objects.txt",
              "final_winners_objects.txt"]:
        objs += [x.strip() for x in open(f) if x.strip()]
    objs = sorted(set(objs))
    rmap = crit_radius_map()
    tasks = [(o, lc, rmap[(o, lc)]) for o in objs for lc in LOADS
             if (o, lc) in rmap]
    print(f"objects: {len(objs)}  units: {len(tasks)}", flush=True)

    with Pool(64) as pool:
        out = []
        for i, res in enumerate(pool.imap_unordered(unit_counts, tasks,
                                                    chunksize=2)):
            if res:
                out.extend(res)
            if (i + 1) % 50 == 0:
                print(f"done {i+1}/{len(tasks)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("loc_prf1_156_micro.csv", index=False)
    print(f"\nunits ok: {df.groupby(['object','load_case']).ngroups}")

    print("\n===== MICRO POOLED =====")
    print(micro(df).to_string())
    for lc in sorted(df.load_case.unique()):
        print(f"\n===== MICRO {lc} =====")
        print(micro(df[df.load_case == lc]).to_string())


if __name__ == "__main__":
    main()
