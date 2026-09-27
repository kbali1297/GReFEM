"""Radius-free hotspot-coverage diagnostic.

For the most severe fine-solution ZZ points (top 0.01% by value, and the
top-20 points), compute the distance from each GT point to the NEAREST
anchor of each method, normalized by h_min (= crit_radius / 4).
This is what the distance-based sizing field actually responds to.

Anchor sets: identical to loc_prf1_156_p99.py symmetric matching.
Output: loc_hotspot_dist.csv (per-unit percentiles of d/h_min).
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


def unit_dist(args):
    obj, lc, r = args
    hmin = r / 4.0
    d = find_base(obj, lc)
    if d is None:
        return None
    try:
        g = glob.glob(os.path.join(
            d, f"{lc}_gemini*_geo_maxprompt_ortho_5views_11grid_1run",
            "refinement_points_prefilt.npy"))
        gp = glob.glob(os.path.join(
            d, f"{lc}_gpt-5.4-mini_geo_maxprompt_ortho_5views_11grid_1run",
            "refinement_points_prefilt.npy"))
        bl_p = os.path.join(d, "refinement_points_baseline_DENSE_dihedral30.0deg.npy")
        if not g or not os.path.exists(bl_p):
            return None
        gre = np.load(g[0])
        bl = np.load(bl_p)
        if len(gre) == 0 or len(bl) == 0:
            return None

        pts, vals = read_pos_points_all(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), load_case=lc)
        if len(vals) == 0:
            return None

        n_min = min(len(gre), len(bl))
        rng_g = np.random.default_rng(1)
        rng_h = np.random.default_rng(2)
        anchors = {}
        anchors["grefem_gemini"] = (gre if len(gre) <= n_min else
                                    gre[rng_g.choice(len(gre), size=n_min,
                                                     replace=False)])
        if gp:
            gptp = np.load(gp[0])
            if len(gptp):
                n_min_gpt = min(len(gptp), len(bl))
                rng_gp = np.random.default_rng(4)
                anchors["grefem_gpt"] = (gptp if len(gptp) <= n_min_gpt else
                                         gptp[rng_gp.choice(len(gptp),
                                                            size=n_min_gpt,
                                                            replace=False)])
        anchors["heuristic_min"] = (bl if len(bl) <= n_min else
                                    bl[rng_h.choice(len(bl), size=n_min,
                                                    replace=False)])
        mech_p = os.path.join(d, f"refinement_points_mech_dihedral30.0deg_{lc}.npy")
        if os.path.exists(mech_p):
            mech = np.load(mech_p)
            if len(mech):
                anchors["mech"] = mech

        # two GT tiers: top 0.01% by value, and the top-20 most severe points
        order = np.argsort(vals)
        k1 = max(20, int(len(vals) * 0.0001))
        gt_001 = pts[order[-k1:]]
        gt_top20 = pts[order[-20:]]

        rows = []
        for m, a in anchors.items():
            tree = cKDTree(a)
            d1, _ = tree.query(gt_001)
            d2, _ = tree.query(gt_top20)
            d1 = d1 / hmin
            d2 = d2 / hmin
            rows.append(dict(
                object=obj, load_case=lc, method=m, n_anchors=len(a),
                n_gt001=len(gt_001),
                d001_med=float(np.median(d1)),
                d001_p90=float(np.percentile(d1, 90)),
                d001_max=float(d1.max()),
                frac001_within1=float(np.mean(d1 <= 1)),
                frac001_within2=float(np.mean(d1 <= 2)),
                frac001_within4=float(np.mean(d1 <= 4)),
                dtop20_med=float(np.median(d2)),
                dtop20_max=float(d2.max()),
                fractop20_within2=float(np.mean(d2 <= 2)),
                fractop20_within4=float(np.mean(d2 <= 4)),
            ))
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
        for i, res in enumerate(pool.imap_unordered(unit_dist, tasks,
                                                    chunksize=2)):
            if res:
                out.extend(res)
            if (i + 1) % 50 == 0:
                print(f"done {i+1}/{len(tasks)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("loc_hotspot_dist.csv", index=False)
    print(f"\nunits ok: {df.groupby(['object','load_case']).ngroups}")

    cols = ["d001_med", "d001_p90", "frac001_within1", "frac001_within2",
            "frac001_within4", "dtop20_med", "fractop20_within2",
            "fractop20_within4"]
    print("\n===== per-unit MEAN over units (distances in h_min units) =====")
    print(df.groupby("method")[cols].mean().round(3).to_string())
    print("\n===== per-unit MEDIAN over units =====")
    print(df.groupby("method")[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
