"""Element-level metrics for GReFEM GPT-5.4-mini meshes only:
elem P/R/F1 (fine elems h<=1.25*h_min vs top-0.1% ZZ hotspots) + median h at hotspots.
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
TRIM = 0.05
PCT = 99.9
FINE_FAC = 1.25


def read_pos_points(pos_file, load_case, frac=TRIM):
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
        return points, values
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_dist = (x_max - x_min) * frac
    y_dist = (y_max - y_min) * frac
    mask = np.ones(len(points), dtype=bool)
    if load_case in ("compression", "bending", "torsion", "shear"):
        mask &= (points[:, 1] < y_max - y_dist) & (points[:, 1] > y_min + y_dist)
    else:
        mask &= (points[:, 0] < x_max - x_dist) & (points[:, 0] > x_min + x_dist)
        mask &= points[:, 1] < y_max - y_dist
    return points[mask], values[mask]


def find_base(obj, lc):
    for b in BASES:
        if os.path.exists(os.path.join(b, obj, f"fine2_mesh_{lc}_zz.pos")):
            return os.path.join(b, obj)
    return None


def unit_metrics(args):
    obj, lc, r = args
    d = find_base(obj, lc)
    if d is None:
        return None
    try:
        import meshio
        g = glob.glob(os.path.join(
            d, "refined_mesh",
            f"{lc}_gpt*_geo_maxprompt_ortho_5views_11grid_1run_refined.msh"))
        if not g:
            return None
        pts, vals = read_pos_points(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), lc)
        if len(vals) == 0:
            return None
        thr = np.percentile(vals, PCT)
        gt = pts[vals >= thr]
        if len(gt) == 0:
            return None
        gt_tree = cKDTree(gt)
        h_cut = FINE_FAC * r / 4.0

        m = meshio.read(g[0])
        tets = None
        for cb in m.cells:
            if cb.type == "tetra":
                tets = cb.data if tets is None else np.vstack([tets, cb.data])
        if tets is None or len(tets) == 0:
            return None
        nodes = m.points
        v0 = nodes[tets[:, 0]]
        cent = (v0 + nodes[tets[:, 1]] + nodes[tets[:, 2]]
                + nodes[tets[:, 3]]) / 4.0
        a = nodes[tets[:, 1]] - v0
        b = nodes[tets[:, 2]] - v0
        c = nodes[tets[:, 3]] - v0
        vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        h = np.cbrt(vol)
        dd_all, _ = gt_tree.query(cent)
        loc = dd_all <= r
        h_loc = float(np.median(h[loc])) if loc.sum() else np.nan
        elem_prec_all = float(loc.mean())
        fine = h <= h_cut
        n_fine = int(fine.sum())
        if n_fine == 0:
            prec, rec, f1 = np.nan, 0.0, 0.0
        else:
            fc = cent[fine]
            prec = float(np.mean(dd_all[fine] <= r))
            f_tree = cKDTree(fc)
            dg, _ = f_tree.query(gt)
            rec = float(np.mean(dg <= r))
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return dict(object=obj, load_case=lc, r=r, n_fine=n_fine,
                    prec=prec, rec=rec, f1=f1, h_loc=h_loc,
                    elem_prec_all=elem_prec_all, n_tot=len(tets))
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
    print(f"units: {len(tasks)}", flush=True)

    with Pool(48) as pool:
        out = [r for r in pool.imap_unordered(unit_metrics, tasks, chunksize=1)
               if r]
    df = pd.DataFrame(out)
    df.to_csv("elem_prf1_gpt.csv", index=False)
    print(f"\nn={len(df)}")
    print(df[["prec", "rec", "f1", "h_loc", "elem_prec_all"]].agg(
        ["mean", "median"]).round(4))


if __name__ == "__main__":
    main()
