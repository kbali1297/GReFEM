"""Element-level P/R/F1: fine elements (h <= 1.25*h_min) vs top-0.1% ZZ hotspots.
Precision: fraction of fine elements within r of a GT hotspot point.
Recall:    fraction of GT hotspot points within r of a fine element centroid.
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


def mesh_paths(d, lc):
    out = {}
    g = glob.glob(os.path.join(
        d, "refined_mesh",
        f"{lc}_gemini*_geo_maxprompt_ortho_5views_11grid_1run_refined.msh"))
    if g:
        out["grefem"] = g[0]
    p = os.path.join(
        d, "refined_mesh",
        f"{lc}_heuristic_baseline_DENSE_dihedral30.0deg_sub_fine2_mesh_refined.msh")
    if os.path.exists(p):
        out["blind_sub"] = p
    p = os.path.join(
        d, "refined_mesh",
        f"{lc}_heuristic_mech_dihedral30.0deg_{lc}_sub_fine2_mesh_refined.msh")
    if os.path.exists(p):
        out["mech_sub"] = p
    return out


def unit_metrics(args):
    obj, lc, r = args
    d = find_base(obj, lc)
    if d is None:
        return None
    try:
        import meshio
        paths = mesh_paths(d, lc)
        if "grefem" not in paths or "blind_sub" not in paths:
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

        rows = []
        for meth, p in paths.items():
            m = meshio.read(p)
            tets = None
            for cb in m.cells:
                if cb.type == "tetra":
                    tets = cb.data if tets is None else np.vstack([tets, cb.data])
            if tets is None or len(tets) == 0:
                continue
            nodes = m.points
            v0 = nodes[tets[:, 0]]
            cent = (v0 + nodes[tets[:, 1]] + nodes[tets[:, 2]]
                    + nodes[tets[:, 3]]) / 4.0
            a = nodes[tets[:, 1]] - v0
            b = nodes[tets[:, 2]] - v0
            c = nodes[tets[:, 3]] - v0
            vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
            h = np.cbrt(vol)
            fine = h <= h_cut
            n_fine = int(fine.sum())
            if n_fine == 0:
                rows.append(dict(object=obj, load_case=lc, method=meth, r=r,
                                 n_fine=0, prec=np.nan, rec=0.0, f1=0.0,
                                 n_gt=len(gt)))
                continue
            fc = cent[fine]
            dd, _ = gt_tree.query(fc)
            prec = float(np.mean(dd <= r))
            f_tree = cKDTree(fc)
            dg, _ = f_tree.query(gt)
            rec = float(np.mean(dg <= r))
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows.append(dict(object=obj, load_case=lc, method=meth, r=r,
                             n_fine=n_fine, prec=prec, rec=rec, f1=f1,
                             n_gt=len(gt)))
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

    with Pool(48) as pool:
        out = []
        for i, res in enumerate(pool.imap_unordered(unit_metrics, tasks,
                                                    chunksize=1)):
            if res:
                out.extend(res)
            if (i + 1) % 50 == 0:
                print(f"done {i+1}/{len(tasks)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("elem_prf1.csv", index=False)

    print("\n=== ELEMENT-level P / R / F1 (fine elems h<=1.25*h_min vs "
          "top-0.1% ZZ hotspots, radius r) ===", flush=True)
    print(df.groupby("method")[["prec", "rec", "f1"]].agg(
        ["mean", "median"]).round(4), flush=True)
    p = df.pivot_table(index=["object", "load_case"], columns="method",
                       values="f1")
    ok = p.dropna(subset=["grefem", "blind_sub"])
    print(f"\npaired F1 grefem vs blind_sub ({len(ok)}): grefem higher in "
          f"{(ok.grefem > ok.blind_sub).mean():.1%}", flush=True)
    if "mech_sub" in p:
        okm = p.dropna(subset=["grefem", "mech_sub"])
        print(f"paired F1 grefem vs mech ({len(okm)}): grefem higher in "
              f"{(okm.grefem > okm.mech_sub).mean():.1%}", flush=True)
    pr = df.pivot_table(index=["object", "load_case"], columns="method",
                        values="rec")
    okr = pr.dropna(subset=["grefem", "blind_sub", "mech_sub"])
    print(f"\npaired recall grefem>blind in {(okr.grefem > okr.blind_sub).mean():.1%}, "
          f"grefem>mech in {(okr.grefem > okr.mech_sub).mean():.1%} "
          f"({len(okr)} units)", flush=True)


if __name__ == "__main__":
    main()
