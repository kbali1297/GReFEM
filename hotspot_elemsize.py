"""Decisive mechanism test: LOCAL ELEMENT SIZE AT TOP ZZ HOTSPOTS.

For each of the 756 rebuttal units, take GT = top-0.1% fine-solve ZZ points
(5% boundary trim to suppress load-application concentrations) and measure, in
each policy's ACTUAL refined mesh (hs1.0, matched budgets):
  - h_loc: median element size (cbrt of tet volume) of tets whose centroid
    lies within r = crit_radius of any GT hotspot point;
  - n_loc: number of such tets (elements spent at the hotspots);
  - n_tot: total tets.
Policies: grefem (gemini max), blind_sub, mech_sub.

If GReFEM has SMALLER h_loc / MORE n_loc at hotspots despite lower anchor
precision, then low precision != under-refining hotspots, and the budget-
dilution mechanism is confirmed.

Output: hotspot_elemsize.csv + summary in log.
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


def read_msh_tets(path):
    """Minimal Gmsh 4.1 / 2.2 ASCII reader returning (nodes, tets)."""
    import meshio
    m = meshio.read(path)
    tets = None
    for cb in m.cells:
        if cb.type == "tetra":
            tets = cb.data if tets is None else np.vstack([tets, cb.data])
    return m.points, tets


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

        rows = []
        for meth, p in paths.items():
            nodes, tets = read_msh_tets(p)
            if tets is None or len(tets) == 0:
                continue
            v0 = nodes[tets[:, 0]]
            cent = (v0 + nodes[tets[:, 1]] + nodes[tets[:, 2]]
                    + nodes[tets[:, 3]]) / 4.0
            a = nodes[tets[:, 1]] - v0
            b = nodes[tets[:, 2]] - v0
            c = nodes[tets[:, 3]] - v0
            vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
            dd, _ = gt_tree.query(cent)
            near = dd <= r
            h_loc = float(np.median(np.cbrt(vol[near]))) if near.any() else np.nan
            rows.append(dict(object=obj, load_case=lc, method=meth, r=r,
                             n_gt=len(gt), n_tot=len(tets),
                             n_loc=int(near.sum()), h_loc=h_loc,
                             h_glob=float(np.median(np.cbrt(vol)))))
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
            if (i + 1) % 25 == 0:
                print(f"done {i+1}/{len(tasks)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("hotspot_elemsize.csv", index=False)

    piv_h = df.pivot_table(index=["object", "load_case"], columns="method",
                           values="h_loc")
    piv_n = df.pivot_table(index=["object", "load_case"], columns="method",
                           values="n_loc")
    print("\n=== median local element size at top-0.1% ZZ hotspots ===",
          flush=True)
    print(piv_h.median().round(3), flush=True)
    print("\n=== median #elements within r of hotspots ===", flush=True)
    print(piv_n.median().round(0), flush=True)
    both = piv_h.dropna(subset=["grefem", "blind_sub"])
    frac = (both.grefem < both.blind_sub).mean()
    ratio = (both.grefem / both.blind_sub).median()
    print(f"\npaired units (grefem vs blind_sub): {len(both)}", flush=True)
    print(f"grefem smaller hotspot elements in {frac:.1%} of units; "
          f"median size ratio grefem/blind = {ratio:.3f}", flush=True)
    bn = piv_n.dropna(subset=["grefem", "blind_sub"])
    fr_n = (bn.grefem > bn.blind_sub).mean()
    rt_n = (bn.grefem / bn.blind_sub.replace(0, np.nan)).median()
    print(f"grefem MORE hotspot elements in {fr_n:.1%} of units; "
          f"median count ratio = {rt_n:.3f}", flush=True)


if __name__ == "__main__":
    main()
