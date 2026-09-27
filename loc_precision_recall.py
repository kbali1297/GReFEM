"""Anchor localization precision/recall vs ground-truth ZZ hotspots.

Ground truth: top-0.1% ZZ points of the fine2 reference (same
read_pos_points_top + boundary trimming as the QoI tables).
Match radius: crit_radius (= 4*h_min) taken from combined_200obj_table.csv.

precision = frac of anchors within r of >=1 GT hotspot
recall    = frac of GT hotspots within r of >=1 anchor

Methods: grefem_max / grefem_mid / grefem_none anchors (prefilt npy) and
heuristic_sub = DENSE dihedral anchors subsampled with rng(0) to the median
GReFEM anchor count (exact replication of compute_local_error_tables.py).
Full-density heuristic also reported.
"""
import glob
import os
import re as _re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from multiprocessing import Pool

ZZ_TOP = 99.9
VARIANTS = ["maxprompt", "midprompt", "noneprompt"]


def read_pos_points_top(pos_file, top_percentile=99.9, load_case=None,
                        remove_boundary_frac=0.02):
    """Verbatim copy of fem_fenics.read_pos_points_top (avoids MPI import)."""
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
        if len(values) == 0:
            return np.empty((0, 3)), np.empty((0,))
    cutoff = np.percentile(values, top_percentile)
    mask = values >= cutoff
    return points[mask], values[mask]


def unit_metrics(args):
    obj, lc, src, r = args
    base = "test_meshes_extra" if src == "extra" else "test_meshes"
    d = os.path.join(base, obj)
    try:
        gt, _ = read_pos_points_top(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), ZZ_TOP, load_case=lc)
        if len(gt) == 0:
            return None
        gt_tree = cKDTree(gt)

        anchors = {}
        counts = []
        for v in VARIANTS:
            g = glob.glob(os.path.join(
                d, f"{lc}_gemini*_geo_{v}_ortho_5views_11grid_1run",
                "refinement_points_prefilt.npy"))
            if not g:
                continue
            pts = np.load(g[0])
            if len(pts):
                anchors[f"grefem_{v[:-6] if v != 'noneprompt' else 'none'}"] = pts
                counts.append(len(pts))
        if not counts:
            return None

        bl = np.load(os.path.join(
            d, "refinement_points_baseline_DENSE_dihedral30.0deg.npy"))
        if len(bl) == 0:
            return None
        anchors["heuristic"] = bl
        n_sub = int(np.median(counts))
        if n_sub < len(bl):
            rng = np.random.default_rng(0)
            anchors["heuristic_sub"] = bl[rng.choice(len(bl), size=n_sub,
                                                     replace=False)]
        else:
            anchors["heuristic_sub"] = bl

        rows = []
        for m, pts in anchors.items():
            d_a2g, _ = gt_tree.query(pts)          # anchor -> nearest GT
            d_g2a, _ = cKDTree(pts).query(gt)      # GT -> nearest anchor
            prec = float(np.mean(d_a2g <= r))
            rec = float(np.mean(d_g2a <= r))
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            rows.append(dict(object=obj, load_case=lc, method=m,
                             n_anchors=len(pts), n_gt=len(gt), r=r,
                             precision=prec, recall=rec, f1=f1,
                             med_gt_dist_r=float(np.median(d_g2a) / r)))
        return rows
    except Exception as e:
        print(f"[fail] {obj}/{lc}: {e}", flush=True)
        return None


def main():
    df = pd.read_csv("combined_200obj_table.csv", dtype={"object": str})
    ok = df[(df.status == "ok") & df.method.isin(["grefem_max",
                                                  "heuristic_sub"])]
    units = (ok.groupby(["object", "load_case"])
               .filter(lambda g: g.method.nunique() == 2)
               .drop_duplicates(["object", "load_case"])
             [["object", "load_case", "src", "crit_radius"]])
    tasks = [tuple(t) for t in units.itertuples(index=False)]
    print(f"units: {len(tasks)}", flush=True)

    with Pool(64) as pool:
        out = []
        for i, res in enumerate(pool.imap_unordered(unit_metrics, tasks,
                                                    chunksize=4)):
            if res:
                out.extend(res)
            if (i + 1) % 200 == 0:
                print(f"done {i+1}/{len(tasks)}", flush=True)

    res = pd.DataFrame(out)
    res.to_csv("loc_precision_recall.csv", index=False)
    print(f"\nrows: {len(res)}  units ok: "
          f"{res.groupby(['object','load_case']).ngroups}")

    agg = dict(precision_mean=("precision", "mean"),
               precision_med=("precision", "median"),
               recall_mean=("recall", "mean"),
               recall_med=("recall", "median"),
               f1_mean=("f1", "mean"), f1_med=("f1", "median"),
               n_anchors=("n_anchors", "median"),
               med_gt_dist_r=("med_gt_dist_r", "median"),
               n=("precision", "size"))
    print("\n===== POOLED =====")
    print(res.groupby("method").agg(**agg).round(3).to_string())
    for lc in sorted(res.load_case.unique()):
        print(f"\n===== {lc} =====")
        print(res[res.load_case == lc].groupby("method").agg(**agg)
              .round(3).to_string())

    # paired grefem_max vs heuristic_sub
    piv = res.pivot_table(index=["object", "load_case"], columns="method",
                          values=["precision", "recall", "f1"])
    from scipy.stats import wilcoxon
    print("\n===== PAIRED grefem_max vs heuristic_sub =====")
    for metric in ["precision", "recall", "f1"]:
        a = piv[(metric, "grefem_max")]
        b = piv[(metric, "heuristic_sub")]
        m = a.notna() & b.notna()
        a, b = a[m], b[m]
        diff = a - b
        try:
            p = wilcoxon(a, b).pvalue
        except ValueError:
            p = float("nan")
        print(f"{metric:9s}: grefem win {np.mean(diff > 0):.1%} | tie "
              f"{np.mean(diff == 0):.1%} | med_diff {diff.median():+.4f} | "
              f"mean g={a.mean():.3f} h_sub={b.mean():.3f} | p={p:.2e} | "
              f"n={m.sum()}")


if __name__ == "__main__":
    main()
