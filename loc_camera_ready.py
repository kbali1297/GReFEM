#!/usr/bin/env python
"""Anchor-level localization P/R/F1 on the 156-object / 780-case benchmark
for all 12 GReFEM backend x prompt variants (5 orthoViews) and the baselines.

GT per case   : fine-reference ZZ points >= 99th percentile (2% boundary trim),
                as in the submission's localization metric.
Match radius  : r = crit_radius = 4 h_min.
Anchor sets   : MLLM variants native (refinement_points_prefilt.npy);
                blind heuristic native and anchor-matched to GReFEM-Gemini
                (L+F+H) per case; load-informed heuristic native.
Cells/view    : parsed from each run's inference.log.

Writes loc_camera_ready.csv (per case) and prints macro/micro summaries.
"""
import ast
import glob
import os

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree

from loc_prf1_156_p99 import crit_radius_map, find_base, read_pos_points_all

LOADS = ["compression", "bending", "torsion",
         "bending_compression", "torsion_compression"]
MODELS = {"gemini-3-flash-preview": "Gemini", "gpt-5.4-mini": "GPT-5.4-mini",
          "claude-haiku-4.5": "Claude-Haiku-4.5",
          "qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B"}
PROMPTS = {"maxprompt": "L+F+H", "midprompt": "L+F", "noneprompt": "L"}


def cells_per_view(log, prompt):
    try:
        with open(log) as f:
            for line in f:
                if line.startswith("Identified cell numbers:"):
                    d = ast.literal_eval(line.split(":", 1)[1].strip())
                    vals = list(d.values())
                    if prompt == "L":  # Load-Only: every key holds the same list
                        vals = vals[:1]
                    tot = np.sum([np.sum(v) for v in vals])
                    return float(tot) / max(len(vals[0]), 1)
    except OSError:
        pass
    return np.nan


def counts(a, gt, gt_tree, r):
    d_a2g, _ = gt_tree.query(a)
    d_g2a, _ = cKDTree(a).query(gt)
    return int(np.sum(d_a2g <= r)), int(np.sum(d_g2a <= r))


def unit(args):
    obj, lc, r = args
    d = find_base(obj, lc)
    if d is None:
        return None
    try:
        pts, vals = read_pos_points_all(
            os.path.join(d, f"fine2_mesh_{lc}_zz.pos"), load_case=lc)
        if len(vals) == 0:
            return None
        gt = pts[vals >= np.percentile(vals, 99.0)]
        gt_tree = cKDTree(gt)
        rows = []

        def add(method, prompt, a, cpv=np.nan):
            ma, mg = counts(a, gt, gt_tree, r)
            rows.append(dict(object=obj, load_case=lc, method=method,
                             prompt=prompt, r=r, n_anchors=len(a),
                             matched_anchors=ma, n_gt=len(gt), matched_gt=mg,
                             cells_per_view=cpv))

        n_ref = None
        for mk, mname in MODELS.items():
            for pk, pname in PROMPTS.items():
                run = glob.glob(os.path.join(
                    d, f"{lc}_{mk}*_geo_{pk}_ortho_5views_11grid_1run"))
                if not run:
                    continue
                f = os.path.join(run[0], "refinement_points_prefilt.npy")
                if not os.path.exists(f):
                    continue
                a = np.load(f)
                if len(a) == 0:
                    continue
                if mname == "Gemini" and pname == "L+F+H":
                    n_ref = len(a)
                add(mname, pname, a, cells_per_view(
                    os.path.join(run[0], "inference.log"), pname))

        bl_p = os.path.join(d, "refinement_points_baseline_DENSE_dihedral30.0deg.npy")
        if os.path.exists(bl_p):
            bl = np.load(bl_p)
            if len(bl):
                add("heuristic_native", "", bl)
                if n_ref:
                    rng = np.random.default_rng(2)
                    sub = bl if len(bl) <= n_ref else bl[
                        rng.choice(len(bl), n_ref, replace=False)]
                    add("heuristic_matched", "", sub)
        mech_p = os.path.join(d, f"refinement_points_mech_dihedral30.0deg_{lc}.npy")
        if os.path.exists(mech_p):
            mech = np.load(mech_p)
            if len(mech):
                add("mech", "", mech)
        return rows
    except Exception as e:
        print(f"[fail] {obj}/{lc}: {e}", flush=True)
        return None


def summarize(df):
    d = df.copy()
    d["P"] = d.matched_anchors / d.n_anchors
    d["R"] = d.matched_gt / d.n_gt
    d["F1"] = np.where(d.P + d.R > 0, 2 * d.P * d.R / (d.P + d.R), 0.0)
    g = d.groupby(["method", "prompt"])
    out = g[["P", "R", "F1"]].mean()
    out["pts_k"] = g.n_anchors.mean() / 1000
    out["cells_view"] = g.cells_per_view.mean()
    out["n"] = g.size()
    return out.round(4)


def main():
    objs = []
    for f in ["final_dense_objects.txt", "final_orig_objects.txt",
              "final_winners_objects.txt"]:
        objs += [x.strip() for x in open(f) if x.strip()]
    objs = sorted(set(objs))
    rmap = crit_radius_map()
    tasks = [(o, lc, rmap[(o, lc)]) for o in objs for lc in LOADS
             if (o, lc) in rmap]
    print(f"objects: {len(objs)}  cases: {len(tasks)}", flush=True)
    out = []
    with Pool(96) as pool:
        for i, res in enumerate(pool.imap_unordered(unit, tasks, chunksize=1)):
            if res:
                out.extend(res)
            if (i + 1) % 100 == 0:
                print(f"done {i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv("loc_camera_ready.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\ncases: {df.groupby(['object', 'load_case']).ngroups}")
    print("\n===== MACRO, all loads =====")
    print(summarize(df).to_string())
    for lc in LOADS:
        print(f"\n===== MACRO {lc} =====")
        print(summarize(df[df.load_case == lc]).to_string())


if __name__ == "__main__":
    main()
