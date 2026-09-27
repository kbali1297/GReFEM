#!/usr/bin/env python
"""Pre-validate that AMBER can build its initial coarse mesh for an object.

AMBER prepares its dataset eagerly and generates the initial uniform mesh via
pygmsh; a single un-meshable ABC part aborts the whole training run. Some ABC
geometries fail this path ("invalid boundary mesh", "not a closed loop", ...)
even though our own mesher handles them. This script tests each candidate
object and appends the verdict to a CSV, so prepare_amber_layout.py can keep
only objects that are known to work.

Usage (shardable):
    python validate_amber_objects.py --shard 0 --nshards 80 \
        --out_csv amber_valid_shards/v0.csv
"""
import argparse
import csv
import functools
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/data/1bali/GReFEM/baselines/AMBER")

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
CAD_DIR = "ABC_CAD_Dataset_small2_augmented"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts_dir", default="ml_baseline/amber_experts")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--divisor", type=float, default=3000.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from src.tasks.expert_geometry_dataset_preparator import mesh_from_geometry_fn
    from src.tasks.domains.gmsh_util import geom_fn_from_file, get_bounding_box

    have = {}
    for p in glob.glob(os.path.join(args.experts_dir, "*.msh")):
        stem = os.path.basename(p)[:-4]
        for lc in sorted(LCS, key=len, reverse=True):
            if stem.endswith("_" + lc):
                have.setdefault(stem[: -(len(lc) + 1)], set()).add(lc)
                break
    objs = sorted(o for o, s in have.items() if len(s) == 5)
    objs = [o for i, o in enumerate(objs) if i % args.nshards == args.shard]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object", "n_tets", "bbox_diagonal",
                                          "status"])
        w.writeheader()
        for obj in objs:
            steps = glob.glob(os.path.join(CAD_DIR, obj, "*.step"))
            row = {"object": obj}
            if not steps:
                row["status"] = "no_step"
                w.writerow(row); f.flush(); continue
            try:
                gf = geom_fn_from_file(steps[0])
                bb = get_bounding_box(geometry_fn=gf)
                dim = len(bb) // 2
                ext = np.asarray(bb[dim:], float) - np.asarray(bb[:dim], float)
                vol = float(np.prod(ext)) / args.divisor
                m = mesh_from_geometry_fn(geometry_fn=gf,
                                          max_initial_element_volume=vol,
                                          dim=dim)
                row.update(n_tets=int(m.t.shape[1]),
                           bbox_diagonal=float(np.linalg.norm(ext)),
                           status="ok")
                print(f"[ok] {obj}: {m.t.shape[1]} tets")
            except (Exception, SystemExit) as e:
                row["status"] = f"fail: {type(e).__name__}: {str(e)[:80]}"
                print(f"[fail] {obj}: {str(e)[:80]}")
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
