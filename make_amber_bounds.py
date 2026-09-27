#!/usr/bin/env python
"""Compute AMBER's sizing-field clip bounds and element cap from the training
expert meshes, so sharded inference workers do not each have to re-prepare the
full training dataset (which is eagerly built and takes minutes).

Replicates mesh_generation_algorithm.py exactly:
    min_sizing_field = (1/1.25) * min over train experts of min(sizing_field)
    max_sizing_field =    1.25  * max over train experts of max(sizing_field)
    max_mesh_elements = 1.5 * max over train experts of n_elements   ("auto")

Usage:
    python make_amber_bounds.py --data_dir /data/1bali/GReFEM/baselines/AMBER/data/grefem \
        --out amber_bounds.json
"""
import argparse
import functools
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/1bali/GReFEM/baselines/AMBER")

print = functools.partial(print, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir",
                    default="/data/1bali/GReFEM/baselines/AMBER/data/grefem")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="amber_bounds.json")
    ap.add_argument("--min_factor", type=float, default=1.25)
    ap.add_argument("--max_factor", type=float, default=1.25)
    ap.add_argument("--element_factor", type=float, default=1.5)
    args = ap.parse_args()

    from src.mesh_util.load_mesh import load_expert_mesh
    from src.mesh_util.sizing_field_util import get_sizing_field
    from src.tasks.domains.mesh_wrapper import MeshWrapper

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.split, "*.msh")))
    print(f"{len(paths)} expert meshes in {args.split}")
    mins, maxs, nels = [], [], []
    for i, p in enumerate(paths):
        try:
            m = MeshWrapper(load_expert_mesh(p))
            sf = get_sizing_field(m, mesh_node_type="element")
            mins.append(float(np.min(sf)))
            maxs.append(float(np.max(sf)))
            nels.append(int(m.num_elements))
        except Exception as e:
            print(f"[warn] {p}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)}")

    out = {
        "min_sizing_field": float(min(mins) / args.min_factor),
        "max_sizing_field": float(max(maxs) * args.max_factor),
        "max_mesh_elements": int(max(nels) * args.element_factor),
        "n_meshes": len(nels),
        "raw_min": float(min(mins)), "raw_max": float(max(maxs)),
        "max_expert_elements": int(max(nels)),
        "median_expert_elements": int(np.median(nels)),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
