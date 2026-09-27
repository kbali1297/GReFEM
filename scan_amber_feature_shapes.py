#!/usr/bin/env python
"""Find AMBER training data points whose vertex features have inconsistent
shapes.

mesh_to_graph.py builds vertex features as
    np.array([degree, sizing_field]).T
with
    degree       = np.unique(mesh_edges, return_counts=True)[1]   # vertices in edges
    sizing_field = get_sizing_field(..., "vertex")                 # nvertices
These lengths differ whenever a mesh has vertices that appear in no edge
(unreferenced / orphan vertices), which raises
"setting an array element with a sequence ... inhomogeneous shape".

Usage:
    python scan_amber_feature_shapes.py --split train --workers 40
"""
import argparse
import functools
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/data/1bali/GReFEM/baselines/AMBER")

print = functools.partial(print, flush=True)

DATA = "/data/1bali/GReFEM/baselines/AMBER/data/grefem"


def check(args):
    idx, split = args
    from src.mesh_util.load_mesh import load_expert_mesh
    from src.mesh_util.sizing_field_util import get_sizing_field
    from src.tasks.expert_geometry_dataset_preparator import mesh_from_geometry_fn
    from src.tasks.domains.gmsh_util import geom_fn_from_file, get_bounding_box
    from src.tasks.domains.mesh_wrapper import MeshWrapper

    stem = os.path.join(DATA, split, f"{idx:03d}")
    out = dict(idx=idx, status="ok")
    try:
        gf = geom_fn_from_file(stem + ".step")
        bb = get_bounding_box(geometry_fn=gf)
        dim = len(bb) // 2
        ext = np.asarray(bb[dim:], float) - np.asarray(bb[:dim], float)
        m = mesh_from_geometry_fn(
            geometry_fn=gf,
            max_initial_element_volume=float(np.prod(ext)) / 3000.0, dim=dim)
        w = MeshWrapper(m)
        deg = np.unique(w.mesh_edges, return_counts=True)[1]
        sf = get_sizing_field(mesh=w, mesh_node_type="vertex")
        out.update(nverts=int(m.p.shape[1]), n_deg=int(len(deg)),
                   n_sf=int(np.shape(sf)[0]), sf_ndim=int(np.ndim(sf)),
                   n_nan_sf=int(np.isnan(np.asarray(sf)).sum()))
        if len(deg) != np.shape(sf)[0]:
            out["status"] = "SHAPE_MISMATCH"
        elif out["n_nan_sf"]:
            out["status"] = "NAN_SIZING_FIELD"
    except (Exception, SystemExit) as e:
        out["status"] = f"error: {type(e).__name__}: {str(e)[:60]}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    tasks = [(i, args.split) for i in range(1, args.n + 1)]
    bad = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(check, tasks, chunksize=2)):
            if r["status"] != "ok":
                bad.append(r)
                print(f"[{r['status']}] idx={r['idx']} "
                      f"nverts={r.get('nverts')} n_deg={r.get('n_deg')} "
                      f"n_sf={r.get('n_sf')} nan={r.get('n_nan_sf')}")
            if (k + 1) % 100 == 0:
                print(f"  scanned {k+1}/{len(tasks)}, bad so far {len(bad)}")
    print(f"\ntotal bad: {len(bad)}/{len(tasks)}")
    if bad:
        idxs = sorted(r["idx"] for r in bad)
        print("bad indices:", idxs)
        open(f"amber_bad_{args.split}.txt", "w").write(
            "\n".join(str(i) for i in idxs) + "\n")


if __name__ == "__main__":
    main()
