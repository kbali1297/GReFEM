#!/usr/bin/env python
"""Parallel coarse-mesh generation for a test dir (resumable: skips objects
whose 5 coarse_mesh_{lc}.msh already exist and are readable).

Usage: python make_coarse_meshes_par.py <test_dir> <objects_file> <workers>
"""
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/data/1bali/GReFEM")
LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]


def one(args):
    td, o = args
    from compute_local_error_tables import compute_h_sizes, count_tets
    from utils import generate_or_refine_mesh
    od = os.path.join(td, o)
    tgts = [os.path.join(od, f"coarse_mesh_{lc}.msh") for lc in LCS]
    if all(os.path.exists(t) and count_tets(t) > 0 for t in tgts):
        return f"[skip] {o}"
    try:
        h_max, h_min, step = compute_h_sizes(od)
        generate_or_refine_mesh(step_or_mesh_path=step, points_of_interest=[],
                                h_min=h_min, h_max=h_max, suffix=None,
                                verbose=False, out_msh=tgts[0])
        n = count_tets(tgts[0])
        if n == 0:
            return f"[fail] {o}: no tets"
        for t in tgts[1:]:
            shutil.copy(tgts[0], t)
        return f"[ok] {o}: {n} tets"
    except Exception as e:
        return f"[fail] {o}: {e}"


if __name__ == "__main__":
    td, objf, nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    objs = [l.strip() for l in open(objf) if l.strip()]
    with ProcessPoolExecutor(nw) as ex:
        for r in ex.map(one, [(td, o) for o in objs]):
            print(r, flush=True)
    print("COARSE DONE", flush=True)
