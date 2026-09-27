#!/usr/bin/env python
"""Generate + solve finer fine-reference meshes (fine2_mesh_*) and their
ZZ fields, so the QoI floor (h_floor) drops and budget-matched candidates
(oracle/heuristic) are no longer clamped by the reference resolution.

Writes per object/load_case:
    fine2_mesh_{lc}.msh, fine2_mesh_{lc}_sol.xdmf(+h5), fine2_mesh_{lc}_zz.pos
Resumable: skips outputs that already exist.
"""
import argparse
import os
import time

os.environ.setdefault("OPAL_PREFIX", "/data/1bali/miniforge3/envs/multi_view_3DQA")
os.environ.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")

import meshio
import numpy as np

from compute_local_error_tables import compute_h_sizes, count_tets
from fem_fenics import _SOLVERS, return_zz_field
from utils import generate_or_refine_mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", default="test_meshes")
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--load_cases", nargs="+", default=["bending", "torsion"])
    ap.add_argument("--fine_factor", type=float, default=16.0,
                    help="h_fine = h_max / fine_factor (original refs used 10).")
    ap.add_argument("--max_elements", type=float, default=6e6,
                    help="Cap on fine-mesh tets (original refs used 2e6).")
    ap.add_argument("--prefix", default="fine2_mesh")
    ap.add_argument("--E", type=float, default=210e9)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--drop_strain_stress", action="store_true",
                    help="Delete the (large) stress/strain xdmf+h5 outputs "
                         "after solving; only u + zz pos are kept.")
    args = ap.parse_args()

    tasks = [(o, lc) for o in args.objects for lc in args.load_cases]
    n_total, n_done = len(tasks), 0
    print(f"[progress] {n_done}/{n_total} objects done", flush=True)

    for obj in args.objects:
        obj_dir = os.path.join(args.test_dir, obj)
        try:
            h_max, _, step_path = compute_h_sizes(obj_dir)
        except Exception as e:
            print(f"[skip] {obj}: h size computation failed: {e}", flush=True)
            n_done += len(args.load_cases)
            print(f"[progress] {n_done}/{n_total} objects done", flush=True)
            continue

        for lc in args.load_cases:
            t0 = time.time()
            msh = os.path.join(obj_dir, f"{args.prefix}_{lc}.msh")
            sol = msh.replace(".msh", "_sol.xdmf")
            pos = msh.replace(".msh", "_zz.pos")
            if (os.path.exists(sol) and os.path.exists(sol.replace(".xdmf", ".h5"))
                    and os.path.exists(pos)):
                print(f"[skip] {obj}/{lc}: reference already solved", flush=True)
                n_done += 1
                print(f"[progress] {n_done}/{n_total} objects done", flush=True)
                continue

            # ---- mesh (uniform h_fine, capped element count) ----
            h_fine = h_max / args.fine_factor
            if not (os.path.exists(msh) and count_tets(msh) > 0):
                while True:
                    generate_or_refine_mesh(step_or_mesh_path=step_path,
                                            h_min=h_fine, h_max=h_fine,
                                            suffix=None, verbose=False,
                                            out_msh=msh)
                    n = count_tets(msh)
                    if n == 0:
                        raise RuntimeError(f"mesh gen failed for {obj}/{lc}")
                    print(f"[mesh] {obj}/{lc}: {n} tets (h_fine {h_fine:.4f})",
                          flush=True)
                    if n <= args.max_elements:
                        break
                    h_fine *= (n / args.max_elements) ** (1.0 / 3.0) * 1.10
                    print(f"[mesh] {obj}/{lc}: over cap, retry h_fine "
                          f"{h_fine:.4f}", flush=True)

            # ---- solve + ZZ recovery ----
            print(f"[solve] {obj}/{lc}: {count_tets(msh)} tets ...", flush=True)
            solver_fn = _SOLVERS[lc]
            u_xdmf, *_, uh = solver_fn(msh, E=args.E, nu=args.nu)
            return_zz_field(msh, uh, output_file=pos)
            if args.drop_strain_stress:
                base = msh[:-4]
                for suffix in ("_stress.xdmf", "_stress.h5",
                               "_strain.xdmf", "_strain.h5"):
                    p = base + suffix
                    if os.path.exists(p):
                        os.remove(p)
            print(f"[done] {obj}/{lc}: {u_xdmf} + {pos} "
                  f"({int(time.time()-t0)}s)", flush=True)
            n_done += 1
            print(f"[progress] {n_done}/{n_total} objects done", flush=True)


if __name__ == "__main__":
    main()
