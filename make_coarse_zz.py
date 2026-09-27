#!/usr/bin/env python
"""Recover the ZZ error field from the ALREADY-SOLVED coarse mesh, giving the
practical solver-informed baseline reviewers asked for ("coarse FEM solve +
ZZ-error-estimator refinement") as opposed to the fine-reference oracle.

Writes  <obj>/coarse_mesh_{lc}_zz.pos  (no new FEM solves: reads the cached
coarse_mesh_{lc}_sol.xdmf). Resumable.

Usage:
  python make_coarse_zz.py --test_dir test_meshes_dense --objects_file dense_cohort.txt
"""
import argparse
import functools
import os
import sys

os.environ.setdefault("OPAL_PREFIX", "/data/1bali/miniforge3/envs/multi_view_3DQA")
os.environ.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--objects_file", required=True)
    ap.add_argument("--load_cases", nargs="+", default=LCS)
    args = ap.parse_args()

    from fem_fenics import return_zz_field

    objs = [l.strip() for l in open(args.objects_file) if l.strip()]
    n_ok = n_skip = n_fail = 0
    for i, o in enumerate(objs):
        od = os.path.join(args.test_dir, o)
        for lc in args.load_cases:
            msh = os.path.join(od, f"coarse_mesh_{lc}.msh")
            sol = os.path.join(od, f"coarse_mesh_{lc}_sol.xdmf")
            pos = os.path.join(od, f"coarse_mesh_{lc}_zz.pos")
            if os.path.exists(pos):
                n_skip += 1
                continue
            if not (os.path.exists(msh) and os.path.exists(sol)):
                print(f"[skip] {o}/{lc}: no coarse mesh/solution")
                n_skip += 1
                continue
            try:
                return_zz_field(msh, sol, output_file=pos)
                n_ok += 1
                print(f"[ok] {o}/{lc} -> {os.path.basename(pos)}")
            except (Exception, SystemExit) as e:
                n_fail += 1
                print(f"[fail] {o}/{lc}: {e}")
        if (i + 1) % 20 == 0:
            print(f"[progress] {i+1}/{len(objs)} objects | ok {n_ok} "
                  f"skip {n_skip} fail {n_fail}")
    print(f"DONE: ok {n_ok}, skip {n_skip}, fail {n_fail}")


if __name__ == "__main__":
    main()
