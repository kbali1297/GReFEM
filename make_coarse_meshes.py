#!/usr/bin/env python
"""Generate coarse_mesh_{lc}.msh for new objects (uniform h_max sizing,
same compute_h_sizes convention), one gmsh call per object, copied to all
load-case filenames. Resumable."""
import argparse, os, shutil, functools
from compute_local_error_tables import compute_h_sizes, count_tets
from utils import generate_or_refine_mesh
print = functools.partial(print, flush=True)
LCS = ['bending','compression','torsion','bending_compression','torsion_compression']

ap = argparse.ArgumentParser()
ap.add_argument('--test_dir', required=True)
ap.add_argument('--objects_file', required=True)
args = ap.parse_args()
objs = [l.strip() for l in open(args.objects_file) if l.strip()]
for i, o in enumerate(objs):
    od = os.path.join(args.test_dir, o)
    tgts = [os.path.join(od, f'coarse_mesh_{lc}.msh') for lc in LCS]
    if all(os.path.exists(t) and count_tets(t) > 0 for t in tgts):
        print(f'[skip] {o}'); continue
    try:
        h_max, h_min, step = compute_h_sizes(od)
        base = tgts[0]
        generate_or_refine_mesh(step_or_mesh_path=step, points_of_interest=[],
                                h_min=h_min, h_max=h_max, suffix=None,
                                verbose=False, out_msh=base)
        n = count_tets(base)
        if n == 0:
            print(f'[fail] {o}: no tets'); continue
        for t in tgts[1:]:
            shutil.copy(base, t)
        print(f'[ok] {i+1}/{len(objs)} {o}: {n} tets (h_max {h_max:.3f})')
    except Exception as e:
        print(f'[fail] {o}: {e}')
