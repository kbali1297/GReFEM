#!/usr/bin/env python
"""Generate geometric-heuristic (dense dihedral) refinement points for the
extra test set. One npy per object, saved as
    <parent>/<obj>/refinement_points_baseline_DENSE_dihedral30.0deg.npy
so compute_local_error_tables.py can use --baseline_dir <parent>.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from tqdm import tqdm

from infer_baseline import upsample_concave_edges, upsample_concave_surfaces
from utils import filter_points_by_density_fast

FEATURE_ANGLE = 30.0
EXP_NAME = f"baseline_DENSE_dihedral{FEATURE_ANGLE}deg"


def process(parent, obj):
    out = os.path.join(parent, obj, f"refinement_points_{EXP_NAME}.npy")
    if os.path.exists(out):
        return obj, -1  # already done
    mesh_path = os.path.join(parent, obj, "renders_pyvista", f"{obj}.obj")
    mesh = trimesh.load_mesh(mesh_path)
    bbox_min, bbox_max = mesh.bounds
    target_spacing = np.linalg.norm(bbox_max - bbox_min) / (66.667 * 2)

    edge_pts = upsample_concave_edges(mesh, feature_angle=FEATURE_ANGLE,
                                      target_spacing=target_spacing)
    edge_pts = filter_points_by_density_fast(edge_pts, target_spacing * 0.75)
    surf_pts = upsample_concave_surfaces(mesh, feature_angle=FEATURE_ANGLE,
                                         target_spacing=target_spacing)
    surf_pts = filter_points_by_density_fast(surf_pts, target_spacing * 0.75)

    parts = [p for p in (edge_pts, surf_pts) if p.shape[0] > 0]
    if parts:
        pts = np.unique(np.round(np.vstack(parts), decimals=4), axis=0)
    else:
        pts = np.empty((0, 3))
    np.save(out, pts)
    return obj, len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="test_meshes_extra")
    ap.add_argument("--objects_file",
                    default="test_meshes_extra/final_objects.txt")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with open(args.objects_file) as fh:
        objects = [l.strip() for l in fh if l.strip()]
    print(f"{len(objects)} objects, {args.workers} workers", flush=True)

    counts, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, args.parent, o): o for o in objects}
        with tqdm(total=len(objects), file=sys.stdout, ascii=True,
                  mininterval=10, unit="obj") as bar:
            for fut in as_completed(futs):
                try:
                    obj, n = fut.result()
                    if n >= 0:
                        counts.append(n)
                except Exception as e:
                    failed.append(futs[fut])
                    print(f"\n[fail] {futs[fut]}: {e}", flush=True)
                bar.update(1)

    if counts:
        print(f"new point sets: {len(counts)} | median {np.median(counts):.0f}"
              f", min {min(counts)}, max {max(counts)}", flush=True)
    print(f"DONE. {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
