#!/usr/bin/env python
"""Screen the ABC pool for feature-dense parts: run the exact DENSE dihedral
30deg detector (same as the heuristic baseline) on every candidate OBJ and
record the anchor count. Geometry-only, no solver — an a-priori selection
criterion for the feature-dense cohort. Writes dense_screen.csv."""
import os, sys, glob, csv
import numpy as np
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed
from infer_baseline import upsample_concave_edges, upsample_concave_surfaces
from utils import filter_points_by_density_fast

POOL = "ABC_CAD_Dataset_small2_augmented"
FEATURE_ANGLE = 30.0

exclude = set(os.path.basename(d.rstrip("/")) for pat in
              ["test_meshes/*/", "test_meshes_extra/*/"] for d in glob.glob(pat))

def one(obj_id):
    objs = glob.glob(os.path.join(POOL, obj_id, "*.obj"))
    if not objs:
        return obj_id, -2, 0.0
    try:
        mesh = trimesh.load_mesh(objs[0])
        bmin, bmax = mesh.bounds
        sp = np.linalg.norm(bmax - bmin) / (66.667 * 2)
        e = upsample_concave_edges(mesh, feature_angle=FEATURE_ANGLE, target_spacing=sp)
        e = filter_points_by_density_fast(e, sp * 0.75)
        s = upsample_concave_surfaces(mesh, feature_angle=FEATURE_ANGLE, target_spacing=sp)
        s = filter_points_by_density_fast(s, sp * 0.75)
        parts = [p for p in (e, s) if p.shape[0] > 0]
        n = len(np.unique(np.round(np.vstack(parts), 4), axis=0)) if parts else 0
        return obj_id, n, float(mesh.volume) if mesh.is_watertight else -1.0
    except Exception:
        return obj_id, -3, 0.0

if __name__ == "__main__":
    ids = sorted(d for d in os.listdir(POOL)
                 if os.path.isdir(os.path.join(POOL, d)) and d not in exclude)
    print(f"{len(ids)} candidates (after excluding {len(exclude)} existing)", flush=True)
    with open("dense_screen.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["object", "n_anchors", "volume"])
        with ProcessPoolExecutor(50) as ex:
            futs = {ex.submit(one, i): i for i in ids}
            for k, fu in enumerate(as_completed(futs)):
                w.writerow(fu.result()); f.flush()
                if (k + 1) % 500 == 0:
                    print(f"{k+1}/{len(ids)}", flush=True)
    print("done -> dense_screen.csv", flush=True)
