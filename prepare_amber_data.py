#!/usr/bin/env python
"""Convert GReFEM (object, load_case) units into AMBER's expert-geometry
dataset layout.

AMBER (Freymuth et al., NeurIPS 2025) learns a sizing field by regressing the
local element size of an *adaptive expert mesh*. Our ZZ-oracle refined meshes
are exactly such experts (adaptive, error-indicator driven); the fine2
reference is uniformly refined and therefore carries no sizing information.

Each data point becomes a triplet, matching AMBER's Mold task layout:
    NNN.brep            normalized geometry (longest bbox side -> 1)
    NNN.vtk             normalized expert mesh (ZZ-oracle refined)
    NNN_features.txt    load-case one-hot (5 floats, one per line)

Geometry is rescaled per object because AMBER has no scale normalization
(gmsh_util.py:39 is commented out) and uses one fixed max_initial_element_volume
for the whole dataset; raw ABC parts span ~1e8 in bounding-box volume.

Usage:
    python prepare_amber_data.py --out_dir /data/1bali/GReFEM/baselines/AMBER/data/grefem_rod
"""
import argparse
import functools
import glob
import json
import os

import meshio
import numpy as np

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
SRC_DIR = {"orig": "test_meshes", "extra": "test_meshes_extra"}
ORACLE = "{lc}_zz_oracle_fine2_mesh_refined.msh"


def normalize_geometry(step_path, out_brep):
    """Import STEP, scale so the longest bbox side is 1, write .brep.
    Returns (scale, original_extents)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("norm")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        bb = gmsh.model.getBoundingBox(-1, -1)
        ext = np.array(bb[3:]) - np.array(bb[:3])
        scale = 1.0 / float(ext.max())
        gmsh.model.occ.dilate(gmsh.model.getEntities(3), 0, 0, 0,
                              scale, scale, scale)
        gmsh.model.occ.synchronize()
        os.makedirs(os.path.dirname(os.path.abspath(out_brep)), exist_ok=True)
        gmsh.write(out_brep)
    finally:
        gmsh.finalize()
    return scale, ext


def write_expert_mesh(msh_path, out_vtk, scale):
    """Read gmsh expert mesh, keep tets only, scale vertices, write vtk."""
    m = meshio.read(msh_path)
    tets = np.concatenate([b.data for b in m.cells if b.type == "tetra"])
    pts = np.asarray(m.points, float) * scale
    meshio.write_points_cells(out_vtk, pts, [("tetra", tets)])
    return len(pts), len(tets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pool_json", default="amber_train_pool.json")
    ap.add_argument("--n_val_objects", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = json.load(open(args.pool_json))
    # objects with all 5 oracle labels, deterministic order
    objs = sorted(o for o, v in pool.items() if len(v["oracle"]) == 5)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(objs))
    val_objs = {objs[i] for i in perm[:args.n_val_objects]}
    print(f"{len(objs)} labelled objects -> "
          f"{len(objs) - len(val_objs)} train / {len(val_objs)} val")

    manifest = {"train": [], "val": []}
    counters = {"train": 0, "val": 0}
    geom_cache = {}

    for obj in objs:
        src = pool[obj]["src"]
        split = "val" if obj in val_objs else "train"
        # pool json stores the source directory name directly
        obj_dir = os.path.join(SRC_DIR.get(src, src), obj)
        steps = glob.glob(os.path.join(obj_dir, "*.step"))
        if not steps:
            print(f"[skip] {obj}: no STEP")
            continue
        for lc in LCS:
            oracle = os.path.join(obj_dir, "refined_mesh", ORACLE.format(lc=lc))
            if not os.path.exists(oracle):
                continue
            counters[split] += 1
            idx = counters[split]
            stem = os.path.join(args.out_dir, split, f"{idx:03d}")
            try:
                if obj not in geom_cache:
                    scale, ext = normalize_geometry(steps[0], stem + ".brep")
                    geom_cache[obj] = (scale, ext, stem + ".brep")
                else:
                    scale, ext, first_brep = geom_cache[obj]
                    # same geometry reused across load cases
                    os.link(first_brep, stem + ".brep") \
                        if not os.path.exists(stem + ".brep") else None
                nv, nt = write_expert_mesh(oracle, stem + ".vtk", scale)
                onehot = [1.0 if lc == l else 0.0 for l in LCS]
                with open(stem + "_features.txt", "w") as f:
                    f.write("\n".join(f"{v}" for v in onehot) + "\n")
                manifest[split].append(dict(
                    idx=idx, object=obj, load_case=lc, src=src, scale=scale,
                    orig_extents=ext.tolist(), expert_msh=oracle,
                    step=steps[0], n_verts=nv, n_tets=nt))
                print(f"[{split} {idx:03d}] {obj}/{lc}: scale {scale:.4g}, "
                      f"expert {nt} tets")
            except Exception as e:
                counters[split] -= 1
                print(f"[fail] {obj}/{lc}: {type(e).__name__}: {e}")

    # AMBER's config wants a test split; point it at val (our real evaluation
    # runs through infer_amber_grefem.py on the held-out GReFEM units).
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(args.out_dir, "val", "*"))):
        dst = os.path.join(test_dir, os.path.basename(p))
        if not os.path.exists(dst):
            os.link(p, dst)

    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"),
              indent=1)
    print(f"\nwrote {counters['train']} train / {counters['val']} val "
          f"data points to {args.out_dir}")
    print(f"num_data_points: train={counters['train']} "
          f"val={counters['val']} test={counters['val']}")


if __name__ == "__main__":
    main()
