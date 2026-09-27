#!/usr/bin/env python
"""Lay out the AMBER GReFEM dataset from the generated expert meshes.

Produces, per data point, in AMBER's expected <split>/<NNN>.<ext> form:
    NNN.step            symlink to the ORIGINAL ABC CAD geometry (not rescaled;
                        OCC scaling corrupts STEP tolerances and the geometry
                        then fails to mesh)
    NNN.msh             symlink to ml_baseline/amber_experts/<cad>_<lc>.msh
    NNN_features.txt    5x load-case one-hot + log10(bbox diagonal)

Splits are by OBJECT (never by load case), so no geometry appears in both
train and val. The evaluation cohort is untouched here: it is held out
entirely and only used at table time.

Usage:
    python prepare_amber_layout.py --n_train_objects 300 --n_val_objects 40
"""
import argparse
import functools
import csv
import glob
import json
import os
import shutil

import numpy as np

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
CAD_DIR = "ABC_CAD_Dataset_small2_augmented"


def bbox_diagonal(step_path):
    """Bounding-box diagonal of a STEP geometry (original units)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("bb")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        bb = gmsh.model.getBoundingBox(-1, -1)
    finally:
        gmsh.finalize()
    ext = np.array(bb[3:]) - np.array(bb[:3])
    return float(np.linalg.norm(ext))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valid_glob", default="amber_valid_shards/v*.csv",
                    help="CSVs from validate_amber_objects.py; only objects "
                         "whose initial mesh AMBER can build are used, and "
                         "their cached bbox diagonal is reused.")
    ap.add_argument("--max_expert_tets", type=int, default=0,
                    help="If >0, keep only objects whose largest expert mesh "
                         "has at most this many tets (see "
                         "--expert_size_csv). AMBER caps generated meshes at "
                         "1.5x the largest training expert, so this bounds "
                         "the per-epoch gmsh cost and also matches the "
                         "supervision resolution to the evaluation budget.")
    ap.add_argument("--expert_size_csv", default="amber_expert_maxsize.csv")
    ap.add_argument("--exclude_objects", default="amber_exclude_objects.txt",
                    help="Objects to drop (one per line). Used for geometries "
                         "whose initial mesh has orphan vertices, which give "
                         "NaN vertex sizing fields in AMBER.")
    ap.add_argument("--experts_dir", default="ml_baseline/amber_experts")
    ap.add_argument("--units_file", default=None,
                    help="lines '<object> <src_dir>'; STEP is looked up in "
                         "<src_dir>/<object>/*.step instead of CAD_DIR.")
    ap.add_argument("--out_dir", default="/data/1bali/GReFEM/baselines/AMBER/data/grefem")
    ap.add_argument("--n_train_objects", type=int, default=300)
    ap.add_argument("--n_val_objects", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # optional per-object source dir for STEP files
    src_map = {}
    if args.units_file and os.path.exists(args.units_file):
        for line in open(args.units_file):
            parts = line.split()
            if len(parts) >= 2:
                src_map[parts[0]] = parts[1]

    def step_paths(obj):
        if obj in src_map:
            return glob.glob(os.path.join(src_map[obj], obj, "*.step"))
        return glob.glob(os.path.join(CAD_DIR, obj, "*.step"))

    # objects that have all 5 expert meshes
    have = {}
    for p in glob.glob(os.path.join(args.experts_dir, "*.msh")):
        stem = os.path.basename(p)[:-4]
        for lc in sorted(LCS, key=len, reverse=True):
            if stem.endswith("_" + lc):
                have.setdefault(stem[: -(len(lc) + 1)], {})[lc] = p
                break
    full = sorted(o for o, d in have.items() if len(d) == 5)
    print(f"{len(have)} objects with experts, {len(full)} with all 5 load cases")

    # keep only objects AMBER can actually build an initial mesh for, and
    # reuse the bounding-box diagonal measured during validation
    valid_diag = {}
    for p in glob.glob(args.valid_glob):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok" and row.get("bbox_diagonal"):
                    valid_diag[row["object"]] = float(row["bbox_diagonal"])
    if valid_diag:
        before = len(full)
        full = [o for o in full if o in valid_diag]
        print(f"validated: {len(full)}/{before} objects mesh successfully")

    if args.max_expert_tets > 0 and os.path.exists(args.expert_size_csv):
        sizes = {}
        with open(args.expert_size_csv, newline="") as f:
            for row in csv.DictReader(f):
                sizes[row["object"]] = int(row["max_expert_tets"])
        before = len(full)
        full = [o for o in full
                if 0 < sizes.get(o, 1 << 60) <= args.max_expert_tets]
        print(f"expert-size filter <= {args.max_expert_tets} tets: "
              f"{len(full)}/{before} objects")

    if args.exclude_objects and os.path.exists(args.exclude_objects):
        excl = set(l.strip() for l in open(args.exclude_objects) if l.strip())
        before = len(full)
        full = [o for o in full if o not in excl]
        print(f"excluded {before - len(full)} objects "
              f"({len(excl)} listed in {args.exclude_objects})")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(full))
    val_objs = [full[i] for i in order[: args.n_val_objects]]
    train_objs = [full[i] for i in
                  order[args.n_val_objects: args.n_val_objects + args.n_train_objects]]
    print(f"train objects {len(train_objs)}, val objects {len(val_objs)}")

    manifest = {}
    for split, objs in [("train", train_objs), ("val", val_objs)]:
        split_dir = os.path.join(args.out_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
        idx = 0
        rows = []
        for obj in objs:
            steps = step_paths(obj)
            if not steps:
                print(f"[skip] {obj}: no STEP")
                continue
            diag = valid_diag.get(obj)
            if diag is None:
                try:
                    diag = bbox_diagonal(steps[0])
                except Exception as e:
                    print(f"[skip] {obj}: bbox failed: {e}")
                    continue
            for lc in LCS:
                idx += 1
                stem = os.path.join(split_dir, f"{idx:03d}")
                os.symlink(os.path.abspath(steps[0]), stem + ".step")
                os.symlink(os.path.abspath(have[obj][lc]), stem + ".msh")
                feats = [1.0 if lc == l else 0.0 for l in LCS]
                feats.append(float(np.log10(max(diag, 1e-12))))
                with open(stem + "_features.txt", "w") as f:
                    f.write("\n".join(f"{v}" for v in feats) + "\n")
                rows.append(dict(idx=idx, object=obj, load_case=lc,
                                 step=steps[0], expert=have[obj][lc],
                                 bbox_diagonal=diag))
        manifest[split] = rows
        print(f"{split}: {idx} data points from {len(objs)} objects")

    # AMBER's Trainer also runs a test pass; point it at val (the real
    # evaluation happens on the held-out GReFEM cohort via our own pipeline).
    test_dir = os.path.join(args.out_dir, "test")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    for p in sorted(glob.glob(os.path.join(args.out_dir, "val", "*"))):
        os.symlink(os.path.realpath(p) if os.path.islink(p) else os.path.abspath(p),
                   os.path.join(test_dir, os.path.basename(p)))

    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"),
              indent=1)
    n_tr, n_va = len(manifest["train"]), len(manifest["val"])
    print(f"\nnum_data_points: train={n_tr} val={n_va} test={n_va}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
