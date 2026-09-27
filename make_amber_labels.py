#!/usr/bin/env python
"""Build AMBER expert meshes (training labels) from the ml_baseline point-cloud
ZZ labels.

AMBER regresses the local element size of an *adaptive* expert mesh, so the
per-point ZZ targets in ml_baseline/labels/<cad>.npz cannot be used directly.
For each (cad, load_case) we rebuild an adaptive mesh with the SAME recipe that
produced the paper's zz_oracle candidate:

    anchors = top-N surface points by target_<load_case>   (N = --top_n,
              default 1563 = median GReFEM anchor count over the eval set)
    h_max, h_min = compute_h_sizes(step)                   (same volume^0.3 law)
    generate_or_refine_mesh(step, anchors, h_min, h_max)

so the supervision lives in the same mesh family as the oracle. No budget
matching here: element budget is an evaluation-time concern, handled at
inference by AMBER's sizing-field scale factor.

Meshes are written next to the labels as
    ml_baseline/amber_experts/<cad>_<load_case>.msh
and are skipped if already present (resumable / shardable).

Usage:
    python make_amber_labels.py --shard 0 --nshards 100 \
        --out_dir ml_baseline/amber_experts
"""
import argparse
import functools
import glob
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_local_error_tables import compute_h_sizes, count_tets  # noqa: E402
from utils import generate_or_refine_mesh  # noqa: E402

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
CAD_DIR = "ABC_CAD_Dataset_small2_augmented"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels_dir", default="ml_baseline/labels")
    ap.add_argument("--out_dir", default="ml_baseline/amber_experts")
    ap.add_argument("--top_n", type=int, default=1563,
                    help="anchors per load case (median GReFEM anchor count)")
    ap.add_argument("--load_cases", nargs="+", default=LCS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cads = sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(args.labels_dir, "*.npz")))
    cads = [c for i, c in enumerate(cads) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(cads)} cads")

    n_ok = n_skip = n_fail = 0
    for ci, cad in enumerate(cads):
        steps = glob.glob(os.path.join(CAD_DIR, cad, "*.step"))
        if not steps:
            print(f"[skip] {cad}: no STEP")
            continue
        try:
            d = np.load(os.path.join(args.labels_dir, f"{cad}.npz"))
            pts = np.asarray(d["points"], float)
        except Exception as e:
            print(f"[fail] {cad}: unreadable labels: {e}")
            continue
        try:
            h_max, h_min, step_path = compute_h_sizes(os.path.join(CAD_DIR, cad))
        except Exception as e:
            print(f"[fail] {cad}: h sizes: {e}")
            continue

        for lc in args.load_cases:
            key = f"target_{lc}"
            out = os.path.join(args.out_dir, f"{cad}_{lc}.msh")
            if os.path.exists(out) and count_tets(out) > 0:
                n_skip += 1
                continue
            if key not in d.files:
                continue
            tgt = np.asarray(d[key], float)
            if not np.isfinite(tgt).any() or tgt.max() <= 0:
                print(f"[skip] {cad}/{lc}: no positive ZZ target")
                continue
            n_anchor = min(args.top_n, int((tgt > 0).sum()))
            anchors = pts[np.argsort(tgt)[::-1][:n_anchor]]
            try:
                generate_or_refine_mesh(
                    step_or_mesh_path=step_path, points_of_interest=anchors,
                    h_min=h_min, h_max=h_max, suffix=None,
                    verbose=False, out_msh=out)
                nt = count_tets(out)
                if nt == 0:
                    if os.path.exists(out):
                        os.remove(out)
                    raise RuntimeError("mesh gen produced no tets")
                n_ok += 1
                print(f"[ok] {cad}/{lc}: {nt} tets from {n_anchor} anchors "
                      f"(h_min {h_min:.3f} h_max {h_max:.3f})")
            except Exception as e:
                n_fail += 1
                print(f"[fail] {cad}/{lc}: {type(e).__name__}: {e}")
                traceback.print_exc()
        if (ci + 1) % 5 == 0:
            print(f"[progress] {ci+1}/{len(cads)} cads | ok {n_ok} "
                  f"skip {n_skip} fail {n_fail}")

    print(f"\nDone. ok {n_ok}, skipped {n_skip}, failed {n_fail}")


if __name__ == "__main__":
    main()
