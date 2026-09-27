#!/usr/bin/env python3
"""
infer_baseline_mech.py

MECHANICS-AWARE geometric heuristic baseline (rebuttal).

Same candidate generator as the blind geometric heuristic
(infer_baseline.py: concave sharp edges + concave smooth surfaces), but
every candidate point is weighted by a load-case-dependent nominal-stress
rule computed from cross-section properties of the part along the loading
axis (y, matching the BCs in fem_fenics.py: y_min clamped, y_max loaded):

    compression : w = 1 / A(y)              (net-section stress)
    bending     : w = |x - x_bar(y)| / I_z(y)  (flexure formula, sigma = M c / I)
    torsion     : w = r(y) / J(y)            (tau = T r / J)
    shear       : w = |x - x_bar(y)| / I_z(y)  (guided cantilever ~ bending)

Section properties A(y), centroid (x_bar, z_bar), I_z(y), J(y) are computed
per voxel layer from a filled voxelization of the surface mesh -- purely
geometric, no solver involved. Only the top --keep_pct fraction of
candidates by weight is kept (the mechanics decides WHERE among the
geometric features to refine, mirroring the selectivity of the LLM), then
the same density filtering as the blind baseline is applied. The final
mesh budget is matched downstream by compute_local_error_tables.py.

Outputs per CAD (into --out_dir/<cad>/):
    refinement_points_mech_dihedral{angle}deg_{load_case}.npy

Usage:
    python infer_baseline_mech.py \
        --test_dir test_meshes --out_dir test_meshes_rebuttal_baseline \
        --load_cases bending compression torsion
"""
import os
import argparse
import numpy as np
import trimesh

from infer_baseline import upsample_concave_edges, upsample_concave_surfaces
from utils import filter_points_by_density_fast


# ------------------------------------------------------------------
# Cross-section properties along the loading axis (y) via voxelization
# ------------------------------------------------------------------

def section_properties(mesh, pitch=None):
    """Per-voxel-layer section properties along y.

    Returns dict of 1D arrays keyed by layer: y_centers, A, x_bar, z_bar,
    I_z (second moment about the z-axis through the layer centroid,
    integrating x-offsets -> flexure in the x-y plane), J (polar second
    moment about the layer centroid).
    """
    if pitch is None:
        pitch = float(np.linalg.norm(mesh.extents)) / 120.0
    vox = mesh.voxelized(pitch)
    try:
        vox = vox.fill()
    except Exception:
        pass  # open meshes: fall back to surface voxels (still monotone in A)
    pts = vox.points  # (N, 3) centers of filled voxels
    if len(pts) == 0:
        raise RuntimeError("Voxelization produced no filled voxels")

    y = pts[:, 1]
    # group voxels into layers of thickness = pitch
    layer_idx = np.floor((y - y.min()) / pitch + 0.5).astype(int)
    layers = np.unique(layer_idx)

    y_centers, A, x_bar, z_bar, I_z, J = [], [], [], [], [], []
    dA = pitch * pitch
    for li in layers:
        m = layer_idx == li
        xs, zs = pts[m, 0], pts[m, 2]
        xb, zb = xs.mean(), zs.mean()
        y_centers.append(y[m].mean())
        A.append(m.sum() * dA)
        x_bar.append(xb)
        z_bar.append(zb)
        I_z.append(((xs - xb) ** 2).sum() * dA)
        J.append((((xs - xb) ** 2) + ((zs - zb) ** 2)).sum() * dA)

    return {k: np.asarray(v, float) for k, v in zip(
        ["y", "A", "x_bar", "z_bar", "I_z", "J"],
        [y_centers, A, x_bar, z_bar, I_z, J])}


def mechanics_weights(points, sec, load_case):
    """Nominal-stress weight for each candidate point under load_case."""
    # nearest layer per point
    li = np.abs(points[:, 1][:, None] - sec["y"][None, :]).argmin(axis=1)
    A = np.maximum(sec["A"][li], 1e-12)
    I_z = np.maximum(sec["I_z"][li], 1e-12)
    J = np.maximum(sec["J"][li], 1e-12)
    x_off = np.abs(points[:, 0] - sec["x_bar"][li])
    r = np.sqrt((points[:, 0] - sec["x_bar"][li]) ** 2
                + (points[:, 2] - sec["z_bar"][li]) ** 2)

    if load_case == "compression":
        w = 1.0 / A
    elif load_case in ("bending", "shear"):
        w = x_off / I_z
    elif load_case == "torsion":
        w = r / J
    elif load_case == "bending_compression":
        w = _normalize(x_off / I_z) + _normalize(1.0 / A)
    elif load_case == "torsion_compression":
        w = _normalize(r / J) + _normalize(1.0 / A)
    else:
        raise ValueError(f"Unknown load_case {load_case}")
    return _normalize(w)


def _normalize(w):
    lo, hi = float(w.min()), float(w.max())
    return np.zeros_like(w) if hi - lo < 1e-30 else (w - lo) / (hi - lo)


# ------------------------------------------------------------------
# Main per-object pipeline
# ------------------------------------------------------------------

def compute_mech_points(mesh_path, load_case, feature_angle=30.0,
                        keep_pct=70.0, min_keep=50, boundary_frac=0.02):
    """Return mechanics-weighted refinement anchors for one CAD/load case."""
    mesh = trimesh.load_mesh(mesh_path)
    bbox_min, bbox_max = mesh.bounds
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    target_spacing = diag / (66.667 * 2)  # identical to blind baseline

    # 1. candidate geometric features (identical to blind baseline)
    edge_pts = upsample_concave_edges(mesh, feature_angle=feature_angle,
                                      target_spacing=target_spacing)
    surf_pts = upsample_concave_surfaces(mesh, feature_angle=feature_angle,
                                         target_spacing=target_spacing)
    parts = [p for p in (edge_pts, surf_pts) if p.shape[0] > 0]
    if not parts:
        return np.empty((0, 3))
    cand = np.unique(np.round(np.vstack(parts), decimals=4), axis=0)

    # 2. trim the load-application bands (same convention as evaluation:
    #    read_pos_points_top drops 2% at the y extremes; the mesher also
    #    ignores anchors there)
    y_lo, y_hi = cand[:, 1].min(), cand[:, 1].max()
    band = boundary_frac * (y_hi - y_lo)
    inner = (cand[:, 1] > y_lo + band) & (cand[:, 1] < y_hi - band)
    if inner.sum() >= min_keep:
        cand = cand[inner]

    # 3. mechanics weighting from voxelized section properties
    sec = section_properties(mesh)
    w = mechanics_weights(cand, sec, load_case)

    # 4. keep the top (100 - keep_pct) percentile by weight
    thr = np.percentile(w, keep_pct)
    keep = w >= thr
    if keep.sum() < min_keep:  # tiny parts: keep the best min_keep
        keep = np.argsort(w)[::-1][:min_keep]
    selected = cand[keep]
    sel_w = w[keep] if keep.dtype == bool else w[keep]

    # 5. density filter, highest-weight points first so they survive thinning
    order = np.argsort(sel_w)[::-1]
    filtered = filter_points_by_density_fast(selected[order],
                                             target_spacing * 0.75)
    return np.asarray(filtered)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", default="test_meshes")
    ap.add_argument("--out_dir", default="test_meshes_rebuttal_baseline")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--load_cases", nargs="+",
                    default=["bending", "compression", "torsion"])
    ap.add_argument("--feature_angle", type=float, default=30.0)
    ap.add_argument("--keep_pct", type=float, default=70.0,
                    help="Weight percentile below which candidates are "
                         "dropped (70 = keep top 30%% by nominal stress).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    objects = args.objects or sorted(
        d for d in os.listdir(args.test_dir)
        if os.path.isdir(os.path.join(args.test_dir, d)))

    for obj in objects:
        mesh_path = os.path.join(args.test_dir, obj, "renders_pyvista",
                                 f"{obj}.obj")
        if not os.path.exists(mesh_path):
            print(f"[skip] {obj}: no surface mesh at {mesh_path}")
            continue
        out_obj_dir = os.path.join(args.out_dir, obj)
        os.makedirs(out_obj_dir, exist_ok=True)
        for lc in args.load_cases:
            out_npy = os.path.join(
                out_obj_dir,
                f"refinement_points_mech_dihedral{args.feature_angle}deg_{lc}.npy")
            if os.path.exists(out_npy) and not args.overwrite:
                print(f"[skip] {obj}/{lc}: exists")
                continue
            try:
                pts = compute_mech_points(
                    mesh_path, lc, feature_angle=args.feature_angle,
                    keep_pct=args.keep_pct)
                np.save(out_npy, pts)
                print(f"[ok] {obj}/{lc}: {len(pts)} anchors -> {out_npy}")
            except Exception as e:
                print(f"[fail] {obj}/{lc}: {e}")


if __name__ == "__main__":
    main()
