#!/usr/bin/env python3
"""
compute_field_error_tables.py

Crit-restricted FIELD-norm error metrics (no order statistics, no scalar
cancellation): for every (object, load_case, candidate) row of the combined
table, interpolate the candidate displacement onto the fine2 reference mesh
restricted to Omega_crit and compute

  rel_energy_norm_crit = ||u_h - u_ref||_E(crit)  / ||u_ref||_E(crit)
  rel_vm_l2_crit       = ||vm_h - vm_ref||_L2(crit) / ||vm_ref||_L2(crit)
  rel_disp_l2_crit     = ||u_h - u_ref||_L2(crit) / ||u_ref||_L2(crit)

Omega_crit = ref-mesh cells whose centroid lies within crit_radius (from the
combined table, = 4*h_min) of the top-percentile ZZ points of the reference.
All meshes/solutions are cached (*_sol.xdmf next to each candidate .msh);
this script only evaluates metrics. Resumable via --done_glob, shardable via
--shard/--nshards over units.

Usage:
    python compute_field_error_tables.py \
        --units_csv combined_200obj_table.csv \
        --shard 0 --nshards 100 \
        --out_csv field_shards/fe_w0.csv
"""
import os
import csv
import glob
import time
import argparse
import functools
import traceback

import numpy as np
import pandas as pd
import ufl
from dolfinx import fem, mesh as dmesh
from scipy.spatial import cKDTree

from fem_fenics import load_solution, read_pos_points_top, eps

print = functools.partial(print, flush=True)

CSV_FIELDS = [
    "object", "load_case", "candidate", "method", "src", "candidate_msh",
    "n_crit_cells_ref", "crit_radius",
    "energy_norm_ref_crit", "vm_l2_ref_crit", "disp_l2_ref_crit",
    "rel_energy_norm_crit", "rel_vm_l2_crit", "rel_disp_l2_crit",
    "status",
]

SRC_DIR = {"orig": "test_meshes", "extra": "test_meshes_extra"}


def load_done_keys(out_csv, done_glob=None):
    done = set()
    paths = [out_csv]
    if done_glob:
        for g in done_glob.split():
            paths += glob.glob(g)
    for p in set(paths):
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((row["object"], row["load_case"],
                              row["candidate"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units_csv", default="combined_200obj_table.csv")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--done_glob", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ref_prefix", default="fine2_mesh")
    ap.add_argument("--zz_top_percentile", type=float, default=99.9)
    ap.add_argument("--E", type=float, default=210e9)
    ap.add_argument("--nu", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.units_csv, dtype={"object": str})
    units = sorted(df.groupby(["object", "load_case"]).groups)
    units = [u for i, u in enumerate(units) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(units)} units")

    done = load_done_keys(args.out_csv, args.done_glob)
    write_header = not os.path.exists(args.out_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fout = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    mu = args.E / (2.0 * (1.0 + args.nu))
    lmbda = args.E * args.nu / ((1.0 + args.nu) * (1.0 - 2.0 * args.nu))

    def sig(u):
        e = eps(u)
        return 2.0 * mu * e + lmbda * ufl.tr(e) * ufl.Identity(3)

    def vm(u):
        s = sig(u)
        dev = s - (ufl.tr(s) / 3.0) * ufl.Identity(3)
        return ufl.sqrt(1.5 * ufl.inner(dev, dev) + 1e-30)

    t0 = time.time()
    unit_times = []
    for ui, (obj, lc) in enumerate(units):
        rows = df[(df["object"] == obj) & (df["load_case"] == lc)]
        pend = [r for _, r in rows.iterrows()
                if (obj, lc, r["candidate"]) not in done]
        if not pend:
            continue
        t_unit = time.time()
        src = rows["src"].iloc[0]
        tdir = os.path.join(SRC_DIR[src], obj)
        ref_xdmf = os.path.join(tdir, f"{args.ref_prefix}_{lc}_sol.xdmf")
        zz_pos = os.path.join(tdir, f"{args.ref_prefix}_{lc}_zz.pos")
        crit_radius = float(rows["crit_radius"].iloc[0])
        print(f"\n[{ui+1}/{len(units)}] {obj}/{lc} "
              f"({len(pend)} candidates, elapsed {int(time.time()-t0)}s)")
        try:
            crit_points, _ = read_pos_points_top(
                zz_pos, args.zz_top_percentile, load_case=lc)
            msh_ref, V_ref, u_ref = load_solution(ref_xdmf)
            tdim = msh_ref.topology.dim
            centroids = msh_ref.geometry.x[msh_ref.geometry.dofmap].mean(axis=1)
            d, _ = cKDTree(np.asarray(crit_points, float)).query(centroids)
            crit_cells = np.flatnonzero(d <= crit_radius).astype(np.int32)
            if len(crit_cells) == 0:
                raise RuntimeError("no crit cells on reference mesh")
            ct = dmesh.meshtags(msh_ref, tdim, crit_cells,
                                np.ones(len(crit_cells), dtype=np.int32))
            dxc = ufl.Measure("dx", domain=msh_ref, subdomain_data=ct)(1)

            u_ci = fem.Function(V_ref)   # candidate on ref mesh (crit cells)
            diff = u_ci - u_ref
            forms = {
                "en_err": fem.form(0.5 * ufl.inner(sig(diff), eps(diff)) * dxc),
                "vm_err": fem.form((vm(u_ci) - vm(u_ref)) ** 2 * dxc),
                "l2_err": fem.form(ufl.inner(diff, diff) * dxc),
            }
            en_ref = np.sqrt(abs(fem.assemble_scalar(fem.form(
                0.5 * ufl.inner(sig(u_ref), eps(u_ref)) * dxc))))
            vm_ref = np.sqrt(abs(fem.assemble_scalar(fem.form(
                vm(u_ref) ** 2 * dxc))))
            l2_ref = np.sqrt(abs(fem.assemble_scalar(fem.form(
                ufl.inner(u_ref, u_ref) * dxc))))
            print(f"  ref: {len(crit_cells)} crit cells / "
                  f"{msh_ref.topology.index_map(tdim).size_local}, "
                  f"|u|_E(crit)={en_ref:.4g}")
        except (Exception, SystemExit) as e:
            print(f"[fail] {obj}/{lc}: reference setup failed: {e}")
            traceback.print_exc()
            for r in pend:
                writer.writerow({"object": obj, "load_case": lc,
                                 "candidate": r["candidate"],
                                 "method": r["method"], "src": src,
                                 "status": f"error_ref: {e}"})
            fout.flush()
            continue

        for r in pend:
            row = {"object": obj, "load_case": lc,
                   "candidate": r["candidate"], "method": r["method"],
                   "src": src, "candidate_msh": r["candidate_msh"],
                   "n_crit_cells_ref": len(crit_cells),
                   "crit_radius": crit_radius,
                   "energy_norm_ref_crit": float(en_ref),
                   "vm_l2_ref_crit": float(vm_ref),
                   "disp_l2_ref_crit": float(l2_ref)}
            t_c = time.time()
            try:
                cand_xdmf = r["candidate_msh"].replace(".msh", "_sol.xdmf")
                _, V_c, u_c = load_solution(cand_xdmf)
                idata = fem.create_interpolation_data(
                    V_ref, V_c, crit_cells, padding=1e-6 * crit_radius)
                u_ci.x.array[:] = 0.0
                u_ci.interpolate_nonmatching(u_c, crit_cells,
                                             interpolation_data=idata)
                en = np.sqrt(abs(fem.assemble_scalar(forms["en_err"])))
                vme = np.sqrt(abs(fem.assemble_scalar(forms["vm_err"])))
                l2 = np.sqrt(abs(fem.assemble_scalar(forms["l2_err"])))
                row.update({
                    "rel_energy_norm_crit": float(en / (en_ref + 1e-30)),
                    "rel_vm_l2_crit": float(vme / (vm_ref + 1e-30)),
                    "rel_disp_l2_crit": float(l2 / (l2_ref + 1e-30)),
                    "status": "ok"})
                print(f"  [ok] {r['method']:18s} "
                      f"E={row['rel_energy_norm_crit']:.4f} "
                      f"vm={row['rel_vm_l2_crit']:.4f} "
                      f"u={row['rel_disp_l2_crit']:.4f} "
                      f"({int(time.time()-t_c)}s)")
            except (Exception, SystemExit) as e:
                row["status"] = f"error: {e}"
                print(f"  [fail] {r['method']}: {e}")
                traceback.print_exc()
            writer.writerow(row)
            fout.flush()

        unit_times.append(time.time() - t_unit)
        avg = float(np.mean(unit_times))
        rem = len(units) - (ui + 1)
        print(f"[progress] {ui+1}/{len(units)} units | avg {avg:.0f}s/unit | "
              f"ETA {int(avg*rem/60)}m "
              f"(~{time.strftime('%H:%M', time.localtime(time.time()+avg*rem))})")

    fout.close()
    print(f"\nDone. Results in {args.out_csv}")


if __name__ == "__main__":
    main()
