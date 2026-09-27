#!/usr/bin/env python
"""Numerical sizing-field sweep tables (stress AND energy) for the rebuttal.

Config A (primary): 6 policies, largest object set -- the element-size
                    sensitivity analysis for the main test set.
Config B (control): adds the uniform-mesh reference, which exists only for the
                    subset of objects that gmsh can mesh uniformly at every
                    sizing level.

Emits markdown tables + per-level paired Wilcoxon tests to stdout and
sweep_tables.md, and the raw aggregates to sweep_tables_{A,B}.csv.
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, trim_mean

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
LABEL = {"coarse": "Coarse mesh, default sizing only (fixed reference)",
         "uniform": "Coarse/uniform mesh at this sizing (h_min=h_max)",
         "grefem_max": "GReFEM (zero-shot MLLM)",
         "heuristic_sub": "Geometric heuristic (blind)",
         "mech_sub": "Load-informed heuristic",
         "zz_coarse": "ZZ from coarse solve (1 solve)",
         "zz_oracle": "ZZ oracle (fine ref, ceiling)"}
A_METHODS = ["coarse", "grefem_max", "heuristic_sub", "mech_sub",
             "zz_coarse", "zz_oracle"]
B_METHODS = ["uniform", "grefem_max", "heuristic_sub", "mech_sub",
             "zz_coarse", "zz_oracle"]


def fam(c, lc):
    if c in ("coarse", "zz_oracle", "zz_coarse", "uniform"):
        return c
    if re.match(rf"{lc}_gemini.*_geo_maxprompt_ortho_5views", c):
        return "grefem_max"
    if c.startswith("heuristic_baseline_DENSE") and c.endswith("_sub"):
        return "heuristic_sub"
    if c.startswith("heuristic_mech_") and c.endswith("_sub"):
        return "mech_sub"
    return None


def raw(patterns, hs, keep):
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            if "aggregated" in f:
                continue
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if not (r.get("status") or "").startswith("ok"):
                        continue
                    if r["object"] not in keep:
                        continue
                    m = fam(r["candidate"], r["load_case"])
                    if m is None:
                        continue
                    try:
                        rows.append({"object": r["object"],
                                     "load_case": r["load_case"], "method": m,
                                     "h_scale": hs,
                                     "vm": float(r["rel_vm_p99_err"]),
                                     "en": float(r["rel_energy_crit_err"]),
                                     "cells": float(r["n_cells"])})
                    except (ValueError, KeyError, TypeError):
                        continue
    return pd.DataFrame(rows)


def load_all(dense, win, orig):
    keep = dense | win | orig
    frames = [raw(["dense_shards/*.csv"], 1.0, dense)]
    for d in sorted(glob.glob("dense_pareto_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), dense))
    pt = pd.read_csv("pareto_table.csv", dtype={"object": str})
    pt = pt[pt["object"].isin(win)].rename(
        columns={"rel_vm_p99_err": "vm", "rel_energy_crit_err": "en",
                 "n_cells": "cells"})
    frames.append(pt[["object", "load_case", "method", "h_scale",
                      "vm", "en", "cells"]])
    frames.append(raw(["winners_shards/*.csv"], 1.0, win))
    for d in sorted(glob.glob("winners_pareto_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), win))
    frames.append(raw(["fine2ref_shards/*.csv", "mb2_shards/*.csv",
                       "spread_shards/*.csv"], 1.0, orig))
    for base in ("orig_shards", "czz_shards", "uni_shards"):
        for d in sorted(glob.glob(f"{base}/hs*/")):
            frames.append(raw([d + "*.csv"],
                              float(os.path.basename(d.rstrip("/"))[2:]), keep))
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["object", "load_case", "method", "h_scale"])


def emit(df, methods, title, note, out_csv, fh):
    levels = sorted(df["h_scale"].unique())
    need = set(methods)
    ok = df.groupby(["object", "load_case"]).apply(
        lambda g: all(need <= set(g[g["h_scale"] == h]["method"])
                      for h in levels), include_groups=False)
    units = set(ok[ok].index)
    d = df[[t in units for t in zip(df["object"], df["load_case"])]]
    n_obj, n_unit = d["object"].nunique(), len(units)

    agg = d.groupby(["h_scale", "method"]).agg(
        vm_t=("vm", lambda s: trim_mean(s.dropna(), .05)),
        vm_m=("vm", "median"),
        en_t=("en", lambda s: trim_mean(s.dropna(), .05)),
        en_m=("en", "median"),
        cells=("cells", "median")).reset_index()
    agg.to_csv(out_csv, index=False)

    w = lambda s: (fh.write(s + "\n"), print(s))
    w(f"\n### {title}")
    w(f"\n{note}")
    w(f"\n**{n_obj} objects, {n_unit} units** complete with all "
      f"{len(methods)} policies at all {len(levels)} sizing levels.\n")

    for metric, mt, mm, name in [("stress", "vm_t", "vm_m",
                                  "Relative peak-stress error "
                                  "(vm_p99 in Omega_crit)"),
                                 ("energy", "en_t", "en_m",
                                  "Relative strain-energy error "
                                  "(Omega_crit)")]:
        w(f"\n**{name}** — 5 %-trimmed mean (median in brackets); "
          f"median realised element count in *italics*\n")
        hdr = "| policy | " + " | ".join(
            f"h×{h:g}" for h in levels) + " |"
        w(hdr)
        w("|" + "---|" * (len(levels) + 1))
        for m in methods:
            cells = []
            for h in levels:
                r = agg[(agg["h_scale"] == h) & (agg["method"] == m)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                best = agg[(agg["h_scale"] == h)
                           & (agg["method"].isin(methods))][mt].min()
                v = f"{r[mt]:.3f} [{r[mm]:.3f}]"
                if abs(r[mt] - best) < 5e-4:
                    v = f"**{v}**"
                cells.append(f"{v}<br>*{r['cells']:,.0f}*")
            w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")

    # per-level paired tests vs GReFEM
    w(f"\n**Paired tests per level** (median Δ = GReFEM − policy; "
      f"negative = GReFEM better; n = {n_unit})\n")
    others = [m for m in methods if m != "grefem_max"]
    w("| level | " + " | ".join(LABEL[m].split(" (")[0] for m in others) + " |")
    w("|" + "---|" * (len(others) + 1))
    for h in levels:
        s = d[d["h_scale"] == h]
        piv = s.pivot_table(index=["object", "load_case"], columns="method",
                            values="vm")
        row = []
        for m in others:
            if m not in piv:
                row.append("—")
                continue
            diff = (piv["grefem_max"] - piv[m]).dropna()
            try:
                p = wilcoxon(diff).pvalue
            except ValueError:
                p = np.nan
            star = "" if p >= 0.05 else ""
            row.append(f"{diff.median():+.3f}, p={p:.2g}{star}")
        w(f"| h×{h:g} | " + " | ".join(row) + " |")
    return agg, n_obj, n_unit


def main():
    dense = set(l.strip() for l in open("final_dense_objects.txt") if l.strip())
    win = set(l.strip() for l in open("final_winners_objects.txt") if l.strip())
    orig = set(l.strip() for l in open("final_orig_objects.txt") if l.strip())
    df = load_all(dense, win, orig)

    with open("sweep_tables.md", "w") as fh:
        emit(df[df["method"].isin(A_METHODS)], A_METHODS,
             "Config A (primary) — element-size sensitivity, 6 policies",
             "All refined policies budget-matched to GReFEM at each sizing "
             "level. `coarse` is the stored unrefined mesh and is the same at "
             "every level (it is not re-scaled), so it appears as a single "
             "reference value.",
             "sweep_tables_A.csv", fh)
        emit(df[df["method"].isin(B_METHODS)], B_METHODS,
             "Config B (control) — adds a uniform mesh re-generated at each "
             "sizing level",
             "The uniform mesh applies the same h-scaling as the refined "
             "policies but with no anchors. Because scaling h upward only "
             "coarsens it, its element counts fall **below** the refined "
             "range; it therefore bounds the no-adaptivity behaviour at each "
             "sizing level but does not provide an equal-element-count "
             "comparison. Object count is reduced because gmsh cannot mesh "
             "every geometry uniformly at the coarsest sizings.",
             "sweep_tables_B.csv", fh)
    print("\nwrote sweep_tables.md, sweep_tables_A.csv, sweep_tables_B.csv")


if __name__ == "__main__":
    main()
