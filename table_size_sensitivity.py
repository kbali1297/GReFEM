#!/usr/bin/env python
"""Element-size (sizing-field) sensitivity table for the rebuttal.

Answers JSWj Q1: "the refinement level is determined by predefined sizing-field
parameters. Could the authors provide more sensitivity analysis on these
parameters to better understand how these design choices affect the final FEM
accuracy?"

The sizing field is scaled by a common factor hs applied to BOTH h_min and
h_max for every refined candidate, holding fixed: the reference solution,
Omega_crit, crit_radius, and the anchor sets. At each level every baseline is
re-matched to the GReFEM element count at that level, so the comparison stays
budget-matched throughout.

Complete-case: only units with all reported candidates at ALL four levels.

Writes size_sensitivity_table.md (markdown) + size_sensitivity_data.csv.
"""
import csv
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, trim_mean

METHODS = ["coarse", "grefem_max", "heuristic_sub", "mech_sub",
           "zz_coarse", "zz_oracle"]
LABEL = {"coarse": "Coarse (no refinement)",
         "grefem_max": "GReFEM (zero-shot)",
         "heuristic_sub": "Geometric heuristic",
         "mech_sub": "Load-informed heuristic",
         "zz_coarse": "ZZ from coarse solve",
         "zz_oracle": "ZZ oracle (ceiling)"}
LEVELS = [1.0, 1.25, 1.6, 2.0]


def fam(c, lc):
    if c in ("coarse", "zz_oracle", "zz_coarse"):
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


def main():
    dense = set(l.strip() for l in open("final_dense_objects.txt") if l.strip())
    win = set(l.strip() for l in open("final_winners_objects.txt") if l.strip())
    orig = set(l.strip() for l in open("final_orig_objects.txt") if l.strip())
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
    for d in sorted(glob.glob("orig_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), orig))
    for d in sorted(glob.glob("czz_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), keep))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(METHODS)].drop_duplicates(
        ["object", "load_case", "method", "h_scale"])

    need = set(METHODS)
    ok = df.groupby(["object", "load_case"]).apply(
        lambda g: all(need <= set(g[g["h_scale"] == h]["method"])
                      for h in LEVELS), include_groups=False)
    units = set(ok[ok].index)
    df = df[[t in units for t in zip(df["object"], df["load_case"])]]
    n_obj, n_unit = df["object"].nunique(), len(units)
    df.to_csv("size_sensitivity_data.csv", index=False)

    out = []
    A = out.append
    A("## Element-size (sizing-field) sensitivity\n")
    A(f"Sizing field scaled by a common factor applied to both `h_min` and "
      f"`h_max` for every refined candidate. Held fixed across levels: the "
      f"reference solution, Ω_crit, `crit_radius`, and each method's anchor "
      f"set. At every level all baselines are re-matched to the GReFEM element "
      f"count, so the comparison remains budget-matched.\n")
    A(f"**{n_obj} objects × 5 loading cases = {n_unit} paired units**, "
      f"complete at all four levels. Values are relative error in Ω_crit "
      f"(5 %-trimmed mean over units); lower is better.\n")

    # --- realised element counts ---
    cells = df.pivot_table(index="h_scale", columns="method", values="cells",
                           aggfunc="mean").reindex(LEVELS)[METHODS]
    A("### Realised element counts (mean, thousands)\n")
    A("| sizing | " + " | ".join(LABEL[m] for m in METHODS) + " |")
    A("|---" * (len(METHODS) + 1) + "|")
    for h in LEVELS:
        A(f"| **h × {h}** | " +
          " | ".join(f"{cells.loc[h, m]/1e3:,.1f}k" for m in METHODS) + " |")
    A("")
    sp = ((cells[[m for m in METHODS if m != "coarse"]].max(axis=1)
           / cells[[m for m in METHODS if m != "coarse"]].min(axis=1) - 1) * 100)
    A(f"Refined candidates agree to within "
      f"{sp.min():.1f}–{sp.max():.1f} % at every level (budget matching holds; "
      f"`coarse` is the unrefined mesh and is not budget-matched).\n")

    # --- per-metric sensitivity tables ---
    for col, name in [("vm", "Peak stress (rel. σ_vM^p99 error in Ω_crit)"),
                      ("en", "Strain energy (rel. error in Ω_crit)")]:
        t = df.pivot_table(index="h_scale", columns="method", values=col,
                           aggfunc=lambda s: trim_mean(s.dropna(), .05)
                           ).reindex(LEVELS)[METHODS]
        A(f"### {name}\n")
        A("| sizing | mean cells | " +
          " | ".join(LABEL[m] for m in METHODS) + " |")
        A("|---" * (len(METHODS) + 2) + "|")
        for h in LEVELS:
            best = t.loc[h, [m for m in METHODS if m != "coarse"]].min()
            cs = cells.loc[h, [m for m in METHODS if m != "coarse"]].mean()
            vals = []
            for m in METHODS:
                v = f"{t.loc[h, m]:.3f}"
                vals.append(f"**{v}**" if abs(t.loc[h, m] - best) < 5e-4 else v)
            A(f"| **h × {h}** | {cs/1e3:,.0f}k | " + " | ".join(vals) + " |")
        A(f"| *Δ (h×2.0 − h×1.0)* | | " +
          " | ".join(f"*{t.loc[2.0, m]-t.loc[1.0, m]:+.3f}*"
                     for m in METHODS) + " |")
        A("")

    # --- paired tests per level ---
    A("### Paired tests at each sizing level (vm_p99)\n")
    A("Median Δ = GReFEM − baseline; negative means GReFEM is better. "
      "Two-sided Wilcoxon signed-rank over the paired units.\n")
    others = [m for m in METHODS if m != "grefem_max"]
    A("| sizing | " + " | ".join(LABEL[m] for m in others) + " |")
    A("|---" * (len(others) + 1) + "|")
    for h in LEVELS:
        piv = df[df["h_scale"] == h].pivot_table(
            index=["object", "load_case"], columns="method", values="vm")
        cellsr = []
        for m in others:
            d = (piv["grefem_max"] - piv[m]).dropna()
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = np.nan
            star = "" if p >= 0.05 else "*"
            cellsr.append(f"{d.median():+.3f}{star} (p={p:.2g})")
        A(f"| **h × {h}** | " + " | ".join(cellsr) + " |")
    A("")
    A("`*` = significant at p < 0.05.\n")

    # --- ranking stability ---
    A("### Ranking stability\n")
    t = df.pivot_table(index="h_scale", columns="method", values="vm",
                       aggfunc=lambda s: trim_mean(s.dropna(), .05)
                       ).reindex(LEVELS)[METHODS]
    for h in LEVELS:
        order = t.loc[h].sort_values().index.tolist()
        A(f"- **h × {h}**: " + " < ".join(LABEL[m] for m in order))
    A("")

    txt = "\n".join(out)
    with open("size_sensitivity_table.md", "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\n[wrote size_sensitivity_table.md and size_sensitivity_data.csv]")


if __name__ == "__main__":
    main()
