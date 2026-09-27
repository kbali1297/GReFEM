#!/usr/bin/env python
"""Pareto figure for the feature-dense cohort: error vs mesh budget over the
sizing sweep (h_scale 2.0 / 1.6 / 1.25 / 1.0), one curve per method.

Levels: hs1.0 comes from dense_shards/ (default sizing), the coarser levels
from dense_pareto_shards/hs*/. Strict inclusion: only objects present in
dense_cohort_table.csv (all 5 load cases, oracle-sane), and only units
complete at EVERY level so all curves rest on the same comparisons.

Usage: python plot_dense_pareto.py [vm|energy]
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import trim_mean

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, VIOLET, AQUA, GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a", "#898781"

SERIES = [  # method, color, linestyle, marker, facecolor, label
    ("grefem_max",    BLUE,   "-",  "o", BLUE,    "GReFEM (zero-shot MLLM)"),
    ("heuristic_sub", ORANGE, "-",  "D", ORANGE,  "Geometric heuristic (blind)"),
    ("mech_sub",      VIOLET, "--", "s", VIOLET,  "Load-informed heuristic"),
    ("zz_oracle",     AQUA,   "-.", "*", AQUA,    "ZZ oracle (fine-ref, ceiling)"),
]
NEED = {"coarse", "grefem_max", "heuristic_sub", "mech_sub", "zz_oracle"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
})


def fam(cand, lc):
    if cand in ("coarse", "zz_oracle"):
        return cand
    if re.match(rf"{lc}_gemini.*_geo_maxprompt_ortho_5views", cand):
        return "grefem_max"
    if cand.startswith("heuristic_baseline_DENSE") and cand.endswith("_sub"):
        return "heuristic_sub"
    if cand.startswith("heuristic_mech_") and cand.endswith("_sub"):
        return "mech_sub"
    return None


def load_level(pattern, hs, keep_objs):
    rows = []
    for f in glob.glob(pattern):
        if "aggregated" in f:
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if not (r.get("status") or "").startswith("ok"):
                    continue
                if r["object"] not in keep_objs:
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
    which = sys.argv[1] if len(sys.argv) > 1 else "vm"
    col, ylab, out = ((("vm"), "Rel. $\\sigma_{vM}^{p99}$ error in $\\Omega_{crit}$",
                       "dense_pareto_figure")
                      if which == "vm" else
                      ("en", "Rel. strain-energy error in $\\Omega_{crit}$",
                       "dense_pareto_figure_energy"))

    # strict set: all 5 load cases complete at every sizing level, duplicate
    # CAD parts removed (see build_dense_strict.py)
    keep = set(l.strip() for l in open("dense_strict_objects.txt")
               if l.strip())

    frames = [load_level("dense_shards/*.csv", 1.0, keep)]
    for d in sorted(glob.glob("dense_pareto_shards/hs*/")):
        hs = float(os.path.basename(d.rstrip("/"))[2:])
        frames.append(load_level(d + "*.csv", hs, keep))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["object", "load_case", "method", "h_scale"])

    levels = sorted(df["h_scale"].unique())
    # units complete at every level
    ok = df.groupby(["object", "load_case"]).apply(
        lambda g: all(NEED <= set(g[g["h_scale"] == h]["method"])
                      for h in levels), include_groups=False)
    units = set(ok[ok].index)
    df = df[[t in units for t in zip(df["object"], df["load_case"])]]
    n_obj, n_unit = df["object"].nunique(), len(units)
    print(f"levels {levels} | {n_obj} objects, {n_unit} units complete at all levels")

    agg = df.groupby(["h_scale", "method"]).agg(
        y=(col, lambda s: trim_mean(s.dropna(), 0.05)),
        med=(col, "median"),
        cells=("cells", "mean")).reset_index()
    agg.to_csv(f"{out}_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, ycol, title in [(axes[0], "y", "Mean over units (5% trimmed)"),
                            (axes[1], "med", "Median over units")]:
        for m, c, ls, mk, mfc, lab in SERIES:
            s = agg[agg["method"] == m].sort_values("cells")
            ax.plot(s["cells"] / 1e3, s[ycol], ls, color=c, marker=mk,
                    markersize=7 if mk == "*" else 4.5, markerfacecolor=mfc,
                    markeredgecolor=c, markeredgewidth=1.1, lw=1.7,
                    label=lab if ax is axes[0] else None, clip_on=False, zorder=3)
        co = agg[agg["method"] == "coarse"].iloc[0]
        ax.plot(co["cells"] / 1e3, co[ycol], "X", color=GRAY, markersize=7,
                zorder=3, label="Coarse (no refinement)" if ax is axes[0] else None)
        ax.annotate("coarse", (co["cells"] / 1e3, co[ycol]),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=8, color=INK2)
        ax.set_title(title, color=INK2, loc="left", pad=6)
        ax.set_xlabel("Mesh cells ($\\times 10^3$, mean at matched budget)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=3)

    axes[0].set_ylabel(ylab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.5, -0.06), handlelength=2.6, columnspacing=1.4)
    fig.suptitle(f"Feature-dense cohort: accuracy vs mesh budget "
                 f"({n_obj} objects $\\times$ 5 load cases = {n_unit} units, "
                 f"budgets matched per level)", fontsize=9, y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf/png and {out}_data.csv")


if __name__ == "__main__":
    main()
