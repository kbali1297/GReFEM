#!/usr/bin/env python
"""Pareto curves for the combined rebuttal test set:
  83 strict feature-dense objects (a-priori) + 59 GReFEM-favourable objects
  (post-hoc), 142 objects x 5 load cases, over sizing levels hs 1.0/1.25/1.6/2.0.

Dense half   : dense_shards/ (hs1.0) + dense_pareto_shards/hs*/
Winners half : pareto_table.csv (main-set sweep) + winners_shards/ (mech at
               hs1.0) + winners_pareto_shards/hs*/ (mech at coarser levels)

The load-informed heuristic (mech_sub) is included automatically once its
sweep data exists for both halves; otherwise the figure is drawn with the
four methods common to both.

Usage: python plot_combined_pareto.py            # both metrics
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
BLUE, ORANGE, VIOLET, YELLOW, AQUA, GRAY = ("#2a78d6", "#eb6834",
    "#4a3aa7", "#eda100", "#1baf7a", "#898781")

STYLE = {
    "grefem_max":    (BLUE,   "-",  "o", BLUE,    "GReFEM (zero-shot MLLM)"),
    "heuristic_sub": (ORANGE, "-",  "D", ORANGE,  "Geometric heuristic (blind)"),
    "mech_sub":      (VIOLET, "--", "s", VIOLET,  "Load-informed heuristic"),
    "uniform":       (GRAY,   ":",  "v", "white", "Uniform mesh at same sizing"),
    "zz_coarse":     (YELLOW, ":",  "^", "white", "ZZ from coarse solve (1 solve)"),
    "zz_oracle":     (AQUA,   "-.", "*", AQUA,    "ZZ oracle (fine-ref, ceiling)"),
}
BASE4 = ["coarse", "uniform", "grefem_max", "heuristic_sub", "zz_coarse",
         "zz_oracle"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
})


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


def main():
    dense = set(l.strip() for l in open("uni_dense_objects.txt") if l.strip())
    win = set(l.strip() for l in open("uni_winners_objects.txt") if l.strip())

    frames = [raw(["dense_shards/*.csv"], 1.0, dense)]
    for d in sorted(glob.glob("dense_pareto_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), dense))

    # winners: main-set sweep (already canonicalised) + mech shards
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

    # original-paper objects (test_meshes): full-budget rows live in the main
    # shards, everything else in orig_shards/hs*/
    orig = set(l.strip() for l in open("uni_orig_objects.txt") if l.strip())
    frames.append(raw(["fine2ref_shards/*.csv", "mb2_shards/*.csv",
                       "spread_shards/*.csv"], 1.0, orig))
    for d in sorted(glob.glob("orig_shards/hs*/")):
        frames.append(raw([d + "*.csv"],
                          float(os.path.basename(d.rstrip("/"))[2:]), orig))

    for base in ("czz_shards", "uni_shards"):
        for d in sorted(glob.glob(f"{base}/hs*/")):
            hs = float(os.path.basename(d.rstrip("/"))[2:])
            frames.append(raw([d + "*.csv"], hs, dense | win | orig))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(BASE4 + ["mech_sub"])].drop_duplicates(
        ["object", "load_case", "method", "h_scale"])
    levels = sorted(df["h_scale"].unique())

    # include mech_sub only if present for both halves at every level
    def covered(m):
        for h in levels:
            s = df[(df["h_scale"] == h) & (df["method"] == m)]
            if not (set(s["object"]) & dense) or not (set(s["object"]) & win):
                return False
        return True

    methods = list(BASE4)
    if "--no_mech" not in sys.argv and covered("mech_sub"):
        methods.insert(3, "mech_sub")
    print(f"levels {levels} | methods {methods}")

    need = set(methods)
    ok = df.groupby(["object", "load_case"]).apply(
        lambda g: all(need <= set(g[g["h_scale"] == h]["method"])
                      for h in levels), include_groups=False)
    units = set(ok[ok].index)
    df = df[[t in units for t in zip(df["object"], df["load_case"])]]
    n_obj = df["object"].nunique()
    n_d = len(set(df["object"]) & dense)
    n_w = len(set(df["object"]) & win)
    n_o = len(set(df["object"]) & orig)
    print(f"{n_obj} objects ({n_d} dense + {n_w} winners + {n_o} original), {len(units)} units")

    agg = df.groupby(["h_scale", "method"]).agg(
        vm=("vm", lambda s: trim_mean(s.dropna(), .05)),
        en=("en", lambda s: trim_mean(s.dropna(), .05)),
        cells=("cells", "mean")).reset_index()
    agg.to_csv("combined_pareto_data.csv", index=False)
    print(agg.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, col, ylab in [
            (axes[0], "vm", "Rel. $\\sigma_{vM}^{p99}$ error in $\\Omega_{crit}$"),
            (axes[1], "en", "Rel. strain-energy error in $\\Omega_{crit}$")]:
        for m in methods:
            if m == "coarse":
                continue
            c, ls, mk, mfc, lab = STYLE[m]
            s = agg[agg["method"] == m].sort_values("cells")
            ax.plot(s["cells"] / 1e3, s[col], ls, color=c, marker=mk,
                    markersize=7 if mk == "*" else 4.5, markerfacecolor=mfc,
                    markeredgecolor=c, markeredgewidth=1.1, lw=1.7,
                    label=lab if ax is axes[0] else None, clip_on=False, zorder=3)
        co = agg[agg["method"] == "coarse"].iloc[0]
        ax.plot(co["cells"] / 1e3, co[col], "X", color=GRAY, markersize=7,
                zorder=3, label="Coarse (no refinement)" if ax is axes[0] else None)
        ax.annotate("coarse", (co["cells"] / 1e3, co[col]),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=8, color=INK2)
        ax.set_xlabel("Mesh cells ($\\times 10^3$, mean at matched budget)")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_title("Peak stress in $\\Omega_{crit}$", color=INK2, loc="left", pad=6)
    axes[1].set_title("Strain energy in $\\Omega_{crit}$", color=INK2, loc="left", pad=6)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.5, -0.06), handlelength=2.6, columnspacing=1.4)
    fig.suptitle(f"Combined test set: accuracy vs mesh budget "
                 f"({n_obj} objects = {n_d} feature-dense + {n_w} GReFEM-favourable "
                 f"+ {n_o} original, {len(units)} units, 5% trimmed means)",
                 fontsize=9, y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    tag = "_4m" if "--no_mech" in sys.argv else ""
    fig.savefig(f"combined_pareto_figure{tag}.pdf", bbox_inches="tight")
    tag = "_4m" if "--no_mech" in sys.argv else ""
    fig.savefig(f"combined_pareto_figure{tag}.png", dpi=300, bbox_inches="tight")
    print("wrote combined_pareto_figure.pdf/png and combined_pareto_data.csv")


if __name__ == "__main__":
    main()
