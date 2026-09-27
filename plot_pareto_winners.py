#!/usr/bin/env python
"""Pareto curves split by object population: GReFEM-favourable objects
(object-level mean grefem_max - heuristic vm_p99 diff < -0.01 at h_scale=1)
vs the remaining objects. Writes pareto_figure_winners.{pdf,png}."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

SERIES = [
    ("grefem_max",    BLUE,   "-",  "o", BLUE,    "GReFEM (max prompt)"),
    ("grefem_mid",    BLUE,   "--", "s", BLUE,    "GReFEM (mid prompt)"),
    ("grefem_none",   BLUE,   ":",  "^", "white", "GReFEM (no-physics prompt)"),
    ("heuristic",     ORANGE, "-",  "D", ORANGE,  "Geometric heuristic (dense)"),
    ("heuristic_sub", ORANGE, "--", "D", "white", "Geometric heuristic (subsampled)"),
    ("zz_oracle",     AQUA,   "-.", "*", AQUA,    "ZZ oracle (solver-informed)"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
})

df = pd.read_csv("pareto_table.csv", dtype={"object": str})
po = pd.read_csv("pareto_winners_perobj.csv", dtype={"object": str}).set_index("object")
winners = set(po[po["mean_diff"] < -0.01].index)

def agg_subset(sub):
    return sub.groupby(["h_scale", "method"]).agg(
        y=("rel_vm_p99_err", lambda s: trim_mean(s.dropna(), 0.05)),
        cells=("n_cells", "mean")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
subsets = [
    (axes[0], df[df["object"].isin(winners)],
     f"GReFEM-favourable objects (n={len(winners)})"),
    (axes[1], df[~df["object"].isin(winners)],
     f"Remaining objects (n={po.shape[0] - len(winners)})"),
]
for ax, sub, title in subsets:
    a = agg_subset(sub)
    for m, c, ls, mk, mfc, lab in SERIES:
        s = a[a["method"] == m].sort_values("cells")
        ax.plot(s["cells"] / 1e3, s["y"], ls, color=c, marker=mk,
                markersize=6.5 if mk == "*" else 4.5, markerfacecolor=mfc,
                markeredgecolor=c, markeredgewidth=1.1, lw=1.7,
                label=lab if ax is axes[0] else None, clip_on=False, zorder=3)
    co = a[a["method"] == "coarse"].iloc[0]
    ax.plot(co["cells"] / 1e3, co["y"], "X", color=GRAY, markersize=7, zorder=3,
            label="Coarse (no refinement)" if ax is axes[0] else None)
    ax.annotate("coarse", (co["cells"] / 1e3, co["y"]),
                textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK2)
    ax.set_title(title, color=INK2, loc="left", pad=6)
    ax.set_xlabel("Mesh cells ($\\times 10^3$, mean at matched budget)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0, 82)
    ax.tick_params(length=3)

axes[0].set_ylabel("Rel. $\\sigma_{vM}^{p99}$ error in $\\Omega_{crit}$\n(5% trimmed mean)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=7.6, bbox_to_anchor=(0.5, -0.04), handlelength=2.6,
           columnspacing=1.2)
fig.suptitle("Accuracy vs budget, split by object population "
             "(winner set defined at full budget; advantage validated on held-out load cases)",
             fontsize=9, y=1.0, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig("pareto_figure_winners.pdf", bbox_inches="tight")
fig.savefig("pareto_figure_winners.png", dpi=300, bbox_inches="tight")
print("wrote pareto_figure_winners.pdf/png")
