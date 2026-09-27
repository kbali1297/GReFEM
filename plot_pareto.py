#!/usr/bin/env python
"""Pareto figure for the rebuttal: relative vm_p99 error (Omega_crit) vs mesh
budget, one curve per method over the 4 sizing levels (h_scale 2.0, 1.6,
1.25, 1.0), pooled over the 735 complete-case units. Writes
pareto_figure.{pdf,png}."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"

SERIES = [  # method, color, ls, marker, mfc, label
    ("grefem_max",   BLUE,   "-",  "o", BLUE,    "GReFEM (max prompt)"),
    ("grefem_mid",   BLUE,   "--", "s", BLUE,    "GReFEM (mid prompt)"),
    ("grefem_none",  BLUE,   ":",  "^", "white", "GReFEM (no-physics prompt)"),
    ("heuristic",    ORANGE, "-",  "D", ORANGE,  "Geometric heuristic (dense)"),
    ("heuristic_sub", ORANGE, "--", "D", "white", "Geometric heuristic (subsampled)"),
    ("zz_oracle",    AQUA,   "-.", "*", AQUA,    "ZZ oracle (solver-informed)"),
]

FAMILY_LABELS = [  # method whose end point anchors the label
    ("grefem_max", "GReFEM", BLUE, (8, 10)),
    ("heuristic", "Heuristic", ORANGE, (8, -14)),
    ("zz_oracle", "ZZ oracle", AQUA, (8, -2)),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK,
})

agg = pd.read_csv("pareto_agg.csv")

METRICS = {
    "vm": ("vm_p99_tmean", "vm_p99_med",
           "Rel. $\\sigma_{vM}^{p99}$ error in $\\Omega_{crit}$",
           "pareto_figure"),
    "energy": ("energy_tmean", "energy_med",
               "Rel. strain-energy error in $\\Omega_{crit}$",
               "pareto_figure_energy"),
}
import sys
which = sys.argv[1] if len(sys.argv) > 1 else "vm"
col_mean, col_med, ylab, outname = METRICS[which]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
for ax, col, title in [
        (axes[0], col_mean, "Mean over units (5% trimmed)"),
        (axes[1], col_med, "Median over units")]:
    for m, c, ls, mk, mfc, lab in SERIES:
        sub = agg[agg["method"] == m].sort_values("cells")
        ax.plot(sub["cells"] / 1e3, sub[col], ls, color=c, marker=mk,
                markersize=6.5 if mk == "*" else 4.5,
                markerfacecolor=mfc, markeredgecolor=c, markeredgewidth=1.1,
                lw=1.7, label=lab, clip_on=False, zorder=3)
    co = agg[agg["method"] == "coarse"].iloc[0]
    ax.plot(co["cells"] / 1e3, co[col], "X", color=GRAY, markersize=7,
            zorder=3, label="Coarse (no refinement)" if col == col_mean else None)
    ax.annotate("coarse", (co["cells"] / 1e3, co[col]),
                textcoords="offset points", xytext=(8, -3),
                fontsize=8, color=INK2)
    for m, lab, c, off in FAMILY_LABELS:
        sub = agg[agg["method"] == m].sort_values("cells").iloc[-1]
        ax.annotate(lab, (sub["cells"] / 1e3, sub[col]),
                    textcoords="offset points", xytext=off,
                    fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=BASE, lw=0.7,
                                    shrinkA=2, shrinkB=2))
    ax.set_title(title, color=INK2, loc="left", pad=6)
    ax.set_xlabel("Mesh cells ($\\times 10^3$, mean at matched budget)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 82)
    ax.tick_params(length=3)

axes[0].set_ylabel(ylab)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=7.6, bbox_to_anchor=(0.5, -0.04), handlelength=2.6,
           columnspacing=1.2)
fig.suptitle("Accuracy vs mesh budget (735 units: 152 objects $\\times$ 5 load cases, "
             "budgets matched per level)", fontsize=9, y=1.0, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(f"{outname}.pdf", bbox_inches="tight")
fig.savefig(f"{outname}.png", dpi=300, bbox_inches="tight")
print(f"wrote {outname}.pdf/png")
