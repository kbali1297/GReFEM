#!/usr/bin/env python
"""Rebuttal Pareto figure: downstream error vs element budget.

Follows the evaluation protocol of the learned-AMR literature -- ASMR
(Freymuth et al., NeurIPS 2023, Fig. 4/5) and AMBER (Freymuth et al., NeurIPS
2025, Fig. 4) both report solution error against number of mesh elements on a
log-log Pareto front, sweeping a single knob (element penalty / sizing-field
scale). Here the knob is the candidate sizing scale h_scale in
{1.0, 1.25, 1.6, 2.0}, giving ~62k/39k/23k/14k tets.

Left panel: absolute relative error of the QoI (p99 von Mises in Omega_crit).
Right panel: fraction of the achievable error reduction realised, i.e.
    (err_coarse - err_method) / (err_coarse - err_oracle)
which is the ASMR-style normalisation by the coarse mesh and makes the
method ordering readable at a glance.

Usage:
    python plot_pareto_figure.py --agg pareto_agg.csv --out pareto_figure.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    "coarse":        dict(label="Coarse (no refinement)", color="#8c8c8c",
                          marker="s", ls=":"),
    "heuristic":     dict(label="Mesh-geometric heuristic", color="#d62728",
                          marker="^", ls="--"),
    "heuristic_sub": dict(label="Heuristic (subsampled)", color="#ff9896",
                          marker="v", ls="--"),
    "grefem_max":    dict(label="GReFEM (max prompt)", color="#1f77b4",
                          marker="o", ls="-"),
    "grefem_mid":    dict(label="GReFEM (mid prompt)", color="#4c9fd4",
                          marker="o", ls="-"),
    "grefem_none":   dict(label="GReFEM (no physics prompt)", color="#aec7e8",
                          marker="o", ls="-"),
    "zz_oracle":     dict(label="ZZ/SPR oracle (solver-informed)",
                          color="#2ca02c", marker="*", ls="-."),
    "amber":         dict(label="AMBER (trained, NeurIPS'25)", color="#9467bd",
                          marker="D", ls="-"),
}
ORDER = ["coarse", "heuristic", "heuristic_sub", "grefem_none", "grefem_mid",
         "grefem_max", "amber", "zz_oracle"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agg", default="pareto_agg.csv")
    ap.add_argument("--metric", default="vm_p99_med",
                    help="column to plot (vm_p99_med, vm_p99_mean, ...)")
    ap.add_argument("--out", default="pareto_figure.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.agg)
    n = int(df["n"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- left: absolute error vs cells ----
    ax = axes[0]
    for m in ORDER:
        sub = df[df["method"] == m].sort_values("cells")
        if sub.empty:
            continue
        st = STYLE[m]
        if m == "coarse":  # single point, draw as horizontal reference
            ax.axhline(sub[args.metric].iloc[0], color=st["color"],
                       ls=st["ls"], lw=1.2, zorder=1)
            ax.plot(sub["cells"], sub[args.metric], color=st["color"],
                    marker=st["marker"], ms=7, ls="none", label=st["label"])
            continue
        ax.plot(sub["cells"], sub[args.metric], color=st["color"],
                marker=st["marker"], ms=6, ls=st["ls"], lw=1.6,
                label=st["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Number of mesh elements")
    ax.set_ylabel(r"rel. error of $\sigma_{vM}^{p99}$ in $\Omega_{crit}$")
    ax.set_title(f"Downstream error vs element budget (n={n} units)")
    ax.grid(alpha=0.3, which="both", lw=0.5)

    # ---- right: fraction of achievable error reduction ----
    ax = axes[1]
    coarse = df[df["method"] == "coarse"][args.metric].iloc[0]
    orc = df[df["method"] == "zz_oracle"].set_index("h_scale")[args.metric]
    for m in ORDER:
        if m in ("coarse",):
            continue
        sub = df[df["method"] == m].sort_values("cells")
        if sub.empty:
            continue
        gain = [(coarse - r[args.metric]) /
                (coarse - orc.loc[r["h_scale"]] + 1e-12)
                for _, r in sub.iterrows()]
        st = STYLE[m]
        ax.plot(sub["cells"], gain, color=st["color"], marker=st["marker"],
                ms=6, ls=st["ls"], lw=1.6, label=st["label"])
    ax.axhline(1.0, color="#2ca02c", lw=0.8, ls=":")
    ax.axhline(0.0, color="#8c8c8c", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Number of mesh elements")
    ax.set_ylabel("fraction of coarse$\\rightarrow$oracle gap closed")
    ax.set_title("Normalised error reduction (1.0 = solver-informed oracle)")
    ax.grid(alpha=0.3, which="both", lw=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        path = args.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
