#!/usr/bin/env python
"""Aggregate the budget-sweep (Pareto) shards: pareto_shards/hs{1.25,1.6,2.0}
plus the existing default-sizing results (hs=1.0) from
combined_200obj_table.csv. Complete-case at the (object, load_case) unit
level ACROSS ALL LEVELS so every curve is computed on the same units.
Rows with status ok_norefine are kept (mesh solved; the method simply could
not spend more budget at that sizing) and counted.

Outputs: pareto_table.csv (per unit/level/method), pareto_agg.csv +
markdown/stdout summary with per-level Wilcoxon grefem vs heuristic.
"""
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LCS = ["bending", "compression", "torsion", "bending_compression",
       "torsion_compression"]
METHODS = ["coarse", "grefem_max", "grefem_mid", "grefem_none",
           "heuristic", "heuristic_sub", "zz_oracle"]

CANON = {
    "coarse": "coarse",
    "zz_oracle": "zz_oracle",
    "heuristic_baseline_DENSE_dihedral30.0deg": "heuristic",
    "heuristic_baseline_DENSE_dihedral30.0deg_sub": "heuristic_sub",
}


def canon(cand, lc):
    if cand in CANON:
        return CANON[cand]
    m = re.match(rf"{lc}_gemini.*_geo_(max|mid|none)prompt_ortho_5views", cand)
    return f"grefem_{m.group(1)}" if m else None


def load_level(d):
    files = [f for f in glob.glob(os.path.join(d, "*_w*.csv"))]
    df = pd.concat([pd.read_csv(f, dtype={"object": str}) for f in files],
                   ignore_index=True)
    df = df[df["status"].isin(["ok", "ok_norefine"])].copy()
    df["method"] = [canon(c, lc) for c, lc in
                    zip(df["candidate"], df["load_case"])]
    df = df.dropna(subset=["method"])
    df = df.drop_duplicates(["object", "load_case", "method"])
    return df


def main():
    frames = []
    for d in sorted(glob.glob("pareto_shards/hs*/")):
        hs = float(os.path.basename(d.rstrip("/"))[2:])
        sub = load_level(d)
        sub["h_scale"] = hs
        frames.append(sub)
        n_nr = (sub["status"] == "ok_norefine").sum()
        print(f"hs{hs}: {len(sub)} rows ({n_nr} ok_norefine), "
              f"{sub.groupby(['object', 'load_case']).ngroups} units")

    # default-sizing level (hs=1.0) from the combined main table
    base = pd.read_csv("combined_200obj_table.csv", dtype={"object": str})
    base = base[base["method"].isin(METHODS)].copy()
    base["h_scale"] = 1.0
    frames.append(base)
    print(f"hs1.0 (main table): {len(base)} rows, "
          f"{base.groupby(['object', 'load_case']).ngroups} units")

    cols = ["object", "load_case", "method", "h_scale", "n_cells", "n_dofs",
            "rel_vm_p99_err", "rel_energy_crit_err", "status",
            "t_candidate_s"]
    df = pd.concat([f.reindex(columns=cols) for f in frames],
                   ignore_index=True)

    # complete case: unit has all 7 methods at ALL levels
    levels = sorted(df["h_scale"].unique())
    cnt = df.groupby(["object", "load_case"]).apply(
        lambda g: all(
            set(g[g["h_scale"] == h]["method"]) >= set(METHODS)
            for h in levels),
        include_groups=False)
    keep = set(cnt[cnt].index)
    df = df[[t in keep for t in zip(df["object"], df["load_case"])]]
    print(f"\ncomplete-case units across {len(levels)} levels: {len(keep)} "
          f"({df['object'].nunique()} objects)")

    df.to_csv("pareto_table.csv", index=False)

    # ---- per-level aggregate ----
    print("\n" + "=" * 100)
    print("PARETO AGGREGATE (pooled over load cases, complete-case units)")
    print("=" * 100)
    from scipy.stats import trim_mean
    tm = lambda s: trim_mean(s.dropna(), 0.05)
    agg = df.groupby(["h_scale", "method"]).agg(
        vm_p99_mean=("rel_vm_p99_err", "mean"),
        vm_p99_tmean=("rel_vm_p99_err", tm),
        vm_p99_med=("rel_vm_p99_err", "median"),
        energy_mean=("rel_energy_crit_err", "mean"),
        energy_tmean=("rel_energy_crit_err", tm),
        energy_med=("rel_energy_crit_err", "median"),
        cells=("n_cells", "mean"),
        t_med=("t_candidate_s", "median"),
        n=("rel_vm_p99_err", "size"),
    )
    agg = agg.reindex(pd.MultiIndex.from_product([levels, METHODS],
                                                 names=["h_scale", "method"]))
    print(agg.round(4).to_string())
    agg.to_csv("pareto_agg.csv")

    # ---- per-level paired tests: grefem vs heuristic ----
    print("\nPaired per level (pooled units), vm_p99:")
    for h in levels:
        sub = df[df["h_scale"] == h]
        piv = sub.pivot_table(index=["object", "load_case"], columns="method",
                              values="rel_vm_p99_err")
        for g in ["grefem_max", "grefem_mid", "grefem_none"]:
            d_ = (piv[g] - piv["heuristic"]).dropna()
            try:
                _, p = wilcoxon(d_)
            except ValueError:
                p = np.nan
            print(f"  hs{h:<5} {g:12s} vs heuristic: n={len(d_):3d} "
                  f"mean {d_.mean():+.4f} med {d_.median():+.4f} "
                  f"win {(d_ < -0.01).mean():.1%} loss {(d_ > 0.01).mean():.1%} "
                  f"p={p:.2e}")

    print("\nWrote pareto_table.csv, pareto_agg.csv")


if __name__ == "__main__":
    main()
