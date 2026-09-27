#!/usr/bin/env python
"""Combined test-set options for the rebuttal, computed on the 4 candidates
available for BOTH cohorts (coarse, grefem_max, heuristic_sub, zz_oracle).

  A  dense cohort            111 objects, a-priori (>6000 dihedral anchors)
  B  full main set           152 objects, a-priori (random ABC sample)
  C  A + B pooled            legitimate combined test set, no outcome selection
  D  59 winner objects       POST-HOC: selected because GReFEM beat the
                             heuristic on this very metric -> reporting a
                             GReFEM-vs-heuristic win here is circular
  E  A + D pooled            what "combined test set incl. winners" would give

Every comparison is paired within (object, load_case) at matched budget.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, trim_mean

METHODS = ["coarse", "grefem_max", "heuristic_sub", "zz_oracle"]


def load_dense():
    d = pd.read_csv("dense_cohort_table.csv", dtype={"object": str})
    return d[d["method"].isin(METHODS)][
        ["object", "load_case", "method", "rel_vm_p99_err",
         "rel_energy_crit_err", "n_cells"]]


def load_main():
    d = pd.read_csv("combined_200obj_table.csv", dtype={"object": str})
    d = d.rename(columns={"heuristic_sub": "heuristic_sub"})
    d = d[d["method"].isin(METHODS)]
    return d[["object", "load_case", "method", "rel_vm_p99_err",
              "rel_energy_crit_err", "n_cells"]]


def strict(df):
    """units with all 4 methods, objects with all 5 load cases"""
    cnt = df.groupby(["object", "load_case"])["method"].nunique()
    df = df[[t in set(cnt[cnt >= len(METHODS)].index)
             for t in zip(df["object"], df["load_case"])]]
    per = df.groupby("object")["load_case"].nunique()
    return df[df["object"].isin(set(per[per == 5].index))]


def report(df, name):
    n_obj = df["object"].nunique()
    n_unit = df.groupby(["object", "load_case"]).ngroups
    tab = df.groupby("method").agg(
        vm_tmean=("rel_vm_p99_err", lambda s: trim_mean(s.dropna(), .05)),
        vm_med=("rel_vm_p99_err", "median"),
        cells=("n_cells", "mean")).reindex(METHODS)
    print(f"\n{'='*88}\n{name}\n  {n_obj} objects, {n_unit} units\n{'='*88}")
    print(tab.round(4).to_string())
    piv = {m: df[df["method"] == m].set_index(["object", "load_case"])
           for m in METHODS}
    a = piv["grefem_max"]
    for other in ["heuristic_sub", "zz_oracle"]:
        b = piv[other]
        common = a.index.intersection(b.index)
        d = (a.loc[common, "rel_vm_p99_err"]
             - b.loc[common, "rel_vm_p99_err"]).dropna()
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
        print(f"  grefem vs {other:14s} n={len(d):4d} "
              f"tmean {trim_mean(d.values, .05):+.4f} med {d.median():+.4f} "
              f"win {(d < -0.01).mean():5.1%} loss {(d > 0.01).mean():5.1%} "
              f"p={p:.3g}")
    c, o = tab.loc["coarse", "vm_tmean"], tab.loc["zz_oracle", "vm_tmean"]
    for m in ["grefem_max", "heuristic_sub"]:
        v = tab.loc[m, "vm_tmean"]
        print(f"  {m:14s} closes {100*(c-v)/(c-o):5.1f}% of coarse->oracle gap")
    return tab


def main():
    dense = strict(load_dense())
    main_all = strict(load_main())
    dense_objs = set(dense["object"])
    main_objs = set(main_all["object"]) - dense_objs
    main_all = main_all[main_all["object"].isin(main_objs)]

    winners = set(l.strip() for l in open("winners_extra.txt") if l.strip())
    win_df = main_all[main_all["object"].isin(winners)]

    report(dense, "A  DENSE COHORT  (a-priori: >6000 dihedral anchors)")
    report(main_all, "B  FULL MAIN SET  (a-priori: random ABC sample)")
    report(pd.concat([dense, main_all], ignore_index=True),
           "C  POOLED A+B  <<< legitimate combined test set, no outcome selection")
    report(win_df, "D  59 WINNER OBJECTS  <<< POST-HOC (selected on this metric)")
    report(pd.concat([dense, win_df], ignore_index=True),
           "E  POOLED A+D  <<< 'combined test set incl. winners' (biased)")


if __name__ == "__main__":
    main()
