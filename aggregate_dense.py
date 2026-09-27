#!/usr/bin/env python
"""Feature-dense cohort rebuttal table.

STRICT inclusion: an object enters the table only if ALL 5 load cases
completed with every reported candidate present. Reported candidates:

    coarse         no refinement (floor)
    grefem_max     GReFEM, geo_max prompt, ortho 5 views
    heuristic_sub  blind geometric heuristic, anchors matched to GReFEM count
    mech_sub       load-informed heuristic (nominal-stress weighted), matched
    zz_oracle      top-percentile ZZ of the fine reference (ceiling)
    [zz_coarse]    ZZ from the coarse solve (practical solver-informed), if run
    [mlgnn_sub]    learned ZZ surrogate, if run

All refined candidates are budget-matched to the grefem_max element count.
Reads every dense_shards/*.csv (sharded + per-unit queue outputs).
"""
import argparse
import csv
import glob
import re

import pandas as pd
from scipy.stats import wilcoxon, trim_mean

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
CORE = ["coarse", "grefem_max", "heuristic_sub", "mech_sub", "zz_oracle"]
OPTIONAL = ["zz_coarse", "mlgnn_sub"]
LABEL = {"coarse": "coarse (no refinement)",
         "grefem_max": "GReFEM (max prompt)",
         "heuristic_sub": "geometric heuristic",
         "mech_sub": "load-informed heuristic",
         "zz_oracle": "ZZ oracle (fine ref)",
         "zz_coarse": "ZZ from coarse solve",
         "mlgnn_sub": "learned ZZ surrogate"}


def fam(cand, lc):
    if cand in ("coarse", "zz_oracle", "zz_coarse"):
        return cand
    if re.match(rf"{lc}_gemini.*_geo_maxprompt_ortho_5views", cand):
        return "grefem_max"
    if cand.startswith("heuristic_baseline_DENSE") and cand.endswith("_sub"):
        return "heuristic_sub"
    if cand.startswith("heuristic_mech_") and cand.endswith("_sub"):
        return "mech_sub"
    if cand.startswith("mlgnn") and cand.endswith("_sub"):
        return "mlgnn_sub"
    return None


def load():
    rows = []
    for f in glob.glob("dense_shards/*.csv"):
        if "aggregated" in f:
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if not (r.get("status") or "").startswith("ok"):
                    continue
                m = fam(r["candidate"], r["load_case"])
                if m is None:
                    continue
                try:
                    rows.append({
                        "object": r["object"], "load_case": r["load_case"],
                        "method": m,
                        "rel_vm_p99_err": float(r["rel_vm_p99_err"]),
                        "rel_energy_crit_err": float(r["rel_energy_crit_err"]),
                        "n_cells": float(r["n_cells"])})
                except (ValueError, KeyError, TypeError):
                    continue
    return pd.DataFrame(rows).drop_duplicates(
        ["object", "load_case", "method"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle_sanity", action="store_true", default=True,
                    help="drop objects with any unit whose oracle error > 1")
    ap.add_argument("--no_oracle_sanity", dest="oracle_sanity",
                    action="store_false")
    ap.add_argument("--extra", nargs="*", default=[], choices=OPTIONAL,
                    help="additionally report these candidates (each one "
                         "shrinks the strict set to objects that have it)")
    args = ap.parse_args()

    df = load()
    methods = CORE + [m for m in args.extra if m in set(df["method"])]
    print(f"raw ok rows: {len(df)} | units "
          f"{df.groupby(['object','load_case']).ngroups} | "
          f"objects {df['object'].nunique()}")
    print(f"reporting methods: {methods}")

    # 1. unit-level complete case over the reported methods
    need = set(methods)
    has = df.groupby(["object", "load_case"])["method"].apply(
        lambda s: need <= set(s))
    df = df[[t in set(has[has].index)
             for t in zip(df["object"], df["load_case"])]]

    # 2. STRICT: object must have all 5 load cases complete
    per_obj = df.groupby("object")["load_case"].nunique()
    df = df[df["object"].isin(set(per_obj[per_obj == len(LCS)].index))]
    print(f"after strict all-5-load-case filter: {df['object'].nunique()} "
          f"objects, {df.groupby(['object','load_case']).ngroups} units")

    # 3. oracle sanity (same convention as the main table)
    if args.oracle_sanity:
        orc = df[df["method"] == "zz_oracle"]
        bad_objs = set(orc[orc["rel_vm_p99_err"] > 1]["object"])
        if bad_objs:
            df = df[~df["object"].isin(bad_objs)]
            print(f"oracle-sanity: dropped {len(bad_objs)} objects "
                  f"(oracle error > 1) -> {df['object'].nunique()} objects, "
                  f"{df.groupby(['object','load_case']).ngroups} units")

    n_obj = df["object"].nunique()
    n_unit = df.groupby(["object", "load_case"]).ngroups
    df.to_csv("dense_cohort_table.csv", index=False)

    def tbl(sub, title):
        t = sub.groupby("method").agg(
            vm_tmean=("rel_vm_p99_err", lambda s: trim_mean(s.dropna(), .05)),
            vm_med=("rel_vm_p99_err", "median"),
            en_tmean=("rel_energy_crit_err",
                      lambda s: trim_mean(s.dropna(), .05)),
            en_med=("rel_energy_crit_err", "median"),
            cells=("n_cells", "mean"),
            n=("rel_vm_p99_err", "size")).reindex(methods)
        print(f"\n{'='*92}\n{title}\n{'='*92}")
        print(t.round(4).to_string())
        return t

    for lc in LCS:
        s = df[df["load_case"] == lc]
        if len(s):
            tbl(s, f"{lc}   ({s['object'].nunique()} objects)")
    tab = tbl(df, f"POOLED   ({n_obj} objects, {n_unit} units)")

    print(f"\n{'='*92}\nPAIRED: grefem_max minus baseline "
          f"(negative = GReFEM better)\n{'='*92}")
    piv = {m: df[df["method"] == m].set_index(["object", "load_case"])
           for m in methods}
    a = piv["grefem_max"]
    for other in [m for m in methods if m != "grefem_max"]:
        b = piv[other]
        common = a.index.intersection(b.index)
        for col, name in [("rel_vm_p99_err", "vm_p99"),
                          ("rel_energy_crit_err", "energy")]:
            d = (a.loc[common, col] - b.loc[common, col]).dropna()
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = float("nan")
            # 5%-trimmed mean: raw means are destroyed by units whose
            # reference energy in Omega_crit is ~0 (blow-up ratios)
            tm = trim_mean(d.values, 0.05) if len(d) > 10 else d.mean()
            print(f"  vs {other:14s} {name:7s} n={len(d):4d} "
                  f"tmean {tm:+.4f} med {d.median():+.4f} "
                  f"win {(d < -0.01).mean():5.1%} loss {(d > 0.01).mean():5.1%}"
                  f"  p={p:.3g}")

    print(f"\n{'='*92}\nORACLE-GAP CLOSURE (pooled, 5% trimmed mean vm_p99)"
          f"\n{'='*92}")
    c, o = tab.loc["coarse", "vm_tmean"], tab.loc["zz_oracle", "vm_tmean"]
    for m in [x for x in methods if x not in ("coarse", "zz_oracle")]:
        v = tab.loc[m, "vm_tmean"]
        print(f"  {LABEL[m]:26s} {v:.4f}   closes {100*(c-v)/(c-o):5.1f}%")

    print(f"\nWrote dense_cohort_table.csv ({n_obj} objects, {n_unit} units)")


if __name__ == "__main__":
    main()
