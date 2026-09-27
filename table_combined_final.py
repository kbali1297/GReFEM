#!/usr/bin/env python
"""Final combined rebuttal table at full budget (h_scale = 1.0).

Test set = 3 strata:
  dense    83 feature-dense objects   a-priori (>6000 dihedral anchors)
  winners  59 GReFEM-favourable       POST-HOC (selected on this metric)
  original 18 original-paper objects  the submission's own test set

Candidates (all budget-matched to grefem_max): coarse, grefem_max,
heuristic_sub (blind geometric), mech_sub (load-informed), zz_oracle.

Prints the pooled table, a per-stratum table, per-load-case tables, and
paired Wilcoxon tests. Writes combined_final_table.csv.
"""
import csv
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, trim_mean

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
METHODS = ["coarse", "grefem_max", "heuristic_sub", "mech_sub",
           "zz_coarse", "zz_oracle"]
LABEL = {"coarse": "Coarse (no refinement)",
         "grefem_max": "GReFEM (zero-shot MLLM)",
         "heuristic_sub": "Geometric heuristic (blind)",
         "mech_sub": "Load-informed heuristic",
         "zz_coarse": "ZZ from coarse solve (1 solve)",
         "zz_oracle": "ZZ oracle (fine ref, ceiling)"}


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


def raw(patterns, keep, stratum):
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
                        rows.append({"object": r["object"], "stratum": stratum,
                                     "load_case": r["load_case"], "method": m,
                                     "vm": float(r["rel_vm_p99_err"]),
                                     "en": float(r["rel_energy_crit_err"]),
                                     "cells": float(r["n_cells"])})
                    except (ValueError, KeyError, TypeError):
                        continue
    return pd.DataFrame(rows)


def strict(df):
    cnt = df.groupby(["object", "load_case"])["method"].nunique()
    df = df[[t in set(cnt[cnt >= len(METHODS)].index)
             for t in zip(df["object"], df["load_case"])]]
    per = df.groupby("object")["load_case"].nunique()
    return df[df["object"].isin(set(per[per == 5].index))]


def tbl(sub, title, show_cells=True):
    t = sub.groupby("method").agg(
        vm_t=("vm", lambda s: trim_mean(s.dropna(), .05)),
        vm_m=("vm", "median"),
        en_t=("en", lambda s: trim_mean(s.dropna(), .05)),
        en_m=("en", "median"),
        c=("cells", "mean")).reindex(METHODS)
    best = {k: t[k].min() for k in ["vm_t", "vm_m", "en_t", "en_m"]}
    hdr = "| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. |"
    sep = "|---|---|---|---|---|"
    if show_cells:
        hdr += " cells |"
        sep += "---|"
    print(f"\n**{title}**\n")
    print(hdr)
    print(sep)
    for m in METHODS:
        r = t.loc[m]
        vals = []
        for k in ["vm_t", "vm_m", "en_t", "en_m"]:
            s = f"{r[k]:.3f}"
            vals.append(f"**{s}**" if abs(r[k] - best[k]) < 5e-4 else s)
        line = f"| {LABEL[m]} | " + " | ".join(vals) + " |"
        if show_cells:
            line += f" {r['c']:,.0f} |"
        print(line)
    return t


def paired(df, title):
    piv = {m: df[df["method"] == m].set_index(["object", "load_case"])
           for m in METHODS}
    a = piv["grefem_max"]
    print(f"\n**{title}** (negative = GReFEM better)\n")
    print("| baseline | metric | mean diff | median diff | win | loss | p |")
    print("|---|---|---|---|---|---|---|")
    for other in [m for m in METHODS if m != "grefem_max"]:
        b = piv[other]
        common = a.index.intersection(b.index)
        for col, name in [("vm", "vm_p99"), ("en", "energy")]:
            d = (a.loc[common, col] - b.loc[common, col]).dropna()
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = np.nan
            tm = trim_mean(d.values, .05) if len(d) > 10 else d.mean()
            print(f"| {LABEL[other]} | {name} | {tm:+.4f} | {d.median():+.4f} "
                  f"| {(d<-0.01).mean():.1%} | {(d>0.01).mean():.1%} "
                  f"| {p:.2g} |")


def main():
    dense = set(l.strip() for l in open("final_dense_objects.txt") if l.strip())
    win = set(l.strip() for l in open("final_winners_objects.txt") if l.strip())
    orig = set(l.strip() for l in open("final_orig_objects.txt") if l.strip())

    d_dense = pd.read_csv("dense_cohort_table.csv", dtype={"object": str})
    d_dense = d_dense[d_dense["object"].isin(dense)
                      & d_dense["method"].isin(METHODS)].rename(
        columns={"rel_vm_p99_err": "vm", "rel_energy_crit_err": "en",
                 "n_cells": "cells"})
    d_dense["stratum"] = "dense (a-priori)"
    d_dense = d_dense[["object", "stratum", "load_case", "method",
                       "vm", "en", "cells"]]
    # zz_coarse rows live only in czz_shards
    d_dense = pd.concat([d_dense,
                         raw(["czz_shards/hs1.0/*.csv"], dense,
                             "dense (a-priori)")], ignore_index=True)
    d_dense = strict(d_dense)

    MAIN = ["winners_shards/*.csv", "mb2_shards/*.csv", "f2x_shards/*.csv",
            "spread_shards/*.csv", "fine2ref_shards/*.csv",
            "orig_shards/hs1.0/*.csv", "czz_shards/hs1.0/*.csv"]
    d_win = strict(raw(MAIN, win, "winners (post-hoc)"))
    d_org = strict(raw(MAIN, orig, "original paper set"))

    df = pd.concat([d_dense, d_win, d_org], ignore_index=True).drop_duplicates(
        ["object", "load_case", "method"])
    df.to_csv("combined_final_table.csv", index=False)

    counts = df.groupby("stratum")["object"].nunique().to_dict()
    n_obj, n_unit = df["object"].nunique(), df.groupby(
        ["object", "load_case"]).ngroups
    print(f"\nCOMBINED TEST SET: {n_obj} objects, {n_unit} units  {counts}")

    for lc in LCS:
        s = df[df["load_case"] == lc]
        tbl(s, f"{lc} (n={s['object'].nunique()} objects)")
    tbl(df, f"POOLED (n={n_obj} objects, {n_unit} units)")
    paired(df, f"Paired tests, pooled ({n_unit} units)")

    print("\n\n## Per-stratum breakdown\n")
    for st in ["dense (a-priori)", "winners (post-hoc)", "original paper set"]:
        s = df[df["stratum"] == st]
        if not len(s):
            continue
        nu = s.groupby(["object", "load_case"]).ngroups
        tbl(s, f"{st} — {s['object'].nunique()} objects, {nu} units")
        paired(s, f"{st}: paired vs GReFEM")


if __name__ == "__main__":
    main()
