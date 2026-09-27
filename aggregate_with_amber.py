#!/usr/bin/env python
"""Add the AMBER row to the main comparison table.

Reads the AMBER QoI rows produced by compute_local_error_tables.py (candidate
label "<lc>_amber_refined" -> method "amber") together with the existing
combined table, restricts to units where every method is present, and prints
per-load-case and pooled mean/median error plus paired Wilcoxon tests of
GReFEM against every baseline.

Usage:
    python aggregate_with_amber.py --amber_glob 'amber_eval_shards/*.csv'
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
BASE_METHODS = ["coarse", "grefem_max", "grefem_mid", "grefem_none",
                "heuristic", "heuristic_sub", "zz_oracle"]


def canon(cand, lc):
    if cand == "coarse":
        return "coarse"
    if cand == "zz_oracle":
        return "zz_oracle"
    if cand == "heuristic_baseline_DENSE_dihedral30.0deg":
        return "heuristic"
    if cand == "heuristic_baseline_DENSE_dihedral30.0deg_sub":
        return "heuristic_sub"
    if cand.endswith("_amberdef_refined") or cand.endswith("_amberdef") or cand == "amberdef":
        return "amberdef"
    if cand.endswith("_amber_refined") or cand.endswith("_amber") or cand == "amber":
        return "amber"
    m = re.match(rf"{lc}_gemini.*_geo_(max|mid|none)prompt_ortho_5views", cand)
    return f"grefem_{m.group(1)}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", default="combined_200obj_table.csv")
    ap.add_argument("--amber_glob", default="amber_eval_shards/*.csv")
    ap.add_argument("--out", default="combined_with_amber.csv")
    args = ap.parse_args()

    base = pd.read_csv(args.combined, dtype={"object": str})
    base = base[base["method"].isin(BASE_METHODS)][
        ["object", "load_case", "method", "src", "rel_vm_p99_err",
         "rel_energy_crit_err", "n_cells"]]

    files = [f for f in glob.glob(args.amber_glob) if "aggregated" not in f]
    if not files:
        raise SystemExit(f"no AMBER result CSVs matched {args.amber_glob}")
    am = pd.concat([pd.read_csv(f, dtype={"object": str}) for f in files],
                   ignore_index=True)
    am = am[am["status"].astype(str).str.startswith("ok")].copy()
    am["method"] = [canon(str(c), lc)
                    for c, lc in zip(am["candidate"], am["load_case"])]
    am = am[am["method"].isin(["amber", "amberdef"])].drop_duplicates(
        ["object", "load_case", "method"])
    am = am[["object", "load_case", "method", "rel_vm_p99_err",
             "rel_energy_crit_err", "n_cells"]]
    am["src"] = "extra"
    print(f"AMBER rows: {len(am)} over {am['object'].nunique()} objects")

    df = pd.concat([base, am], ignore_index=True)
    methods = BASE_METHODS + ["amber", "amberdef"]
    cnt = df.groupby(["object", "load_case"])["method"].nunique()
    complete = set(cnt[cnt >= len(methods)].index)
    df = df[[t in complete for t in zip(df["object"], df["load_case"])]]
    print(f"complete units (all {len(methods)} methods): {len(complete)}, "
          f"objects {df['object'].nunique()}")
    df.to_csv(args.out, index=False)

    for scope in LCS + ["POOLED"]:
        sub = df if scope == "POOLED" else df[df["load_case"] == scope]
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["object", "load_case"], columns="method",
                              values="rel_vm_p99_err")
        pe = sub.pivot_table(index=["object", "load_case"], columns="method",
                             values="rel_energy_crit_err")
        pc = sub.pivot_table(index=["object", "load_case"], columns="method",
                             values="n_cells")
        print(f"\n{'='*88}\n{scope}  (n={len(piv)} units)\n{'='*88}")
        print(f"{'method':14s} {'vm_p99 mean':>12s} {'median':>8s} "
              f"{'energy mean':>12s} {'median':>8s} {'cells':>9s}")
        for m in methods:
            if m not in piv:
                continue
            print(f"{m:14s} {piv[m].mean():12.4f} {piv[m].median():8.4f} "
                  f"{pe[m].mean():12.4f} {pe[m].median():8.4f} "
                  f"{pc[m].mean():9.0f}")
        if scope == "POOLED":
            print("\ngrefem_max vs others (paired Wilcoxon, vm_p99):")
            for m in methods:
                if m in ("grefem_max",) or m not in piv:
                    continue
                x, y = piv["grefem_max"], piv[m]
                k = x.notna() & y.notna()
                try:
                    _, p = wilcoxon(x[k], y[k])
                except ValueError:
                    p = np.nan
                d = (x[k] - y[k])
                print(f"  vs {m:14s} n={int(k.sum()):3d} "
                      f"mean_diff {d.mean():+.4f} med {d.median():+.4f} "
                      f"win {(d < -0.01).mean():.1%} loss {(d > 0.01).mean():.1%} "
                      f"p={p:.2e}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
