#!/usr/bin/env python
"""Aggregate the AMBER pareto sweep evals into rebuttal-style tables.

Reads amber_sweep_eval_shards/[AB]_*.csv (per-unit eval shards, skipping the
*_aggregated.csv summaries), parses candidate labels
{lc}_amberf_h{ddd} (adapted) / {lc}_amberfd_h{ddd} (default) into
(method, h_scale), and compares per h-level against grefem_max from
size_sensitivity_data.csv on the shared units (paired Wilcoxon).
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

SUF2H = {"100": 1.0, "125": 1.25, "160": 1.6, "200": 2.0}
PAT = re.compile(r"_(amberf d?|amberfd|amberf)_h(\d{3})$")


def trimmed(x, p=0.05):
    x = np.sort(np.asarray(x, float))
    k = int(len(x) * p)
    return float(np.mean(x[k:len(x) - k])) if len(x) > 2 * k else float(np.mean(x))


def main():
    rows = []
    for p in glob.glob("amber_sweep_eval_shards/[AB]_*.csv"):
        if "_aggregated" in p:
            continue
        try:
            df = pd.read_csv(p, dtype={"object": str})
        except Exception:
            continue
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df[df["candidate"].str.contains("amberf", na=False)]
    m = df["candidate"].str.extract(r"_(amberfd|amberf)_h(\d{3})$")
    df["method"] = m[0]
    df["h_scale"] = m[1].map(SUF2H)
    df = df.dropna(subset=["method", "h_scale"])
    df = df[df["status"].isin(["ok", "ok_norefine"])]
    df = df.dropna(subset=["rel_vm_p99_err", "rel_energy_crit_err"])
    df = df.drop_duplicates(subset=["object", "load_case", "method", "h_scale"],
                            keep="last")

    ss = pd.read_csv("size_sensitivity_data.csv", dtype={"object": str})
    g = ss[ss.method == "grefem_max"].set_index(["object", "load_case", "h_scale"])

    print(f"{len(df)} AMBER eval rows "
          f"({df.drop_duplicates(['object','load_case']).shape[0]} units)")
    out = []
    for method, name in [("amberf", "AMBER (adapted)"),
                         ("amberfd", "AMBER (default)")]:
        for h in [1.0, 1.25, 1.6, 2.0]:
            sub = df[(df.method == method) & (df.h_scale == h)]
            if sub.empty:
                continue
            # paired against grefem_max on shared units
            keys = [(o, lc, h) for o, lc in zip(sub["object"], sub["load_case"])]
            mask = [k in g.index for k in keys]
            subp = sub[mask]
            gref = g.loc[[(o, lc, h) for o, lc in
                          zip(subp["object"], subp["load_case"])]]
            dvm = gref["vm"].values - subp["rel_vm_p99_err"].values
            den = gref["en"].values - subp["rel_energy_crit_err"].values
            wvm = stats.wilcoxon(dvm, alternative="less").pvalue if len(dvm) > 10 else np.nan
            wen = stats.wilcoxon(den, alternative="less").pvalue if len(den) > 10 else np.nan
            out.append({
                "method": name, "h_scale": h, "N": len(sub),
                "vm_trim": trimmed(sub.rel_vm_p99_err),
                "vm_med": float(sub.rel_vm_p99_err.median()),
                "en_trim": trimmed(sub.rel_energy_crit_err),
                "en_med": float(sub.rel_energy_crit_err.median()),
                "cells_med": float(sub.n_cells.median()),
                "N_paired": len(subp),
                "d_vm_med": float(np.median(dvm)) if len(dvm) else np.nan,
                "p_vm(grefem<amber)": wvm,
                "d_en_med": float(np.median(den)) if len(den) else np.nan,
                "p_en(grefem<amber)": wen,
            })
    # grefem reference rows for the same table
    for h in [1.0, 1.25, 1.6, 2.0]:
        gr = ss[(ss.method == "grefem_max") & (ss.h_scale == h)]
        out.append({"method": "GReFEM (max)", "h_scale": h, "N": len(gr),
                    "vm_trim": trimmed(gr.vm), "vm_med": float(gr.vm.median()),
                    "en_trim": trimmed(gr.en), "en_med": float(gr.en.median()),
                    "cells_med": float(gr.cells.median())})
    res = pd.DataFrame(out).sort_values(["h_scale", "method"])
    res.to_csv("amber_pareto_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False,
                        float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
