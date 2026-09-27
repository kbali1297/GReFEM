#!/usr/bin/env python
"""Final combined aggregation: 170-object extra set (f2x_shards) + original
30-object set (fine2ref_shards). Complete-case at unit level, ortho candidates
only, oracle-sanity exclusion, per-load-case mean/median tables + Wilcoxon."""
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LCS = ["bending", "compression", "torsion", "bending_compression", "torsion_compression"]
DROP_ORIG = ["Electrical_Parts_Servos_SG-90_SG90-4-arms-horn", "00210021"]

CANON = {
    "coarse": "coarse",
    "zz_oracle": "zz_oracle",
    "zz_oracle_spread": "zz_oracle_spread",
    "heuristic_baseline_DENSE_dihedral30.0deg": "heuristic",
    "heuristic_baseline_DENSE_dihedral30.0deg_sub": "heuristic_sub",
}

def canon(cand, lc):
    if cand in CANON:
        return CANON[cand]
    m = re.match(rf"{lc}_gemini.*_geo_(max|mid|none)prompt_ortho_5views", cand)
    if m:
        return f"grefem_{m.group(1)}"
    return None  # random views etc. -> exclude

def load(globpat, objs=None):
    files = [f for f in glob.glob(globpat) if "aggregated" not in f]
    df = pd.concat([pd.read_csv(f, dtype={"object": str}) for f in files],
                   ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    if objs is not None:
        df = df[df["object"].isin(objs)]
    df["method"] = [canon(c, lc) for c, lc in zip(df["candidate"], df["load_case"])]
    df = df.dropna(subset=["method"])
    df = df.drop_duplicates(["object", "load_case", "method"])
    return df

objs_extra = set(l.strip() for l in open("test_meshes_extra/analysis_objects.txt") if l.strip())
df_x = load("f2x_shards/*_f2x*_w*.csv", objs_extra)
df_o = load("fine2ref_shards/*_f2_w*.csv")
df_o = df_o[~df_o["object"].isin(DROP_ORIG)]
df_sx = load("spread_shards/extra_*_w*.csv", objs_extra)
df_so = load("spread_shards/orig_*_w*.csv")
df_so = df_so[~df_so["object"].isin(DROP_ORIG)]
df_sx = df_sx[df_sx["method"] == "zz_oracle_spread"]
df_so = df_so[df_so["method"] == "zz_oracle_spread"]
df_x = pd.concat([df_x, df_sx], ignore_index=True)
df_o = pd.concat([df_o, df_so], ignore_index=True)

METHODS = ["coarse", "grefem_max", "grefem_mid", "grefem_none",
           "heuristic", "heuristic_sub", "zz_oracle", "zz_oracle_spread"]

df = pd.concat([df_x.assign(src="extra"), df_o.assign(src="orig")], ignore_index=True)

# complete-case units: all 8 methods ok
cnt = df.groupby(["object", "load_case"])["method"].nunique()
complete = set(cnt[cnt >= 8].index)
df = df[[t in complete for t in zip(df["object"], df["load_case"])]]

# oracle sanity exclusion: unit dropped if BOTH oracle variants have
# rel_vm_p99_err > 1 (keep units where at least one oracle is sane)
orc = df[df["method"].isin(["zz_oracle", "zz_oracle_spread"])]
orc_min = orc.groupby(["object", "load_case"])["rel_vm_p99_err"].min()
bad_units = set(orc_min[orc_min > 1].index)
df = df[[t not in bad_units for t in zip(df["object"], df["load_case"])]]

print(f"extra objects: {df[df.src=='extra']['object'].nunique()}, "
      f"orig objects: {df[df.src=='orig']['object'].nunique()}")
print(f"units: complete {len(complete)}, oracle-excluded {len(bad_units)}, "
      f"final {df.groupby(['object','load_case']).ngroups}")

METRICS = ["rel_vm_p99_err", "rel_energy_crit_err"]

for lc in LCS:
    sub = df[df["load_case"] == lc]
    piv = {m: sub[sub["method"] == m].set_index("object") for m in METHODS}
    n_units = sub["object"].nunique()
    print(f"\n{'='*100}\nLOAD CASE: {lc}   (n = {n_units} objects)\n{'='*100}")
    tab = sub.groupby("method").agg(
        vm_p99_mean=("rel_vm_p99_err", "mean"),
        vm_p99_median=("rel_vm_p99_err", "median"),
        energy_mean=("rel_energy_crit_err", "mean"),
        energy_median=("rel_energy_crit_err", "median"),
        n_cells_mean=("n_cells", "mean"),
        n=("rel_vm_p99_err", "size"),
    ).reindex(METHODS)
    print(tab.round(4).to_string())
    # paired Wilcoxon: grefem_max vs baselines, both metrics
    print("\nWilcoxon (paired, two-sided), grefem_max vs:")
    a = piv["grefem_max"]
    for other in ["coarse", "heuristic", "heuristic_sub", "grefem_none",
                  "zz_oracle", "zz_oracle_spread"]:
        b = piv[other]
        common = a.index.intersection(b.index)
        for met in METRICS:
            x, y = a.loc[common, met], b.loc[common, met]
            try:
                stat, p = wilcoxon(x, y)
            except ValueError:
                p = np.nan
            print(f"  vs {other:14s} {met:22s} n={len(common):3d} "
                  f"med_diff={float((x-y).median()):+.4f} p={p:.2e}")

# overall pooled table
print(f"\n{'='*100}\nPOOLED over all load cases\n{'='*100}")
tab = df.groupby("method").agg(
    vm_p99_mean=("rel_vm_p99_err", "mean"),
    vm_p99_median=("rel_vm_p99_err", "median"),
    energy_mean=("rel_energy_crit_err", "mean"),
    energy_median=("rel_energy_crit_err", "median"),
    n=("rel_vm_p99_err", "size"),
).reindex(METHODS)
print(tab.round(4).to_string())

df.to_csv("combined_200obj_table.csv", index=False)
print("\nWrote combined_200obj_table.csv")
