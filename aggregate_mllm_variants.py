"""Aggregate the MLLM prompt/model-variant campaign (gpt-5.4-mini max/mid/none,
gemini-3-flash mid/none) into the rebuttal-table metrics, matching the
conventions of the existing tables:
  - cohort: 773 (object, load_case) units (grefem_max @ h1.0 in
    size_sensitivity_data.csv)
  - per (experiment, h_scale): 5%-trimmed mean and median of
    rel_vm_p99_err and rel_energy_crit_err, mean n_cells, n units
  - ok and ok_norefine rows both count; dedupe (object, load_case) keep=last
Sources: all shard CSVs anywhere (the variants appear in many older shard dirs)
plus the new mllm_shards/{hs*,pool} CSVs.
"""
import csv, glob, re, sys
import numpy as np
import pandas as pd

ss = pd.read_csv("size_sensitivity_data.csv", dtype={"object": str})
coh = set(map(tuple, ss[(ss.method == "grefem_max") & (ss.h_scale == 1.0)][
    ["object", "load_case"]].values))

EXPS = {
    "gpt-5.4-mini_geo_maxprompt": "GPT-max",
    "gpt-5.4-mini_geo_midprompt": "GPT-mid",
    "gpt-5.4-mini_geo_noneprompt": "GPT-none",
    "gemini-3-flash-preview_geo_midprompt": "GReFEM-mid",
    "gemini-3-flash-preview_geo_noneprompt": "GReFEM-none",
}
HS = ["1.0", "1.25", "1.6", "2.0"]

rows = []
files = [p for p in glob.glob("*shards*/**/*.csv", recursive=True)
         if "_aggregated" not in p and "done_manifest" not in p]
for p in files:
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        need = {"object", "load_case", "candidate", "status",
                "rel_vm_p99_err", "rel_energy_crit_err"}
        if not rd.fieldnames or not need <= set(rd.fieldnames):
            continue
        for r in rd:
            if r["status"] not in ("ok", "ok_norefine"):
                continue
            exp = next((e for e in EXPS if e in r["candidate"]), None)
            if exp is None or "_5views_11grid_1run" not in r["candidate"]:
                continue
            if (r["object"], r["load_case"]) not in coh:
                continue
            m = re.search(r"_hs([0-9.]+?)(?:_fine2_mesh)?_refined",
                          r.get("candidate_msh", "") or "")
            h = {"2": "2.0"}.get(m.group(1), m.group(1)) if m else "1.0"
            if h not in HS:
                continue
            rows.append({
                "object": r["object"], "load_case": r["load_case"],
                "exp": exp, "h": h,
                "vm": float(r["rel_vm_p99_err"] or "nan"),
                "en": float(r["rel_energy_crit_err"] or "nan"),
                "cells": float(r.get("n_cells") or "nan"),
            })

df = pd.DataFrame(rows).drop_duplicates(["object", "load_case", "exp", "h"],
                                        keep="last")

def tmean(x, frac=0.05):
    x = np.sort(x[~np.isnan(x)])
    k = int(len(x) * frac)
    return float(np.mean(x[k:len(x) - k])) if len(x) > 2 * k else float("nan")

out = []
for exp, name in EXPS.items():
    for h in HS:
        s = df[(df.exp == exp) & (df.h == h)]
        out.append({
            "method": name, "h": h, "n": len(s),
            "vm_tmean": round(tmean(s.vm.values), 3),
            "vm_med": round(float(np.nanmedian(s.vm.values)), 3),
            "en_tmean": round(tmean(s.en.values), 3),
            "en_med": round(float(np.nanmedian(s.en.values)), 3),
            "cells": int(np.nanmean(s.cells.values)),
        })
res = pd.DataFrame(out)
res.to_csv("mllm_variant_table.csv", index=False)
print(res.to_string(index=False))

# reference: GReFEM (gemini maxprompt) from size_sensitivity_data for context
print("\nGReFEM (grefem_max) reference from size_sensitivity_data.csv:")
for h in [1.0, 1.25, 1.6, 2.0]:
    g = ss[(ss.method == "grefem_max") & (ss.h_scale == h)]
    print(f"  h{h}: vm_tmean={tmean(g.vm.values):.3f} "
          f"vm_med={np.nanmedian(g.vm.values):.3f} "
          f"en_tmean={tmean(g.en.values):.3f} "
          f"en_med={np.nanmedian(g.en.values):.3f} "
          f"cells={np.nanmean(g.cells.values):.0f} n={len(g)}")
