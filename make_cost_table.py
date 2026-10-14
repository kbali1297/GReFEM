#!/usr/bin/env python
"""Per-case cost table (h x1.0): mean cells, median max-displacement error and
median candidate solve time per policy, on the cohort of combined_final_table.csv.

Usage: python make_cost_table.py [cohort_csv]
"""
import csv
import glob
import re
import sys

import numpy as np
import pandas as pd

COHORT = sys.argv[1] if len(sys.argv) > 1 else "combined_final_table.csv"
GREFEM_LFH = ["gemini-3-flash-preview_geo_maxprompt", "gpt-5.4-mini_geo_maxprompt",
              "claude-haiku-4.5_geo_maxprompt",
              "qwen3-vl-235b-a22b-instruct_geo_maxprompt"]
ORDER = ["coarse", "grefem", "heuristic_sub", "mech_sub", "zz_coarse", "zz_oracle"]


def family(c):
    if c in ("coarse", "zz_oracle", "zz_coarse"):
        return c, ""
    if c.startswith("heuristic_baseline_DENSE") and c.endswith("_sub"):
        return "heuristic_sub", ""
    if c.startswith("heuristic_mech_") and c.endswith("_sub"):
        return "mech_sub", ""
    if "_5views_11grid_1run" in c:
        k = next((k for k in GREFEM_LFH if k in c), None)
        if k:
            return "grefem", k
    return None, None


def main():
    coh = pd.read_csv(COHORT, dtype={"object": str})
    cohort = set(map(tuple, coh[["object", "load_case"]].values))
    rows = []
    need = {"object", "load_case", "candidate", "status", "rel_max_disp_err"}
    # same per-case sources as table_combined_final.py; MLLM variants from any shard
    base_src = (glob.glob("dense_shards/*.csv") + glob.glob("czz_shards/hs1.0/*.csv")
                + [f for pat in ["winners_shards/*.csv", "mb2_shards/*.csv",
                                 "f2x_shards/*.csv", "spread_shards/*.csv",
                                 "fine2ref_shards/*.csv", "orig_shards/hs1.0/*.csv",
                                 "add2_shards/hs1.0/*.csv"] for f in glob.glob(pat)])
    base_src = set(base_src)
    for p in glob.glob("*shards*/**/*.csv", recursive=True):
        if "_aggregated" in p or "done_manifest" in p or p.startswith("amber"):
            continue
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            if not rd.fieldnames or not need <= set(rd.fieldnames):
                continue
            for r in rd:
                if r["status"] not in ("ok", "ok_norefine"):
                    continue
                if (r["object"], r["load_case"]) not in cohort:
                    continue
                fam, be = family(r["candidate"])
                if fam is None or (fam != "grefem" and p not in base_src):
                    continue
                m = re.search(r"_hs([0-9.]+?)(?:_fine2_mesh)?_refined",
                              r.get("candidate_msh", "") or "")
                if (float(m.group(1)) if m else 1.0) != 1.0 or "/hs1.25/" in p \
                        or "/hs1.6/" in p or "/hs2.0/" in p:
                    continue
                try:
                    rows.append({"object": r["object"], "load_case": r["load_case"],
                                 "method": fam, "backend": be,
                                 "disp": float(r["rel_max_disp_err"] or "nan"),
                                 "t": float(r.get("t_candidate_s") or "nan")})
                except ValueError:
                    continue
    df = pd.DataFrame(rows).drop_duplicates(
        ["object", "load_case", "method", "backend"], keep="last")
    print(f"cohort cases: {len(cohort)}")
    # solve time only where every non-MLLM policy has a timing (t_candidate_s is
    # wall time of solve + QoI, near zero when the solution was cached)
    timed = df.dropna(subset=["t"])
    common = None
    for m in ORDER:
        k = set(map(tuple, timed[timed.method == m][["object", "load_case"]].values))
        common = k if common is None else common & k
    for m in ORDER:
        d = df[df.method == m]
        t = timed[(timed.method == m).values & np.array(
            [x in common for x in zip(timed.object, timed.load_case)])]
        print(f"{m:14s} n={len(d):5d}  max-disp err med={d.disp.median():.3f}  "
              f"solve time med={t.t.median():.1f}s (n={len(t)}, {len(common)} cases)")


if __name__ == "__main__":
    main()
