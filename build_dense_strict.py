#!/usr/bin/env python
"""Build the strict dense-cohort object list for the final tables/figures:

  * object must have ALL 5 load cases complete (all 5 candidates ok)
    at EVERY sizing level (hs 1.0, 1.25, 1.6, 2.0)
  * duplicate CAD parts (same geometry, different ABC id) removed

Writes dense_strict_objects.txt and prints the attrition breakdown.
"""
import csv
import glob
import os
import re
import collections

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
NEED = {"coarse", "grefem_max", "heuristic_sub", "mech_sub", "zz_oracle"}

# duplicate CAD parts identified by inspection (anchor count -> object id);
# 7064 maps to two ids, only the plain-tube duplicate 00140013 is dropped.
DUPES = ["00910041",  # 9605
         "00200028",  # 10335
         "00850026",  # 7819
         "00560036",  # 7573
         "00080044",  # 7070
         "00670047",  # 7065
         "00140013",  # 7064 (plain tube; 00910048 kept, distinct bracket)
         "00470022",  # 6986
         "00790048",  # 6820
         "00410019",  # 6742
         "00520082",  # 6685
         "00250065"]  # 6116


def fam(c, lc):
    if c in ("coarse", "zz_oracle"):
        return c
    if re.match(rf"{lc}_gemini.*_geo_maxprompt_ortho_5views", c):
        return "grefem_max"
    if c.startswith("heuristic_baseline_DENSE") and c.endswith("_sub"):
        return "heuristic_sub"
    if c.startswith("heuristic_mech_") and c.endswith("_sub"):
        return "mech_sub"
    return None


def complete_units(patterns):
    have = collections.defaultdict(set)
    for pat in patterns:
        for f in glob.glob(pat):
            if "aggregated" in f:
                continue
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if (r.get("status") or "").startswith("ok"):
                        m = fam(r["candidate"], r["load_case"])
                        if m:
                            have[(r["object"], r["load_case"])].add(m)
    return {k for k, v in have.items() if not NEED - v}


def main():
    levels = {"hs1.0": ["dense_shards/*.csv"]}
    for d in sorted(glob.glob("dense_pareto_shards/hs*/")):
        levels[os.path.basename(d.rstrip("/"))] = [d + "*.csv"]

    per_level = {}
    for name, pats in levels.items():
        per_level[name] = complete_units(pats)
        objs = {o for o, _ in per_level[name]}
        print(f"{name:8s}: {len(per_level[name]):4d} complete units, "
              f"{len(objs)} objects")

    # objects complete for all 5 load cases at every level
    common = set.intersection(*per_level.values())
    by_obj = collections.Counter(o for o, _ in common)
    strict = sorted(o for o, c in by_obj.items() if c == len(LCS))
    print(f"\nunits complete at ALL levels: {len(common)}")
    print(f"objects with all 5 load cases at all levels: {len(strict)}")

    dropped = [o for o in strict if o in DUPES]
    final = [o for o in strict if o not in DUPES]
    print(f"duplicate parts removed: {len(dropped)} {dropped}")
    print(f"FINAL strict object set: {len(final)}")

    with open("dense_strict_objects.txt", "w") as fh:
        fh.write("\n".join(final) + "\n")
    print("wrote dense_strict_objects.txt")


if __name__ == "__main__":
    main()
