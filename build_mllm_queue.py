"""Build the work queue for the MLLM pool workers: one line per pending
(h_scale, test_dir, object, load_case) unit, i.e. units where at least one
target candidate (gpt-5.4-mini max/mid/none, gemini-3 mid/none with an
existing refinement_points_prefilt.npy) has no ok row yet in the done
manifests or the mllm shard/pool CSVs. Heaviest work (hs1.0) first.
Line format: hs|test_dir|object|load_case
"""
import csv, glob, os
import pandas as pd

ss = pd.read_csv("size_sensitivity_data.csv", dtype={"object": str})
coh = ss[(ss.method == "grefem_max") & (ss.h_scale == 1.0)][
    ["object", "load_case"]].drop_duplicates()
units = sorted(map(tuple, coh.values))

roots = {}
for d in ["test_meshes", "test_meshes_dense", "test_meshes_extra"]:
    for o in os.listdir(d):
        roots[o] = d

EXPS = ["gpt-5.4-mini_geo_maxprompt", "gpt-5.4-mini_geo_midprompt",
        "gpt-5.4-mini_geo_noneprompt", "gemini-3-flash-preview_geo_midprompt",
        "gemini-3-flash-preview_geo_noneprompt"]
SUF = "_ortho_5views_11grid_1run"
HS = ["1.0", "1.25", "1.6", "2.0"]

done = {h: set() for h in HS}  # (object, load_case, candidate)
srcs = {h: [f"mllm_shards/done_manifest_hs{h}.csv"] for h in HS}
for p in glob.glob("mllm_shards/hs*/*_w*.csv"):
    srcs[p.split("/hs")[1].split("/")[0]].append(p)
# pool CSVs carry the hs in the row's candidate_msh -> parse generically
import re
for h in HS:
    for p in srcs[h]:
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done[h].add((r["object"], r["load_case"], r["candidate"]))
for p in glob.glob("mllm_shards/pool/*.csv"):
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            m = re.search(r"_hs([0-9.]+)(?:_fine2_mesh)?_refined", r.get("candidate_msh", ""))
            h = m.group(1) if m else "1.0"
            if h in done:
                done[h].add((r["object"], r["load_case"], r["candidate"]))

qlines = []
for h in HS:
    for o, lc in units:
        root = roots[o]
        pend = False
        for e in EXPS:
            npy = f"{root}/{o}/{lc}_{e}{SUF}/refinement_points_prefilt.npy"
            if not os.path.exists(npy):
                continue
            if (o, lc, f"{lc}_{e}{SUF}") not in done[h]:
                pend = True
                break
        if pend:
            qlines.append(f"{h}|{root}|{o}|{lc}")

# hs1.0 (heaviest) first so the long pole starts immediately
order = {"1.0": 0, "1.25": 1, "1.6": 2, "2.0": 3}
qlines.sort(key=lambda l: order[l.split("|")[0]])
with open("mllm_queue.txt", "w") as f:
    f.write("\n".join(qlines) + ("\n" if qlines else ""))
from collections import Counter
print(f"queue: {len(qlines)} unit-tasks",
      dict(Counter(l.split('|')[0] for l in qlines)))
