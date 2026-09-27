"""Build per-h_scale done-manifests + cohort object lists for the MLLM
(gpt-5.4-mini x3, gemini mid/none catch-up) Pareto evaluation campaign.

Scans every existing shard CSV, attributes each ok row to its h_scale via
the hs tag in candidate_msh, and writes mllm_shards/done_manifest_hs{H}.csv
(columns: object,load_case,candidate,status) usable via --done_glob so
compute_local_error_tables.py skips already-solved candidates exactly.
'coarse' rows are h-independent and copied into all four manifests.
"""
import csv, glob, os, re
import pandas as pd

os.makedirs("mllm_shards", exist_ok=True)

ss = pd.read_csv("size_sensitivity_data.csv", dtype={"object": str})
coh = ss[(ss.method == "grefem_max") & (ss.h_scale == 1.0)][
    ["object", "load_case"]].drop_duplicates()
coh_units = set(map(tuple, coh.values))
coh_objs = sorted(coh["object"].unique())
print(f"cohort: {len(coh_units)} units, {len(coh_objs)} objects")

roots = {}
for d in ["test_meshes", "test_meshes_dense", "test_meshes_extra"]:
    for o in os.listdir(d):
        roots[o] = d
by_root = {}
for o in coh_objs:
    by_root.setdefault(roots[o], []).append(o)
for r, objs in sorted(by_root.items()):
    tag = {"test_meshes": "orig", "test_meshes_dense": "dense",
           "test_meshes_extra": "extra"}[r]
    with open(f"mllm_shards/cohort_{tag}.txt", "w") as f:
        f.write("\n".join(objs) + "\n")
    print(f"  {tag}: {len(objs)} objects")

HS = ["1.0", "1.25", "1.6", "2.0"]
done = {h: set() for h in HS}  # (object, load_case, candidate)
for p in glob.glob("*shards*/**/*.csv", recursive=True):
    if "_aggregated" in p or p.startswith("mllm_shards"):
        continue
    try:
        df = pd.read_csv(p, dtype={"object": str}, on_bad_lines="skip")
    except Exception:
        continue
    if not {"object", "load_case", "candidate", "status"} <= set(df.columns):
        continue
    df = df[df["status"] == "ok"]
    for _, r in df.iterrows():
        cand = str(r["candidate"])
        key = (str(r["object"]), str(r["load_case"]), cand)
        if (key[0], key[1]) not in coh_units:
            continue
        if cand == "coarse":
            for h in HS:
                done[h].add(key)
            continue
        if "_5views_11grid_1run" not in cand:
            continue
        msh = str(r.get("candidate_msh", ""))
        m = re.search(r"_hs([0-9.]+?)(?:_fine2_mesh)?_refined", msh)
        h = m.group(1) if m else "1.0"
        h = {"2": "2.0"}.get(h, h)
        if h in done:
            done[h].add(key)

for h in HS:
    out = f"mllm_shards/done_manifest_hs{h}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "load_case", "candidate", "status"])
        for k in sorted(done[h]):
            w.writerow(list(k) + ["ok"])
    print(f"{out}: {len(done[h])} done keys")

# remaining-work summary
exps = ["gpt-5.4-mini_geo_maxprompt", "gpt-5.4-mini_geo_midprompt",
        "gpt-5.4-mini_geo_noneprompt", "gemini-3-flash-preview_geo_midprompt",
        "gemini-3-flash-preview_geo_noneprompt"]
for e in exps:
    line = []
    for h in HS:
        got = {(o, lc) for o, lc, c in done[h] if e in c}
        line.append(f"h{h}:{len(coh_units - got)}")
    print(f"TODO {e}: " + " ".join(line))
