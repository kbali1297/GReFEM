#!/usr/bin/env python
"""Camera-ready downstream tables on the full 156-object / 780-case benchmark.

Sources (all per-unit, relative errors inside Omega_crit):
  combined_final_table.csv      coarse, grefem_max (Gemini L+F+H), heuristic_sub,
                                mech_sub, zz_coarse, zz_oracle  @ h x1.0 (+ stratum)
  size_sensitivity_data.csv     same six policies @ h x{1,1.25,1.6,2}
  *shards*/**/*.csv             the 11 other MLLM backend x prompt variants
  amber_sweep_eval_shards/      AMBER load-conditioned (amberf) / geometry-only (amberfd)

Writes camera_ready_units.csv (long format) and camera_ready_tables.md.
"""
import csv
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import trim_mean, wilcoxon

LCS = ["compression", "bending", "torsion",
       "bending_compression", "torsion_compression"]
LC_LABEL = {"compression": "Comp.", "bending": "Bend.", "torsion": "Tors.",
            "bending_compression": "Bend.+Comp.",
            "torsion_compression": "Tors.+Comp."}
HS = [1.0, 1.25, 1.6, 2.0]

VARIANTS = {  # candidate substring -> (backend, prompt)
    "gemini-3-flash-preview_geo_maxprompt": ("Gemini", "L+F+H"),
    "gemini-3-flash-preview_geo_midprompt": ("Gemini", "L+F"),
    "gemini-3-flash-preview_geo_noneprompt": ("Gemini", "L"),
    "gpt-5.4-mini_geo_maxprompt": ("GPT-5.4-mini", "L+F+H"),
    "gpt-5.4-mini_geo_midprompt": ("GPT-5.4-mini", "L+F"),
    "gpt-5.4-mini_geo_noneprompt": ("GPT-5.4-mini", "L"),
    "claude-haiku-4.5_geo_maxprompt": ("Claude-Haiku-4.5", "L+F+H"),
    "claude-haiku-4.5_geo_midprompt": ("Claude-Haiku-4.5", "L+F"),
    "claude-haiku-4.5_geo_noneprompt": ("Claude-Haiku-4.5", "L"),
    "qwen3-vl-235b-a22b-instruct_geo_maxprompt": ("Qwen3-VL-235B", "L+F+H"),
    "qwen3-vl-235b-a22b-instruct_geo_midprompt": ("Qwen3-VL-235B", "L+F"),
    "qwen3-vl-235b-a22b-instruct_geo_noneprompt": ("Qwen3-VL-235B", "L"),
}
BACKENDS = ["Gemini", "GPT-5.4-mini", "Claude-Haiku-4.5", "Qwen3-VL-235B"]
PROMPTS = ["L+F+H", "L+F", "L"]
BASELINES = ["coarse", "heuristic_sub", "mech_sub", "zz_coarse",
             "amber", "amber_geo", "zz_oracle"]
BLABEL = {"coarse": "Coarse (no refinement)",
          "heuristic_sub": "Geometric heuristic (blind)",
          "mech_sub": "Load-informed heuristic",
          "zz_coarse": "ZZ from coarse solve",
          "amber": "AMBER, load-conditioned",
          "amber_geo": "AMBER, geometry-only",
          "zz_oracle": "ZZ from fine reference solve"}
STRATA = {"dense (a-priori)": "A", "winners (post-hoc)": "B",
          "original paper set": "C"}


def tm(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(trim_mean(x, 0.05)) if len(x) else float("nan")


def load_variants(cohort):
    rows = []
    files = [p for p in glob.glob("*shards*/**/*.csv", recursive=True)
             if "_aggregated" not in p and "done_manifest" not in p]
    need = {"object", "load_case", "candidate", "status",
            "rel_vm_p99_err", "rel_energy_crit_err"}
    for p in files:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            if not rd.fieldnames or not need <= set(rd.fieldnames):
                continue
            for r in rd:
                if r["status"] not in ("ok", "ok_norefine"):
                    continue
                c = r["candidate"]
                if "_5views_11grid_1run" not in c:
                    continue
                key = next((k for k in VARIANTS if k in c), None)
                if key is None or (r["object"], r["load_case"]) not in cohort:
                    continue
                m = re.search(r"_hs([0-9.]+?)(?:_fine2_mesh)?_refined",
                              r.get("candidate_msh", "") or "")
                h = float(m.group(1)) if m else 1.0
                if h not in HS:
                    continue
                try:
                    rows.append({"object": r["object"],
                                 "load_case": r["load_case"],
                                 "backend": VARIANTS[key][0],
                                 "prompt": VARIANTS[key][1], "h": h,
                                 "vm": float(r["rel_vm_p99_err"] or "nan"),
                                 "en": float(r["rel_energy_crit_err"] or "nan"),
                                 "cells": float(r.get("n_cells") or "nan")})
                except ValueError:
                    continue
    df = pd.DataFrame(rows).drop_duplicates(
        ["object", "load_case", "backend", "prompt", "h"], keep="last")
    df["method"] = "grefem"
    return df


def load_amber():
    parts = []
    for p in glob.glob("amber_sweep_eval_shards/[AB]_*.csv"):
        if "_aggregated" in p:
            continue
        try:
            parts.append(pd.read_csv(p, dtype={"object": str}))
        except Exception:
            continue
    df = pd.concat(parts, ignore_index=True)
    m = df["candidate"].str.extract(r"_(amberfd|amberf)_h(\d{3})$")
    df["method"] = m[0].map({"amberf": "amber", "amberfd": "amber_geo"})
    df["h"] = m[1].map({"100": 1.0, "125": 1.25, "160": 1.6, "200": 2.0})
    df = df.dropna(subset=["method", "h"])
    df = df[df["status"].isin(["ok", "ok_norefine"])]
    df = df.rename(columns={"rel_vm_p99_err": "vm",
                            "rel_energy_crit_err": "en", "n_cells": "cells"})
    df = df.dropna(subset=["vm", "en"]).drop_duplicates(
        ["object", "load_case", "method", "h"], keep="last")
    return df[["object", "load_case", "method", "h", "vm", "en", "cells"]]


def build():
    main = pd.read_csv("combined_final_table.csv", dtype={"object": str})
    strat = main.drop_duplicates("object").set_index("object")["stratum"]
    main["h"] = 1.0
    ss = pd.read_csv("size_sensitivity_data.csv", dtype={"object": str})
    ss = ss.rename(columns={"h_scale": "h"})
    ss = ss[ss.h != 1.0]
    base = pd.concat([main.drop(columns="stratum"), ss], ignore_index=True)
    base["backend"], base["prompt"] = "", ""
    g = base.method == "grefem_max"
    base.loc[g, ["backend", "prompt"]] = ["Gemini", "L+F+H"]
    base.loc[g, "method"] = "grefem"

    cohort = set(map(tuple, main[["object", "load_case"]].values))
    var = load_variants(cohort)
    var = var[~((var.backend == "Gemini") & (var.prompt == "L+F+H"))]
    amb = load_amber()
    amb = amb[[(o, lc) in cohort for o, lc in zip(amb.object, amb.load_case)]]
    amb["backend"], amb["prompt"] = "", ""

    df = pd.concat([base, var, amb], ignore_index=True)
    df["stratum"] = df["object"].map(strat).map(STRATA)
    df.to_csv("camera_ready_units.csv", index=False)
    return df


def sel(df, method, backend="", prompt="", h=1.0):
    s = df[(df.method == method) & (df.h == h)]
    if method == "grefem":
        s = s[(s.backend == backend) & (s.prompt == prompt)]
    return s


def row_stats(s):
    return dict(n=s.groupby(["object", "load_case"]).ngroups,
                vm_t=tm(s.vm), vm_m=float(s.vm.median()),
                en_t=tm(s.en), en_m=float(s.en.median()),
                cells=float(s.cells.mean()))


def paired(a, b):
    k = ["object", "load_case"]
    j = a[k + ["vm", "en"]].merge(b[k + ["vm", "en"]], on=k,
                                  suffixes=("_a", "_b"))
    out = {"n": len(j)}
    for c in ["vm", "en"]:
        d = (j[f"{c}_a"] - j[f"{c}_b"]).dropna()
        out[f"{c}_med"] = float(d.median())
        out[f"{c}_win"] = float((d < -0.01).mean())
        out[f"{c}_loss"] = float((d > 0.01).mean())
        out[f"{c}_p"] = float(wilcoxon(d).pvalue) if len(d) > 10 else np.nan
    return out


def main():
    df = build()
    L = []
    P = L.append
    n_obj = df[df.method == "coarse"].object.nunique()
    P(f"# Camera-ready tables ({n_obj} objects)\n")
    P("Strata objects: " + str(df.drop_duplicates("object")
                               .stratum.value_counts().to_dict()) + "\n")

    def policy_list():
        out = [("Coarse (no refinement)", "coarse", "", "")]
        for p in PROMPTS:
            for b in BACKENDS:
                out.append((f"GReFEM ({b}) {p}", "grefem", b, p))
        for m in BASELINES[1:]:
            out.append((BLABEL[m], m, "", ""))
        return out

    # ---- Table 1: pooled, all 156 + outcome-independent subset A+C ----
    for title, sub in [("ALL 156 objects", df),
                       ("A+C (outcome-independent)", df[df.stratum.isin(["A", "C"])]),
                       ("A (feature-dense)", df[df.stratum == "A"]),
                       ("B (post-hoc GReFEM-favourable)", df[df.stratum == "B"]),
                       ("C (original set)", df[df.stratum == "C"])]:
        P(f"\n## Main table @h1.0 — {title}\n")
        P("| policy | n | vm tmean | vm med | en tmean | en med | cells |")
        P("|---|---|---|---|---|---|---|")
        for lab, m, b, p in policy_list():
            s = sel(sub, m, b, p)
            if not len(s):
                continue
            r = row_stats(s)
            P(f"| {lab} | {r['n']} | {r['vm_t']:.3f} | {r['vm_m']:.3f} | "
              f"{r['en_t']:.3f} | {r['en_m']:.3f} | {r['cells']:,.0f} |")
        # backend-pooled GReFEM per prompt
        for p in PROMPTS:
            s = sub[(sub.method == "grefem") & (sub.h == 1.0) & (sub.prompt == p)]
            r = row_stats(s)
            P(f"| GReFEM (4-backend pool) {p} | {r['n']} | {r['vm_t']:.3f} | "
              f"{r['vm_m']:.3f} | {r['en_t']:.3f} | {r['en_m']:.3f} | "
              f"{r['cells']:,.0f} |")

    # ---- gap closed ----
    P("\n## Fraction of coarse->fine-ref gap closed (vm tmean)\n")
    for title, sub in [("all", df), ("A+C", df[df.stratum.isin(["A", "C"])])]:
        c = tm(sel(sub, "coarse").vm)
        o = tm(sel(sub, "zz_oracle").vm)
        for lab, m, b, p in [("GReFEM Gemini L+F+H", "grefem", "Gemini", "L+F+H"),
                             ("Geom heur", "heuristic_sub", "", ""),
                             ("Load-inf", "mech_sub", "", ""),
                             ("ZZ coarse", "zz_coarse", "", ""),
                             ("AMBER load", "amber", "", ""),
                             ("AMBER geo", "amber_geo", "", "")]:
            v = tm(sel(sub, m, b, p).vm)
            P(f"- {title}: {lab}: {v:.3f} -> {(c - v) / (c - o):.1%}")

    # ---- paired tests ----
    for title, sub in [("all", df), ("A+C", df[df.stratum.isin(["A", "C"])]),
                       ("A", df[df.stratum == "A"]), ("C", df[df.stratum == "C"]),
                       ("B", df[df.stratum == "B"])]:
        P(f"\n## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — {title}\n")
        P("| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |")
        P("|---|---|---|---|---|---|---|---|")
        a = sel(sub, "grefem", "Gemini", "L+F+H")
        for m in BASELINES:
            r = paired(a, sel(sub, m))
            P(f"| {BLABEL[m]} | {r['n']} | {r['vm_med']:+.4f} | "
              f"{r['vm_win']:.1%}/{r['vm_loss']:.1%} | {r['vm_p']:.2g} | "
              f"{r['en_med']:+.4f} | {r['en_win']:.1%}/{r['en_loss']:.1%} | "
              f"{r['en_p']:.2g} |")
        # 4-backend mean per case vs baselines
        gp = sub[(sub.method == "grefem") & (sub.h == 1.0) & (sub.prompt == "L+F+H")]
        gp = gp.groupby(["object", "load_case"])[["vm", "en"]].mean().reset_index()
        P(f"\n4-backend per-case mean (L+F+H) minus baseline — {title}\n")
        for m in BASELINES:
            r = paired(gp, sel(sub, m))
            P(f"| {BLABEL[m]} | {r['n']} | {r['vm_med']:+.4f} | "
              f"{r['vm_win']:.1%}/{r['vm_loss']:.1%} | {r['vm_p']:.2g} | "
              f"{r['en_med']:+.4f} | {r['en_win']:.1%}/{r['en_loss']:.1%} | "
              f"{r['en_p']:.2g} |")

    # ---- prompt ablation, paired within backend ----
    P("\n## Prompt ablation, paired (L+F+H minus other) per backend, all 156\n")
    for b in BACKENDS:
        a = sel(df, "grefem", b, "L+F+H")
        for p in ["L+F", "L"]:
            r = paired(a, sel(df, "grefem", b, p))
            P(f"- {b}: L+F+H vs {p}: n={r['n']} vm medΔ={r['vm_med']:+.4f} "
              f"p={r['vm_p']:.2g}; en medΔ={r['en_med']:+.4f} p={r['en_p']:.2g}")

    # ---- per load case ----
    P("\n## Per load case @h1.0 (vm tmean / en tmean), all 156\n")
    P("| policy | " + " | ".join(LC_LABEL[l] for l in LCS) + " |")
    P("|---|" + "---|" * len(LCS))
    pl = [("Coarse (no refinement)", "coarse", "", "")]
    pl += [(f"GReFEM ({b}) L+F+H", "grefem", b, "L+F+H") for b in BACKENDS]
    pl += [(BLABEL[m], m, "", "") for m in BASELINES[1:]]
    for lab, m, b, p in pl:
        cells = []
        for lc in LCS:
            s = sel(df[df.load_case == lc], m, b, p)
            cells.append(f"{tm(s.vm):.3f}/{tm(s.en):.3f}")
        P(f"| {lab} | " + " | ".join(cells) + " |")
    P("\nPer load case, GReFEM pooled over backends, per prompt:\n")
    for p in PROMPTS:
        cells = []
        for lc in LCS:
            s = df[(df.method == "grefem") & (df.h == 1.0) & (df.prompt == p)
                   & (df.load_case == lc)]
            cells.append(f"{tm(s.vm):.3f}/{tm(s.en):.3f}")
        P(f"| GReFEM 4-backend pool {p} | " + " | ".join(cells) + " |")
    P("\nPaired GReFEM (Gemini L+F+H) minus baseline per load case (vm medΔ, p):\n")
    for m in ["heuristic_sub", "mech_sub", "zz_coarse", "amber", "amber_geo"]:
        cells = []
        for lc in LCS:
            sub = df[df.load_case == lc]
            r = paired(sel(sub, "grefem", "Gemini", "L+F+H"), sel(sub, m))
            cells.append(f"{r['vm_med']:+.3f} (p={r['vm_p']:.2g})")
        P(f"| {BLABEL[m]} | " + " | ".join(cells) + " |")

    # ---- sensitivity across experimental dimensions (GReFEM only) ----
    P("\n## Sensitivity: std of trimmed-mean error across levels of each dimension\n")
    g1 = df[(df.method == "grefem") & (df.h == 1.0)]
    gall = df[df.method == "grefem"]
    dims = [("MLLM backend", g1, "backend"), ("Prompt", g1, "prompt"),
            ("Load case", g1, "load_case"), ("Sizing level", gall, "h")]
    P("| dimension | levels | vm std | vm range | en std | en range |")
    P("|---|---|---|---|---|---|")
    for name, s, col in dims:
        v = s.groupby(col).vm.apply(tm)
        e = s.groupby(col).en.apply(tm)
        P(f"| {name} | {len(v)} | {v.std(ddof=0):.3f} | {v.max() - v.min():.3f} | "
          f"{e.std(ddof=0):.3f} | {e.max() - e.min():.3f} |")
    # policy dimension (reference): spread across all non-oracle refined policies
    pol = []
    for lab, m, b, p in [("g", "grefem", "Gemini", "L+F+H")] + \
            [(m, m, "", "") for m in ["heuristic_sub", "mech_sub", "zz_coarse",
                                      "amber"]]:
        s = sel(df, m, b, p)
        pol.append((tm(s.vm), tm(s.en)))
    pol = np.array(pol)
    P(f"| Refinement policy (ref.) | {len(pol)} | {pol[:, 0].std():.3f} | "
      f"{np.ptp(pol[:, 0]):.3f} | {pol[:, 1].std():.3f} | {np.ptp(pol[:, 1]):.3f} |")

    # ---- sizing sweep: backend pooled + AMBER variants ----
    P("\n## Sizing sweep (vm tmean / en tmean)\n")
    P("| h | n units | GReFEM Gemini | GReFEM 4-pool | Geom | Load-inf | ZZc | "
      "AMBER load | AMBER geo | ZZ fine |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for h in HS:
        def f(s):
            return f"{tm(s.vm):.3f}/{tm(s.en):.3f}"
        gp = df[(df.method == "grefem") & (df.h == h) & (df.prompt == "L+F+H")]
        cells = [f(sel(df, "grefem", "Gemini", "L+F+H", h)), f(gp)] + \
            [f(sel(df, m, h=h)) for m in ["heuristic_sub", "mech_sub",
                                          "zz_coarse", "amber", "amber_geo",
                                          "zz_oracle"]]
        n = sel(df, "grefem", "Gemini", "L+F+H", h).shape[0]
        P(f"| {h} | {n} | " + " | ".join(cells) + " |")
    P("\nPaired GReFEM (Gemini L+F+H) minus AMBER per sizing level:\n")
    for h in HS:
        a = sel(df, "grefem", "Gemini", "L+F+H", h)
        for m in ["amber", "amber_geo"]:
            r = paired(a, sel(df, m, h=h))
            P(f"- h{h} {BLABEL[m]}: n={r['n']} vm medΔ={r['vm_med']:+.4f} "
              f"p={r['vm_p']:.2g}; en medΔ={r['en_med']:+.4f} p={r['en_p']:.2g}")

    txt = "\n".join(L)
    open("camera_ready_tables.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
