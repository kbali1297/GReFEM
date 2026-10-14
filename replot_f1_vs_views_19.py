#!/usr/bin/env python
"""F1@r vs number of views restricted to the 19 benchmark parts of the original
submission set. Same data (df_neurips_2026.csv), same micro-pooling and same
blind-heuristic anchor subsample as the published 30-part figure; --check
reproduces the published 30-part numbers first."""
import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_analysis_rebuttal import precision_recall_f1

ROOT = "/data/1bali/GReFEM"
OUT = (f"{ROOT}/GReFEM_NeurIPS_2026_camera_ready (1)/figures/"
       "micro_F1_vs_num_views.pdf")
REMOVE_30 = ['00520044', '00530042', '00530061', '00210058', '00210018',
             '00200037', '00200039', '00200050', '00210005', '00230017']
PROMPT = {"geo_max": "Load+Features+Heuristics", "geo_mid": "Load+Features"}
MODEL = {"gemini-3-flash-preview": "Gemini-3-flash",
         "claude-haiku-4.5": "Claude-Haiku-4.5",
         "gpt-5.4-mini": "GPT-5.4-mini",
         "qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B"}
LOADS = ['compression', 'torsion', 'bending', 'torsion_compression', 'bending_compression']


def micro_f1(g):
    p = g.matched_refine / g.num_refine
    r = g.matched_stress / g.num_stress
    return 2 * p * r / (p + r)


def mllm_curves(df, meshes, view_type='ortho'):
    src = df[df.mesh_name.isin(meshes) & df.prompt_type.isin(['geo_max', 'geo_mid'])
             & df.num_views_inference.between(1, 10) & (df.view_type == view_type)
             & df[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].notna().all(axis=1)]
    g = src.groupby(['num_views_inference', 'llm_model', 'prompt_type'], as_index=False)[
        ['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum()
    g['micro_F1'] = micro_f1(g)
    n = src.groupby(['num_views_inference', 'llm_model', 'prompt_type']).mesh_name.nunique()
    return g, n


def baseline_counts():
    """Per-(mesh, load) counts of the blind heuristic with the published global subsample."""
    sizes = json.load(open(f"{ROOT}/cad_element_sizes.json"))
    r_thresh = {m: 4.0 * v['h_min'] for m, v in sizes.items()}
    zz = pickle.load(open(f"{ROOT}/zz_points_dicts/zz_points_dict_neurips.pkl", 'rb'))
    trees = pickle.load(open(f"{ROOT}/zz_points_dicts/zz_points_tree_dict_neurips.pkl", 'rb'))
    bdir, exp = f"{ROOT}/test_meshes_rebuttal_baseline", "baseline_DENSE_dihedral30.0deg"
    pts, order = {}, []
    for m in r_thresh:
        if m in REMOVE_30:
            continue
        try:
            pts[m] = np.load(f"{bdir}/{m}/refinement_points_{exp}.npy")
            order.append(m)
        except FileNotFoundError:
            pass
    total = sum(len(pts[m]) for m in order)
    target = 370851 // 5
    mask = np.zeros(total, dtype=bool)
    mask[:target] = True
    np.random.seed(42)
    np.random.shuffle(mask)
    rows, i = [], 0
    for m in order:
        sel = pts[m][mask[i:i + len(pts[m])]]
        i += len(pts[m])
        for lc in LOADS:
            key = f"{m}_{lc}"
            if key not in zz:
                continue
            _, _, _, ms, mr, ns, nr = precision_recall_f1(zz[key], sel, r=r_thresh[m], tree_stress=trees[key])
            rows.append(dict(mesh_name=m, load_case=lc, matched_stress=ms, matched_refine=mr,
                             num_stress=ns, num_refine=nr))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(f"{ROOT}/df_neurips_2026.csv", dtype={'mesh_name': str}, low_memory=False)
    parts19 = [l.strip() for l in open(f"{ROOT}/final_orig_objects.txt") if l.strip()]
    assert len(parts19) == 19
    base = baseline_counts()

    if args.check:
        m30 = [m for m in sorted(df.mesh_name.unique()) if m not in REMOVE_30]
        g30, _ = mllm_curves(df, m30)
        pub = pd.read_csv(f"{ROOT}/logs_plots/micro_F1_all_models_geo_max_geo_mid_table.csv")
        mrg = g30.merge(pub, on=['num_views_inference', 'llm_model', 'prompt_type'])
        print(f"30-part check: {len(m30)} parts, {len(mrg)}/{len(pub)} rows matched, "
              f"max |dF1| = {(mrg.micro_F1_x - mrg.micro_F1_y).abs().max():.2e}")
        print(f"30-part blind heuristic F1 = {micro_f1(base[['matched_refine','num_refine','matched_stress','num_stress']].sum()):.4f} (published 0.5362)")

    g, n = mllm_curves(df, parts19)
    b19 = base[base.mesh_name.isin(parts19)]
    bf1 = micro_f1(b19[['matched_refine', 'num_refine', 'matched_stress', 'num_stress']].sum())
    print(f"19 parts: MLLM parts per point min/max = {n.min()}/{n.max()}; "
          f"heuristic covers {b19.mesh_name.nunique()} parts, {len(b19)} cases, F1 = {bf1:.4f}")
    g['label'] = g.llm_model.map(MODEL) + " | " + g.prompt_type.map(PROMPT)
    piv = g.pivot_table(index='label', columns='num_views_inference', values='micro_F1')
    print(piv.round(3).to_string())
    # random-view inference exists only for Gemini
    gr, nr = mllm_curves(df, parts19, view_type='random')
    gr['label'] = gr.llm_model.map(MODEL) + " | " + gr.prompt_type.map(PROMPT) + ", random views"
    print(f"random views: models {gr.llm_model.unique().tolist()}, parts per point min/max = {nr.min()}/{nr.max()}")
    print(gr.pivot_table(index='label', columns='num_views_inference', values='micro_F1').round(3).to_string())
    pd.concat([g.assign(view_type='ortho'), gr.assign(view_type='random')]).to_csv(
        f"{ROOT}/logs_plots/micro_F1_19parts_table.csv", index=False)

    plt.style.use("default")
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    plt.rcParams["mathtext.fontset"] = "stix"
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    # hue key and sorting identical to the published figure, so colours/markers match
    g['hue'] = g.llm_model + " | " + g.prompt_type.map({'geo_max': 'Explicit+', 'geo_mid': 'Explicit'})
    hues = sorted(g.hue.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(hues)))
    markers = ["o", "s", "^", "D", "v", "p", "X", "*"]
    for i, h in enumerate(hues):
        sub = g[g.hue == h].sort_values("num_views_inference")
        ax.plot(sub.num_views_inference.astype(float), sub.micro_F1.astype(float),
                marker=markers[i % len(markers)], linewidth=1.8, markersize=4,
                label=sub.label.iloc[0], color=colors[i])
    hue_color = dict(zip(hues, colors))
    hue_marker = {h: markers[i % len(markers)] for i, h in enumerate(hues)}
    gr['hue'] = gr.llm_model + " | " + gr.prompt_type.map({'geo_max': 'Explicit+', 'geo_mid': 'Explicit'})
    for h in sorted(gr.hue.unique()):
        sub = gr[gr.hue == h].sort_values("num_views_inference")
        ax.plot(sub.num_views_inference.astype(float), sub.micro_F1.astype(float),
                marker=hue_marker[h], markerfacecolor="white", linestyle=":", linewidth=1.8,
                markersize=4, label=sub.label.iloc[0], color=hue_color[h])
    ax.axhline(bf1, color="black", linestyle="--", linewidth=1.2,
               label=f"Geometric heuristic (blind) ({bf1:.4f})")
    ax.set_xlabel("k = No. of Grid inference views", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1@r", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6.5, framealpha=0.9, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="solid: orthoViews; dotted: random views", title_fontsize=6.5)
    ax.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout(pad=0.4)
    plt.savefig(OUT, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
