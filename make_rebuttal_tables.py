#!/usr/bin/env python
"""Render rebuttal-ready markdown tables (per load case + pooled) from
combined_200obj_table.csv (produced by aggregate_f2x_combined.py).

Format matches the rebuttal table style:
  | method | vm_p99 mean | vm_p99 med | energy mean | energy med | cells |
Best (lowest) value per error column is bolded; oracle rows are marked.
"""
import argparse
import pandas as pd

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
METHODS = ["coarse", "grefem_max", "grefem_mid", "grefem_none",
           "heuristic", "heuristic_sub", "zz_oracle", "zz_oracle_spread"]
ERR_COLS = ["vm_p99_mean", "vm_p99_med", "energy_mean", "energy_med"]


def make_table(sub, with_cells=True):
    tab = sub.groupby("method").agg(
        vm_p99_mean=("rel_vm_p99_err", "mean"),
        vm_p99_med=("rel_vm_p99_err", "median"),
        energy_mean=("rel_energy_crit_err", "mean"),
        energy_med=("rel_energy_crit_err", "median"),
        cells=("n_cells", "mean"),
    ).reindex(METHODS)

    # best (lowest) per error column among non-oracle methods gets bolded;
    # oracle values bolded separately when they beat everything
    lines = []
    hdr = ["method"] + [c.replace("_", " ").replace("med", "med.")
                        for c in ERR_COLS]
    if with_cells:
        hdr.append("cells")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    best = {c: tab[c].min() for c in ERR_COLS}
    for m, row in tab.iterrows():
        cells = [m]
        for c in ERR_COLS:
            v = f"{row[c]:.3f}"
            cells.append(f"**{v}**" if abs(row[c] - best[c]) < 5e-4 else v)
        if with_cells:
            cells.append(f"{row['cells']:,.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="combined_200obj_table.csv")
    ap.add_argument("--out", default="rebuttal_tables.md")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"object": str})
    out = []
    for lc in LCS:
        sub = df[df["load_case"] == lc]
        if sub.empty:
            continue
        n = sub["object"].nunique()
        out.append(f"**{lc} (n={n})**\n")
        out.append(make_table(sub))
        out.append("")
    n_units = df.groupby(["object", "load_case"]).ngroups
    out.append(f"**pooled over all load cases (n={n_units} units, "
               f"{df['object'].nunique()} objects)**\n")
    out.append(make_table(df, with_cells=False))
    out.append("")

    text = "\n".join(out)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
