#!/usr/bin/env python
"""Emit the method-applicability table for the rebuttal.

Reviewers ask why we do not compare against "other refinement methods stated
in related work". This table states, per published method, what it requires and
whether it can be applied to our setting at all (3D tetrahedral meshes derived
from raw CAD STEP files, five mechanical load cases, no per-task solver loop).

Facts recorded here were checked against the papers and the public code in
July 2026; see notes column for the blocking reason.
"""
import argparse

ROWS = [
    # name, venue, dim, supervision, solver-in-loop, public code, applicable, note
    dict(method="Uniform refinement", venue="classical", dim="2D/3D",
         supervision="none", solver="no", code="n/a", applicable="yes",
         note="reported as the coarse baseline"),
    dict(method="ZZ/SPR error-indicator AMR",
         venue="Zienkiewicz & Zhu 1992", dim="2D/3D",
         supervision="none", solver="yes (per part)", code="n/a",
         applicable="yes",
         note="our solver-informed oracle; the same estimator ASMR uses as "
              "its reference heuristic and AMBER uses to build expert meshes"),
    dict(method="Curvature/feature sizing heuristic", venue="classical (gmsh)",
         dim="2D/3D", supervision="none", solver="no", code="yes",
         applicable="yes",
         note="our mesh-geometric heuristic baseline (+ load-aware variant)"),
    dict(method="MeshingNet3D", venue="Adv. Eng. Softw. 2021", dim="3D",
         supervision="expert error fields", solver="yes (training data)",
         code="no", applicable="reimplementation only",
         note="closest published match (tet meshes, linear elasticity) but no "
              "public code; built on FreeFem++/Tetgen"),
    dict(method="GraphMesh", venue="ICCS 2024", dim="2D",
         supervision="expert sizing fields", solver="yes (training data)",
         code="no", applicable="no",
         note="mean-value-coordinate features restrict it to polygonal 2D "
              "domains; AMBER could only run it on their Poisson set"),
    dict(method="ASMR / ASMR++", venue="NeurIPS 2023 / ML 2026", dim="2D",
         supervision="RL reward", solver="yes (every training step)",
         code="yes", applicable="no",
         note="2D triangular meshes on scikit-fem domains; needs a solver in "
              "the RL loop per task; no 3D CAD path"),
    dict(method="RL-AMR (Yang'23, VDGN, Foucart'23)",
         venue="AISTATS/AAMAS/JCP 2023", dim="2D",
         supervision="RL reward", solver="yes (every training step)",
         code="partial", applicable="no",
         note="structured/2D refinement, per-problem training"),
    dict(method="MeshDQN", venue="AIP Advances 2023", dim="2D",
         supervision="RL reward (drag/lift)", solver="yes", code="yes",
         applicable="no",
         note="coarsens 2D CFD airfoil meshes to preserve aerodynamic "
              "coefficients; different task"),
    dict(method="AMBER", venue="NeurIPS 2025", dim="2D/3D",
         supervision="adaptive expert meshes", solver="yes (training labels)",
         code="yes", applicable="yes (retrained here)",
         note="only learned method we could apply; no pretrained weights "
              "exist, so we train it on ZZ-oracle expert meshes over 200 "
              "held-out ABC parts x 5 load cases"),
    dict(method="GReFEM (ours)", venue="-", dim="3D",
         supervision="none (zero-shot MLLM)", solver="no", code="yes",
         applicable="yes", note="no training, no solve at inference"),
]

COLS = [("method", "Method"), ("venue", "Venue"), ("dim", "Domain"),
        ("supervision", "Supervision"), ("solver", "Solver in loop"),
        ("code", "Public code"), ("applicable", "Applicable here")]


def markdown():
    out = ["| " + " | ".join(c[1] for c in COLS) + " |",
           "|" + "---|" * len(COLS)]
    for r in ROWS:
        out.append("| " + " | ".join(str(r[c[0]]) for c in COLS) + " |")
    out.append("")
    out.append("Notes:")
    for r in ROWS:
        out.append(f"- **{r['method']}**: {r['note']}")
    return "\n".join(out)


def latex():
    spec = "l l l l c c c"
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(c[1] for c in COLS) + r" \\", r"\midrule"]
    for r in ROWS:
        cells = [str(r[c[0]]).replace("&", r"\&") for c in COLS]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="applicability_table.md")
    ap.add_argument("--latex_out", default="applicability_table.tex")
    a = ap.parse_args()
    md = markdown()
    open(a.out, "w").write(md + "\n")
    open(a.latex_out, "w").write(latex() + "\n")
    print(md)
    print(f"\n[wrote {a.out}, {a.latex_out}]")
