# GReFEM

**Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided 3D Mesh Refinement**

Kartik Bali, Mahish K. Guru, Christian J. Cyron, Roland Aydin

[![arXiv](https://img.shields.io/badge/arXiv-2607.08798-b31b1b.svg)](https://arxiv.org/abs/2607.08798)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

---

## Overview

Adaptive volumetric finite-element meshing dictates the computational budget of
a CAE/FEM analysis. It traditionally requires either iterative PDE solvers in the
loop or heavily-supervised data-driven surrogates trained on large-scale
simulation data. **GReFEM** (*Geometric Reasoning Enhanced Multimodal LLMs for
Finite Element Meshing*) asks whether the high-level semantic understanding of
off-the-shelf Multimodal Large Language Models (MLLMs) can serve as a viable,
zero-shot geometric proxy for mesh refinement.

GReFEM uses an MLLM to visually localize stress-critical regions from
physics-guided textual prompts and rendered views of a CAD part, then converts
those 2D selections into a 3D volumetric sizing field. To bridge 2D MLLM
pre-training and 3D geometry, we introduce **orthoViews**, a supervised
view-selection module that maximizes the observability of key geometric features.

```
CAD (STEP) ──► orthoViews (view selection) ──► rendered views + physics prompt
                                                      │
                                                      ▼
                                            MLLM anchor prediction (2D)
                                                      │
                                    MV-RaySeg (2D → 3D volumetric anchors)
                                                      │
                                      distance-based sizing field ──► gmsh mesh
                                                      │
                                             FEM solve (dolfinx)
```

## Repository structure

### Core pipeline
| File | Role |
|------|------|
| `ortho_views.py` | orthoViews: view-candidate generation + geometric feature detection (FreeCAD) |
| `model.py` / `train.py` / `data.py` | orthoViews view-selector (DINOv2 backbone) training and dataset |
| `general_prompts.py` | Physics-guided prompt templates (Load / Load+Features / Load+Features+Heuristics) |
| `infer.py` | MLLM anchor prediction (view filtering, multi-view aggregation) |
| `infer_meshpoints.py` | MV-RaySeg: project 2D MLLM points to 3D volumetric anchors |
| `generate_renders.py` | Orthographic/perspective view rendering + load visualization |
| `generate_mesh_and_simulate.py` / `*_parallel.py` | Sizing field → gmsh mesh → FEM solve → QoI |
| `fem_fenics.py` | dolfinx linear-elasticity solver, ZZ error indicator, QoI extraction |
| `cad_utils.py`, `utils.py`, `cad_element_sizes.py` | CAD/mesh utilities |

### Baselines
| File | Policy |
|------|--------|
| `infer_baseline.py` | Blind geometric heuristic (concave sharp edges + smooth concave surfaces) |
| `infer_baseline_mech.py` | Load-informed heuristic (voxel cross-section properties + closed-form nominal stress) |
| `make_coarse_zz.py`, `compute_zz_pos.py` | ZZ-from-coarse-solve solver-informed policy |
| `solve_fine_reference.py`, `make_uniform_meshes.py` | Fine-reference solve (≈5·10⁶ elements) and ZZ upper bound |
| `build_amber_cache.py`, `make_amber_labels*.py`, `prepare_amber_data.py`, `infer_amber_grefem.py` | Retrained AMBER learned-AMR baseline |

### Aggregation, tables & metrics
| File | Output |
|------|--------|
| `aggregate_f2x_combined.py` | `combined_200obj_table.csv` (main per-case data) |
| `aggregate_with_amber.py` | `combined_with_amber.csv` (main table incl. AMBER) |
| `make_rebuttal_tables.py` | Main downstream-error markdown table |
| `sweep_tables.py` | Sizing sweep tables (`sweep_tables_{A,B}.csv`) |
| `compute_local_error_tables.py`, `compute_field_error_tables.py` | Per-case QoI errors in Ω_crit |
| `elem_prf1.py`, `loc_prf1_156_micro.py`, `fine_elem_precision.py` | Localization precision/recall/F1 |
| `make_applicability_table.py` | Related-work applicability table |
| `plot_*.py` | Pareto figures |

## Results

GReFEM is evaluated on **156 CAD parts** sampled uniformly at random from the ABC
dataset, each under **5 loading cases** (780 paired case-level comparisons). Every
case is scored against an independent high-fidelity reference solve (uniform mesh
of ≈5·10⁶ elements). Errors are relative to the reference quantity of interest
inside a fixed critical region Ω_crit — the neighbourhood of the reference
solution's top-0.1-percentile Zienkiewicz–Zhu (ZZ) error points. All refinement
policies are compared at a **matched element budget** (same number of refinement
anchors, matched element counts), so differences reflect *where* elements are
placed, not how many are spent.

### Downstream FEM error at a matched budget

Relative error in peak von Mises stress (σ_vM^p99, the failure-relevant design
quantity) and strain energy (the natural FEM error norm), lower is better:

| method | information used | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|-|-|-|-|-|-|-|
| Coarse (no refinement) | — | 0.738 | 0.786 | 0.944 | 0.967 | 25 062 |
| **GReFEM (Gemini)** | rendered views + physics prompt, **no solve** | **0.628** | **0.652** | 0.880 | 0.919 | 95 486 |
| GReFEM (GPT-5.4-mini) | rendered views + physics prompt | 0.630 | 0.652 | 0.872 | 0.915 | 97 257 |
| Geometric heuristic (blind) | geometry only | 0.672 | 0.696 | 0.891 | 0.927 | 94 086 |
| Load-informed heuristic | geometry + hand-derived closed-form stress | 0.567 | 0.591 | 0.868 | 0.914 | 88 149 |
| ZZ from coarse solve | **1 FEM solve** + ZZ estimator | 0.671 | 0.700 | 0.885 | 0.924 | 92 700 |
| ZZ from fine reference solve | **reference solve** (unattainable bound) | 0.551 | 0.581 | 0.851 | 0.909 | 82 333 |
| AMBER (learned AMR) | 440 supervised expert meshes + load | 0.666 | 0.679 | 0.875 | 0.929 | 91 679 |

**Zero-shot GReFEM beats the blind geometric heuristic, the solver-informed
coarse-ZZ policy, and the supervised learned-AMR baseline (AMBER) on peak-stress
error — with no FEM solve, no training on simulation data, and no hand-derived
mechanics.** Its full forward cost is view rendering plus a single MLLM call
(median ≈7 s). Measured as the fraction of the coarse-to-fine-reference gap
closed, GReFEM recovers **59%** entirely solver-free.

The one policy that outperforms it, the load-informed heuristic, requires an
expert to hand-derive beam-theory stress formulas per load family and silently
loses validity outside the slender-member regime; GReFEM adapts to a new load
family by editing a textual prompt.

### The gain comes from the physics prompt, not the MLLM brand

Ablating the prompt at the same matched budget degrades error monotonically and
identically across both MLLM backends:

| prompt | Gemini vm_p99 | GPT-5.4-mini vm_p99 |
|-|-|-|
| Load + Features + Heuristics | **0.628** | **0.630** |
| Load + Features | 0.636 | 0.644 |
| Load-Only | 0.650 | 0.666 |

### Robust across element budgets

Sizing sweep (relative σ_vM^p99 / energy error in Ω_crit; GReFEM pools both
backends per case, ~1,560 cases per level):

| sizing | mean cells | GReFEM | Geom. heur. | ZZ coarse | AMBER |
|-|-|-|-|-|-|
| h × 1.0 | 91k | **0.630**/0.876 | 0.672/0.891 | 0.671/0.886 | 0.666/0.875 |
| h × 1.25 | 56k | **0.688**/0.902 | 0.688/0.895 | 0.715/0.905 | 0.699/0.901 |
| h × 1.6 | 33k | **0.724**/0.925 | 0.745/0.934 | 0.728/0.925 | 0.744/0.933 |
| h × 2.0 | 20k | **0.750**/0.939 | 0.764/0.942 | 0.758/0.943 | 0.765/0.944 |

GReFEM attains the lowest peak-stress error among all solver-free, derivation-free
policies at **every** budget level.

The full experimental discussion, baseline definitions and additional tables
(cost breakdown, localization precision/recall, displacement error) are in
[REBUTTAL_RESPONSES.md](REBUTTAL_RESPONSES.md).

## Data (HuggingFace)

The case-level result tables backing every table above are consolidated under
[`hf_dataset/`](hf_dataset/) with a full manifest and column reference in
[`hf_dataset/README.md`](hf_dataset/README.md), ready to upload to HuggingFace.

## Environment

```bash
conda env create -f environment.yml
```

Core dependencies: `dolfinx`/FEniCSx, `gmsh`, `FreeCAD`, `trimesh`, `pyvista`,
`torch` + `torchvision` (DINOv2), and an MLLM API client (Gemini / GPT).
The AMBER baseline uses the code under `baselines/AMBER/` (not vendored here).

## Citation

```bibtex
@article{bali2026grefem,
  title   = {GReFEM: Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided 3D Mesh Refinement},
  author  = {Bali, Kartik and Guru, Mahish K. and Cyron, Christian J. and Aydin, Roland},
  journal = {arXiv preprint arXiv:2607.08798},
  year    = {2026}
}
```

## License

Released under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/), matching
the paper.
