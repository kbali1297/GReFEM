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

## Rebuttal

The rebuttal responses (submission 17439) are in
[`REBUTTAL_RESPONSES.md`](REBUTTAL_RESPONSES.md). They expand the evaluation to
**156 CAD parts × 5 load cases (780 paired cases)** and compare seven refinement
policies at a matched element budget, with downstream FEM error (peak von Mises
stress σ_vM^p99 and strain energy in Ω_crit) as the primary criterion, plus a
retrained learned-AMR baseline (AMBER) and a load-informed mechanics heuristic.

Key findings:
- At a matched budget, GReFEM outperforms the blind geometric heuristic, the
  coarse-solve ZZ policy, and AMBER on peak-stress error, **with no FEM solve and
  no hand-derived mechanics** (median forward cost ≈7 s: render + one MLLM call).
- GReFEM closes **59%** of the coarse-to-fine-reference error gap solver-free.
- A hand-derived load-informed heuristic beats GReFEM where beam-theory reasoning
  applies (reported openly as a boundary of the method); GReFEM's value is
  generality — adapting to a new load family is a prompt edit, not per-family
  engineering.

## Data (HuggingFace)

The case-level result tables backing every rebuttal table are consolidated under
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
