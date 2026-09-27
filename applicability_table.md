| Method | Venue | Domain | Supervision | Solver in loop | Public code | Applicable here |
|---|---|---|---|---|---|---|
| Uniform refinement | classical | 2D/3D | none | no | n/a | yes |
| ZZ/SPR error-indicator AMR | Zienkiewicz & Zhu 1992 | 2D/3D | none | yes (per part) | n/a | yes |
| Curvature/feature sizing heuristic | classical (gmsh) | 2D/3D | none | no | yes | yes |
| MeshingNet3D | Adv. Eng. Softw. 2021 | 3D | expert error fields | yes (training data) | no | reimplementation only |
| GraphMesh | ICCS 2024 | 2D | expert sizing fields | yes (training data) | no | no |
| ASMR / ASMR++ | NeurIPS 2023 / ML 2026 | 2D | RL reward | yes (every training step) | yes | no |
| RL-AMR (Yang'23, VDGN, Foucart'23) | AISTATS/AAMAS/JCP 2023 | 2D | RL reward | yes (every training step) | partial | no |
| MeshDQN | AIP Advances 2023 | 2D | RL reward (drag/lift) | yes | yes | no |
| AMBER | NeurIPS 2025 | 2D/3D | adaptive expert meshes | yes (training labels) | yes | yes (retrained here) |
| GReFEM (ours) | - | 3D | none (zero-shot MLLM) | no | yes | yes |

Notes:
- **Uniform refinement**: reported as the coarse baseline
- **ZZ/SPR error-indicator AMR**: our solver-informed oracle; the same estimator ASMR uses as its reference heuristic and AMBER uses to build expert meshes
- **Curvature/feature sizing heuristic**: our mesh-geometric heuristic baseline (+ load-aware variant)
- **MeshingNet3D**: closest published match (tet meshes, linear elasticity) but no public code; built on FreeFem++/Tetgen
- **GraphMesh**: mean-value-coordinate features restrict it to polygonal 2D domains; AMBER could only run it on their Poisson set
- **ASMR / ASMR++**: 2D triangular meshes on scikit-fem domains; needs a solver in the RL loop per task; no 3D CAD path
- **RL-AMR (Yang'23, VDGN, Foucart'23)**: structured/2D refinement, per-problem training
- **MeshDQN**: coarsens 2D CFD airfoil meshes to preserve aerodynamic coefficients; different task
- **AMBER**: only learned method we could apply; no pretrained weights exist, so we train it on ZZ-oracle expert meshes over 200 held-out ABC parts x 5 load cases
- **GReFEM (ours)**: no training, no solve at inference
