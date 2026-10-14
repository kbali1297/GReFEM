---
license: cc-by-4.0
language:
  - en
pretty_name: GReFEM Evaluation Data
tags:
  - finite-element-method
  - adaptive-mesh-refinement
  - multimodal-llm
  - computer-aided-engineering
  - cad
size_categories:
  - 10K<n<100K
---

# GReFEM — Evaluation Data

Case-level result tables backing the tables and figures of the camera-ready paper
**GReFEM: Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided
3D Mesh Refinement** (arXiv:[2607.08798](https://arxiv.org/abs/2607.08798)).

All downstream numbers are computed on **156 CAD parts** from the ABC and
FreeCAD datasets, disjoint from the orthoViews training and validation splits,
each under **5 loading cases** (bending, compression, torsion,
bending+compression, torsion+compression), giving **780 paired geometry–load
cases**. Every case is scored against an independent high-fidelity reference
solve (uniform fine mesh, ≈5·10⁶ tetrahedra). Errors are relative to the
reference quantity of interest inside a fixed critical region Ω_crit (the
neighbourhood, of radius 4·h_min, of the reference solution's top-0.1% ZZ error
points). Tables in the paper aggregate cases by the **5%-trimmed mean** and the
median.

Refinement policies (all refined policies matched to GReFEM's anchor count and
element budget): `coarse` (unrefined), `grefem` (four MLLMs × three prompts),
blind geometric heuristic (`heuristic_sub` / `blind_sub`), load-informed
heuristic (`mech_sub` / `mech`), `zz_coarse` (ZZ from one coarse solve),
`zz_oracle` (ZZ from the fine reference solve, an oracle reference) and `amber`
(AMBER retrained with load conditioning on 440 expert meshes).

## Layout

| Folder | File | Backs | Rows |
|--------|------|-------|------|
| `main_downstream_error/` | `camera_ready_units.csv` | Main downstream table, sizing sweep, per-load-case and sensitivity tables (all policies, MLLMs, prompts, sizing levels) | 58,336 |
| `main_downstream_error/` | `combined_final_table.csv` | Main table, non-MLLM policies + GReFEM-Gemini (L+F+H) at h×1.0 | 4,680 |
| `sizing_sweep/` | `size_sensitivity_data.csv` | Sizing sweep, per case (h×1.0/1.25/1.6/2.0), GReFEM-Gemini (L+F+H) and baselines | 18,552 |
| `localization/` | `loc_camera_ready.csv` | Anchor-level localization table (12 MLLM–prompt variants + heuristics) | 11,367 |
| `localization/` | `elem_prf1.csv` | Element-level localization table (P/R/F1) | 2,340 |
| `localization/` | `hotspot_elemsize.csv` | Element size at reference hotspots (element-level table) | 2,340 |
| `localization/` | `micro_F1_19parts_table.csv` | F1@r vs. number of views (orthoViews and random views), 19 parts of the original submission set | 100 |
| `localization/` | `df_neurips_2026.csv` | Raw view-count grid of the original submission set (source of the F1-vs-views figure) | 48,000 |

## Column reference

Common columns: `object` (part ID), `load_case`, `method` (policy), `vm` /
`en` (relative σ_vM^p99 / strain-energy error in Ω_crit vs. the reference
solve), `cells` (element count of the candidate mesh). `stratum` records which
curation batch a part was added in (`original paper set`, `dense (a-priori)`,
`winners (post-hoc)`); it is not used by any table in the paper.

### `main_downstream_error/camera_ready_units.csv`
`object, load_case, method, vm, en, cells, h, backend, prompt, stratum` — one row
per case, policy and sizing level `h` ∈ {1.0, 1.25, 1.6, 2.0}. For
`method = grefem`, `backend` ∈ {Gemini, GPT-5.4-mini, Claude-Haiku-4.5,
Qwen3-VL-235B} and `prompt` ∈ {L+F+H, L+F, L} (Load+Features+Heuristics,
Load+Features, Load-Only). `amber` is the load-conditioned AMBER reported in the
paper; `amber_geo` is AMBER without load conditioning (not reported in the
paper). AMBER fails to mesh ≈6% of cases, so it has fewer rows.

### `main_downstream_error/combined_final_table.csv`
`object, stratum, load_case, method, vm, en, cells` at h×1.0 for `coarse`,
`grefem_max` (Gemini, L+F+H), `heuristic_sub`, `mech_sub`, `zz_coarse`,
`zz_oracle`.

### `sizing_sweep/size_sensitivity_data.csv`
`object, load_case, method, h_scale, vm, en, cells` — one row per case per sizing
level; the sizing field is scaled by `h_scale` (both h_min and h_max) and every
baseline is re-matched to GReFEM's element count at that level.

### `localization/loc_camera_ready.csv`
`object, load_case, method, prompt, r, n_anchors, matched_anchors, n_gt,
matched_gt, cells_per_view` — anchor-level counts against the reference ZZ points
above the 99th percentile, match radius `r` = 4·h_min. Precision =
`matched_anchors / n_anchors`, recall = `matched_gt / n_gt`. `method` is an MLLM
(with `prompt`), `heuristic_native`, `heuristic_matched` (reduced to
GReFEM-Gemini's anchor count) or `mech`.

### `localization/elem_prf1.csv`
`object, load_case, method, r, n_fine, prec, rec, f1, n_gt` — element-level
precision/recall/F1; predictions are deeply refined elements (size ≤ 1.25·h_min),
ground truth the top-0.1% reference ZZ hotspots within radius `r`.

### `localization/hotspot_elemsize.csv`
`object, load_case, method, r, n_gt, n_tot, n_loc, h_loc` — `h_loc` is the
median element size (cube root of tetrahedron volume) of elements within `r` of
a reference hotspot; `n_loc` the number of such elements, `n_tot` all elements.

### `localization/micro_F1_19parts_table.csv`
`num_views_inference, llm_model, prompt_type, matched_refine, num_refine,
matched_stress, num_stress, micro_F1, label, view_type` — micro-pooled F1@r per
view count on the 19 benchmark parts of the original submission set;
`view_type` ∈ {ortho (orthoViews), random} (random views were run for Gemini only).

### `localization/df_neurips_2026.csv`
Raw view-count grid of the original submission set (40 meshes, 1–10 views,
orthoViews and random views): `llm_model, mesh_name, load_case, view_type,
num_views_inference, grid_size, prompt_type, run, num_refinement_points,
precision, recall, F1, rel_L2, rel_E, total_cells, matched_stress,
matched_refine, num_stress, num_refine`.

## Provenance / reproduction

These CSVs are produced from the raw solver shards by scripts in the
[GReFEM repository](https://github.com/kbali1297/GReFEM):

- `compute_local_error_tables.py` → per-case Ω_crit errors (solver shards)
- `table_combined_final.py` → `combined_final_table.csv`
- `table_size_sensitivity.py` → `size_sensitivity_data.csv`
- `make_camera_ready_tables.py` → `camera_ready_units.csv` and `camera_ready_tables.md`
- `loc_camera_ready.py` → `loc_camera_ready.csv`
- `elem_prf1.py`, `hotspot_elemsize.py` → element-level localization CSVs
- `replot_f1_vs_views_19.py` → `micro_F1_19parts_table.csv` (from `df_neurips_2026.csv`)

See the repository `README.md` and `camera_ready_tables.md` for the full pipeline
and table definitions.

## Citation

```bibtex
@article{bali2026grefem,
  title   = {GReFEM: Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided 3D Mesh Refinement},
  author  = {Bali, Kartik and Guru, Mahish K. and Cyron, Christian J. and Aydin, Roland},
  journal = {arXiv preprint arXiv:2607.08798},
  year    = {2026}
}
```

Licensed CC BY 4.0, matching the paper.
