---
license: cc-by-4.0
language:
  - en
pretty_name: GReFEM Rebuttal Evaluation Data
tags:
  - finite-element-method
  - adaptive-mesh-refinement
  - multimodal-llm
  - computer-aided-engineering
  - cad
size_categories:
  - 10K<n<100K
---

# GReFEM — Rebuttal Evaluation Data

Case-level result tables backing the rebuttal experiments for the paper
**GReFEM: Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided
3D Mesh Refinement** (arXiv:[2607.08798](https://arxiv.org/abs/2607.08798)).

All numbers are computed on **156 CAD parts** sampled uniformly at random from
the ABC dataset (the 30 objects of the original submission plus 126 newly
drawn), each under **5 loading cases** (bending, compression, torsion,
bending+compression, torsion+compression), giving **780 paired case-level
comparisons**. Every (geometry, load) case is scored against an independent
high-fidelity reference solve (uniform fine mesh of ≈5·10⁶ elements). Errors are
reported relative to the reference quantity of interest inside a fixed critical
region Ω_crit (the neighbourhood of the reference solution's top-0.1-percentile
Zienkiewicz–Zhu error points).

Refinement policies compared (matched element budget): `coarse`,
`grefem_*` (GReFEM with Gemini / GPT backends and prompt ablations), blind
geometric `heuristic`, load-informed heuristic (`mech`), `zz_coarse`
(ZZ from one coarse solve), `zz_oracle` (ZZ from the fine reference solve,
an unattainable upper bound), and `amber` (retrained learned-AMR baseline).

## Layout

| Folder | File | Backs rebuttal table | Rows |
|--------|------|----------------------|------|
| `main_downstream_error/` | `combined_200obj_table.csv` | Main downstream error table (per-case, all metrics) | 7,336 |
| `main_downstream_error/` | `combined_with_amber.csv` | Main table incl. AMBER (complete-case units) | 7,200 |
| `sizing_sweep/` | `size_sensitivity_data.csv` | Sizing sweep (per-case, h×1.0/1.25/1.6/2.0) | 18,552 |
| `sizing_sweep/` | `sweep_tables_A.csv` / `sweep_tables_B.csv` | Sizing sweep (aggregated, config A/B) | 24 each |
| `sizing_sweep/` | `pareto_table.csv` | Raw per-case Pareto (cells vs. error, all h) | 20,580 |
| `localization/` | `loc_prf1_156_micro.csv` | Localization precision/recall (anchor level) | 2,268 |
| `localization/` | `elem_prf1.csv` | Element-level P/R/F1 at hotspots | 2,340 |
| `localization/` | `elem_prf1_gpt.csv` | Element-level P/R/F1, GPT backend | 780 |
| `localization/` | `fine_elem_precision.csv` | Fine-mesh element precision | 2,340 |
| `localization/` | `df_neurips_2026.csv` | Full localization + prompt/backend/view grid | 48,000 |
| `amber/` | `amber_pareto_table.csv` | AMBER vs. GReFEM (paired, per h level) | 12 |
| `prompt_ablation/` | `mllm_variant_table.csv` | Prompt / MLLM ablation summary | 20 |
| `field_error/` | `field_error_table.csv` | Field-level (L2) error norms in Ω_crit | 7,336 |

## Column reference

### `main_downstream_error/combined_200obj_table.csv`
Per (object, load_case, candidate) row. Key columns:
- `object`, `load_case`, `method`, `src` (`orig` = original 30 / `extra` = added 126)
- `candidate`, `candidate_msh` — refined mesh identifier
- `n_dofs`, `n_cells` — problem size / element count
- `energy_total`, `energy_crit`, `vm_p99_crit`, `vm_max_crit`, `max_disp` — raw QoIs
- `rel_energy_err`, `rel_energy_crit_err`, `rel_vm_p99_err`, `rel_vm_max_err`,
  `rel_max_disp_err` — relative errors vs. the fine reference
- `n_crit_cells`, `crit_radius`, `n_crit_points` — Ω_crit definition
- `status`

`combined_with_amber.csv` is the complete-case subset with the AMBER method added
(`object, load_case, method, src, rel_vm_p99_err, rel_energy_crit_err, n_cells`).

### `sizing_sweep/size_sensitivity_data.csv`
`object, load_case, method, h_scale, vm, en, cells` — one row per case per sizing
level (`h_scale` ∈ {1.0, 1.25, 1.6, 2.0}); `vm`/`en` are relative σ_vM^p99 /
strain-energy error in Ω_crit. `sweep_tables_{A,B}.csv` are the aggregated
(trimmed-mean / median) versions.

### `localization/elem_prf1.csv`
`object, load_case, method, r, n_fine, prec, rec, f1, n_gt` — element-level
precision/recall/F1 where predictions are deeply refined elements (size ≤ 1.25·h_min)
and ground truth are reference hotspots within radius `r`.

### `localization/df_neurips_2026.csv`
Full experiment grid: `llm_model, mesh_name, load_case, view_type,
num_views_inference, grid_size, prompt_type, run, num_refinement_points,
precision, recall, F1, rel_L2, rel_E, total_cells, matched_stress,
matched_refine, num_stress, num_refine`.

### `amber/amber_pareto_table.csv`
`method, h_scale, N, vm_trim, vm_med, en_trim, en_med, cells_med, N_paired,
d_vm_med, p_vm(grefem<amber), d_en_med, p_en(grefem<amber)` — paired GReFEM-vs-AMBER
comparison with Wilcoxon p-values per sizing level.

## Provenance / reproduction

These CSVs are produced from the raw solver shards by the aggregation scripts in
the [GReFEM repository](https://github.com/kbali1297/GReFEM):

- `aggregate_f2x_combined.py` → `combined_200obj_table.csv`
- `aggregate_with_amber.py` → `combined_with_amber.csv`
- `sweep_tables.py` → `sweep_tables_{A,B}.csv`
- `compute_local_error_tables.py` → per-case error rows (Ω_crit QoIs)
- `elem_prf1.py`, `loc_prf1_156_micro.py`, `fine_elem_precision.py` → localization CSVs

See the repository `README.md` and `REBUTTAL_RESPONSES.md` for the full pipeline
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
