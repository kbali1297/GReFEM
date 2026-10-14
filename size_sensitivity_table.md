## Element-size (sizing-field) sensitivity

Sizing field scaled by a common factor applied to both `h_min` and `h_max` for every refined candidate. Held fixed across levels: the reference solution, Ω_crit, `crit_radius`, and each method's anchor set. At every level all baselines are re-matched to the GReFEM element count, so the comparison remains budget-matched.

**156 objects × 5 loading cases = 773 paired units**, complete at all four levels. Values are relative error in Ω_crit (5 %-trimmed mean over units); lower is better.

### Realised element counts (mean, thousands)

| sizing | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|
| **h × 1.0** | 24.9k | 95.4k | 94.0k | 88.1k | 92.5k | 82.3k |
| **h × 1.25** | 24.9k | 57.1k | 56.9k | 54.4k | 55.8k | 52.2k |
| **h × 1.6** | 24.9k | 33.2k | 33.2k | 31.9k | 32.6k | 31.0k |
| **h × 2.0** | 24.9k | 20.6k | 20.5k | 20.0k | 20.2k | 19.4k |

Refined candidates agree to within 5.9–15.8 % at every level (budget matching holds; `coarse` is the unrefined mesh and is not budget-matched).

### Peak stress (rel. σ_vM^p99 error in Ω_crit)

| sizing | mean cells | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|---|
| **h × 1.0** | 90k | 0.738 | 0.627 | 0.673 | 0.566 | 0.673 | **0.550** |
| **h × 1.25** | 55k | 0.738 | 0.692 | 0.687 | 0.627 | 0.717 | **0.608** |
| **h × 1.6** | 32k | 0.738 | 0.724 | 0.745 | **0.686** | 0.728 | 0.700 |
| **h × 2.0** | 20k | 0.738 | 0.756 | 0.766 | 0.734 | 0.760 | **0.728** |
| *Δ (h×2.0 − h×1.0)* | | *+0.000* | *+0.129* | *+0.093* | *+0.168* | *+0.087* | *+0.178* |

### Strain energy (rel. error in Ω_crit)

| sizing | mean cells | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|---|
| **h × 1.0** | 90k | 0.944 | 0.880 | 0.892 | 0.869 | 0.887 | **0.850** |
| **h × 1.25** | 55k | 0.944 | 0.906 | 0.895 | 0.890 | 0.905 | **0.869** |
| **h × 1.6** | 32k | 0.944 | 0.928 | 0.934 | **0.915** | 0.926 | 0.915 |
| **h × 2.0** | 20k | 0.944 | 0.941 | 0.943 | **0.928** | 0.944 | 0.931 |
| *Δ (h×2.0 − h×1.0)* | | *+0.000* | *+0.061* | *+0.051* | *+0.059* | *+0.057* | *+0.081* |

### Paired tests at each sizing level (vm_p99)

Median Δ = GReFEM − baseline; negative means GReFEM is better. Two-sided Wilcoxon signed-rank over the paired units.

| sizing | Coarse (no refinement) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|
| **h × 1.0** | -0.082* (p=1.9e-33) | -0.026* (p=8.4e-11) | +0.039* (p=5.5e-10) | -0.039* (p=3.2e-11) | +0.037* (p=7.3e-12) |
| **h × 1.25** | -0.042* (p=1.9e-08) | -0.003 (p=0.66) | +0.028* (p=1.3e-11) | -0.023* (p=2.3e-05) | +0.045* (p=6e-16) |
| **h × 1.6** | -0.008 (p=0.081) | -0.007* (p=0.00059) | +0.016* (p=1.3e-06) | -0.000 (p=0.23) | +0.008* (p=0.00016) |
| **h × 2.0** | +0.003 (p=0.063) | -0.000 (p=0.075) | +0.001* (p=0.0008) | +0.000 (p=0.37) | +0.005* (p=5.5e-05) |

`*` = significant at p < 0.05.

### Ranking stability

- **h × 1.0**: ZZ oracle (ceiling) < Load-informed heuristic < GReFEM (zero-shot) < Geometric heuristic < ZZ from coarse solve < Coarse (no refinement)
- **h × 1.25**: ZZ oracle (ceiling) < Load-informed heuristic < Geometric heuristic < GReFEM (zero-shot) < ZZ from coarse solve < Coarse (no refinement)
- **h × 1.6**: Load-informed heuristic < ZZ oracle (ceiling) < GReFEM (zero-shot) < ZZ from coarse solve < Coarse (no refinement) < Geometric heuristic
- **h × 2.0**: ZZ oracle (ceiling) < Load-informed heuristic < Coarse (no refinement) < GReFEM (zero-shot) < ZZ from coarse solve < Geometric heuristic

