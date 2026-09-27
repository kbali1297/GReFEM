## Element-size (sizing-field) sensitivity

Sizing field scaled by a common factor applied to both `h_min` and `h_max` for every refined candidate. Held fixed across levels: the reference solution, Ω_crit, `crit_radius`, and each method's anchor set. At every level all baselines are re-matched to the GReFEM element count, so the comparison remains budget-matched.

**156 objects × 5 loading cases = 773 paired units**, complete at all four levels. Values are relative error in Ω_crit (5 %-trimmed mean over units); lower is better.

### Realised element counts (mean, thousands)

| sizing | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|
| **h × 1.0** | 25.3k | 95.8k | 94.4k | 88.5k | 93.0k | 82.7k |
| **h × 1.25** | 25.3k | 57.4k | 57.3k | 54.7k | 56.1k | 52.5k |
| **h × 1.6** | 25.3k | 33.4k | 33.4k | 32.2k | 32.8k | 31.2k |
| **h × 2.0** | 25.3k | 20.7k | 20.7k | 20.2k | 20.3k | 19.6k |

Refined candidates agree to within 5.8–15.8 % at every level (budget matching holds; `coarse` is the unrefined mesh and is not budget-matched).

### Peak stress (rel. σ_vM^p99 error in Ω_crit)

| sizing | mean cells | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|---|
| **h × 1.0** | 91k | 0.737 | 0.629 | 0.672 | 0.566 | 0.671 | **0.551** |
| **h × 1.25** | 56k | 0.737 | 0.692 | 0.688 | 0.628 | 0.715 | **0.609** |
| **h × 1.6** | 33k | 0.737 | 0.723 | 0.745 | **0.687** | 0.728 | 0.701 |
| **h × 2.0** | 20k | 0.737 | 0.754 | 0.764 | 0.733 | 0.758 | **0.727** |
| *Δ (h×2.0 − h×1.0)* | | *+0.000* | *+0.125* | *+0.092* | *+0.167* | *+0.087* | *+0.176* |

### Strain energy (rel. error in Ω_crit)

| sizing | mean cells | Coarse (no refinement) | GReFEM (zero-shot) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|---|---|
| **h × 1.0** | 91k | 0.943 | 0.881 | 0.891 | 0.868 | 0.886 | **0.851** |
| **h × 1.25** | 56k | 0.943 | 0.906 | 0.895 | 0.891 | 0.905 | **0.868** |
| **h × 1.6** | 33k | 0.943 | 0.927 | 0.934 | **0.915** | 0.925 | 0.915 |
| **h × 2.0** | 20k | 0.943 | 0.940 | 0.942 | **0.928** | 0.943 | 0.931 |
| *Δ (h×2.0 − h×1.0)* | | *+0.000* | *+0.059* | *+0.051* | *+0.060* | *+0.057* | *+0.080* |

### Paired tests at each sizing level (vm_p99)

Median Δ = GReFEM − baseline; negative means GReFEM is better. Two-sided Wilcoxon signed-rank over the paired units.

| sizing | Coarse (no refinement) | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle (ceiling) |
|---|---|---|---|---|---|
| **h × 1.0** | -0.081* (p=1.1e-31) | -0.025* (p=4.3e-10) | +0.041* (p=3.3e-10) | -0.037* (p=4e-10) | +0.037* (p=6.2e-12) |
| **h × 1.25** | -0.042* (p=2.7e-08) | -0.004 (p=0.6) | +0.026* (p=6.4e-11) | -0.023* (p=3.2e-05) | +0.045* (p=1.8e-15) |
| **h × 1.6** | -0.008 (p=0.084) | -0.007* (p=0.00046) | +0.015* (p=3.3e-06) | -0.000 (p=0.21) | +0.006* (p=0.0004) |
| **h × 2.0** | +0.003 (p=0.076) | -0.000 (p=0.077) | +0.002* (p=0.0012) | +0.000 (p=0.35) | +0.006* (p=6.2e-05) |

`*` = significant at p < 0.05.

### Ranking stability

- **h × 1.0**: ZZ oracle (ceiling) < Load-informed heuristic < GReFEM (zero-shot) < ZZ from coarse solve < Geometric heuristic < Coarse (no refinement)
- **h × 1.25**: ZZ oracle (ceiling) < Load-informed heuristic < Geometric heuristic < GReFEM (zero-shot) < ZZ from coarse solve < Coarse (no refinement)
- **h × 1.6**: Load-informed heuristic < ZZ oracle (ceiling) < GReFEM (zero-shot) < ZZ from coarse solve < Coarse (no refinement) < Geometric heuristic
- **h × 2.0**: ZZ oracle (ceiling) < Load-informed heuristic < Coarse (no refinement) < GReFEM (zero-shot) < ZZ from coarse solve < Geometric heuristic

