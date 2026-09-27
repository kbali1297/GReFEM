
### Config A (primary) — element-size sensitivity, 6 policies

All refined policies budget-matched to GReFEM at each sizing level. `coarse` is the stored unrefined mesh and is the same at every level (it is not re-scaled), so it appears as a single reference value.

**156 objects, 773 units** complete with all 6 policies at all 4 sizing levels.


**Relative peak-stress error (vm_p99 in Omega_crit)** — 5 %-trimmed mean (median in brackets); median realised element count in *italics*

| policy | h×1 | h×1.25 | h×1.6 | h×2 |
|---|---|---|---|---|
| Coarse mesh, default sizing only (fixed reference) | 0.737 [0.784]<br>*7,627* | 0.737 [0.784]<br>*7,627* | 0.737 [0.784]<br>*7,627* | 0.737 [0.784]<br>*7,627* |
| GReFEM (zero-shot MLLM) | 0.629 [0.653]<br>*77,607* | 0.692 [0.718]<br>*47,337* | 0.723 [0.754]<br>*27,064* | 0.754 [0.795]<br>*16,168* |
| Geometric heuristic (blind) | 0.672 [0.696]<br>*75,580* | 0.688 [0.714]<br>*47,228* | 0.745 [0.774]<br>*27,208* | 0.764 [0.798]<br>*16,109* |
| Load-informed heuristic | 0.566 [0.589]<br>*70,729* | 0.628 [0.653]<br>*45,623* | **0.687 [0.716]**<br>*26,036* | 0.733 [0.771]<br>*15,905* |
| ZZ from coarse solve (1 solve) | 0.671 [0.700]<br>*75,877* | 0.715 [0.755]<br>*47,228* | 0.728 [0.767]<br>*26,923* | 0.758 [0.800]<br>*15,855* |
| ZZ oracle (fine ref, ceiling) | **0.551 [0.584]**<br>*61,392* | **0.609 [0.642]**<br>*41,815* | 0.701 [0.742]<br>*24,843* | **0.727 [0.775]**<br>*15,231* |

**Relative strain-energy error (Omega_crit)** — 5 %-trimmed mean (median in brackets); median realised element count in *italics*

| policy | h×1 | h×1.25 | h×1.6 | h×2 |
|---|---|---|---|---|
| Coarse mesh, default sizing only (fixed reference) | 0.943 [0.967]<br>*7,627* | 0.943 [0.967]<br>*7,627* | 0.943 [0.967]<br>*7,627* | 0.943 [0.967]<br>*7,627* |
| GReFEM (zero-shot MLLM) | 0.881 [0.921]<br>*77,607* | 0.906 [0.942]<br>*47,337* | 0.927 [0.956]<br>*27,064* | 0.940 [0.970]<br>*16,168* |
| Geometric heuristic (blind) | 0.891 [0.928]<br>*75,580* | 0.895 [0.930]<br>*47,228* | 0.934 [0.961]<br>*27,208* | 0.942 [0.970]<br>*16,109* |
| Load-informed heuristic | 0.868 [0.914]<br>*70,729* | 0.891 [0.931]<br>*45,623* | **0.915 [0.948]**<br>*26,036* | **0.928 [0.964]**<br>*15,905* |
| ZZ from coarse solve (1 solve) | 0.886 [0.925]<br>*75,877* | 0.905 [0.952]<br>*47,228* | 0.925 [0.957]<br>*26,923* | 0.943 [0.972]<br>*15,855* |
| ZZ oracle (fine ref, ceiling) | **0.851 [0.909]**<br>*61,392* | **0.868 [0.921]**<br>*41,815* | 0.915 [0.956]<br>*24,843* | 0.931 [0.968]<br>*15,231* |

**Paired tests per level** (median Δ = GReFEM − policy; negative = GReFEM better; n = 773)

| level | Coarse mesh, default sizing only | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle |
|---|---|---|---|---|---|
| h×1 | -0.081, p=1.1e-31 | -0.025, p=4.3e-10 | +0.041, p=3.3e-10 | -0.037, p=4e-10 | +0.037, p=6.2e-12 |
| h×1.25 | -0.042, p=2.7e-08 | -0.004, p=0.6 | +0.026, p=6.4e-11 | -0.023, p=3.2e-05 | +0.045, p=1.8e-15 |
| h×1.6 | -0.008, p=0.084 | -0.007, p=0.00046 | +0.015, p=3.3e-06 | -0.000, p=0.21 | +0.006, p=0.0004 |
| h×2 | +0.003, p=0.076 | -0.000, p=0.077 | +0.002, p=0.0012 | +0.000, p=0.35 | +0.006, p=6.2e-05 |

### Config B (control) — adds a uniform mesh re-generated at each sizing level

The uniform mesh applies the same h-scaling as the refined policies but with no anchors. Because scaling h upward only coarsens it, its element counts fall **below** the refined range; it therefore bounds the no-adaptivity behaviour at each sizing level but does not provide an equal-element-count comparison. Object count is reduced because gmsh cannot mesh every geometry uniformly at the coarsest sizings.

**138 objects, 642 units** complete with all 6 policies at all 4 sizing levels.


**Relative peak-stress error (vm_p99 in Omega_crit)** — 5 %-trimmed mean (median in brackets); median realised element count in *italics*

| policy | h×1 | h×1.25 | h×1.6 | h×2 |
|---|---|---|---|---|
| Coarse/uniform mesh at this sizing (h_min=h_max) | 0.803 [0.853]<br>*4,515* | 0.817 [0.868]<br>*3,346* | 0.765 [0.844]<br>*2,451* | 0.778 [0.844]<br>*1,865* |
| GReFEM (zero-shot MLLM) | 0.630 [0.656]<br>*72,213* | 0.692 [0.717]<br>*45,873* | 0.733 [0.765]<br>*25,699* | 0.766 [0.802]<br>*15,858* |
| Geometric heuristic (blind) | 0.683 [0.709]<br>*70,290* | 0.697 [0.724]<br>*45,220* | 0.751 [0.780]<br>*26,080* | 0.773 [0.804]<br>*15,464* |
| Load-informed heuristic | 0.580 [0.604]<br>*67,304* | 0.633 [0.656]<br>*43,127* | **0.694 [0.718]**<br>*24,896* | 0.739 [0.777]<br>*15,524* |
| ZZ from coarse solve (1 solve) | 0.682 [0.714]<br>*70,936* | 0.722 [0.766]<br>*45,936* | 0.737 [0.780]<br>*26,050* | 0.762 [0.806]<br>*15,576* |
| ZZ oracle (fine ref, ceiling) | **0.554 [0.585]**<br>*57,396* | **0.619 [0.655]**<br>*39,840* | 0.712 [0.755]<br>*23,968* | **0.738 [0.783]**<br>*14,896* |

**Relative strain-energy error (Omega_crit)** — 5 %-trimmed mean (median in brackets); median realised element count in *italics*

| policy | h×1 | h×1.25 | h×1.6 | h×2 |
|---|---|---|---|---|
| Coarse/uniform mesh at this sizing (h_min=h_max) | 0.968 [0.984]<br>*4,515* | 0.965 [0.985]<br>*3,346* | 0.961 [0.979]<br>*2,451* | 0.963 [0.982]<br>*1,865* |
| GReFEM (zero-shot MLLM) | 0.882 [0.921]<br>*72,213* | 0.907 [0.941]<br>*45,873* | 0.931 [0.958]<br>*25,699* | 0.944 [0.972]<br>*15,858* |
| Geometric heuristic (blind) | 0.897 [0.932]<br>*70,290* | 0.902 [0.937]<br>*45,220* | 0.937 [0.962]<br>*26,080* | 0.945 [0.971]<br>*15,464* |
| Load-informed heuristic | 0.876 [0.926]<br>*67,304* | 0.892 [0.935]<br>*43,127* | **0.916 [0.949]**<br>*24,896* | **0.928 [0.963]**<br>*15,524* |
| ZZ from coarse solve (1 solve) | 0.891 [0.928]<br>*70,936* | 0.909 [0.958]<br>*45,936* | 0.929 [0.960]<br>*26,050* | 0.943 [0.973]<br>*15,576* |
| ZZ oracle (fine ref, ceiling) | **0.852 [0.913]**<br>*57,396* | **0.870 [0.925]**<br>*39,840* | 0.921 [0.958]<br>*23,968* | 0.934 [0.969]<br>*14,896* |

**Paired tests per level** (median Δ = GReFEM − policy; negative = GReFEM better; n = 642)

| level | Coarse/uniform mesh at this sizing | Geometric heuristic | Load-informed heuristic | ZZ from coarse solve | ZZ oracle |
|---|---|---|---|---|---|
| h×1 | -0.151, p=1.5e-49 | -0.031, p=2.1e-12 | +0.026, p=3.5e-06 | -0.044, p=2.4e-12 | +0.035, p=1.4e-09 |
| h×1.25 | -0.112, p=9.7e-37 | -0.010, p=0.058 | +0.024, p=5.6e-08 | -0.027, p=1.4e-06 | +0.042, p=1.6e-11 |
| h×1.6 | -0.041, p=8.8e-06 | -0.003, p=0.0079 | +0.014, p=2.2e-05 | -0.000, p=0.26 | +0.003, p=0.0027 |
| h×2 | -0.030, p=0.0067 | +0.000, p=0.16 | +0.003, p=0.00018 | +0.000, p=0.77 | +0.006, p=4.8e-05 |
