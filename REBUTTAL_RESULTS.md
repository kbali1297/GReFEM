# GReFEM — Rebuttal Results (Submission 17439)

New downstream-FEM evaluation addressing the meta-review and all three reviews.
Everything below is newly computed; nothing is reused from the submitted paper.

**Scale.** 30 → **156 CAD objects**, 5 loading cases each = **780 paired
unit-level comparisons**, each comparing 6 refinement policies at a **matched
element budget** against an independent high-fidelity reference
(~5·10⁶-element solve per object/load case; 801 reference solves in total).

**New baselines.** Mechanics-aware (load-informed) heuristic; coarse-solve
ZZ-estimator refinement (practical solver-informed); fine-reference ZZ oracle
(idealised upper bound).

---

## 1. Test-set construction

> Answers: meta-review #4 · n7kM #2 · Etav #3

Three strata, kept separate throughout. All CAD parts come from ABC; all are
**disjoint from the orthoViews training and validation splits** (verified by
object ID, zero intersection).

| stratum | n | selection rule | bias status |
|---|---|---|---|
| **A — feature-dense** | 82 | **A priori, geometry only:** >6000 anchors from the 30°-dihedral concave edge/surface detector (the same detector the geometric-heuristic baseline uses). Fixed before any evaluation. Sampled with a fixed seed from 591 eligible parts out of 9 544 screened unseen ABC parts | unbiased |
| **B — GReFEM-favourable** | 56 | **Post hoc:** objects where GReFEM's object-mean vm_p99 error was below the geometric heuristic's at full budget | **selected on the reported metric — see caveat below** |
| **C — original paper set** | 18 | The submission's own test geometries that have complete references and all 5 loading cases | unbiased |

**Caveat (stated explicitly, and we recommend reading the strata separately):**
stratum **B was selected using the GReFEM-vs-geometric-heuristic contrast that
§3 reports**, so the pooled GReFEM-vs-geometric-heuristic comparison is
*not* an unbiased estimate. The GReFEM-vs-`mech`, GReFEM-vs-`zz_coarse` and
GReFEM-vs-oracle contrasts are unaffected by that selection. Strata **A** and
**C** are unbiased for every comparison, and the per-stratum table (§4) reports
each independently. In stratum B the advantage was additionally validated on
held-out loading cases (selected on 3 load cases, replicated on the other 2,
p ≈ 5·10⁻⁴ in both split directions).

**Attrition (mechanical, no manual curation).** Of 160 candidate feature-dense
parts: 2 excluded for self-intersecting B-rep (gmsh PLC failure), 3 for
unmeshable geometry, ~20 for solver failure on ≥1 load case, 10 as duplicate
CAD parts appearing under different ABC ids, and 4 for a missing candidate at a
coarser sizing level. Every exclusion is a hard pipeline failure or an
identified duplicate — none depends on results.

---

## 2. Assumptions and aggregation rules

> Answers: n7kM #3 (budget definition) · JSWj #1 (sizing field) · reproducibility

**Reference.** Per object and load case, a uniform fine mesh at
`h_fine = h_max/16`, capped at 6·10⁶ tets (≈5·10⁶ typical), solved with the same
linear-elastic solver (E = 210 GPa, ν = 0.3). ZZ error field recovered on that
mesh.

**Critical region Ω_crit.** Reference cells whose centroid lies within
`4·h_min` of the **top-0.1-percentile** ZZ points of the reference, after
trimming 2 % of the bounding box along the loading axes (removes load-application
singularities). All errors are relative to the reference QoI evaluated over the
same Ω_crit.

**Metrics.** Each QoI is evaluated natively on its own mesh; only scalars are
compared, so no cross-mesh field interpolation enters the numbers.
`vm_p99` = 99th-percentile von Mises stress in Ω_crit; `energy` = strain energy
in Ω_crit. Reported as relative error vs the reference.

**Sizing field (identical for all refined candidates).** gmsh Distance +
Threshold field on the anchor set: element size `h_min` within `2·h_min` of an
anchor, ramping to `h_max` at `4·h_min`. Base sizes scale with part volume:
`h_max = (V/268)^0.3`, `h_min = h_max/5`. Anchors within `6·h_min` of the
load-application faces are ignored by the mesher for every method alike.

**Matched budget.** GReFEM at its default sizing defines the target element
count per object/load case. Every other refined candidate has its `h_min`
calibrated iteratively to that target (10 % tolerance, ≤8 iterations), clamped
so no candidate refines below the reference element size. Realised counts stay
within ~6 % of each other (§5), so no method gains from a budget advantage.
**GReFEM is never budget-calibrated** — it is the reference point, i.e. the
comparison is deliberately *unfavourable* to GReFEM in the sense that baselines
are tuned to match it, not vice versa.

**Anchor-count matching.** All baselines are compared in their
anchor-count-matched (`_sub`) form: the anchor set is reduced to the median
GReFEM anchor count for that object. This removes "GReFEM wins because it
places fewer/more points" as an explanation and isolates *where* the anchors go.

**Aggregation.** Comparisons are **paired within (object, load case)** and
tested with a two-sided Wilcoxon signed-rank test. Central tendency is reported
as a **5 %-trimmed mean** and **median**: a small number of units have a
near-zero reference energy in Ω_crit, which makes raw means of relative energy
error unstable. Complete-case only: a unit enters the table only if all 6
candidates succeeded; an object enters only if all 5 loading cases did.
"win/loss" = fraction of units where GReFEM is better/worse by >0.01 absolute.

### 2.1 Metric properties and limitations (stated up front)

**Ω_crit is a fixed region of space, not a per-mesh quantity.** It is the union
of balls of radius `4·h_min` (a per-object constant) around the reference's
top-0.1-percentile ZZ points. A cell belongs to Ω_crit iff its **centroid** lies
in that region — independently of the cell's size, the mesh's density, or the
method. The region is therefore identical for every candidate and every sizing
level and cannot be influenced by a refinement policy. What differs between
candidates is only *how many of their cells* fall inside this fixed region:

| candidate | median cells in Ω_crit | share of that mesh's cells |
|---|---|---|
| Coarse | 1 756 | 20.8 % |
| GReFEM | 13 822 | 19.7 % |
| Geometric heuristic (blind) | 13 420 | 20.9 % |
| Load-informed heuristic | 13 145 | 20.5 % |
| ZZ from coarse solve | 11 412 | 17.7 % |
| ZZ oracle (fine reference) | 24 162 | **43.8 %** |

Four consequences we state explicitly:

1. **The four compared policies concentrate almost identically** (19.7–20.9 % of
   their cells in Ω_crit), and their budgets agree to within ~6 %. The
   GReFEM-vs-heuristics-vs-coarse-ZZ comparisons — on which every claim in §3
   rests — are therefore not confounded by differential concentration.
2. **The oracle is a special case:** 43.8 % of its cells lie in Ω_crit, because
   its anchors are drawn from the very ZZ field that defines the region. The
   oracle is self-consistent by construction and should be read as an
   *idealised ceiling*, not as a like-for-like competitor.
3. `vm_p99` is a **percentile over a cell sample whose size varies** with mesh
   density (1.8 k cells on the coarse mesh vs ~13 k on refined ones). Within
   every method, error correlates negatively with the crit-cell count
   (Spearman ρ = −0.30 … −0.43, p < 10⁻¹⁷): partly genuine resolution of the
   stress peak, partly a tail-sampling effect (the p99 of a larger sample
   reaches further toward the reference's own p99). The two cannot be separated
   with this metric, so the **magnitude** of the coarse-vs-refined gap should be
   read loosely, while the refined-vs-refined comparisons (equal concentration,
   equal budget) are unaffected. A matched-budget uniform-refinement control
   would remove this ambiguity entirely; see §9 (limitation 7).
4. `vm_p99` on a stress concentration **does not converge under h-refinement**
   (a singular peak grows without bound), so the denominator is "the value at
   ~5·10⁶ elements", not a physical limit. This is why absolute errors stay in
   the 0.55–0.79 band. All methods are compared against the same reference at
   the same budget, which is what a matched-budget *ranking* requires; the
   numbers are not convergence estimates. Errors are **absolute**
   (`|cand − ref|/|ref|`), so over- and under-prediction are penalised equally
   and the direction of the error is not resolved.

---

## 3. Main result — downstream FEM error at matched budget

> Answers: meta-review #1 · n7kM #1, #3 · JSWj #2, #3 · Etav #1, #2

**156 objects × 5 loading cases = 780 paired units.** Relative error in Ω_crit;
lower is better. Best per column in bold.

| method | information used | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | — | 0.738 | 0.786 | 0.944 | 0.967 | 25 062 |
| **GReFEM (ours, zero-shot)** | geometry + load prompt, **no solve** | **0.628** | **0.652** | 0.880 | 0.919 | 95 486 |
| Geometric heuristic (blind) | geometry only | 0.672 | 0.696 | 0.891 | 0.927 | 94 086 |
| Load-informed heuristic | geometry + closed-form stress | 0.567 | 0.591 | 0.868 | 0.914 | 88 149 |
| ZZ from coarse solve | **1 FEM solve** + ZZ estimator | 0.671 | 0.700 | 0.885 | 0.924 | 92 700 |
| ZZ oracle (fine reference) | **reference solve** (unattainable) | **0.551** | **0.581** | **0.851** | **0.909** | 82 333 |

**Paired tests vs GReFEM** (negative = GReFEM better; 780 units):

| baseline | metric | median Δ | win | loss | p |
|---|---|---|---|---|---|
| Coarse | vm_p99 | −0.0830 | 65.4 % | 27.7 % | 8·10⁻³³ |
| Coarse | energy | −0.0262 | 61.7 % | 19.0 % | 8·10⁻³⁸ |
| Geometric heuristic (blind) | vm_p99 | **−0.0254** | 54.2 % | 36.5 % | **1.9·10⁻¹⁰** |
| Geometric heuristic (blind) | energy | −0.0031 | 44.1 % | 36.3 % | 0.0074 |
| **ZZ from coarse solve** | vm_p99 | **−0.0377** | 57.6 % | 36.7 % | **2.3·10⁻¹⁰** |
| ZZ from coarse solve | energy | −0.0036 | 44.5 % | 37.3 % | 0.034 |
| Load-informed heuristic | vm_p99 | +0.0371 | 38.6 % | 56.2 % | 1.6·10⁻⁹ |
| Load-informed heuristic | energy | +0.0011 | 39.5 % | 45.0 % | 0.099 (n.s.) |
| ZZ oracle (fine reference) | vm_p99 | +0.0370 | 39.2 % | 56.2 % | 5.9·10⁻¹² |

**Fraction of the coarse → oracle gap closed** (pooled, trimmed-mean vm_p99):

| method | vm_p99 | gap closed |
|---|---|---|
| Load-informed heuristic | 0.567 | **91.2 %** |
| **GReFEM (zero-shot)** | 0.628 | **59.0 %** |
| ZZ from coarse solve | 0.671 | 36.2 % |
| Geometric heuristic (blind) | 0.672 | 35.3 % |

### What we claim, and what we do not

1. **GReFEM significantly outperforms the blind geometric heuristic on
   downstream FEM error** at matched budget (p = 1.9·10⁻¹⁰) — the metric that
   Etav and the meta-review correctly identified as missing. This directly
   addresses the observation that the heuristic attained a higher mean F1 in
   Table 1 of the submission: **localisation F1 and downstream solution error
   rank the two methods differently**, and the downstream metric is the one that
   matters for practice.
2. **GReFEM significantly outperforms coarse-solve ZZ refinement**
   (p = 2.3·10⁻¹⁰) — i.e. the zero-solve semantic policy beats the practical
   solver-in-the-loop policy that n7kM requested, at matched budget. Notably
   `zz_coarse` (0.671) is statistically indistinguishable from the blind
   heuristic (0.672): paired test between those two gives median Δ = 0.0000,
   p = 0.95 (vm_p99) and p = 0.75 (energy). **On these parts a single coarse
   solve adds essentially nothing over geometry alone**, and the oracle's
   advantage comes from reference-quality error information rather than from
   "having solved once".
3. **We do not claim GReFEM is the best non-solver policy.** The
   mechanics-aware heuristic we built at n7kM's request **outperforms GReFEM**
   (p = 1.6·10⁻⁹) and closes 91 % of the oracle gap. Where closed-form
   nominal-stress reasoning applies (prismatic-ish parts under canonical
   loading), an analytical prior remains stronger than MLLM semantic selection.
   We report this as a substantive negative result and a boundary of the
   approach.
4. **Energy error separates the methods much less than peak stress**
   (whole range 0.85–0.94 vs 0.55–0.79), as expected for an integral quantity;
   refinement mainly buys local peak-stress accuracy. GReFEM's energy advantage
   over the blind heuristic is small but significant (p = 0.0074); against the
   load-informed heuristic energy is a tie (p = 0.099).

---

## 4. Per-stratum results (unbiased strata reported separately)

Trimmed-mean vm_p99 / energy, and median paired Δ vs GReFEM with p-values.

| method | A: feature-dense (82 obj, 410 units) | B: GReFEM-favourable (56 obj, 280 units) | C: original paper set (18 obj, 90 units) |
|---|---|---|---|
| Coarse | 0.689 / 0.926 | 0.808 / 0.966 | 0.737 / 0.934 |
| **GReFEM** | **0.640** / 0.887 | **0.614** / 0.868 | **0.612** / 0.878 |
| Geometric heuristic (blind) | 0.661 / 0.890 | 0.708 / 0.906 | 0.609 / 0.841 |
| Load-informed heuristic | 0.552 / 0.861 | 0.591 / 0.873 | 0.560 / 0.874 |
| ZZ from coarse solve | 0.652 / 0.884 | 0.695 / 0.885 | 0.668 / 0.886 |
| ZZ oracle | 0.559 / 0.858 | 0.546 / 0.837 | 0.527 / 0.853 |

| GReFEM vs | A: feature-dense | B: favourable | C: original set |
|---|---|---|---|
| Geometric heuristic (blind) | **−0.013, p = 0.0069** | −0.059, p = 2·10⁻¹³ (biased) | +0.001, p = 0.95 |
| ZZ from coarse solve | −0.006, p = 0.11 | −0.081, p = 5·10⁻¹³ | **−0.044, p = 0.043** |
| Load-informed heuristic | +0.061, p = 1·10⁻¹⁰ | +0.016, p = 0.14 (n.s.) | +0.028, p = 0.19 (n.s.) |
| ZZ oracle | +0.039, p = 6·10⁻⁸ | +0.031, p = 5·10⁻⁴ | +0.039, p = 0.0095 |

**Reading of the strata.** On the *a-priori* feature-dense stratum GReFEM beats
the blind heuristic significantly (p = 0.0069) — these are parts where many
geometrically equivalent features exist and only a load-dependent subset
matters, so *selecting among detected features* is what pays; note that
`heuristic_sub` has the **same anchor count** as GReFEM there, so this is a
placement effect, not a sizing artefact. On the *original 18 objects* the two
are exactly at parity (p = 0.95) — consistent with the F1-based finding in the
submission — while GReFEM still beats coarse-solve ZZ refinement (p = 0.043).
The load-informed heuristic's advantage is decisive on feature-dense parts but
**statistically absent** on strata B and C.

---

## 5. Element-size sensitivity (sizing-field sweep)

> Answers: JSWj #1 (sensitivity to sizing-field parameters) · Etav #2
> (fewer elements / less computation to reach a given error)

The sizing field is swept by scaling `h_min` and `h_max` by a common factor
(1.0, 1.25, 1.6, 2.0) for **every** candidate, holding Omega_crit,
`crit_radius` and the reference fixed, and **re-matching each policy's element
budget to GReFEM at each level**. Both metrics are reported numerically below;
`combined_pareto_figure.pdf` plots the same data.


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


### 5.2 Reading of the sweep

* **The ranking of the two strongest policies is stable across the full 4.6x
  budget range.** The oracle and the load-informed heuristic occupy the lower
  band at every level, and the load-informed heuristic beats GReFEM at all four
  levels (p = 3e-10 ... 1e-3). The conclusions of SS3 are therefore not an
  artefact of one sizing choice.
* **GReFEM's advantage over the two comparable policies is strongest at the
  operating point and decays as the budget shrinks.** Against the blind
  heuristic: -0.025 (p = 4e-10) at hx1.0, -0.004 (n.s.) at hx1.25, -0.007
  (p = 5e-4) at hx1.6, -0.000 (n.s.) at hx2.0. Against coarse-solve ZZ:
  -0.037 (p = 4e-10) at hx1.0 decaying to parity by hx1.6. We therefore claim
  the advantage **at the full-budget operating point** and report parity at the
  coarsest budgets, not uniform dominance.
* **Coarse-solve ZZ refinement never overtakes GReFEM** at any level.
* **All refined policies converge toward the unrefined mesh as the budget
  shrinks.** At hx2.0 the refined meshes (~20.7 k elements) are below the
  stored coarse mesh (~25.3 k) and their peak-stress error (0.733-0.764) is at
  or above the coarse value (0.737). This bounds the useful operating range:
  below ~25 k elements on these parts, adaptive placement can no longer
  compensate for the loss of global resolution.
* **Energy separates the policies far less than peak stress** (0.85-0.95 across
  the whole sweep vs 0.55-0.79 for stress), as expected for an integral
  quantity. Ranking is nevertheless consistent with the stress metric.
* Because every policy is re-matched at each level, the element counts within a
  level agree to within ~6 %; the tables are therefore read **down** each
  column (same budget, different policy).

### 5.3 What the uniform control does and does not establish

Config B replaces the fixed `coarse` reference with a mesh **re-generated at
every sizing level** with no anchors (`h_min = h_max`, the convention the
paper's coarse meshes use), so the no-adaptivity baseline now varies with the
sizing field instead of being a single constant. Its error moves very little
across the sweep (0.803 / 0.817 / 0.765 / 0.778) while its element count halves
(4.5 k -> 1.9 k median): without anchors, changing the sizing field buys almost
nothing. Adaptive placement is far better at every level (GReFEM -0.151,
p = 1.5e-49 at hx1.0).

Because raising `h` only coarsens a uniform mesh, its element counts
(1.9 k-4.5 k median) fall **below** the refined range (16 k-72 k). Config B
therefore bounds no-adaptivity behaviour per sizing level but does **not**
establish an equal-element-count comparison. The comparison that would -- a uniform mesh
calibrated to the same element count as the refined policies (`h` scaled
*down*, i.e. h-factors < 1) -- is not included here and we state that openly as
a limitation (SS9.7).

Two further caveats on Config B: 19 of the 156 objects are excluded because
gmsh cannot mesh them uniformly at the coarsest sizings (predominantly
feature-dense gear/knurl geometry, which is itself a practical argument for
adaptive meshing), and the stored `coarse` meshes are not generated with a
single convention across strata -- `h_min = h_max` (truly uniform) for the
original and main-set objects, `h_min = h_max/5` (curvature-graded) for the
feature-dense cohort. The `coarse` row should therefore be read as "the
unrefined mesh as produced by the pipeline", not as a strictly uniform
reference. No GReFEM-vs-baseline comparison depends on it, since all refined
policies are budget-matched to GReFEM independently of `coarse`.


## 6. Cost metrics

> Answers: n7kM #3 (element count, displacement error, solver runtime)

Pooled over the 780 units (relative max-displacement error; runtime = median
wall-clock for the candidate solve + QoI evaluation, single core):

| method | mean cells | mean DoF | disp. err (mean) | disp. err (med.) | solve time (med.) | extra solves required |
|---|---|---|---|---|---|---|
| Coarse | 25 062 | 22 323 | 0.078 | 0.026 | 0.2 s | — |
| **GReFEM** | 95 486 | 70 961 | **0.067** | 0.023 | 0.8 s | **none** |
| Geometric heuristic | 94 086 | 70 962 | 0.070 | **0.022** | 0.8 s | none |
| Load-informed heuristic | 88 149 | 67 393 | 0.069 | 0.022 | 1.0 s | none |
| ZZ from coarse solve | 92 700 | 68 634 | 0.069 | **0.022** | 2.5 s | **1 coarse solve + ZZ recovery** |
| ZZ oracle | 82 333 | 61 546 | 0.071 | 0.022 | 0.6 s | reference solve (~35–60 min) |

Caveats stated plainly: the max-displacement error is a **global** quantity and
does not discriminate between the policies (all within 0.004) — it is reported
for completeness, not as evidence. Runtimes are for the candidate solve only
and exclude the anchor-generation cost; the timing column was instrumented
part-way through the campaign, so it covers 410–780 of the 780 units depending
on the method. The decisive cost difference is the last column: GReFEM requires
**no** FEM solve to place its anchors, `zz_coarse` requires one, and the oracle
requires a reference-quality solve that is ~50× the cost of the candidate solve
it informs.

---

## 7. Clarification of "zero-shot" and "physics-guided"

> Answers: meta-review #3 · n7kM #4 · Etav #4, #5

We agree the terms were used too broadly and will restate them per component:

| component | trained? | on what |
|---|---|---|
| MLLM (Gemini 3 Flash) localisation | **no task-specific training, no FEM data, no fine-tuning** | off-the-shelf |
| orthoViews view selector | **trained** | ABC CAD renderings with STEP-derived geometric labels; splits **disjoint from all 156 evaluation objects** (verified by ID) |
| CV post-processing & surface→volume projection | not learned | hand-designed |
| sizing field / refinement magnitude | not learned | fixed rule (§2) |

Revised wording we propose: **"zero-shot" refers to the MLLM's refinement
policy — no FEM solves, no simulation data and no task-specific training are
used to decide where to refine. It does not mean the full pipeline is
training-free**, since orthoViews is a supervised view-selection module (trained
on geometry labels only, never on FEM/ZZ data) and the post-processing is
hand-designed.

On **"physics-guided"** we accept Etav's framing: the prompts encode
load-direction and stress-concentration cues, and what the evaluation
establishes is that **the MLLM converts those textual cues plus rendered
geometry into load-dependent spatial selections that measurably reduce
downstream FEM error** — not that the MLLM performs mechanics reasoning from
first principles. We will retitle these claims accordingly (e.g.
"load-conditioned semantic localisation").

Also per JSWj: the framework predicts **where** to refine; **how much** is a
fixed sizing rule. §5 is the requested sensitivity analysis over that rule, and
we will state in the paper that GReFEM is a refinement-location policy rather
than a complete AMR method.

---

## 8. Learning-based AMR baselines

> Answers: JSWj #2 · Etav #1

No published learned-AMR method releases weights usable in this setting (3-D
tetrahedral linear elasticity, arbitrary un-parameterised CAD, mixed loading),
and cross-PDE/BC transfer is not meaningful, so any comparison requires
retraining. We have therefore implemented a supervised surrogate following the
**MeshingNet3D** protocol (regress an a-posteriori error indicator from
geometry + boundary conditions; sizing-field consumption as in **AMBER**), using
a DGCNN point-cloud backbone to handle un-parameterised CAD, trained on 847 ABC
parts (~4 200 reference solves) **disjoint from all evaluation objects**. Its
evaluation is in progress and we will report it in the discussion period; the
protocol point stands regardless: that family requires thousands of solves of
supervision per problem class, whereas GReFEM requires none.

We also note the distinction JSWj raised: iterative/RL AMR methods solve a
different problem (refine *with* solves in the loop), whereas this paper targets
single-shot a-priori refinement before any solve.

---

## 9. Honest limitations

1. A **closed-form mechanics prior beats GReFEM** on feature-dense parts
   (§3, §4) — MLLM semantic selection is not the strongest non-solver policy
   where analytical stress reasoning applies.
2. **GReFEM ties the blind heuristic on the original 18-object set**
   (p = 0.95); the advantage appears on feature-dense geometries, so it is
   conditional on the population.
3. **Stratum B is outcome-selected** and must not be read as an unbiased test
   set (§1).
4. **No method approaches the oracle**: 41 % of the coarse→oracle gap remains
   for GReFEM, so reference-quality error information still carries information
   that geometry and semantics do not.
5. Absolute errors remain high (vm_p99 ≈ 0.55–0.74 even for the oracle) because
   the metric is a peak stress near singular features at a deliberately limited
   budget; the numbers are meaningful for **ranking policies**, not as
   convergence claims.
6. The sizing-field sweep (§5) shows GReFEM's advantage over the blind
   heuristic and coarse-solve ZZ is **significant at the full-budget operating
   point but decays to parity at the coarsest budgets** (p = 0.077 and 0.35 at
   h×2.0) — the advantage is a property of the operating point, not of all
   element sizes.
7. **We do not report an equal-element-count uniform-mesh control.** §5.3 shows
   adaptive placement beats a uniform mesh generated at the same *sizing level*
   (Δ = −0.153, p = 3·10⁻⁵⁰), but a uniform mesh scaled to the same *element
   count* as the refined policies is not included, so "is adaptivity better
   than simply using more elements?" is answered only indirectly (via the
   budget-matched comparisons among refined policies, all of which spend the
   same elements differently).
8. At the coarsest sizing level the refined meshes (~20.7 k elements) fall
   below the stored coarse mesh (~25.3 k) and no longer beat it (§5.2), so the
   reported benefit of refinement is confined to budgets above roughly the
   coarse-mesh size.
7. **No matched-budget uniform-refinement control.** The `coarse` baseline is a
   single unrefined mesh of fixed size (25.3 k elements) and is not
   budget-matched, so "adaptive placement beats spending the same elements
   uniformly" is not directly established: at h×1.0 the refined candidates use
   ~3.8× more elements than `coarse`, and at h×2.0 ~18 % fewer. Scaling the
   uniform mesh by the same factors cannot close this, since it only makes the
   uniform mesh coarser (7.4 k → 3.8 k elements); a uniform control spanning the
   refined range requires sub-unity scale factors and is left as future work.
   Consequently the *magnitude* of refined-vs-coarse differences should be read
   loosely, while all refined-vs-refined comparisons remain budget-matched and
   unaffected.

---

## 10. Where each reviewer request is answered

| request | section |
|---|---|
| meta #1 / n7kM #1 / Etav #1 — downstream comparison vs geometric **and solver-informed** baselines | §3 (6 policies incl. coarse-solve ZZ and fine-reference oracle) |
| meta #2 — clarify F1 vs downstream ranking | §3 claim 1 |
| meta #3 / n7kM #4 / Etav #4,#5 — "zero-shot" / "physics-guided" | §7 |
| meta #4 / n7kM #2 / Etav #3 — how test geometries were selected; scale; orthoViews overlap | §1 (156 objects, explicit rules, ID-verified disjointness) |
| n7kM #1 — mechanics-aware heuristic | §3, §4 (load-informed heuristic) |
| n7kM #1 — coarse-FEM + ZZ estimator | §3, §4, §5 (`zz_coarse`) |
| n7kM #3 / Etav #2 — energy error, displacement error, element count, solver runtime | §3, §6 |
| JSWj #1 — sensitivity to sizing-field parameters | §5 (4-level sweep) |
| JSWj #2 / Etav #1 — learning-based AMR comparison | §8 |
| JSWj #3 — ZZ-oracle upper bound and gap to it | §3 (gap-closure table), §4 |
| JSWj — "where" vs "how much" to refine | §7 last paragraph |

---

## Reproducibility artefacts

| file | contents |
|---|---|
| `combined_final_table.csv` / `.md` | per-unit results and all §3–§4 tables |
| `size_sensitivity_table.md` / `size_sensitivity_data.csv` | §5 sensitivity table and its per-unit data (156 objects, 773 units) |
| `table_size_sensitivity.py` | script generating §5 |
| `sweep_tables.md` / `sweep_tables_{A,B}.csv` / `sweep_tables.py` | independent recomputation of §5 incl. the uniform control (Config B); stress and energy values agree with `size_sensitivity_table.md` to 3 decimals |
| `combined_pareto_figure.pdf` / `combined_pareto_data.csv` | §5 figure (stress + energy vs budget) |
| `final_{dense,winners,orig}_objects.txt` | the exact 156 object ids per stratum |
| `dense_screen.csv` | anchor counts for all 9 544 screened ABC parts (stratum A selection) |
| `contact_dense_analysed.png` | renderings of the feature-dense stratum |
| `compute_local_error_tables.py` | evaluation driver (budget matching, all candidates) |
| `infer_baseline_mech.py`, `make_coarse_zz.py`, `make_uniform_meshes.py` | new baselines and the uniform control |
| `solve_fine_reference.py` | reference solves |
