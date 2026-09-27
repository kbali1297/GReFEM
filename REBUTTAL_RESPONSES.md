# GReFEM - Rebuttal Responses (Submission 17439)

The tables and baseline definitions are reproduced in each reply so that every
response is self-contained. The learned-AMR baseline (**AMBER**, retrained from
scratch for this setting) is now included in each reply. All results below are newly
computed on 156 CAD objects sampled uniformly at random from the ABC dataset
(the 30 objects of the original submission plus 126 newly drawn), each under 5
loading cases, giving 780 paired case-level comparisons. Both the original 30
and the additional 126 were drawn at random with no selection for outcome,
geometry, or difficulty. Every (geometry,load) case is evaluated against
an independent high-fidelity reference solve (a uniform fine mesh of
approximately 5·10⁶ elements; 801 reference solves in total). Errors are
reported relative to the reference quantity of interest within a fixed critical
region Ω_crit, defined as the spatial neighbourhood of the reference solution's
top-0.1-percentile Zienkiewicz–Zhu (ZZ) error points.

The four points raised in the meta-review are addressed within the reviewer
responses below: downstream baseline comparisons appear in all three replies;
the relationship between localisation F1 and downstream error is discussed under
n7kM (W2) and Etav (W3); the scoping of the terms "zero-shot" and
"physics-guided" is treated under n7kM (W4) and Etav (W4/W5); and the
test-geometry selection procedure is described under n7kM (W2).

-

## Response to the Meta-Review (Area Chair MyrJ)

We thank the Area Chair for the clear summary and the four concrete criteria. Each has been addressed with new experiments, reported in full in the reviewer responses below; we summarise here where each answer can be found.

**1. "Add downstream comparison against geometric and solver-informed baselines."**
We compare seven refinement policies at a *matched element budget* on 156 CAD parts × 5 load cases (780 paired cases): the unrefined coarse mesh, GReFEM, the blind geometric heuristic, a new *load-informed* (mechanics-aware) heuristic, a solver-informed **ZZ-from-coarse-solve** policy, a **ZZ-from-fine-reference-solve** upper bound, and a retrained learned-AMR baseline (**AMBER**). All policies receive the same number of refinement anchors as GReFEM and matched element counts, so differences reflect more on *where* smaller elements are placed. Downstream error is measured as peak von Mises stress (σ_vM^p99, the failure-relevant quantity) and strain energy (the natural FEM error norm) against independent ~5·10⁶-element reference solves. GReFEM outperforms the blind heuristic, the coarse-solve ZZ policy, and AMBER on peak-stress error; the closed-form load-informed heuristic outperforms GReFEM where beam-theory reasoning applies, which we report openly as a boundary of the method. GReFEM nonetheless retains a distinct practical value: the heuristic's weighting formulas must be derived by hand for each family of loading cases and silently lose validity outside the slender-member regime they assume, whereas GReFEM adapts to a new load family by editing a textual prompt, requires no per-family engineering, and closes 59% of the coarse-to-fine-reference gap with no solve and no hand-derived mechanics. Full tables: n7kM (Tables 1–3), JSWj (Tables 1–2), Etav (Tables 1–2).

**2. "Explain and clarify the F1-based scores."**
F1 counts overlap with ZZ-critical cells at a fixed top-k and weights every critical cell equally, regardless of severity; a method can therefore attain high F1 by covering many mildly-critical cells while under-resolving the few peak-stress cells that dominate solution error. This is exactly what the new data show: the blind heuristic's higher mean F1 does *not* carry over to downstream FEM error, where GReFEM is the stronger method at a matched budget. We now treat downstream error, not F1 overlap, as the primary criterion. Details: n7kM ("Localisation F1") and Etav ("Localization F1").

**3. "Clarify the 'zero-shot' and 'physics-guided' claims."**
We accept both points and apologize for the unintended confusion. "Zero-shot" refers strictly to the MLLM refinement policy (no task-specific tuning); orthoViews is a supervised view selector trained on geometry labels only, so the full pipeline is not training-free. "Physics-guided" now refers to the narrower empirical claim that the MLLM converts textual load/stress cues plus rendered geometry into load-dependent spatial selections that measurably refine the FEM mesh where stress concentrations must be resolved. This is supported by a new prompt ablation (Load-Only to Load+Features+Heuristics improves error monotonically, identically across two MLLM backends), showing the gain comes from the physics cues and from how faithfully the model follows them when predicting anchor points on the geometry. Details: n7kM (Q4), Etav (W4/W5).

**4. "Explain how the 30 test geometries are selected."**
The original 30 objects were sampled uniformly at random from the ABC dataset, with no selection for outcome, geometry, or difficulty, and are verified by object identifier to be disjoint from the orthoViews training/validation splits. The evaluation is now expanded with 126 further parts drawn the same way, to 156 objects and 780 cases; attrition is purely mechanical (B-rep/meshing/solver failures, de-duplication) and never depends on results. Details: n7kM (Q2), Etav (Q3).

We hope these additions, in particular the seven-policy downstream comparison with solver-informed and learned baselines at matched budgets, meet the bar set out in the meta-review, and we remain eager to provide any further analysis during the discussion window.

-

## Response to Reviewer n7kM

We thank the reviewer for the specific and constructive suggestions. We have implemented both requested baselines and expanded the evaluation from 30 to 156 CAD geometries, additional 126 randomly sampled from the ABC dataset [1].

We first define the refinement policies compared throughout, all evaluated at a matched element budget:

- **Coarse:** the unrefined uniform mesh (fixed floor, no refinement).
- **GReFEM (@5orthoviews):** Our framework with trained ortho-views selection and zero shot MLLM inference.
- **Geometric heuristic (blind):** the submission's load-agnostic detector of concave sharp edges and smooth concave entities.
- **Load-informed heuristic:** the same geometric candidates re-weighted by a closed-form nominal stress heuristic (see W1a), so the loading case decides which features matter; no solver.
- **ZZ from coarse solve:** refinement anchor points placed at the top-percentile ZZ error points of a single coarse FEM solve (see W1b).
- **ZZ from fine reference solve:** refinement anchor points at the top-percentile ZZ points of the fine reference solution (uniform mesh of ~5·10⁶ elements); an unattainable upper bound.
- **AMBER (learned AMR):** learned sizing-field method [2], retrained from scratch with load-case conditioning; described after the sizing sweep.

All baselines are matched to GReFEM: each is given the *same number of refinement anchor points* as GReFEM on that case, and the sizing field is tuned so the element count matches GReFEM's as closely as the mesher allows. AMBER is the only exception: it emits a continuous sizing field, so its element count is matched to GReFEM's via a global scaling factor.

**Q1a Mechanics-aware heuristic.** We added the load-aware counterpart of the blind heuristic. It keeps the *same* geometric candidate points and changes only *which* of them are refined, using
a solver-free estimate of nominal stress:

1. **Cross-sections.** We voxel-fill the part and slice it into thin layers along the loading axis y (clamped at y_min, loaded at y_max). For each layer we compute purely geometric section properties: area A(y), centroid (x̄, z̄), second moment of area I_z(y), and polar moment J(y).
2. **Nominal-stress weight.** Each candidate is assigned to its nearest layer and given the textbook stress for the load case: w = 1/A (compression, net-section stress), w =|x − x̄|/I_z (bending, σ = Mc/I), and w = r/J (torsion, τ = Tr/J), with r the distance to the layer centroid; combined loads sum the normalised terms.
3. **Selection.** We keep the top stress-weighted 30% of candidates and density-filter to the matched anchor budget.

Geometry thus decides *where* refinement may go and beam-theory stress decides *which* of those locations matter under the given load; this mirrors the "Load+Feature+Heuristics" prompt in our appendix. While this analytical prior outperforms GReFEM where closed-form reasoning is valid (Table 1, 3), it must, however, be derived by hand for each family of loading cases, whereas GReFEM can generalize through a textual prompt for more potential applications.

**Q1b Coarse-solve ZZ estimator.** We also added the practical solver-informed baseline. We perform a single coarse FEM solve, recover the ZZ error field on that coarse mesh, and place anchors at its top-percentile error points. Although this policy consumes one real solve, **GReFEM outperforms
it**. The same estimator applied to the fine reference solve gives the "ZZ from fine reference solve" rows, an unattainable upper bound.

Table 1
|method|information used|vm_p99 mean|vm_p99 med.|energy mean|energy med.|cells|
|-|-|-|-|-|-|-|
|Coarse (no refinement)|—|0.738|0.786|0.944|0.967|25 062|
|GReFEM (Gemini)|Load + Features + Heuristics|0.628|0.652|0.880|0.919|95 486|
|GReFEM (GPT-5.4-mini)|Load + Features + Heuristics|0.630|0.652|0.872|0.915|97 257|
|GReFEM (Gemini), reduced prompt|Load + Features|0.636|0.667|0.873|0.920|97 851|
|GReFEM (GPT-5.4-mini), reduced prompt|Load + Features|0.644|0.665|0.879|0.921|93 296|
|GReFEM (Gemini), minimal prompt|Load-Only|0.650|0.675|0.881|0.923|94 187|
|GReFEM (GPT-5.4-mini), minimal prompt|Load-Only|0.666|0.694|0.889|0.927|93 677|
|Geometric heuristic (blind)|geometry only|0.672|0.696|0.891|0.927|94 086|
|Load-informed Geometric heuristic|geometry + closed-form stress heuristic|0.567|0.591|0.868|0.914|88 149|
|ZZ from coarse solve|**1 FEM solve** + ZZ estimator|0.671|0.700|0.885|0.924|92 700|
|ZZ from fine reference solve|**fine reference solve**|**0.551**|**0.581**|**0.851**|**0.909**|82 333|
|AMBER (learned AMR)|440 supervised expert meshes + load|0.666|0.679|0.875|0.929|91 679|

Rows 3–7 ablate GReFEM's MLLM and prompt at the same matched budget: swapping Gemini for GPT-5.4-mini leaves accuracy unchanged, while stripping the load/stress cues from the prompt degrades error monotonically; the gain comes from the textual physics cues, not from a particular MLLM.

**Q2 Scale and selection.** The evaluation now comprises 30+126 CAD parts sampled uniformly at random from the ABC dataset [1], all verified by object identifier to be disjoint from the orthoViews training and validation splits. The original 30-object test set was itself drawn at random, and the 126 additions were sampled the same way. We thus believe it is highly representative of typical CAD geometries. 

**Q3 End-to-end FEM comparison.** Energy error is reported in the main table; the remaining cost metrics, pooled over the 156 X 5 (geometries X load cases) cases, are given below. Peak von Mises stress (σ_vM^p99) is the failure-relevant design quantity, strain energy the natural finite-element error norm, and displacement the global compliance check as requested; all are measured inside Ω_crit on the fine reference mesh.

**Localisation F1**: We thank you for advocating FEM errors as metrics. Here we see as per our main text, precision matters much more for FEM accuracy than F1. F1 counts the overlap between predicted anchors and ZZ-critical cells at a fixed top-k and weights every critical cell equally, regardless of its severity. A policy can therefore raise F1 by covering many mildly-critical cells while under-resolving the few peak-stress cells that dominate solution error. This is what we observe: the blind heuristic's higher mean F1 does not carry over to downstream error, and on the decision-relevant downstream metric GReFEM is the stronger method. 

Table 2
|method|mean cells|L2 disp. err|solve time (med.)|anchor-placement cost (med.)|
|-|-|-|-|-|
|Coarse|25 062|0.026|0.2 s|—|
|**GReFEM**|95 486|0.023|0.8 s|**~7 s: render + 1 MLLM call; no solve**|
|Geometric heuristic|94 086|0.022|0.8 s|<1 s; no solve|
|Load-informed heuristic|88 149|0.022|1.0 s|<1 s; no solve|
|ZZ from coarse solve|92 700|0.022|2.5 s|1 coarse solve + ZZ|
|ZZ from fine reference solve|82 333|0.022|0.6 s|reference solve (~35–60 min)|

Maximum-displacement error is global and noisy (all values within 0.004); we report it for completeness. The decisive distinction is the final column: GReFEM's full forward cost is view rendering plus a single MLLM round-trip (median 7 s, mean 10 s over 1,903 runs), with no FEM solve; the coarse-ZZ policy requires a solve, and the fine-reference ZZ policy a reference-quality solve. The related question of how few elements are needed to reach a given error is answered by the sizing sweep below (each cell gives relative σ_vM^p99 / strain-energy error in Ω_crit); GReFEM in general beats ZZ coarse and the blind heuristic on both metrics.

Table 3 (σ_vM^p99 / energy relative error in Ω_crit)
|sizing|mean cells|GReFEM*|Geom. heur.|Load-inf.|ZZ coarse|ZZ fine ref.|AMBER|
|-|-|-|-|-|-|-|-|
|h × 1.0|91k|0.630/0.876|0.672/0.891|0.566/0.868|0.671/0.886|**0.551**/**0.851**|0.666/0.875|
|h × 1.25|56k|0.688/0.902|0.688/0.895|0.628/0.891|0.715/0.905|**0.609**/**0.868**|0.699/0.901|
|h × 1.6|33k|0.724/0.925|0.745/0.934|**0.687**/**0.915**|0.728/0.925|0.701/0.915|0.744/0.933|
|h × 2.0|20k|0.750/0.939|0.764/0.942|0.733/**0.928**|0.758/0.943|**0.727**/0.931|0.765/0.944|

*The GReFEM column pools the Gemini and GPT-5.4-mini backends per case (~1,560 cases per level); either backend alone matches within 0.005.

**Learned-AMR baseline (AMBER; column above).** We retrained **AMBER** [2], an iterative graph network that predicts a continuous sizing field, from scratch on this task: 440 expert meshes (88 ABC training parts × 5 load cases, disjoint from all evaluation objects), each expert mesh built by projecting the fine reference solve's top-percentile ZZ refinement points onto the coarse-resolution mesh, with load-case conditioning added and training run on H100 GPUs, on 4–20× more expert meshes than the original AMBER datasets (≈20–100 each), to its validation plateau. The predicted field is globally scaled to GReFEM's budget at each sweep level; it fails to mesh ~5% of cases. Zero-shot GReFEM attains lower peak-stress error at every sizing level, with strain-energy differences within 0.006. We attribute this to data diversity: AMBER may need more varied geometries, whereas GReFEM inherits an internet-scale geometric prior at no training cost; on a narrow part family with many experts a supervised sizing field would likely excel GReFEM's prior.

**Q4 Scope of "zero-shot".** We agree and now scope the term to the MLLM part of our framework, which uses no task-specific tuning. The orthoViews module is supervised on geometry labels only. The full pipeline is therefore not training-free. We apologize for the confusion and will state this explicitly.

In the light of above concerns addressed, we hope this rebuttal encourages a positive review.

[1] Koch, Sebastian, et al. "Abc: A big cad model dataset for geometric deep learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

[2] Freymuth, Niklas, et al. "Amber: Adaptive mesh generation by iterative mesh resolution prediction." Advances in Neural Information Processing Systems 38 (2026): 23797-23835.

## Response to Reviewer JSWj

We thank the reviewer for the supportive assessment and for the distinction between *where* and *how much* to refine, which we now adopt in the paper. All results below are on 156 ABC parts [1] (the original 30 plus 126 additions) sampled uniformly at random.

We evaluate the following policies at a matched element budget:

- **Coarse:** the unrefined uniform mesh (fixed floor, no refinement).
- **GReFEM (@5orthoviews):** Our framework with trained ortho-views selection and zero shot MLLM inference.
- **Geometric heuristic (blind):** the submission's load-agnostic detector of concave sharp edges and smooth concave entities.
- **Load-informed heuristic:** the same geometric candidates re-weighted by a closed-form nominal stress heuristic (see W1a), so the loading case decides which features matter; no solver.
- **ZZ from coarse solve:** refinement anchor points placed at the top-percentile ZZ error points of a single coarse FEM solve (see W1b).
- **ZZ from fine reference solve:** refinement anchor points at the top-percentile ZZ points of the fine reference solution (uniform mesh of ~5·10⁶ elements); an unattainable upper bound.
- **AMBER (learned AMR):** learned sizing-field method [2], retrained from scratch (due to inavailability of trained official checkpoints) with load-case conditioning; described after the sizing sweep.

All baselines are matched to GReFEM: each receives the *same number of refinement anchor points* as GReFEM on that case, and its sizing field is tuned so the element count matches GReFEM's as closely as the mesher allows, so the differences reflect *where* anchors are placed rather than *how many* elements are spent. AMBER is the only exception: it emits a continuous sizing field, so its element count is matched to GReFEM's via a global scaling factor.

*How the load-informed heuristic works.* It keeps the *same* geometric candidates as the blind heuristic and changes only *which* are refined: we slice the voxelised part into layers along the loading axis, compute per-layer section properties (A, centroid, I_z, J) with no solver, weight each candidate by the textbook nominal stress for its load case (formulas above), keep the top stress-weighted 30%, and density-filter to the matched budget; geometry decides *where* refinement may go, beam-theory stress *which* locations matter.

**Q1/W1 Sensitivity to the sizing field.** We conducted a four-level sizing sweep by applying a common scale factor to h_min and h_max, and re-matched every baseline to the GReFEM element budget at each level. We see that GReFEM in general beats ZZ coarse and blind geometric heuristic for relative σ_vM^p99 and strain energy in Ω_crit.

Table 1
|sizing|mean cells|GReFEM*|Geom. heur.|Load-inf.|ZZ coarse|ZZ fine ref.|AMBER|
|-|-|-|-|-|-|-|-|
|h × 1.0|91k|0.630/0.876|0.672/0.891|0.566/0.868|0.671/0.886|0.551/0.851|0.666/0.875|
|h × 1.25|56k|0.688/0.902|0.688/0.895|0.628/0.891|0.715/0.905|0.609/0.868|0.699/0.901|
|h × 1.6|33k|0.724/0.925|0.745/0.934|0.687/0.915|0.728/0.925|0.701/0.915|0.744/0.933|
|h × 2.0|20k|0.750/0.939|0.764/0.942|0.733/0.928|0.758/0.943|0.727/0.931|0.765/0.944|

*The GReFEM column pools the Gemini and GPT-5.4-mini backends per case (~1,560 cases per level); either backend alone matches within 0.005.

Accuracy is materially sensitive to the sizing field, and the sensitivity varies across policies: coarsening h by a factor of two increases the fine-reference ZZ policy's error by 0.176 but the blind heuristic's by only 0.092, indicating that accurate element placement pays off higher at sufficiently larger resolutions. The two best policies (the fine-reference ZZ policy and the load-informed heuristic) retain their ranking at every level. GReFEM's advantage over the blind heuristic is largest at the finest budget, narrows to parity at h × 1.25, and re-opens at coarser budgets.

**Q2/W3 ZZ-based upper bound.** We added the ZZ estimator applied to the fine reference solve (uniform mesh of ~5·10⁶ elements), whose anchors are the reference solution's own top-percentile ZZ points and which is therefore a self-consistent, unattainable ceiling. We quantify the gap in peak von Mises stress (σ_vM^p99, the failure-relevant quantity) and strain energy (the natural finite-element error norm), both inside Ω_crit so that the
smooth bulk does not dilute the comparison at the concentrations refinement targets.

Table 2
|method|information used|vm_p99 mean|vm_p99 med.|energy mean|energy med.|cells|
|-|-|-|-|-|-|-|
|Coarse (no refinement)|—|0.738|0.786|0.944|0.967|25 062|
|**GReFEM (Gemini)**|Load + Features + Heuristics, **no solve**|**0.628**|**0.652**|0.880|0.919|95 486|
|GReFEM (GPT-5.4-mini)|Load + Features + Heuristics|0.630|0.652|0.872|0.915|97 257|
|GReFEM (Gemini), reduced prompt|Load + Features|0.636|0.667|0.873|0.920|97 851|
|GReFEM (GPT-5.4-mini), reduced prompt|Load + Features|0.644|0.665|0.879|0.921|93 296|
|GReFEM (Gemini), minimal prompt|Load-Only|0.650|0.675|0.881|0.923|94 187|
|GReFEM (GPT-5.4-mini), minimal prompt|Load-Only|0.666|0.694|0.889|0.927|93 677|
|Geometric heuristic (blind)|geometry only|0.672|0.696|0.891|0.927|94 086|
|Load-informed heuristic|geometry + closed-form stress|0.567|0.591|0.868|0.914|88 149|
|ZZ from coarse solve|**1 FEM solve** + ZZ estimator|0.671|0.700|0.885|0.924|92 700|
|ZZ from fine reference solve|**reference solve** (unattainable)|**0.551**|**0.581**|**0.851**|**0.909**|82 333|
|AMBER (learned AMR)|440 supervised expert meshes + load|0.666|0.679|0.875|0.929|91 679|

Rows 3–7 ablate GReFEM's MLLM and prompt at the same matched budget: swapping Gemini for GPT-5.4-mini leaves accuracy unchanged, while stripping the load/stress cues degrades error monotonically; the gain comes from the textual physics cues, not a particular MLLM.

Measured as the fraction of the coarse-to-fine-reference gap closed (mean vm_p99), GReFEM closes 59% of the gap without any solve, the load-informed heuristic closes 91%, and the coarse-solve ZZ policy and blind heuristic each close approximately 35%.

**Q3 Surface prediction versus internal volumetric distribution.** We agree that a discussion on the robustness of this method for different surface and volumetric mesh distribution is necessary and would very much like to clarify. The resulting volumetric mesh is robust to a divergence between the surface-based prediction and the internal distribution, for two reasons rooted in the pipeline. First, as given in the **MV-RaySeg** section in our paper, we do not project a two-dimensional pixel onto the surface alone but sample three-dimensional points along the projected ray segment into the interior, which creates an extended volumetric buffer rather than a purely superficial anchor set. Second, the distance-based sizing field (Eqs. 6 and 7 in the main text) maps proximity to a linearly varying target element size, so a slight misalignment of the two-dimensional projection shifts the three-dimensional refinement only marginally. Thus interior refinement is therefore governed by proximity to a volumetric anchor buffer under a smoothly varying sizing field, not by an exact surface correspondence. 

The quantities we now report (peak stress, strain energy, and displacement) are downstream FEM errors rather than measures of surface overlap, so any residual mismatch between the surface prediction and the internal distribution is already penalised by the metric, and GReFEM nonetheless improves downstream error at a matched budget. We adopt the reviewer's distinction: GReFEM is a refinement-location policy, the refinement magnitude is set by a fixed rule, and Q1 provides the requested sensitivity analysis over that rule. We will state this scope explicitly given the chance.

**W2 - Learned-AMR baseline (AMBER; the AMBER entries in the tables above).** No published learned-AMR method releases weights usable in this setting, and iterative or reinforcement-learning AMR addresses a different problem (refining *with* solves in the loop). We therefore retrained **AMBER** [2], the strongest published single-shot learned-AMR method (an iterative graph network that predicts a continuous sizing field), from scratch on this task. Supervision: 440 expert adaptive meshes (88 ABC training parts × 5 load cases, disjoint from all 156 evaluation objects), each expert mesh built by projecting the fine reference solve's top-percentile ZZ refinement points onto the coarse-resolution mesh. We conditioned it on the load case and trained to its validation plateau on H100 GPUs; our 440 expert meshes are 4–20× more than the original AMBER datasets (≈20–100 meshes each). At test time its sizing field is globally scaled to GReFEM's element count at each sweep level; it fails to mesh ~5% of cases.

Zero-shot GReFEM attains lower peak-stress error at every sizing level (median paired Δ up to −0.024), with strain-energy differences ≤ 0.007 either way: we believe this difference to be because of data diversity: AMBER [2] might need larger scale of varied geometries to adapt, whereas GReFEM inherits an internet-scale geometric prior at zero training cost; though on a narrow part family with abundant experts we would definitely expect a supervised sizing field to excel a prompt-based prior like GReFEM.

We would like to thank you for the discussion and positive score and hope that this rebuttal helps encourage an even higher positive score.

### Follow-up response to Reviewer JSWj

We thank the reviewer for the continued engagement and for maintaining the positive assessment. The remaining concern is fair, and we accept it: GReFEM currently decides only *where* to refine, while the refinement magnitude: h_min, h_max and the D_min/D_max ramp of the distance-based sizing field (Eqs. 6–7) are hand-set.

We would like to state however that choosing per-anchor magnitudes requires an estimate of local error *severity*, and severity is exactly the information that is unavailable without a solve (or training) for a zero shot MLLM and is what the ZZ estimator extracts from a solution field. Every solver-free policy we compare (both heuristics included) therefore shares the same fixed sizing rule, and our study deliberately isolates the sub-problem that MLLMs can plausibly solve zero-shot today: semantic localization. We will state this scope explicitly, given the chance.

[1] Koch, Sebastian, et al. "Abc: A big cad model dataset for geometric deep learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

[2] Freymuth, Niklas, et al. "Amber: Adaptive mesh generation by iterative mesh resolution prediction." Advances in Neural Information Processing Systems 38 (2026): 23797-23835.

## Response to Reviewer Etav

We thank the reviewer for the detailed critique. We accept the observations regarding the terms "zero-shot" and "physics-guided" and have added the downstream evidence and additional baselines that were requested.

**W1 Additional baselines.** We now compare seven policies at a matched element budget:

- **Coarse:** the unrefined uniform mesh (fixed floor, no refinement).
- **GReFEM (@5 orthoviews):** Our framework with trained ortho-views selection and zero shot MLLM inference.
- **Geometric heuristic (blind):** the submission's load-agnostic detector of concave sharp edges and smooth concave entities.
- **Load-informed heuristic:** the same geometric candidates re-weighted by a closed-form nominal stress heuristic (see W1a), so the loading case decides which features matter; no solver.
- **ZZ from coarse solve:** refinement anchor points placed at the top-percentile ZZ error points of a single coarse FEM solve (see W1b).
- **ZZ from fine reference solve:** refinement anchor points at the top-percentile ZZ points of the fine reference solution (uniform mesh of ~5·10⁶ elements); an unattainable upper bound.
- **AMBER (learned AMR):** learned sizing-field method [2], retrained (due to lack of trained public ckpts) from scratch with load-case conditioning; described after the sizing sweep.

All baselines are matched to GReFEM: each receives the *same number of refinement anchor points* as GReFEM on that case, and its sizing field is tuned so the element count matches GReFEM's as closely as the mesher allows, so the differences reflect *where* anchors are placed rather than *how many* elements are spent. The learned-AMR baseline (AMBER, the AMBER rows and columns in the tables under Q1–Q2) is the only exception: it emits a continuous sizing field rather than discrete anchors, so its element count is matched to GReFEM's via a global scaling factor.

*How the load-informed heuristic works.* It keeps the *same* geometric
candidates as the blind heuristic and changes only *which* are refined: we slice the voxelised part into layers along the loading axis, compute per-layer section properties (A, centroid, I_z, J) with no solver, weight each candidate by the textbook nominal stress for its load case (formulas above), keep the top stress-weighted 30%, and density-filter to the matched budget: geometry decides *where* refinement may go, loading case dictates *which* locations matter.

Table 1
|method|information used|vm_p99 mean|vm_p99 med.|energy mean|energy med.|cells|
|-|-|-|-|-|-|-|
|Coarse (no refinement)|—|0.738|0.786|0.944|0.967|25 062|
|**GReFEM (Gemini)**|Load + Features + Heuristics, **no solve**|**0.628**|**0.652**|**0.880**|**0.919**|95 486|
|GReFEM (GPT-5.4-mini)|Load + Features + Heuristics|0.630|0.652|0.872|0.915|97 257|
|GReFEM (Gemini), reduced prompt|Load + Features|0.636|0.667|0.873|0.920|97 851|
|GReFEM (GPT-5.4-mini), reduced prompt|Load + Features|0.644|0.665|0.879|0.921|93 296|
|GReFEM (Gemini), minimal prompt|Load-Only|0.650|0.675|0.881|0.923|94 187|
|GReFEM (GPT-5.4-mini), minimal prompt|Load-Only|0.666|0.694|0.889|0.927|93 677|
|Geometric heuristic (blind)|geometry only|0.672|0.696|0.891|0.927|94 086|
|Load-informed heuristic|geometry + closed-form stress|0.567|0.591|0.868|0.914|88 149|
|ZZ from coarse solve|**1 FEM solve** + ZZ estimator|0.671|0.700|0.885|0.924|92 700|
|ZZ from fine reference solve|**reference solve** (unattainable)|**0.551**|**0.581**|**0.851**|**0.909**|82 333|
|AMBER (learned AMR)|440 supervised expert meshes + load|0.666|0.679|0.875|0.929|91 679|

Rows 3–7 ablate GReFEM's MLLM and prompt at the same matched budget: swapping Gemini for GPT-5.4-mini leaves accuracy unchanged, while stripping the load/stress cues degrades error monotonically; the gain comes from the textual physics cues, not a particular MLLM.

**Q2 - Whether refinement improves downstream quality.** We thank the reviewer for this question. We measure it on the quantities that matter for structural design and FEM convergence: peak von Mises stress (σ_vM^p99, which governs yield and fatigue) and strain energy (the natural finite-element error norm), all inside Ω_crit computed on the reference solution so that the smooth bulk does not mask the differences at the concentrations. At a matched budget, GReFEM outperforms the blind heuristic and the coarse-solve ZZ policy on downstream peak-stress error for varied mesh sizing. The related question of reaching a given error with fewer elements is answered directly by the sizing sweep (each cell gives the relative σ_vM^p99 / strain-energy error in Ω_crit). 

Table 2 (σ_vM^p99 / energy relative error in Ω_crit)
|sizing|mean cells|GReFEM*|Geom. heur.|Load-inf.|ZZ coarse|ZZ fine ref.|AMBER|
|-|-|-|-|-|-|-|-|
|h × 1.0|91k|0.630/0.876|0.672/0.891|0.566/0.868|0.671/0.886|0.551/0.851|0.666/0.875|
|h × 1.25|56k|0.688/0.902|0.688/0.895|0.628/0.891|0.715/0.905|0.609/0.868|0.699/0.901|
|h × 1.6|33k|0.724/0.925|0.745/0.934|0.687/0.915|0.728/0.925|0.701/0.915|0.744/0.933|
|h × 2.0|20k|0.750/0.939|0.764/0.942|0.733/0.928|0.758/0.943|0.727/0.931|0.765/0.944|

*The GReFEM column pools the Gemini and GPT-5.4-mini backends per case (~1,560 cases per level); either backend alone matches within 0.005.

**Learned-AMR baseline (AMBER; the AMBER entries above).** We retrained **AMBER** [2], an iterative graph network predicting a continuous sizing field, from scratch on this task: 440 expert adaptive meshes (88 ABC training parts × 5 load cases, disjoint from all 156 evaluation objects), each expert mesh built by projecting the fine reference solve's top-percentile ZZ refinement points onto the coarse-resolution mesh. We adapted the network by conditioning it on the load case and trained it to its validation plateau on H100 GPUs; our 440 expert meshes are 4–20× more than the original AMBER datasets (≈20–100 meshes each). Its sizing field is globally scaled to GReFEM's element budget at every sweep level; it fails to mesh ~5% of cases. Zero-shot GReFEM attains lower peak-stress error than AMBER at every sizing level (median paired Δ up to −0.024, largest at h×1.0 and h×1.6), with strain-energy differences within 0.006 in either direction:we believe this difference to be because of data diversity: AMBER [2] might need larger scale of varied geometries to train and adapt, whereas GReFEM inherits an internet-scale geometric prior at zero training cost; though on a narrow part family with abundant experts we would definitely expect a supervised sizing field to excel a prompt-based prior like GReFEM.

**Q3 - Sample size and the F1 comparison.** The evaluation now spans 156 objects and 780 different geometry+loading cases. The original 30-object test set in the submission was drawn uniformly at random from the ABC dataset [1] with no bias, and the 126 additional parts were sampled the same way. 

**Localization F1** (point raised in the review and meta-review): F1 counts the overlap between predicted anchors and ZZ-critical cells at a fixed top-k and weights every critical cell equally, regardless of severity, so a method can score a high F1 by covering many mildly-critical cells while under-resolving the few peak-stress cells that dominate the solution error. Here we see as per our main text, precision matters much more for FEM accuracy than F1. On the downstream FEM metric GReFEM is the stronger method. 

**W4 and W5 - Instruction-following and the terms "zero-shot" and "physics-guided".** We accept these points and understand the confusion. We scope "zero-shot" to the MLLM refinement policy alone, which uses no FEM solves, no simulation data, and no task-specific tuning; the orthoViews module is a supervised view selector trained on STEP files only, so the full pipeline is not training-free, as we now state. Regarding "physics-guided", the prompts encode load-direction and stress-concentration cues, and what our evaluation establishes is the narrower, empirical claim that the MLLM converts these textual cues, together with the rendered geometry, into load-dependent spatial selections that measurably reduce downstream FEM error; it does not perform mechanics reasoning from first principles. We will retitle these claims accordingly if given the chance.

In light of the addressed concerns, we hope these points a encourage a positive reassessment of our work.

### Follow-up response to Reviewer Etav

We thank the reviewer for engaging with the rebuttal and for raising the score. We address the three remaining points directly.

**1. Significance of the MLLM given the load-informed heuristic's stronger numbers.** We report the load-informed heuristic's win openly, but we would again like to point out that it is a *hand-derived mechanical model*. For each family of loading cases an expert must identify the governing nominal-stress formula that embeds the beam-theory assumptions for simple load cases (slender members, axis-aligned loads, known clamp/load faces) our benchmark happens to satisfy. Outside that regime (multi-axial or distributed loads, contact, thermal stress, arbitrary boundary conditions, load cases with no textbook formula) the weighting rule is *undefined* until an expert derives a new one. GReFEM requires none of this: adapting to a new load family is a prompt edit. We see the two as complementary points on an automation-vs-expertise trade-off, and we will state this framing explicitly: the significance of the MLLM is not that it beats every expert-designed prior on the regime that prior was designed for, but that it provides a competitive, general, derivation-free policy where such priors are unavailable.

**2. Updated localization metrics on the 156-object set.** We agree these were owed and provide them now, computed on all 780 cases. We transport the same metrics to the element level of the final meshes, where "predictions" are the deeply refined elements (size ≤ 1.25 h_min): element precision is the fraction of these within r of a true hotspot, element recall the fraction of true hotspots covered by one.

|method|P|R|F1|elem. size at hotspots (med.)|vm_p99 mean|
|-|-|-|-|-|-|
|GReFEM|0.318|0.907|0.432|0.45|0.628|
|Geometric heuristic (blind)|0.313|0.926|0.427|0.49|0.672|
|Load-informed heuristic|0.349|0.832|0.431|0.42|0.567|

While for the expanded set our localization metrics originally claimed in the paper do dim, we still show that GReFEM consistently places elements with higher precision than the blind heuristic. 

**3. Revision of the evaluation section.** We agree, and are committed to it: all experiments above are already run and we will reorganize with the new object set and downstream FEM error becomes the primary criterion, and the localization analysis is restated through the element-level metrics above. 

