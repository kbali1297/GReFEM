
COMBINED TEST SET: 156 objects, 780 units  {'dense (a-priori)': 82, 'original paper set': 18, 'winners (post-hoc)': 56}

**bending (n=156 objects)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.732 | 0.777 | 0.939 | 0.965 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.625 | 0.637 | 0.869 | **0.906** | 93,088 |
| Geometric heuristic (blind) | 0.699 | 0.726 | 0.905 | 0.936 | 91,283 |
| Load-informed heuristic | **0.570** | **0.614** | **0.868** | 0.922 | 85,883 |
| ZZ from coarse solve (1 solve) | 0.684 | 0.697 | 0.895 | 0.916 | 90,413 |
| ZZ oracle (fine ref, ceiling) | 0.574 | 0.619 | 0.875 | 0.927 | 83,850 |

**compression (n=156 objects)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.739 | 0.791 | 0.948 | 0.969 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.661 | 0.693 | 0.900 | 0.932 | 92,530 |
| Geometric heuristic (blind) | 0.696 | 0.731 | 0.910 | 0.938 | 91,577 |
| Load-informed heuristic | 0.632 | 0.665 | 0.887 | 0.929 | 87,224 |
| ZZ from coarse solve (1 solve) | 0.706 | 0.737 | 0.916 | 0.943 | 89,550 |
| ZZ oracle (fine ref, ceiling) | **0.606** | **0.630** | **0.858** | **0.914** | 83,394 |

**torsion (n=156 objects)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.728 | 0.767 | 0.942 | 0.971 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.625 | 0.647 | 0.868 | 0.913 | 97,002 |
| Geometric heuristic (blind) | 0.659 | 0.673 | 0.878 | 0.920 | 95,988 |
| Load-informed heuristic | **0.512** | **0.505** | **0.849** | **0.887** | 88,769 |
| ZZ from coarse solve (1 solve) | 0.650 | 0.693 | 0.877 | 0.921 | 94,485 |
| ZZ oracle (fine ref, ceiling) | 0.570 | 0.606 | 0.857 | 0.909 | 89,265 |

**bending_compression (n=156 objects)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.743 | 0.800 | 0.941 | 0.965 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.602 | 0.624 | 0.850 | 0.904 | 97,596 |
| Geometric heuristic (blind) | 0.647 | 0.683 | 0.860 | 0.906 | 95,991 |
| Load-informed heuristic | 0.555 | 0.560 | 0.832 | 0.892 | 89,782 |
| ZZ from coarse solve (1 solve) | 0.631 | 0.672 | 0.838 | 0.901 | 94,900 |
| ZZ oracle (fine ref, ceiling) | **0.477** | **0.493** | **0.795** | **0.843** | 78,654 |

**torsion_compression (n=156 objects)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.743 | 0.789 | 0.943 | 0.962 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.619 | 0.651 | 0.897 | 0.930 | 97,212 |
| Geometric heuristic (blind) | 0.652 | 0.687 | 0.888 | 0.941 | 95,591 |
| Load-informed heuristic | 0.563 | 0.591 | 0.888 | 0.933 | 89,089 |
| ZZ from coarse solve (1 solve) | 0.673 | 0.707 | 0.878 | 0.936 | 94,153 |
| ZZ oracle (fine ref, ceiling) | **0.523** | **0.560** | **0.855** | **0.925** | 76,501 |

**POOLED (n=156 objects, 780 units)**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.738 | 0.786 | 0.944 | 0.967 | 25,062 |
| GReFEM (zero-shot MLLM) | 0.628 | 0.652 | 0.880 | 0.919 | 95,486 |
| Geometric heuristic (blind) | 0.672 | 0.696 | 0.891 | 0.927 | 94,086 |
| Load-informed heuristic | 0.567 | 0.591 | 0.868 | 0.914 | 88,149 |
| ZZ from coarse solve (1 solve) | 0.671 | 0.700 | 0.885 | 0.924 | 92,700 |
| ZZ oracle (fine ref, ceiling) | **0.551** | **0.581** | **0.851** | **0.909** | 82,333 |

**Paired tests, pooled (780 units)** (negative = GReFEM better)

| baseline | metric | mean diff | median diff | win | loss | p |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | vm_p99 | -0.1030 | -0.0830 | 65.4% | 27.7% | 8e-33 |
| Coarse (no refinement) | energy | -0.0579 | -0.0262 | 61.7% | 19.0% | 7.7e-38 |
| Geometric heuristic (blind) | vm_p99 | -0.0411 | -0.0254 | 54.2% | 36.5% | 1.9e-10 |
| Geometric heuristic (blind) | energy | -0.0134 | -0.0031 | 44.1% | 36.3% | 0.0074 |
| Load-informed heuristic | vm_p99 | +0.0571 | +0.0371 | 38.6% | 56.2% | 1.6e-09 |
| Load-informed heuristic | energy | +0.0101 | +0.0011 | 39.5% | 45.0% | 0.099 |
| ZZ from coarse solve (1 solve) | vm_p99 | -0.0398 | -0.0377 | 57.6% | 36.7% | 2.3e-10 |
| ZZ from coarse solve (1 solve) | energy | -0.0072 | -0.0036 | 44.5% | 37.3% | 0.034 |
| ZZ oracle (fine ref, ceiling) | vm_p99 | +0.0721 | +0.0370 | 39.2% | 56.2% | 5.9e-12 |
| ZZ oracle (fine ref, ceiling) | energy | +0.0256 | +0.0059 | 36.7% | 47.4% | 0.00066 |


## Per-stratum breakdown


**dense (a-priori) — 82 objects, 410 units**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.689 | 0.710 | 0.926 | 0.951 | 39,251 |
| GReFEM (zero-shot MLLM) | 0.640 | 0.657 | 0.887 | 0.927 | 116,290 |
| Geometric heuristic (blind) | 0.661 | 0.677 | 0.890 | 0.924 | 114,626 |
| Load-informed heuristic | **0.552** | **0.572** | 0.861 | **0.904** | 108,359 |
| ZZ from coarse solve (1 solve) | 0.652 | 0.691 | 0.884 | 0.923 | 112,910 |
| ZZ oracle (fine ref, ceiling) | 0.559 | 0.599 | **0.858** | 0.914 | 101,832 |

**dense (a-priori): paired vs GReFEM** (negative = GReFEM better)

| baseline | metric | mean diff | median diff | win | loss | p |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | vm_p99 | -0.0442 | -0.0231 | 53.2% | 36.8% | 3.7e-05 |
| Coarse (no refinement) | energy | -0.0358 | -0.0096 | 49.0% | 25.9% | 8.3e-09 |
| Geometric heuristic (blind) | vm_p99 | -0.0223 | -0.0131 | 50.2% | 40.0% | 0.0069 |
| Geometric heuristic (blind) | energy | -0.0073 | -0.0015 | 42.2% | 37.8% | 0.27 |
| Load-informed heuristic | vm_p99 | +0.0801 | +0.0606 | 33.7% | 60.0% | 1e-10 |
| Load-informed heuristic | energy | +0.0214 | +0.0075 | 32.9% | 48.5% | 0.00095 |
| ZZ from coarse solve (1 solve) | vm_p99 | -0.0119 | -0.0062 | 49.5% | 42.4% | 0.11 |
| ZZ from coarse solve (1 solve) | energy | -0.0003 | -0.0000 | 40.0% | 40.2% | 0.97 |
| ZZ oracle (fine ref, ceiling) | vm_p99 | +0.0742 | +0.0385 | 37.3% | 56.8% | 6.4e-08 |
| ZZ oracle (fine ref, ceiling) | energy | +0.0247 | +0.0082 | 32.9% | 48.8% | 0.0011 |

**winners (post-hoc) — 56 objects, 280 units**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.808 | 0.853 | 0.966 | 0.984 | 5,651 |
| GReFEM (zero-shot MLLM) | 0.614 | 0.635 | 0.868 | **0.908** | 67,717 |
| Geometric heuristic (blind) | 0.708 | 0.736 | 0.906 | 0.941 | 66,559 |
| Load-informed heuristic | 0.591 | 0.625 | 0.873 | 0.934 | 62,566 |
| ZZ from coarse solve (1 solve) | 0.695 | 0.734 | 0.885 | 0.927 | 65,989 |
| ZZ oracle (fine ref, ceiling) | **0.546** | **0.578** | **0.837** | 0.911 | 56,161 |

**winners (post-hoc): paired vs GReFEM** (negative = GReFEM better)

| baseline | metric | mean diff | median diff | win | loss | p |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | vm_p99 | -0.1866 | -0.1786 | 82.1% | 15.0% | 1.4e-27 |
| Coarse (no refinement) | energy | -0.0944 | -0.0578 | 78.9% | 9.3% | 1.5e-30 |
| Geometric heuristic (blind) | vm_p99 | -0.0801 | -0.0590 | 62.5% | 28.2% | 2.4e-13 |
| Geometric heuristic (blind) | energy | -0.0352 | -0.0074 | 48.2% | 30.7% | 4.9e-05 |
| Load-informed heuristic | vm_p99 | +0.0243 | +0.0155 | 45.0% | 51.4% | 0.14 |
| Load-informed heuristic | energy | -0.0036 | -0.0071 | 47.9% | 38.2% | 0.2 |
| ZZ from coarse solve (1 solve) | vm_p99 | -0.0793 | -0.0810 | 68.6% | 27.9% | 5.2e-13 |
| ZZ from coarse solve (1 solve) | energy | -0.0181 | -0.0084 | 48.6% | 33.9% | 0.0037 |
| ZZ oracle (fine ref, ceiling) | vm_p99 | +0.0646 | +0.0310 | 41.8% | 55.0% | 0.0005 |
| ZZ oracle (fine ref, ceiling) | energy | +0.0262 | -0.0011 | 42.1% | 44.3% | 0.3 |

**original paper set — 18 objects, 90 units**

| method | vm_p99 mean | vm_p99 med. | energy mean | energy med. | cells |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.737 | 0.801 | 0.934 | 0.972 | 20,810 |
| GReFEM (zero-shot MLLM) | 0.612 | 0.644 | 0.878 | 0.907 | 87,099 |
| Geometric heuristic (blind) | 0.609 | 0.639 | **0.841** | 0.905 | 86,153 |
| Load-informed heuristic | 0.560 | 0.580 | 0.874 | 0.916 | 75,676 |
| ZZ from coarse solve (1 solve) | 0.668 | 0.681 | 0.886 | 0.919 | 83,734 |
| ZZ oracle (fine ref, ceiling) | **0.527** | **0.557** | 0.853 | **0.899** | 74,925 |

**original paper set: paired vs GReFEM** (negative = GReFEM better)

| baseline | metric | mean diff | median diff | win | loss | p |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | vm_p99 | -0.1151 | -0.1022 | 68.9% | 25.6% | 3.3e-06 |
| Coarse (no refinement) | energy | -0.0388 | -0.0328 | 65.6% | 17.8% | 9.1e-05 |
| Geometric heuristic (blind) | vm_p99 | -0.0019 | +0.0007 | 46.7% | 46.7% | 0.95 |
| Geometric heuristic (blind) | energy | +0.0312 | +0.0073 | 40.0% | 46.7% | 0.13 |
| Load-informed heuristic | vm_p99 | +0.0444 | +0.0280 | 41.1% | 53.3% | 0.19 |
| Load-informed heuristic | energy | +0.0014 | +0.0067 | 43.3% | 50.0% | 0.89 |
| ZZ from coarse solve (1 solve) | vm_p99 | -0.0443 | -0.0442 | 60.0% | 37.8% | 0.043 |
| ZZ from coarse solve (1 solve) | energy | -0.0056 | -0.0137 | 52.2% | 34.4% | 0.27 |
| ZZ oracle (fine ref, ceiling) | vm_p99 | +0.0872 | +0.0392 | 40.0% | 56.7% | 0.0095 |
| ZZ oracle (fine ref, ceiling) | energy | +0.0270 | +0.0124 | 36.7% | 51.1% | 0.13 |
