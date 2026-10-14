# Camera-ready tables (156 objects)

Strata objects: {'A': 81, 'B': 56, 'C': 19}


## Main table @h1.0 — ALL 156 objects

| policy | n | vm tmean | vm med | en tmean | en med | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | 780 | 0.739 | 0.786 | 0.944 | 0.967 | 24,660 |
| GReFEM (Gemini) L+F+H | 780 | 0.626 | 0.649 | 0.879 | 0.919 | 95,052 |
| GReFEM (GPT-5.4-mini) L+F+H | 779 | 0.630 | 0.651 | 0.873 | 0.915 | 96,481 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 780 | 0.643 | 0.652 | 0.880 | 0.925 | 87,216 |
| GReFEM (Qwen3-VL-235B) L+F+H | 780 | 0.651 | 0.672 | 0.886 | 0.924 | 82,774 |
| GReFEM (Gemini) L+F | 774 | 0.634 | 0.665 | 0.872 | 0.920 | 97,095 |
| GReFEM (GPT-5.4-mini) L+F | 776 | 0.641 | 0.663 | 0.878 | 0.920 | 92,620 |
| GReFEM (Claude-Haiku-4.5) L+F | 780 | 0.634 | 0.654 | 0.877 | 0.920 | 95,691 |
| GReFEM (Qwen3-VL-235B) L+F | 780 | 0.646 | 0.675 | 0.889 | 0.930 | 84,970 |
| GReFEM (Gemini) L | 775 | 0.622 | 0.659 | 0.865 | 0.915 | 106,007 |
| GReFEM (GPT-5.4-mini) L | 774 | 0.635 | 0.667 | 0.860 | 0.907 | 119,378 |
| GReFEM (Claude-Haiku-4.5) L | 780 | 0.637 | 0.660 | 0.871 | 0.914 | 112,773 |
| GReFEM (Qwen3-VL-235B) L | 780 | 0.639 | 0.656 | 0.878 | 0.915 | 91,521 |
| Geometric heuristic (blind) | 780 | 0.672 | 0.696 | 0.892 | 0.928 | 93,683 |
| Load-informed heuristic | 780 | 0.567 | 0.591 | 0.869 | 0.914 | 87,746 |
| ZZ from coarse solve | 780 | 0.672 | 0.700 | 0.887 | 0.925 | 92,249 |
| AMBER, load-conditioned | 735 | 0.625 | 0.660 | 0.869 | 0.915 | 84,856 |
| AMBER, geometry-only | 736 | 0.609 | 0.633 | 0.849 | 0.895 | 94,713 |
| ZZ from fine reference solve | 780 | 0.549 | 0.578 | 0.850 | 0.908 | 81,929 |
| GReFEM (4-backend pool) L+F+H | 780 | 0.637 | 0.655 | 0.879 | 0.921 | 90,379 |
| GReFEM (4-backend pool) L+F | 780 | 0.639 | 0.665 | 0.879 | 0.923 | 92,585 |
| GReFEM (4-backend pool) L | 780 | 0.633 | 0.662 | 0.869 | 0.913 | 107,399 |

## Main table @h1.0 — A+C (outcome-independent)

| policy | n | vm tmean | vm med | en tmean | en med | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | 500 | 0.701 | 0.731 | 0.931 | 0.954 | 35,306 |
| GReFEM (Gemini) L+F+H | 500 | 0.632 | 0.654 | 0.885 | 0.922 | 110,360 |
| GReFEM (GPT-5.4-mini) L+F+H | 499 | 0.622 | 0.650 | 0.872 | 0.911 | 109,509 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 500 | 0.641 | 0.650 | 0.884 | 0.921 | 102,242 |
| GReFEM (Qwen3-VL-235B) L+F+H | 500 | 0.654 | 0.672 | 0.894 | 0.929 | 93,578 |
| GReFEM (Gemini) L+F | 494 | 0.631 | 0.667 | 0.877 | 0.925 | 111,094 |
| GReFEM (GPT-5.4-mini) L+F | 496 | 0.636 | 0.655 | 0.879 | 0.920 | 105,823 |
| GReFEM (Claude-Haiku-4.5) L+F | 500 | 0.623 | 0.639 | 0.874 | 0.912 | 110,486 |
| GReFEM (Qwen3-VL-235B) L+F | 500 | 0.645 | 0.671 | 0.894 | 0.931 | 97,857 |
| GReFEM (Gemini) L | 495 | 0.621 | 0.652 | 0.872 | 0.915 | 118,177 |
| GReFEM (GPT-5.4-mini) L | 494 | 0.638 | 0.665 | 0.866 | 0.905 | 133,332 |
| GReFEM (Claude-Haiku-4.5) L | 500 | 0.625 | 0.652 | 0.873 | 0.914 | 125,579 |
| GReFEM (Qwen3-VL-235B) L | 500 | 0.635 | 0.645 | 0.884 | 0.915 | 103,114 |
| Geometric heuristic (blind) | 500 | 0.652 | 0.672 | 0.883 | 0.920 | 108,873 |
| Load-informed heuristic | 500 | 0.554 | 0.575 | 0.866 | 0.907 | 101,848 |
| ZZ from coarse solve | 500 | 0.659 | 0.690 | 0.887 | 0.923 | 106,954 |
| AMBER, load-conditioned | 479 | 0.606 | 0.639 | 0.864 | 0.905 | 94,786 |
| AMBER, geometry-only | 478 | 0.596 | 0.616 | 0.846 | 0.889 | 107,725 |
| ZZ from fine reference solve | 500 | 0.551 | 0.580 | 0.858 | 0.906 | 96,359 |
| GReFEM (4-backend pool) L+F+H | 500 | 0.637 | 0.654 | 0.884 | 0.921 | 103,919 |
| GReFEM (4-backend pool) L+F | 500 | 0.634 | 0.656 | 0.881 | 0.923 | 106,302 |
| GReFEM (4-backend pool) L | 500 | 0.630 | 0.653 | 0.875 | 0.912 | 120,015 |

## Main table @h1.0 — A (feature-dense)

| policy | n | vm tmean | vm med | en tmean | en med | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | 405 | 0.692 | 0.713 | 0.928 | 0.951 | 38,868 |
| GReFEM (Gemini) L+F+H | 405 | 0.636 | 0.656 | 0.886 | 0.926 | 116,711 |
| GReFEM (GPT-5.4-mini) L+F+H | 404 | 0.627 | 0.652 | 0.876 | 0.915 | 115,690 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 405 | 0.644 | 0.650 | 0.885 | 0.922 | 107,586 |
| GReFEM (Qwen3-VL-235B) L+F+H | 405 | 0.661 | 0.676 | 0.897 | 0.933 | 95,333 |
| GReFEM (Gemini) L+F | 399 | 0.633 | 0.667 | 0.878 | 0.927 | 117,171 |
| GReFEM (GPT-5.4-mini) L+F | 401 | 0.644 | 0.659 | 0.884 | 0.924 | 110,847 |
| GReFEM (Claude-Haiku-4.5) L+F | 405 | 0.618 | 0.638 | 0.870 | 0.911 | 116,206 |
| GReFEM (Qwen3-VL-235B) L+F | 405 | 0.654 | 0.690 | 0.901 | 0.940 | 101,770 |
| GReFEM (Gemini) L | 400 | 0.625 | 0.655 | 0.876 | 0.915 | 123,093 |
| GReFEM (GPT-5.4-mini) L | 399 | 0.633 | 0.663 | 0.864 | 0.907 | 138,619 |
| GReFEM (Claude-Haiku-4.5) L | 405 | 0.620 | 0.651 | 0.867 | 0.910 | 131,979 |
| GReFEM (Qwen3-VL-235B) L | 405 | 0.635 | 0.649 | 0.884 | 0.915 | 106,496 |
| Geometric heuristic (blind) | 405 | 0.660 | 0.676 | 0.891 | 0.923 | 115,100 |
| Load-informed heuristic | 405 | 0.554 | 0.574 | 0.864 | 0.904 | 108,739 |
| ZZ from coarse solve | 405 | 0.654 | 0.691 | 0.886 | 0.923 | 113,348 |
| AMBER, load-conditioned | 390 | 0.594 | 0.626 | 0.856 | 0.902 | 101,191 |
| AMBER, geometry-only | 394 | 0.597 | 0.618 | 0.847 | 0.892 | 110,632 |
| ZZ from fine reference solve | 405 | 0.559 | 0.597 | 0.860 | 0.914 | 102,126 |
| GReFEM (4-backend pool) L+F+H | 405 | 0.642 | 0.657 | 0.886 | 0.923 | 108,826 |
| GReFEM (4-backend pool) L+F | 405 | 0.637 | 0.661 | 0.884 | 0.925 | 111,479 |
| GReFEM (4-backend pool) L | 405 | 0.628 | 0.653 | 0.874 | 0.912 | 125,002 |

## Main table @h1.0 — B (post-hoc GReFEM-favourable)

| policy | n | vm tmean | vm med | en tmean | en med | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | 280 | 0.808 | 0.853 | 0.966 | 0.984 | 5,651 |
| GReFEM (Gemini) L+F+H | 280 | 0.614 | 0.635 | 0.868 | 0.908 | 67,717 |
| GReFEM (GPT-5.4-mini) L+F+H | 280 | 0.645 | 0.663 | 0.875 | 0.925 | 73,263 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 280 | 0.646 | 0.663 | 0.871 | 0.928 | 60,382 |
| GReFEM (Qwen3-VL-235B) L+F+H | 280 | 0.645 | 0.672 | 0.872 | 0.916 | 63,481 |
| GReFEM (Gemini) L+F | 280 | 0.638 | 0.662 | 0.862 | 0.907 | 72,398 |
| GReFEM (GPT-5.4-mini) L+F | 280 | 0.650 | 0.678 | 0.875 | 0.921 | 69,232 |
| GReFEM (Claude-Haiku-4.5) L+F | 280 | 0.655 | 0.691 | 0.884 | 0.931 | 69,271 |
| GReFEM (Qwen3-VL-235B) L+F | 280 | 0.647 | 0.681 | 0.880 | 0.928 | 61,958 |
| GReFEM (Gemini) L | 280 | 0.624 | 0.671 | 0.851 | 0.915 | 84,492 |
| GReFEM (GPT-5.4-mini) L | 280 | 0.631 | 0.672 | 0.846 | 0.911 | 94,759 |
| GReFEM (Claude-Haiku-4.5) L | 280 | 0.657 | 0.679 | 0.867 | 0.915 | 89,906 |
| GReFEM (Qwen3-VL-235B) L | 280 | 0.645 | 0.684 | 0.865 | 0.918 | 70,821 |
| Geometric heuristic (blind) | 280 | 0.708 | 0.736 | 0.906 | 0.941 | 66,559 |
| Load-informed heuristic | 280 | 0.591 | 0.625 | 0.873 | 0.934 | 62,566 |
| ZZ from coarse solve | 280 | 0.695 | 0.734 | 0.885 | 0.927 | 65,989 |
| AMBER, load-conditioned | 256 | 0.658 | 0.687 | 0.878 | 0.927 | 66,278 |
| AMBER, geometry-only | 258 | 0.633 | 0.672 | 0.854 | 0.919 | 70,606 |
| ZZ from fine reference solve | 280 | 0.546 | 0.578 | 0.837 | 0.911 | 56,161 |
| GReFEM (4-backend pool) L+F+H | 280 | 0.637 | 0.659 | 0.871 | 0.920 | 66,211 |
| GReFEM (4-backend pool) L+F | 280 | 0.647 | 0.678 | 0.876 | 0.923 | 68,214 |
| GReFEM (4-backend pool) L | 280 | 0.639 | 0.675 | 0.857 | 0.914 | 84,995 |

## Main table @h1.0 — C (original set)

| policy | n | vm tmean | vm med | en tmean | en med | cells |
|---|---|---|---|---|---|---|
| Coarse (no refinement) | 95 | 0.731 | 0.779 | 0.932 | 0.970 | 20,120 |
| GReFEM (Gemini) L+F+H | 95 | 0.612 | 0.641 | 0.881 | 0.907 | 83,284 |
| GReFEM (GPT-5.4-mini) L+F+H | 95 | 0.599 | 0.631 | 0.852 | 0.902 | 83,226 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 95 | 0.621 | 0.650 | 0.875 | 0.916 | 79,462 |
| GReFEM (Qwen3-VL-235B) L+F+H | 95 | 0.614 | 0.648 | 0.874 | 0.917 | 86,095 |
| GReFEM (Gemini) L+F | 95 | 0.623 | 0.668 | 0.865 | 0.920 | 85,572 |
| GReFEM (GPT-5.4-mini) L+F | 95 | 0.605 | 0.626 | 0.851 | 0.899 | 84,613 |
| GReFEM (Claude-Haiku-4.5) L+F | 95 | 0.639 | 0.649 | 0.882 | 0.926 | 86,102 |
| GReFEM (Qwen3-VL-235B) L+F | 95 | 0.609 | 0.648 | 0.859 | 0.907 | 81,173 |
| GReFEM (Gemini) L | 95 | 0.606 | 0.641 | 0.856 | 0.921 | 97,479 |
| GReFEM (GPT-5.4-mini) L | 95 | 0.654 | 0.670 | 0.874 | 0.899 | 111,126 |
| GReFEM (Claude-Haiku-4.5) L | 95 | 0.645 | 0.659 | 0.892 | 0.927 | 98,292 |
| GReFEM (Qwen3-VL-235B) L | 95 | 0.636 | 0.633 | 0.876 | 0.916 | 88,694 |
| Geometric heuristic (blind) | 95 | 0.615 | 0.639 | 0.847 | 0.911 | 82,326 |
| Load-informed heuristic | 95 | 0.553 | 0.580 | 0.873 | 0.915 | 72,469 |
| ZZ from coarse solve | 95 | 0.675 | 0.683 | 0.889 | 0.921 | 79,696 |
| AMBER, load-conditioned | 89 | 0.657 | 0.690 | 0.895 | 0.911 | 66,717 |
| AMBER, geometry-only | 84 | 0.593 | 0.605 | 0.840 | 0.873 | 94,091 |
| ZZ from fine reference solve | 95 | 0.519 | 0.550 | 0.845 | 0.896 | 71,773 |
| GReFEM (4-backend pool) L+F+H | 95 | 0.613 | 0.645 | 0.874 | 0.910 | 83,017 |
| GReFEM (4-backend pool) L+F | 95 | 0.620 | 0.648 | 0.868 | 0.910 | 84,365 |
| GReFEM (4-backend pool) L | 95 | 0.638 | 0.655 | 0.879 | 0.915 | 98,898 |

## Fraction of coarse->fine-ref gap closed (vm tmean)

- all: GReFEM Gemini L+F+H: 0.626 -> 59.9%
- all: Geom heur: 0.672 -> 35.3%
- all: Load-inf: 0.567 -> 90.7%
- all: ZZ coarse: 0.672 -> 35.3%
- all: AMBER load: 0.625 -> 60.5%
- all: AMBER geo: 0.609 -> 68.7%
- A+C: GReFEM Gemini L+F+H: 0.632 -> 45.7%
- A+C: Geom heur: 0.652 -> 32.3%
- A+C: Load-inf: 0.554 -> 98.2%
- A+C: ZZ coarse: 0.659 -> 28.1%
- A+C: AMBER load: 0.606 -> 63.2%
- A+C: AMBER geo: 0.596 -> 70.2%

## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — all

| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |
|---|---|---|---|---|---|---|---|
| Coarse (no refinement) | 780 | -0.0871 | 65.8%/27.3% | 1.3e-34 | -0.0262 | 61.8%/18.7% | 2.9e-39 |
| Geometric heuristic (blind) | 780 | -0.0266 | 54.6%/36.3% | 3.6e-11 | -0.0035 | 44.4%/35.9% | 0.0037 |
| Load-informed heuristic | 780 | +0.0368 | 38.7%/56.0% | 2.6e-09 | +0.0009 | 39.5%/44.9% | 0.12 |
| ZZ from coarse solve | 780 | -0.0394 | 58.1%/36.2% | 1.8e-11 | -0.0039 | 44.9%/37.2% | 0.02 |
| AMBER, load-conditioned | 735 | +0.0001 | 48.2%/47.8% | 0.72 | +0.0041 | 39.3%/46.1% | 0.037 |
| AMBER, geometry-only | 736 | +0.0249 | 42.5%/53.0% | 0.0082 | +0.0171 | 34.8%/52.7% | 8e-08 |
| ZZ from fine reference solve | 780 | +0.0370 | 39.2%/56.2% | 6.9e-12 | +0.0059 | 36.7%/47.4% | 0.00061 |

4-backend per-case mean (L+F+H) minus baseline — all

| Coarse (no refinement) | 780 | -0.0934 | 67.7%/26.5% | 2.1e-38 | -0.0430 | 68.2%/16.4% | 1.4e-52 |
| Geometric heuristic (blind) | 780 | -0.0307 | 56.7%/37.3% | 1.3e-09 | -0.0144 | 53.1%/31.7% | 1.3e-07 |
| Load-informed heuristic | 780 | +0.0414 | 38.5%/56.4% | 5.4e-13 | -0.0068 | 48.1%/39.2% | 0.47 |
| ZZ from coarse solve | 780 | -0.0406 | 58.8%/34.2% | 1.1e-10 | -0.0127 | 52.3%/32.6% | 4.2e-07 |
| AMBER, load-conditioned | 735 | +0.0154 | 44.2%/52.0% | 0.057 | -0.0006 | 44.8%/41.6% | 0.61 |
| AMBER, geometry-only | 736 | +0.0227 | 42.8%/52.6% | 0.00058 | +0.0038 | 40.1%/48.2% | 0.00016 |
| ZZ from fine reference solve | 780 | +0.0571 | 36.4%/58.8% | 3.1e-17 | +0.0000 | 42.8%/45.6% | 0.023 |

## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — A+C

| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |
|---|---|---|---|---|---|---|---|
| Coarse (no refinement) | 500 | -0.0350 | 56.6%/34.2% | 3.9e-10 | -0.0136 | 52.2%/24.0% | 2.6e-13 |
| Geometric heuristic (blind) | 500 | -0.0131 | 50.2%/40.8% | 0.0068 | -0.0007 | 42.2%/38.8% | 0.59 |
| Load-informed heuristic | 500 | +0.0580 | 35.2%/58.6% | 3.3e-10 | +0.0073 | 34.8%/48.6% | 0.0038 |
| ZZ from coarse solve | 500 | -0.0161 | 52.2%/40.8% | 0.0047 | -0.0021 | 42.8%/39.0% | 0.47 |
| AMBER, load-conditioned | 479 | +0.0203 | 45.3%/51.1% | 0.034 | +0.0079 | 37.2%/48.9% | 0.0024 |
| AMBER, geometry-only | 478 | +0.0401 | 40.0%/55.2% | 0.00029 | +0.0248 | 31.0%/56.5% | 1.5e-09 |
| ZZ from fine reference solve | 500 | +0.0385 | 37.8%/56.8% | 2.6e-09 | +0.0095 | 33.6%/49.2% | 0.0003 |

4-backend per-case mean (L+F+H) minus baseline — A+C

| Coarse (no refinement) | 500 | -0.0441 | 59.4%/33.8% | 1.8e-12 | -0.0294 | 62.0%/20.8% | 5.8e-24 |
| Geometric heuristic (blind) | 500 | -0.0142 | 50.8%/41.8% | 0.011 | -0.0111 | 50.6%/35.2% | 0.02 |
| Load-informed heuristic | 500 | +0.0492 | 35.4%/58.6% | 5.8e-12 | -0.0000 | 44.6%/43.4% | 0.38 |
| ZZ from coarse solve | 500 | -0.0221 | 53.0%/39.4% | 0.0034 | -0.0109 | 50.4%/35.0% | 0.0014 |
| AMBER, load-conditioned | 479 | +0.0213 | 42.4%/53.9% | 0.0066 | +0.0000 | 43.6%/43.0% | 0.26 |
| AMBER, geometry-only | 478 | +0.0397 | 40.4%/55.2% | 4e-05 | +0.0098 | 38.1%/50.0% | 1.4e-05 |
| ZZ from fine reference solve | 500 | +0.0596 | 37.2%/57.8% | 4.5e-11 | +0.0000 | 42.2%/45.8% | 0.071 |

## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — A

| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |
|---|---|---|---|---|---|---|---|
| Coarse (no refinement) | 405 | -0.0253 | 53.8%/36.0% | 5.2e-06 | -0.0098 | 49.6%/24.9% | 6.1e-10 |
| Geometric heuristic (blind) | 405 | -0.0154 | 50.6%/39.8% | 0.0038 | -0.0015 | 42.5%/37.5% | 0.2 |
| Load-informed heuristic | 405 | +0.0604 | 33.8%/59.8% | 3.5e-10 | +0.0071 | 33.1%/48.1% | 0.0021 |
| ZZ from coarse solve | 405 | -0.0074 | 49.9%/42.0% | 0.063 | -0.0000 | 40.5%/40.0% | 0.85 |
| AMBER, load-conditioned | 390 | +0.0385 | 42.8%/53.8% | 0.0013 | +0.0107 | 34.4%/51.0% | 0.00013 |
| AMBER, geometry-only | 394 | +0.0441 | 39.8%/55.6% | 0.00047 | +0.0233 | 30.5%/55.8% | 1.1e-07 |
| ZZ from fine reference solve | 405 | +0.0376 | 37.5%/56.5% | 2e-07 | +0.0078 | 33.3%/48.4% | 0.0023 |

4-backend per-case mean (L+F+H) minus baseline — A

| Coarse (no refinement) | 405 | -0.0354 | 56.3%/36.0% | 9.2e-08 | -0.0221 | 59.3%/21.7% | 1.2e-17 |
| Geometric heuristic (blind) | 405 | -0.0151 | 51.1%/41.2% | 0.012 | -0.0114 | 50.6%/33.6% | 0.0069 |
| Load-informed heuristic | 405 | +0.0522 | 34.6%/59.3% | 7.4e-12 | +0.0000 | 42.5%/44.2% | 0.19 |
| ZZ from coarse solve | 405 | -0.0151 | 50.9%/40.5% | 0.069 | -0.0076 | 48.1%/36.0% | 0.023 |
| AMBER, load-conditioned | 390 | +0.0345 | 37.9%/57.9% | 8.1e-05 | +0.0026 | 41.3%/44.9% | 0.026 |
| AMBER, geometry-only | 394 | +0.0483 | 39.8%/55.3% | 2.7e-05 | +0.0142 | 37.6%/50.8% | 5.9e-05 |
| ZZ from fine reference solve | 405 | +0.0553 | 37.0%/58.0% | 8.5e-09 | +0.0000 | 42.0%/45.9% | 0.12 |

## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — C

| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |
|---|---|---|---|---|---|---|---|
| Coarse (no refinement) | 95 | -0.1019 | 68.4%/26.3% | 3.2e-06 | -0.0247 | 63.2%/20.0% | 0.00017 |
| Geometric heuristic (blind) | 95 | -0.0029 | 48.4%/45.3% | 0.71 | +0.0051 | 41.1%/44.2% | 0.21 |
| Load-informed heuristic | 95 | +0.0372 | 41.1%/53.7% | 0.12 | +0.0133 | 42.1%/50.5% | 0.69 |
| ZZ from coarse solve | 95 | -0.0601 | 62.1%/35.8% | 0.012 | -0.0142 | 52.6%/34.7% | 0.25 |
| AMBER, load-conditioned | 89 | -0.0489 | 56.2%/39.3% | 0.048 | -0.0087 | 49.4%/39.3% | 0.36 |
| AMBER, geometry-only | 84 | +0.0229 | 40.5%/53.6% | 0.28 | +0.0392 | 33.3%/59.5% | 0.0025 |
| ZZ from fine reference solve | 95 | +0.0600 | 38.9%/57.9% | 0.0033 | +0.0137 | 34.7%/52.6% | 0.045 |

4-backend per-case mean (L+F+H) minus baseline — C

| Coarse (no refinement) | 95 | -0.1030 | 72.6%/24.2% | 1.5e-06 | -0.0561 | 73.7%/16.8% | 1.9e-07 |
| Geometric heuristic (blind) | 95 | -0.0081 | 49.5%/44.2% | 0.53 | -0.0101 | 50.5%/42.1% | 0.87 |
| Load-informed heuristic | 95 | +0.0302 | 38.9%/55.8% | 0.08 | -0.0216 | 53.7%/40.0% | 0.59 |
| ZZ from coarse solve | 95 | -0.0530 | 62.1%/34.7% | 0.006 | -0.0246 | 60.0%/30.5% | 0.013 |
| AMBER, load-conditioned | 89 | -0.0470 | 61.8%/36.0% | 0.025 | -0.0180 | 53.9%/34.8% | 0.044 |
| AMBER, geometry-only | 84 | +0.0158 | 42.9%/54.8% | 0.5 | +0.0061 | 40.5%/46.4% | 0.092 |
| ZZ from fine reference solve | 95 | +0.0746 | 37.9%/56.8% | 0.0013 | -0.0007 | 43.2%/45.3% | 0.34 |

## Paired Wilcoxon, GReFEM (Gemini L+F+H) minus baseline — B

| baseline | n | vm medΔ | vm win/loss | vm p | en medΔ | en win/loss | en p |
|---|---|---|---|---|---|---|---|
| Coarse (no refinement) | 280 | -0.1786 | 82.1%/15.0% | 1.4e-27 | -0.0578 | 78.9%/9.3% | 1.5e-30 |
| Geometric heuristic (blind) | 280 | -0.0590 | 62.5%/28.2% | 2.4e-13 | -0.0074 | 48.2%/30.7% | 4.9e-05 |
| Load-informed heuristic | 280 | +0.0155 | 45.0%/51.4% | 0.14 | -0.0071 | 47.9%/38.2% | 0.2 |
| ZZ from coarse solve | 280 | -0.0810 | 68.6%/27.9% | 5.2e-13 | -0.0084 | 48.6%/33.9% | 0.0037 |
| AMBER, load-conditioned | 256 | -0.0239 | 53.5%/41.4% | 0.024 | -0.0011 | 43.4%/41.0% | 0.52 |
| AMBER, geometry-only | 258 | +0.0053 | 47.3%/48.8% | 0.74 | +0.0020 | 41.9%/45.7% | 0.4 |
| ZZ from fine reference solve | 280 | +0.0310 | 41.8%/55.0% | 0.0005 | -0.0011 | 42.1%/44.3% | 0.3 |

4-backend per-case mean (L+F+H) minus baseline — B

| Coarse (no refinement) | 280 | -0.1656 | 82.5%/13.6% | 4.9e-27 | -0.0683 | 79.3%/8.6% | 1.9e-30 |
| Geometric heuristic (blind) | 280 | -0.0590 | 67.1%/29.3% | 6.8e-11 | -0.0193 | 57.5%/25.4% | 1.1e-08 |
| Load-informed heuristic | 280 | +0.0209 | 43.9%/52.5% | 0.0041 | -0.0142 | 54.3%/31.8% | 0.012 |
| ZZ from coarse solve | 280 | -0.0708 | 69.3%/25.0% | 2.8e-11 | -0.0166 | 55.7%/28.2% | 3e-05 |
| AMBER, load-conditioned | 256 | +0.0020 | 47.7%/48.4% | 0.63 | -0.0033 | 46.9%/39.1% | 0.5 |
| AMBER, geometry-only | 258 | +0.0001 | 47.3%/47.7% | 0.74 | -0.0001 | 43.8%/45.0% | 0.61 |
| ZZ from fine reference solve | 280 | +0.0496 | 35.0%/60.7% | 1.2e-07 | -0.0010 | 43.9%/45.4% | 0.17 |

## Prompt ablation, paired (L+F+H minus other) per backend, all 156

- Gemini: L+F+H vs L+F: n=774 vm medΔ=+0.0000 p=0.25; en medΔ=+0.0000 p=0.28
- Gemini: L+F+H vs L: n=775 vm medΔ=+0.0024 p=0.36; en medΔ=+0.0017 p=0.048
- GPT-5.4-mini: L+F+H vs L+F: n=776 vm medΔ=-0.0000 p=0.21; en medΔ=-0.0000 p=0.29
- GPT-5.4-mini: L+F+H vs L: n=774 vm medΔ=-0.0017 p=0.33; en medΔ=+0.0061 p=0.0075
- Claude-Haiku-4.5: L+F+H vs L+F: n=780 vm medΔ=+0.0000 p=0.31; en medΔ=+0.0002 p=0.25
- Claude-Haiku-4.5: L+F+H vs L: n=780 vm medΔ=+0.0030 p=0.93; en medΔ=+0.0037 p=0.054
- Qwen3-VL-235B: L+F+H vs L+F: n=780 vm medΔ=+0.0000 p=0.72; en medΔ=+0.0000 p=0.24
- Qwen3-VL-235B: L+F+H vs L: n=780 vm medΔ=+0.0043 p=0.097; en medΔ=+0.0019 p=0.16

## Per load case @h1.0 (vm tmean / en tmean), all 156

| policy | Comp. | Bend. | Tors. | Bend.+Comp. | Tors.+Comp. |
|---|---|---|---|---|---|
| Coarse (no refinement) | 0.739/0.948 | 0.733/0.940 | 0.730/0.943 | 0.742/0.941 | 0.744/0.943 |
| GReFEM (Gemini) L+F+H | 0.659/0.899 | 0.623/0.869 | 0.622/0.868 | 0.600/0.850 | 0.617/0.896 |
| GReFEM (GPT-5.4-mini) L+F+H | 0.655/0.889 | 0.637/0.880 | 0.623/0.871 | 0.607/0.851 | 0.624/0.863 |
| GReFEM (Claude-Haiku-4.5) L+F+H | 0.671/0.897 | 0.654/0.889 | 0.639/0.862 | 0.608/0.858 | 0.637/0.879 |
| GReFEM (Qwen3-VL-235B) L+F+H | 0.673/0.901 | 0.660/0.885 | 0.643/0.878 | 0.626/0.859 | 0.642/0.894 |
| Geometric heuristic (blind) | 0.698/0.911 | 0.697/0.905 | 0.658/0.877 | 0.645/0.860 | 0.656/0.892 |
| Load-informed heuristic | 0.633/0.888 | 0.574/0.872 | 0.514/0.851 | 0.550/0.831 | 0.562/0.888 |
| ZZ from coarse solve | 0.710/0.919 | 0.688/0.898 | 0.652/0.879 | 0.630/0.838 | 0.671/0.877 |
| AMBER, load-conditioned | 0.616/0.860 | 0.654/0.893 | 0.638/0.864 | 0.610/0.867 | 0.602/0.859 |
| AMBER, geometry-only | 0.621/0.860 | 0.625/0.874 | 0.600/0.847 | 0.586/0.825 | 0.608/0.829 |
| ZZ from fine reference solve | 0.607/0.858 | 0.572/0.874 | 0.570/0.859 | 0.472/0.791 | 0.523/0.857 |

Per load case, GReFEM pooled over backends, per prompt:

| GReFEM 4-backend pool L+F+H | 0.666/0.898 | 0.645/0.883 | 0.633/0.872 | 0.611/0.856 | 0.631/0.886 |
| GReFEM 4-backend pool L+F | 0.674/0.899 | 0.638/0.875 | 0.621/0.871 | 0.615/0.861 | 0.645/0.890 |
| GReFEM 4-backend pool L | 0.675/0.892 | 0.645/0.870 | 0.623/0.861 | 0.600/0.849 | 0.623/0.873 |

Paired GReFEM (Gemini L+F+H) minus baseline per load case (vm medΔ, p):

| Geometric heuristic (blind) | -0.021 (p=0.013) | -0.064 (p=4.7e-07) | -0.027 (p=0.02) | -0.030 (p=0.007) | -0.005 (p=0.038) |
| Load-informed heuristic | +0.032 (p=0.034) | +0.013 (p=0.042) | +0.065 (p=7.9e-06) | +0.027 (p=0.049) | +0.042 (p=0.0038) |
| ZZ from coarse solve | -0.040 (p=0.0012) | -0.040 (p=5.4e-05) | -0.039 (p=0.0046) | -0.030 (p=0.082) | -0.047 (p=0.0011) |
| AMBER, load-conditioned | +0.053 (p=0.035) | -0.017 (p=0.15) | -0.022 (p=0.5) | -0.031 (p=0.53) | +0.029 (p=0.15) |
| AMBER, geometry-only | +0.044 (p=0.065) | +0.008 (p=0.6) | +0.045 (p=0.087) | +0.023 (p=0.3) | +0.006 (p=0.42) |

## Sensitivity: std of trimmed-mean error across levels of each dimension

| dimension | levels | vm std | vm range | en std | en range |
|---|---|---|---|---|---|
| MLLM backend | 4 | 0.006 | 0.018 | 0.005 | 0.014 |
| Prompt | 3 | 0.002 | 0.005 | 0.005 | 0.011 |
| Load case | 5 | 0.021 | 0.063 | 0.014 | 0.041 |
| Sizing level | 4 | 0.044 | 0.116 | 0.024 | 0.064 |
| Refinement policy (ref.) | 5 | 0.039 | 0.105 | 0.009 | 0.023 |

## Sizing sweep (vm tmean / en tmean)

| h | n units | GReFEM Gemini | GReFEM 4-pool | Geom | Load-inf | ZZc | AMBER load | AMBER geo | ZZ fine |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 780 | 0.626/0.879 | 0.637/0.879 | 0.672/0.892 | 0.567/0.869 | 0.672/0.887 | 0.625/0.869 | 0.609/0.849 | 0.549/0.850 |
| 1.25 | 773 | 0.692/0.906 | 0.689/0.903 | 0.687/0.895 | 0.627/0.890 | 0.717/0.905 | 0.657/0.891 | 0.632/0.878 | 0.608/0.869 |
| 1.6 | 773 | 0.724/0.928 | 0.728/0.928 | 0.745/0.934 | 0.686/0.915 | 0.728/0.926 | 0.716/0.927 | 0.684/0.909 | 0.700/0.915 |
| 2.0 | 773 | 0.756/0.941 | 0.749/0.938 | 0.766/0.943 | 0.734/0.928 | 0.760/0.944 | 0.736/0.931 | 0.729/0.938 | 0.728/0.931 |

Paired GReFEM (Gemini L+F+H) minus AMBER per sizing level:

- h1.0 AMBER, load-conditioned: n=735 vm medΔ=+0.0001 p=0.72; en medΔ=+0.0041 p=0.037
- h1.0 AMBER, geometry-only: n=736 vm medΔ=+0.0249 p=0.0082; en medΔ=+0.0171 p=8e-08
- h1.25 AMBER, load-conditioned: n=739 vm medΔ=+0.0205 p=0.00057; en medΔ=+0.0030 p=0.015
- h1.25 AMBER, geometry-only: n=737 vm medΔ=+0.0395 p=1.9e-10; en medΔ=+0.0141 p=2.3e-08
- h1.6 AMBER, load-conditioned: n=739 vm medΔ=-0.0063 p=0.88; en medΔ=-0.0001 p=0.95
- h1.6 AMBER, geometry-only: n=737 vm medΔ=+0.0328 p=5.8e-08; en medΔ=+0.0085 p=2.9e-06
- h2.0 AMBER, load-conditioned: n=739 vm medΔ=+0.0120 p=0.011; en medΔ=+0.0021 p=0.04
- h2.0 AMBER, geometry-only: n=732 vm medΔ=+0.0228 p=0.00012; en medΔ=+0.0016 p=0.042
