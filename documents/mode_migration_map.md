# Mode migration map after cleanup

| New `ga_type` | Method folder | Equivalent old `ga_type` | Meaning |
|---:|---|---:|---|
| 0 | `standard_ga` | 0 | Standard GA baseline |
| 1 | `empirical_linkage_legacy` | 1 | Legacy empirical-linkage GA |
| 2 | `blur_ga_pairwise_lasso` | 6 | Theory-aligned pairwise Lasso linkage |
| 3 | `blur_ga_main_pairwise_lasso` | 7 | Theory-aligned partial main+pairwise Lasso linkage |

Removed old `ga_type` values: `2`, `3`, `4`, and `5`.
