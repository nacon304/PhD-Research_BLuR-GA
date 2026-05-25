# Implementation cleanup summary

This version removes the unused experimental modes and renumbers the active Lasso methods.

## New mode mapping

| New id | Method folder | Old id |
|---:|---|---:|
| 0 | `standard_ga` | 0 |
| 1 | `empirical_linkage_legacy` | 1 |
| 2 | `blur_ga_pairwise_lasso` | 6 |
| 3 | `blur_ga_main_pairwise_lasso` | 7 |

Removed code paths: dense pairwise ridge, dense main+pairwise ridge, sparse augmented ElasticNet/Lasso, and sparse excess-response regression.

The residualized main+pairwise Lasso optimization is preserved: y and pairwise columns are residualized together using one control projection, avoiding the former repeated least-squares bottleneck.
