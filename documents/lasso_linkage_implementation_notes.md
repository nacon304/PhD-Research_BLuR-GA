# Lasso linkage implementation notes

This cleaned version keeps only the active regression-linkage methods.

| `ga_type` | Folder name | Meaning | Previous id |
|---:|---|---|---:|
| 0 | `standard_ga` | Standard GA baseline | 0 |
| 1 | `empirical_linkage_legacy` | Legacy empirical linkage | 1 |
| 2 | `blur_ga_pairwise_lasso` | Pairwise Lasso on bipolar pairwise terms | 6 |
| 3 | `blur_ga_main_pairwise_lasso` | Partial main+pairwise Lasso with unpenalized main controls | 7 |

Old dense ridge / ElasticNet / excess-response modes have been removed.

## Pairwise Lasso (`ga_type=2`)

The model uses bipolar variables `u_i = 2x_i - 1` and pairwise columns `u_i u_j`. Columns are centered and scaled to satisfy the Lasso proof normalization as closely as possible.

## Main + Pairwise Lasso (`ga_type=3`)

The model residualizes both the response and pairwise design against the intercept and main-effect controls, then fits Lasso only on the residualized pairwise columns. Main effects are not graph edges.

## Key CLI knobs

- `--lr-sparse-alpha`: manual Lasso penalty used when `--lr-auto-alpha false`.
- `--lr-auto-alpha`: compute the theory-guided lambda at each fit.
- `--lr-auto-min-samples`: require the PSLE/MPSLE-inspired minimum archive size before refitting.
- `--lr-stability-subsamples`: optional stability-selection refits for edge robustness.
