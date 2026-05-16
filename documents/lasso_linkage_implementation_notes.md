# Lasso Linkage Implementation Notes

This project now keeps the original regression-linkage stages unchanged and adds two theory-aligned Lasso stages.

## Method IDs

| `ga_type` | Folder name | Meaning |
|---:|---|---|
| 0 | `standard_ga` | Standard GA baseline. |
| 1 | `empirical_linkage_legacy` | Original empirical linkage/VIG variant. |
| 2 | `blur_ga_stage1_pairwise` | Original pairwise dense ridge/OLS-like regression. |
| 3 | `blur_ga_stage2_main_pairwise` | Original main + pairwise dense ridge/OLS-like regression. |
| 4 | `blur_ga_stage3_sparse` | Backward-compatible sparse augmented main+pairwise ElasticNet/Lasso-style regression. |
| 5 | `blur_ga_stage4_sparse_excess` | Backward-compatible sparse augmented model with excess-response transformation. |
| 6 | `blur_ga_pairwise_lasso` | New theory-aligned Pairwise + Lasso. |
| 7 | `blur_ga_main_pairwise_lasso` | New theory-aligned Main + Pairwise + Lasso using unpenalized main-effect controls. |

## New Pairwise + Lasso (`ga_type=6`)

The estimator uses bipolar chromosome encoding

```text
u_i = 2 x_i - 1
```

and constructs pairwise columns

```text
z_ij = u_i u_j.
```

The response is centered, the pairwise columns are centered and normalized to satisfy approximately

```text
(1/n) ||z_ij||_2^2 = 1,
```

then sklearn's Lasso is fitted with objective

```text
1/(2n) ||y - Z theta||_2^2 + alpha ||theta||_1.
```

Only non-zero pairwise coefficients become linkage graph edges.

## New Main + Pairwise + Lasso (`ga_type=7`)

This implements the partial/residualized Lasso version from the proof. Main effects are controls, not linkage edges.

The control matrix is

```text
U0 = [1, U], where U_i = 2x_i - 1.
```

The estimator residualizes both the response and the pairwise design:

```text
y_tilde = M_U0 y
Z_tilde = M_U0 Z
```

then fits

```text
argmin_theta 1/(2n) ||y_tilde - Z_tilde theta||_2^2 + alpha ||theta||_1.
```

This avoids the stronger and less suitable assumption that Lasso must recover both main-effect support and pairwise support.

## Important implementation details

- Existing `ga_type=0..5` behavior is preserved.
- The new estimator code is organized in `blur_ga/lr_linkage.py` using small OOP components:
  - `PairwiseDesignBuilder`
  - `ResponseTransformer`
  - `DenseRidgeLinkageEstimator`
  - `PenalizedAugmentedLinkageEstimator`
  - `PairwiseLassoEstimator`
  - `PartialMainPairwiseLassoEstimator`
- The same graph output API is preserved through `RegressionVIG`, so existing result writers and graph UI scripts continue to work.
- `--lr-sparse-alpha` is the Lasso penalty parameter for `ga_type=6` and `ga_type=7`.
- `--lr-edge-top-k` can still be used to cap graph density for visualization.
