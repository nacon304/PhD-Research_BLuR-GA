from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

RegressionStage = Literal[
    "pairwise_lasso",
    "main_pairwise_lasso",
]
RegressionSolver = Literal[
    "auto",
    "sklearn_full",
    "dual_ridge",
    "matrix_free",  # backward-compatible alias for dual_ridge
]
LREncoding = Literal["binary", "spin"]


def _num_pairs(n_features: int) -> int:
    n = int(n_features)
    return max(0, n * (n - 1) // 2)


def _full_pair_indices(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular pair indices without constructing Python tuple lists."""
    i, j = np.triu_indices(int(n_features), k=1)
    return i.astype(np.int32, copy=False), j.astype(np.int32, copy=False)


@dataclass(frozen=True)
class RegressionLinkageFit:
    """A fitted regression-linkage graph at one generation.

    Only non-zero/active coefficients need to be returned.  Older versions kept
    a Python list with every possible pair, which is expensive for d >> 500.  The
    graph update only consumes edges with positive weights, so sparse arrays keep
    the same output semantics while avoiding a large Python-object memory cost.
    """

    generation: int
    n_samples: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    coefficients: np.ndarray
    weights: np.ndarray
    stability: np.ndarray
    selected_counts: np.ndarray


@dataclass(frozen=True)
class _PairCoefficientData:
    pair_i: np.ndarray
    pair_j: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class _DenseLinearModel:
    coef_: np.ndarray
    intercept_: float = 0.0


@dataclass(frozen=True)
class _DesignData:
    """Design matrix plus pairwise edge indices."""

    X: np.ndarray
    pair_start: int
    pair_i: np.ndarray
    pair_j: np.ndarray


@dataclass(frozen=True)
class _StandardizedDesign:
    X: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def unscale_coefficients(self, coef: np.ndarray) -> np.ndarray:
        coef = np.asarray(coef, dtype=float)
        out = np.zeros_like(coef, dtype=float)
        mask = self.scales > 0.0
        out[mask] = coef[mask] / self.scales[mask]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class _RegressionEstimator(Protocol):
    alpha: float

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> _PairCoefficientData:
        """Return active pair coefficients."""


@dataclass
class RegressionVIG:
    """Weighted interaction graph learned from GA search history via regression.

    This class intentionally mimics the save/edge-table API of EmpiricalVIG so
    ResultWriter and the existing graph-evolution UI can consume its outputs.

    The public `weight` matrix stores non-negative edge importance used by the
    visualizer. The signed regression coefficient is stored separately in
    `coefficient`; positive and negative epistasis can therefore be inspected
    without breaking the old `eVIG_*` output schema.
    """

    n_features: int
    weight: np.ndarray = field(init=False)
    coefficient: np.ndarray = field(init=False)
    stability: np.ndarray = field(init=False)
    selected_count: np.ndarray = field(init=False)
    tested_count: np.ndarray = field(init=False)
    first_seen_generation: np.ndarray = field(init=False)
    last_updated_generation: np.ndarray = field(init=False)
    last_fit_generation: int = -1
    last_n_samples: int = 0

    def __post_init__(self) -> None:
        shape = (self.n_features, self.n_features)
        self.weight = np.zeros(shape, dtype=float)
        self.coefficient = np.zeros(shape, dtype=float)
        self.stability = np.zeros(shape, dtype=float)
        self.selected_count = np.zeros(shape, dtype=np.int32)
        self.tested_count = np.zeros(shape, dtype=np.int32)
        self.first_seen_generation = np.full(shape, -1, dtype=np.int32)
        self.last_updated_generation = np.full(shape, -1, dtype=np.int32)

    @property
    def n_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.weight > 0.0, k=1)))

    @property
    def n_tested_pairs(self) -> int:
        return int(np.count_nonzero(np.triu(self.tested_count > 0, k=1)))

    def adjacency_matrix(self) -> np.ndarray:
        return self.weight.copy()

    def coefficient_matrix(self) -> np.ndarray:
        return self.coefficient.copy()

    def update_from_fit(self, fit: RegressionLinkageFit, *, min_abs_weight: float = 0.0, top_k: int | None = None) -> None:
        self.last_fit_generation = int(fit.generation)
        self.last_n_samples = int(fit.n_samples)
        new_weight = np.zeros_like(self.weight)
        new_coef = np.zeros_like(self.coefficient)
        new_stability = np.zeros_like(self.stability)
        new_selected = np.zeros_like(self.selected_count)
        new_tested = np.zeros_like(self.tested_count)

        if fit.weights.size == 0:
            self.weight = new_weight
            self.coefficient = new_coef
            self.stability = new_stability
            self.selected_count = new_selected
            self.tested_count = new_tested
            return

        mask = np.asarray(fit.weights > max(0.0, float(min_abs_weight))).reshape(-1)
        if not np.any(mask):
            self.weight = new_weight
            self.coefficient = new_coef
            self.stability = new_stability
            self.selected_count = new_selected
            self.tested_count = new_tested
            return

        idx = np.flatnonzero(mask)
        order = np.lexsort((fit.pair_j[idx], fit.pair_i[idx], -fit.weights[idx]))
        idx = idx[order]
        if top_k is not None:
            idx = idx[: int(top_k)]

        for k in idx:
            i = int(fit.pair_i[k])
            j = int(fit.pair_j[k])
            w = float(fit.weights[k])
            c = float(fit.coefficients[k])
            st = float(fit.stability[k])
            sc = int(fit.selected_counts[k])
            new_weight[i, j] = new_weight[j, i] = w
            new_coef[i, j] = new_coef[j, i] = c
            new_stability[i, j] = new_stability[j, i] = st
            new_selected[i, j] = new_selected[j, i] = sc
            new_tested[i, j] = new_tested[j, i] = fit.n_samples
            if self.first_seen_generation[i, j] < 0:
                self.first_seen_generation[i, j] = self.first_seen_generation[j, i] = int(fit.generation)
            self.last_updated_generation[i, j] = self.last_updated_generation[j, i] = int(fit.generation)

        self.weight = new_weight
        self.coefficient = new_coef
        self.stability = new_stability
        self.selected_count = new_selected
        self.tested_count = new_tested

    def touch_fit_metadata(self, generation: int, n_samples: int) -> None:
        """Refresh fit metadata when the regression input archive is unchanged."""
        self.last_fit_generation = int(generation)
        self.last_n_samples = int(n_samples)
        active = self.weight > 0.0
        self.last_updated_generation[active] = int(generation)

    def edge_table(
        self,
        *,
        min_weight: float = 0.0,
        top_k: int | None = None,
    ) -> list[tuple[int, int, float, int, int, float, int, int]]:
        rows: list[tuple[int, int, float, int, int, float, int, int]] = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                if self.weight[i, j] > 0.0 and self.weight[i, j] >= min_weight:
                    rows.append(
                        (
                            i,
                            j,
                            float(self.weight[i, j]),
                            int(self.selected_count[i, j]),
                            int(self.tested_count[i, j]),
                            float(abs(self.coefficient[i, j])),
                            int(self.first_seen_generation[i, j]),
                            int(self.last_updated_generation[i, j]),
                        )
                    )
        rows.sort(key=lambda r: (-r[2], r[0], r[1]))
        if top_k is not None:
            rows = rows[: int(top_k)]
        return rows

    def tested_pair_table(self) -> list[tuple[int, int, int, int, float]]:
        rows: list[tuple[int, int, int, int, float]] = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                if self.tested_count[i, j] > 0:
                    rows.append((i, j, int(self.tested_count[i, j]), int(self.selected_count[i, j]), float(self.weight[i, j])))
        rows.sort(key=lambda r: (-r[2], r[0], r[1]))
        return rows

    def save_matrix(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.weight, delimiter=",", fmt="%.8f")

    def save_coefficient_matrix(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.coefficient, delimiter=",", fmt="%.8f")

    def save_edges(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(
                "feature_i,feature_j,weight,coefficient,sign,stability,selected_count,tested_count,"
                "weight_sum,first_seen_generation,last_updated_generation\n"
            )
            for i in range(self.n_features):
                for j in range(i + 1, self.n_features):
                    if self.weight[i, j] <= 0.0:
                        continue
                    coef = float(self.coefficient[i, j])
                    sign = 1 if coef > 0 else (-1 if coef < 0 else 0)
                    f.write(
                        f"{i},{j},{self.weight[i, j]:.12f},{coef:.12f},{sign},"
                        f"{self.stability[i, j]:.12f},{int(self.selected_count[i, j])},"
                        f"{int(self.tested_count[i, j])},{abs(coef):.12f},"
                        f"{int(self.first_seen_generation[i, j])},{int(self.last_updated_generation[i, j])}\n"
                    )

    def save_tested_pairs(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("feature_i,feature_j,tested_count,selected_count,weight,coefficient,stability\n")
            for i, j, tested, selected, weight in self.tested_pair_table():
                f.write(
                    f"{i},{j},{tested},{selected},{weight:.12f},"
                    f"{self.coefficient[i, j]:.12f},{self.stability[i, j]:.12f}\n"
                )


class PairwiseDesignBuilder:
    """Build main-effect and pairwise interaction designs.

    ``encoding="binary"`` keeps the original feature-selection variables
    ``x in {0,1}`` and uses active-pair terms ``x_i*x_j``.  A positive
    coefficient is therefore a co-selection effect.

    ``encoding="spin"`` uses ``u = 2x - 1`` and pair terms ``u_i*u_j``.  A
    positive coefficient then means same-state linkage (00/11 vs 01/10).
    """

    def __init__(self, n_features: int, *, include_main: bool, encoding: LREncoding = "binary") -> None:
        self.n_features = int(n_features)
        self.include_main = bool(include_main)
        if encoding not in {"binary", "spin"}:
            raise ValueError("encoding must be either binary or spin")
        self.encoding: LREncoding = encoding
        self._pair_i, self._pair_j = _full_pair_indices(self.n_features)

    @property
    def n_pairs(self) -> int:
        return int(self._pair_i.size)

    def _main_matrix(self, X_bits: np.ndarray) -> np.ndarray:
        X = np.asarray(X_bits, dtype=np.float64)
        if self.encoding == "binary":
            return X
        return 2.0 * X - 1.0

    def build(self, X_bits: np.ndarray) -> _DesignData:
        main = self._main_matrix(X_bits)
        if self.n_pairs == 0:
            pairs = np.empty((main.shape[0], 0), dtype=float)
        else:
            pairs = main[:, self._pair_i] * main[:, self._pair_j]
        if self.include_main:
            X_design = np.hstack([main, pairs])
            pair_start = self.n_features
        else:
            X_design = pairs
            pair_start = 0
        return _DesignData(np.ascontiguousarray(X_design, dtype=np.float64), pair_start, self._pair_i, self._pair_j)

    def main_effect_matrix_with_intercept(self, X_bits: np.ndarray) -> np.ndarray:
        main = self._main_matrix(X_bits)
        return np.hstack([np.ones((main.shape[0], 1), dtype=float), main])


class PairwiseLassoEstimator:
    """Pairwise Lasso on pseudo-Boolean interactions: y ~= sum theta_ij u_i u_j."""

    def __init__(self, *, design_builder: PairwiseDesignBuilder, alpha: float, random_state: int) -> None:
        self.design_builder = design_builder
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> _PairCoefficientData:
        design = self.design_builder.build(X_bits)
        z = _standardize_for_lasso(design.X)
        model = _fit_sklearn_lasso(z.X, y, alpha=self.alpha, random_state=self.random_state)
        coef = z.unscale_coefficients(np.asarray(model.coef_, dtype=float))
        return _active_pair_coefficients(design.pair_i, design.pair_j, coef)


class PartialMainPairwiseLassoEstimator:
    """Main-controlled pairwise Lasso using unpenalized main controls."""

    def __init__(self, *, design_builder: PairwiseDesignBuilder, alpha: float, random_state: int) -> None:
        self.design_builder = design_builder
        self.pairwise_builder = PairwiseDesignBuilder(
            self.design_builder.n_features,
            include_main=False,
            encoding=self.design_builder.encoding,
        )
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> _PairCoefficientData:
        pair_design = self.pairwise_builder.build(X_bits)
        controls = self.design_builder.main_effect_matrix_with_intercept(X_bits)
        y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        resid = _residualize(np.hstack([y_col, pair_design.X]), controls)
        y_resid = resid[:, 0]
        z_resid = resid[:, 1:]
        z = _standardize_for_lasso(z_resid)
        model = _fit_sklearn_lasso(z.X, y_resid, alpha=self.alpha, random_state=self.random_state)
        coef = z.unscale_coefficients(np.asarray(model.coef_, dtype=float))
        return _active_pair_coefficients(pair_design.pair_i, pair_design.pair_j, coef)


class DualPairwiseRidgeEstimator:
    """Exact dual ridge solver for pairwise regression-linkage.

    This replaces the older proximal matrix-free approximation on high-dimensional
    datasets.  It solves the same pairwise regression model in dual form by
    building only the n x n pairwise kernel, never the n x d(d-1)/2 design.
    For type 3, main effects are treated as unpenalized controls by residualizing
    the pairwise kernel and target with respect to [1, main effects].
    """

    def __init__(
        self,
        *,
        n_features: int,
        alpha: float,
        random_state: int,
        encoding: LREncoding = "binary",
        include_main_controls: bool = False,
        max_active_edges: int | None = None,
    ) -> None:
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        if encoding not in {"binary", "spin"}:
            raise ValueError("encoding must be either binary or spin")
        self.encoding: LREncoding = encoding
        self.include_main_controls = bool(include_main_controls)
        self.max_active_edges = max_active_edges

    def _encoded(self, X_bits: np.ndarray) -> np.ndarray:
        X = np.asarray(X_bits, dtype=np.float64)
        if self.encoding == "binary":
            return X
        return 2.0 * X - 1.0

    def _pair_kernel(self, Z: np.ndarray) -> np.ndarray:
        # K_ab = sum_{i<j} phi_ij(a) phi_ij(b), without materializing phi.
        G = np.ascontiguousarray(Z @ Z.T, dtype=np.float64)
        if self.encoding == "binary":
            return 0.5 * G * (G - 1.0)
        return 0.5 * (G * G - float(self.n_features))

    @staticmethod
    def _residualize_kernel(K: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
        if Q is None or Q.size == 0:
            return K
        # M K M, where M = I - Q Q^T.  This keeps type-3 main effects as
        # controls without building residualized pair columns.
        left = K - Q @ (Q.T @ K)
        return np.ascontiguousarray(left - (left @ Q) @ Q.T, dtype=np.float64)

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> _PairCoefficientData:
        Z = self._encoded(X_bits)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = Z.shape
        if n < 2 or d < 2:
            empty = np.empty(0, dtype=np.int32)
            return _PairCoefficientData(empty, empty, np.empty(0, dtype=float))

        K = self._pair_kernel(Z)
        alpha_target = y_arr
        q_controls: np.ndarray | None = None
        if self.include_main_controls:
            controls = np.hstack([np.ones((n, 1), dtype=np.float64), Z])
            q_controls = _orthonormal_basis(controls)
            alpha_target = _residualize_with_basis(y_arr, q_controls)
            K = self._residualize_kernel(K, q_controls)

        ridge = max(float(self.alpha), 1e-10)
        A = K + ridge * np.eye(n, dtype=np.float64)
        try:
            alpha_dual = np.linalg.solve(A, alpha_target)
        except np.linalg.LinAlgError:
            alpha_dual = np.linalg.lstsq(A, alpha_target, rcond=None)[0]
        alpha_eff = _residualize_with_basis(alpha_dual, q_controls) if q_controls is not None else alpha_dual

        # beta_ij = sum_t alpha_t z_ti z_tj = (Z^T diag(alpha) Z)_ij.
        coef_matrix = Z.T @ (alpha_eff.reshape(-1, 1) * Z)
        np.fill_diagonal(coef_matrix, 0.0)
        return _active_pair_coefficients_from_matrix(
            coef_matrix,
            tol=1e-10,
            top_k=self.max_active_edges,
        )


class MatrixFreePairwiseLassoEstimator:
    """Matrix-free proximal Lasso for pseudo-Boolean pairwise interactions.

    It optimizes the same standardized pairwise objective as the dense Lasso path
    without creating Z in R^{n x d(d-1)/2}.  GA masks ``x`` are converted to
    ``u = 2x - 1`` internally, and pair columns exist implicitly as
    ``z_ij = u_i * u_j``.
    """

    def __init__(
        self,
        *,
        n_features: int,
        alpha: float,
        random_state: int,
        include_main_controls: bool = False,
        max_iter: int = 8,
        tol: float = 1e-4,
        step_scale: float = 1.0,
        dtype: str = "float64",
    ) -> None:
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.include_main_controls = bool(include_main_controls)
        self.max_iter = int(max(1, max_iter))
        self.tol = float(max(0.0, tol))
        self.step_scale = float(max(1e-12, step_scale))
        self.dtype = np.float32 if str(dtype) == "float32" else np.float64

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> _PairCoefficientData:
        X0 = np.asarray(X_bits, dtype=self.dtype)
        X = (self.dtype(2.0) * X0 - self.dtype(1.0)).astype(self.dtype, copy=False)
        y_arr = np.asarray(y, dtype=self.dtype).reshape(-1)
        n, d = X.shape
        if n < 2 or d < 2:
            empty = np.empty(0, dtype=np.int32)
            return _PairCoefficientData(empty, empty, np.empty(0, dtype=float))

        # Pseudo-Boolean interaction moments. For z_ij = u_i*u_j in {-1,+1},
        # E[z_ij^2] = 1, hence Var(z_ij) = 1 - E[z_ij]^2.  Constant pair
        # columns are ignored by setting inv_scale to zero.
        pair_mean = (X.T @ X).astype(self.dtype, copy=False) / self.dtype(n)
        np.fill_diagonal(pair_mean, 0.0)
        pair_var = np.maximum(self.dtype(1.0) - pair_mean * pair_mean, self.dtype(0.0)).astype(self.dtype, copy=False)
        active_pair = pair_var > self.dtype(1e-12)
        inv_scale = np.zeros_like(pair_var, dtype=self.dtype)
        inv_scale[active_pair] = self.dtype(1.0) / np.sqrt(pair_var[active_pair]).astype(self.dtype, copy=False)
        np.fill_diagonal(inv_scale, 0.0)

        y_target = y_arr
        q_controls: np.ndarray | None = None
        if self.include_main_controls:
            controls = np.hstack([np.ones((n, 1), dtype=self.dtype), X])
            q_controls = _orthonormal_basis(controls)
            y_target = _residualize_with_basis(y_arr, q_controls).astype(self.dtype, copy=False)

        # W stores standardized pair coefficients symmetrically.  The diagonal is
        # fixed at zero because self-interactions are not graph edges.
        W = np.zeros((d, d), dtype=self.dtype)
        p = max(1, int(np.count_nonzero(np.triu(active_pair, k=1))))
        # Conservative ISTA step for standardized pseudo-Boolean pair columns.
        # The older binary-monomial path used n/p; for u_i*u_j columns this can
        # be too aggressive on small archives because all pair columns are active.
        step0 = self.dtype(self.step_scale / float(max(1, p)))

        for _ in range(self.max_iter):
            pred = _matrix_free_pair_predict(X, W, pair_mean, inv_scale)
            if q_controls is not None:
                pred = _residualize_with_basis(pred, q_controls).astype(self.dtype, copy=False)
            resid = (pred - y_target).astype(self.dtype, copy=False)
            grad_raw = (X.T @ (resid[:, None] * X)).astype(self.dtype, copy=False) / self.dtype(n)
            resid_mean = self.dtype(float(np.mean(resid)))
            grad = (grad_raw - resid_mean * pair_mean) * inv_scale
            np.fill_diagonal(grad, 0.0)

            # Fixed-step ISTA with a cheap numerical safety loop.  Full objective
            # backtracking is too expensive at high d because each objective check
            # requires another implicit pairwise prediction.
            step = step0
            W_new = W
            for _bt in range(8):
                W_candidate = _soft_threshold_matrix(W - step * grad, float(self.alpha) * float(step))
                np.fill_diagonal(W_candidate, 0.0)
                W_candidate = ((W_candidate + W_candidate.T) * self.dtype(0.5)).astype(self.dtype, copy=False)
                max_abs = float(np.max(np.abs(W_candidate))) if W_candidate.size else 0.0
                if np.isfinite(max_abs) and max_abs <= 1e6:
                    W_new = W_candidate
                    break
                step = self.dtype(float(step) * 0.5)

            diff = float(np.max(np.abs(W_new - W)))
            W = W_new
            if self.tol > 0.0 and diff < self.tol:
                break

        raw_coef = (W * inv_scale).astype(np.float64, copy=False)
        np.fill_diagonal(raw_coef, 0.0)
        return _active_pair_coefficients_from_matrix(raw_coef)


def _matrix_free_pair_predict(X: np.ndarray, Wstd: np.ndarray, pair_mean: np.ndarray, inv_scale: np.ndarray) -> np.ndarray:
    A = Wstd * inv_scale
    quad = 0.5 * np.einsum("bi,ij,bj->b", X, A, X, optimize=True)
    offset = 0.5 * np.sum(Wstd * pair_mean * inv_scale)
    return np.asarray(quad - offset, dtype=X.dtype)


def _matrix_free_objective(
    X: np.ndarray,
    Wstd: np.ndarray,
    pair_mean: np.ndarray,
    inv_scale: np.ndarray,
    y_target: np.ndarray,
    q_controls: np.ndarray | None,
    alpha: float,
) -> float:
    pred = _matrix_free_pair_predict(X, Wstd, pair_mean, inv_scale)
    if q_controls is not None:
        pred = _residualize_with_basis(pred, q_controls).astype(X.dtype, copy=False)
    resid = pred - y_target
    return float(0.5 * np.mean(np.square(resid)) + float(alpha) * np.sum(np.abs(np.triu(Wstd, k=1))))


def _soft_threshold_matrix(X: np.ndarray, tau: float) -> np.ndarray:
    return (np.sign(X) * np.maximum(np.abs(X) - float(tau), 0.0)).astype(X.dtype, copy=False)


def _orthonormal_basis(controls: np.ndarray) -> np.ndarray:
    C = np.asarray(controls, dtype=np.float64)
    if C.size == 0:
        return np.empty((C.shape[0], 0), dtype=np.float64)
    try:
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        if s.size == 0:
            return np.empty((C.shape[0], 0), dtype=np.float64)
        tol = np.finfo(float).eps * max(C.shape) * float(s[0])
        rank = int(np.sum(s > tol))
        return np.ascontiguousarray(U[:, :rank], dtype=np.float64)
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(C)
        return np.ascontiguousarray(Q, dtype=np.float64)


def _residualize_with_basis(v: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if Q is None or Q.size == 0:
        return arr.copy()
    return np.nan_to_num(arr - Q @ (Q.T @ arr), nan=0.0, posinf=0.0, neginf=0.0)


def _active_pair_coefficients(pair_i: np.ndarray, pair_j: np.ndarray, coef: np.ndarray, *, tol: float = 1e-12) -> _PairCoefficientData:
    coef_arr = np.nan_to_num(np.asarray(coef, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(coef_arr) > float(tol)
    if not np.any(mask):
        empty = np.empty(0, dtype=np.int32)
        return _PairCoefficientData(empty, empty, np.empty(0, dtype=float))
    return _PairCoefficientData(
        np.asarray(pair_i[mask], dtype=np.int32),
        np.asarray(pair_j[mask], dtype=np.int32),
        np.asarray(coef_arr[mask], dtype=np.float64),
    )


def _active_pair_coefficients_from_matrix(
    coef_matrix: np.ndarray,
    *,
    tol: float = 1e-6,
    top_k: int | None = None,
) -> _PairCoefficientData:
    abs_upper = np.triu(np.abs(coef_matrix), k=1)
    mask = abs_upper > float(tol)
    if not np.any(mask):
        empty = np.empty(0, dtype=np.int32)
        return _PairCoefficientData(empty, empty, np.empty(0, dtype=float))
    if top_k is not None and int(top_k) > 0 and int(top_k) < int(np.count_nonzero(mask)):
        flat = abs_upper.ravel()
        k = int(top_k)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(-flat[idx])]
        pair_i, pair_j = np.unravel_index(idx, coef_matrix.shape)
    else:
        pair_i, pair_j = np.nonzero(mask)
    coefs = coef_matrix[pair_i, pair_j]
    return _PairCoefficientData(
        np.asarray(pair_i, dtype=np.int32),
        np.asarray(pair_j, dtype=np.int32),
        np.asarray(coefs, dtype=np.float64),
    )


def _fit_sklearn_lasso(X: np.ndarray, y: np.ndarray, *, alpha: float, random_state: int):
    try:
        from sklearn.linear_model import Lasso
    except Exception as exc:  # pragma: no cover - environment-specific.
        raise ImportError("scikit-learn is required for Lasso regression-linkage learning") from exc
    model = Lasso(
        alpha=float(alpha),
        fit_intercept=False,
        max_iter=10000,
        tol=1e-4,
        selection="cyclic",
        random_state=int(random_state),
    )
    return model.fit(np.ascontiguousarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))


def _standardize_for_lasso(X: np.ndarray) -> _StandardizedDesign:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D design matrix.")
    if X_arr.shape[0] == 0:
        return _StandardizedDesign(X_arr.copy(), np.zeros(X_arr.shape[1]), np.ones(X_arr.shape[1]))
    means = np.mean(X_arr, axis=0)
    centered = X_arr - means
    norm = np.linalg.norm(centered, axis=0)
    target = np.sqrt(max(1, X_arr.shape[0]))
    scales = norm / target
    Xs = np.zeros_like(centered, dtype=float)
    mask = scales > 1e-12
    Xs[:, mask] = centered[:, mask] / scales[mask]
    return _StandardizedDesign(
        X=np.ascontiguousarray(np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64),
        means=means,
        scales=scales,
    )


def _residualize(A: np.ndarray, controls: np.ndarray) -> np.ndarray:
    A_arr = np.asarray(A, dtype=np.float64)
    C = np.asarray(controls, dtype=np.float64)
    if A_arr.shape[0] != C.shape[0]:
        raise ValueError("A and controls must have the same number of rows.")
    if C.size == 0:
        return A_arr.copy()
    try:
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        if s.size == 0:
            return A_arr.copy()
        tol = np.finfo(float).eps * max(C.shape) * float(s[0])
        rank = int(np.sum(s > tol))
        if rank <= 0:
            return A_arr.copy()
        Q = U[:, :rank]
        resid = A_arr - Q @ (Q.T @ A_arr)
    except np.linalg.LinAlgError:
        fitted = C @ (np.linalg.pinv(C) @ A_arr)
        resid = A_arr - fitted
    return np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)


class RegressionLinkageLearner:
    """Fit pairwise linkage weights from evaluated GA solutions."""

    def __init__(
        self,
        *,
        n_features: int,
        stage: RegressionStage,
        sparse_alpha: float = 0.001,
        auto_alpha: bool = True,
        alpha_c: float = 1.0,
        delta: float = 0.05,
        stability_subsamples: int = 0,
        stability_fraction: float = 0.75,
        random_state: int = 1,
        solver: RegressionSolver = "auto",
        encoding: LREncoding = "binary",
        edge_top_k: int | None = None,
        matrix_free_threshold: int = 700,
        matrix_free_max_iter: int = 8,
        matrix_free_tol: float = 1e-4,
        matrix_free_step_scale: float = 1.0,
        matrix_free_dtype: str = "float64",
    ) -> None:
        self.n_features = int(n_features)
        self.stage = stage
        self.sparse_alpha = float(sparse_alpha)
        self.auto_alpha = bool(auto_alpha)
        self.alpha_c = float(alpha_c)
        self.delta = float(delta)
        self.stability_subsamples = int(stability_subsamples)
        self.stability_fraction = float(stability_fraction)
        self.random_state = int(random_state)
        self.solver = str(solver)
        if encoding not in {"binary", "spin"}:
            raise ValueError("encoding must be either binary or spin")
        self.encoding: LREncoding = encoding
        self.edge_top_k = None if edge_top_k is None else int(edge_top_k)
        self.matrix_free_threshold = int(matrix_free_threshold)
        self.matrix_free_max_iter = int(matrix_free_max_iter)
        self.matrix_free_tol = float(matrix_free_tol)
        self.matrix_free_step_scale = float(matrix_free_step_scale)
        self.matrix_free_dtype = str(matrix_free_dtype)
        self.n_pairs = _num_pairs(self.n_features)
        self._rng = np.random.default_rng(self.random_state)
        self._estimator = self._build_estimator()

    @property
    def include_main_effects(self) -> bool:
        return self.stage == "main_pairwise_lasso"

    @property
    def use_sparse_model(self) -> bool:
        return True

    @property
    def active_solver(self) -> str:
        if self.solver == "matrix_free":
            # Backward-compatible CLI value; the approximate matrix-free path has
            # been replaced by the exact dual-ridge high-dimensional path.
            return "dual_ridge"
        if self.solver == "dual_ridge":
            return "dual_ridge"
        if self.solver == "sklearn_full":
            return "sklearn_full"
        return "dual_ridge" if self.n_features >= self.matrix_free_threshold else "sklearn_full"

    def _build_estimator(self) -> _RegressionEstimator:
        solver = self.active_solver
        if solver == "dual_ridge":
            return DualPairwiseRidgeEstimator(
                n_features=self.n_features,
                alpha=self.sparse_alpha,
                random_state=self.random_state,
                encoding=self.encoding,
                include_main_controls=self.include_main_effects,
                max_active_edges=self.edge_top_k,
            )
        if self.stage == "pairwise_lasso":
            return PairwiseLassoEstimator(
                design_builder=PairwiseDesignBuilder(self.n_features, include_main=False, encoding=self.encoding),
                alpha=self.sparse_alpha,
                random_state=self.random_state,
            )
        if self.stage == "main_pairwise_lasso":
            return PartialMainPairwiseLassoEstimator(
                design_builder=PairwiseDesignBuilder(self.n_features, include_main=True, encoding=self.encoding),
                alpha=self.sparse_alpha,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown regression linkage stage: {self.stage!r}")

    @property
    def use_theory_alpha(self) -> bool:
        return self.auto_alpha

    def theory_alpha(self, y_reg: np.ndarray, *, n_samples: int | None = None, p_columns: int | None = None) -> float:
        """Return lambda_g = c_lambda sigma_hat sqrt(2 log(2 p_g / delta) / n_g)."""
        if not self.use_theory_alpha:
            return float(self.sparse_alpha)
        y_arr = np.asarray(y_reg, dtype=np.float64).reshape(-1)
        n = int(n_samples if n_samples is not None else y_arr.size)
        p = int(p_columns if p_columns is not None else self.n_pairs)
        if n < 2 or p < 1:
            return float(self.sparse_alpha)
        sigma_hat = float(np.std(y_arr, ddof=1))
        if not np.isfinite(sigma_hat):
            sigma_hat = 0.0
        if sigma_hat <= 1e-12:
            return 0.0
        delta = min(max(float(self.delta), 1e-12), 1.0 - 1e-12)
        log_term = max(0.0, float(np.log((2.0 * max(1, p)) / delta)))
        return float(self.alpha_c) * sigma_hat * float(np.sqrt((2.0 * log_term) / float(n)))

    def _fit_with_current_alpha(self, X_bits: np.ndarray, y_reg: np.ndarray, *, alpha: float) -> _PairCoefficientData:
        old_alpha = getattr(self._estimator, "alpha", None)
        if old_alpha is not None:
            setattr(self._estimator, "alpha", float(alpha))
        try:
            return self._estimator.fit(X_bits, y_reg)
        finally:
            if old_alpha is not None:
                setattr(self._estimator, "alpha", old_alpha)

    def fit(self, chromosomes: np.ndarray, fitness: np.ndarray, generations: np.ndarray, *, generation: int) -> RegressionLinkageFit:
        X_bits = np.asarray(chromosomes, dtype=np.float64)
        y = np.asarray(fitness, dtype=np.float64)
        if X_bits.ndim != 2 or X_bits.shape[1] != self.n_features:
            raise ValueError("chromosomes must have shape (n_samples, n_features).")
        if len(X_bits) != len(y):
            raise ValueError("chromosomes and fitness must have the same length.")
        if len(X_bits) < 2:
            empty = np.empty(0, dtype=np.int32)
            return RegressionLinkageFit(generation, len(X_bits), empty, empty, np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=np.int32))

        y_reg = y - float(np.mean(y))
        alpha_g = self.theory_alpha(y_reg, n_samples=len(X_bits), p_columns=self.n_pairs)
        pair_data = self._fit_with_current_alpha(X_bits, y_reg, alpha=alpha_g)
        pair_coef = np.nan_to_num(pair_data.coefficients, nan=0.0, posinf=0.0, neginf=0.0)

        stability = np.ones_like(pair_coef, dtype=float)
        selected_counts = (np.abs(pair_coef) > 1e-12).astype(np.int32)
        if self.stability_subsamples > 0 and len(X_bits) >= 4 and self.active_solver == "sklearn_full":
            stability, selected_counts = self._stability_selection_dense(X_bits, y_reg, pair_data)

        weights = np.abs(pair_coef) * stability
        return RegressionLinkageFit(
            generation=int(generation),
            n_samples=int(len(X_bits)),
            pair_i=pair_data.pair_i,
            pair_j=pair_data.pair_j,
            coefficients=pair_coef,
            weights=weights,
            stability=stability,
            selected_counts=selected_counts,
        )

    def _stability_selection_dense(self, X_bits: np.ndarray, y: np.ndarray, base_data: _PairCoefficientData) -> tuple[np.ndarray, np.ndarray]:
        """Stability counts for active dense-solver edges.

        Stability selection is intentionally limited to the dense small-d path;
        for large matrix-free runs, repeatedly fitting the full pairwise model is
        usually too expensive and defaults remain zero subsamples.
        """
        n = len(y)
        if base_data.coefficients.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=np.int32)
        key_to_pos = {(int(i), int(j)): pos for pos, (i, j) in enumerate(zip(base_data.pair_i, base_data.pair_j))}
        counts = np.zeros(base_data.coefficients.size, dtype=np.int32)
        m = max(2, min(n, int(round(self.stability_fraction * n))))
        for _ in range(self.stability_subsamples):
            idx = self._rng.choice(n, size=m, replace=False)
            y_sub = y[idx]
            alpha_sub = self.theory_alpha(y_sub, n_samples=m, p_columns=self.n_pairs)
            data = self._fit_with_current_alpha(X_bits[idx], y_sub, alpha=alpha_sub)
            for i, j in zip(data.pair_i, data.pair_j):
                pos = key_to_pos.get((int(i), int(j)))
                if pos is not None:
                    counts[pos] += 1
        stability = counts.astype(float) / max(1, self.stability_subsamples)
        return stability, counts

    # Helper kept for notebooks/debugging.
    def _design_matrix(self, X_bits: np.ndarray) -> tuple[np.ndarray, int]:
        builder = PairwiseDesignBuilder(
            self.n_features,
            include_main=self.include_main_effects,
            encoding=self.encoding,
        )
        design = builder.build(X_bits)
        return design.X, design.pair_start


def ga_type_to_regression_stage(ga_type: int) -> RegressionStage | None:
    mapping: dict[int, RegressionStage] = {
        2: "pairwise_lasso",
        3: "pairwise_lasso",
        4: "pairwise_lasso",
    }
    return mapping.get(int(ga_type))


def regression_stage_label(ga_type: int) -> str:
    stage = ga_type_to_regression_stage(ga_type)
    if stage is None:
        return "none"
    return stage
