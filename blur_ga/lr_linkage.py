from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

RegressionStage = Literal[
    "pairwise_lasso",
    "main_pairwise_lasso",
]


@dataclass(frozen=True)
class RegressionLinkageFit:
    """A fitted regression-linkage graph at one generation."""

    generation: int
    n_samples: int
    pairs: list[tuple[int, int]]
    coefficients: np.ndarray
    weights: np.ndarray
    stability: np.ndarray
    selected_counts: np.ndarray


@dataclass(frozen=True)
class _DenseLinearModel:
    coef_: np.ndarray
    intercept_: float = 0.0


@dataclass(frozen=True)
class _DesignData:
    """Design matrix plus the slice containing pairwise edge coefficients."""

    X: np.ndarray
    pair_start: int
    pairs: list[tuple[int, int]]


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
    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return one coefficient per pair in `pairs`."""


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
        self.selected_count = np.zeros(shape, dtype=int)
        self.tested_count = np.zeros(shape, dtype=int)
        self.first_seen_generation = np.full(shape, -1, dtype=int)
        self.last_updated_generation = np.full(shape, -1, dtype=int)

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

        candidate_rows: list[tuple[int, int, float, float, float, int]] = []
        for idx, (i, j) in enumerate(fit.pairs):
            w = float(fit.weights[idx])
            c = float(fit.coefficients[idx])
            st = float(fit.stability[idx])
            sc = int(fit.selected_counts[idx])
            if w >= min_abs_weight and w > 0.0:
                candidate_rows.append((int(i), int(j), w, c, st, sc))
        candidate_rows.sort(key=lambda r: (-r[2], r[0], r[1]))
        if top_k is not None:
            candidate_rows = candidate_rows[: int(top_k)]

        for i, j, w, c, st, sc in candidate_rows:
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
        """Refresh fit metadata when the regression input archive is unchanged.

        Re-fitting on the identical archive would produce the same coefficients,
        weights, stability, and selected counts.  This method preserves the
        generation-level metadata that output files expect without paying the
        regression cost again.
        """
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
    """Build bipolar main-effect and pairwise-monomial designs.

    BLuR-GA now keeps only the theory-aligned Lasso linkage stages.  Therefore
    all regression designs use u_i = 2 x_i - 1 instead of the older 0/1 dense
    regression encoding.
    """

    def __init__(self, n_features: int, *, include_main: bool) -> None:
        self.n_features = int(n_features)
        self.include_main = bool(include_main)
        self.pairs = [(i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)]
        self._pair_i = np.asarray([i for i, _ in self.pairs], dtype=int) if self.pairs else np.empty(0, dtype=int)
        self._pair_j = np.asarray([j for _, j in self.pairs], dtype=int) if self.pairs else np.empty(0, dtype=int)

    def _main_matrix(self, X_bits: np.ndarray) -> np.ndarray:
        X = np.asarray(X_bits, dtype=np.float64)
        return 2.0 * X - 1.0

    def build(self, X_bits: np.ndarray) -> _DesignData:
        main = self._main_matrix(X_bits)
        if len(self.pairs) == 0:
            pairs = np.empty((main.shape[0], 0), dtype=float)
        else:
            pairs = main[:, self._pair_i] * main[:, self._pair_j]
        if self.include_main:
            X_design = np.hstack([main, pairs])
            pair_start = self.n_features
        else:
            X_design = pairs
            pair_start = 0
        return _DesignData(np.ascontiguousarray(X_design, dtype=np.float64), pair_start, self.pairs)

    def main_effect_matrix_with_intercept(self, X_bits: np.ndarray) -> np.ndarray:
        main = self._main_matrix(X_bits)
        return np.hstack([np.ones((main.shape[0], 1), dtype=float), main])


class PairwiseLassoEstimator:
    """Theory-aligned pairwise Lasso: y ~= sum theta_ij u_i u_j.

    The response is centered before this estimator is called. Pairwise columns
    are built with u_i = 2x_i - 1, centered, and normalized to ||Z_j||_2=sqrt(n),
    matching the fixed-design Lasso proof assumptions as closely as possible.
    """

    def __init__(self, *, design_builder: PairwiseDesignBuilder, alpha: float, random_state: int) -> None:
        self.design_builder = design_builder
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = self.design_builder.build(X_bits)
        z = _standardize_for_lasso(design.X)
        model = _fit_sklearn_lasso(z.X, y, alpha=self.alpha, random_state=self.random_state)
        return z.unscale_coefficients(np.asarray(model.coef_, dtype=float))


class PartialMainPairwiseLassoEstimator:
    """Theory-aligned Main+Pairwise Lasso using unpenalized main controls.

    This implements the residualized/partial Lasso theorem:

        y_tilde = M_[1,U] y,     Z_tilde = M_[1,U] Z,
        theta_hat = argmin 1/(2n)||y_tilde - Z_tilde theta||^2 + lambda||theta||_1.

    Main effects are controls and are not treated as graph edges.
    """

    def __init__(self, *, design_builder: PairwiseDesignBuilder, alpha: float, random_state: int) -> None:
        self.design_builder = design_builder
        self.pairwise_builder = PairwiseDesignBuilder(
            self.design_builder.n_features,
            include_main=False,
        )
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, X_bits: np.ndarray, y: np.ndarray) -> np.ndarray:
        pair_design = self.pairwise_builder.build(X_bits)
        controls = self.design_builder.main_effect_matrix_with_intercept(X_bits)
        y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        # Residualize y and all pairwise columns with a single projection. This
        # avoids recomputing the same control decomposition for thousands of
        # pairwise columns.
        resid = _residualize(np.hstack([y_col, pair_design.X]), controls)
        y_resid = resid[:, 0]
        z_resid = resid[:, 1:]
        z = _standardize_for_lasso(z_resid)
        model = _fit_sklearn_lasso(z.X, y_resid, alpha=self.alpha, random_state=self.random_state)
        return z.unscale_coefficients(np.asarray(model.coef_, dtype=float))


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
    # sklearn Lasso uses 1/(2n)||y-Xb||^2 + alpha||b||_1. Scaling columns to
    # sqrt(n) makes each column satisfy (1/n)||Z_j||^2 = 1, as in the proof.
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
        # Orthogonal projection using only the left singular vectors of the
        # control design. This computes the same least-squares residual M_C A as
        # np.linalg.lstsq(C, A), but avoids solving for every RHS coefficient
        # when A has thousands of pairwise columns.
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
        self.pairs = [(i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)]
        self._rng = np.random.default_rng(self.random_state)
        self._estimator = self._build_estimator()

    @property
    def include_main_effects(self) -> bool:
        return self.stage == "main_pairwise_lasso"

    @property
    def use_sparse_model(self) -> bool:
        return True

    def _build_estimator(self) -> _RegressionEstimator:
        if self.stage == "pairwise_lasso":
            return PairwiseLassoEstimator(
                design_builder=PairwiseDesignBuilder(self.n_features, include_main=False),
                alpha=self.sparse_alpha,
                random_state=self.random_state,
            )
        if self.stage == "main_pairwise_lasso":
            return PartialMainPairwiseLassoEstimator(
                design_builder=PairwiseDesignBuilder(self.n_features, include_main=True),
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
        p = int(p_columns if p_columns is not None else len(self.pairs))
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

    def _fit_with_current_alpha(self, X_bits: np.ndarray, y_reg: np.ndarray, *, alpha: float) -> np.ndarray:
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
            zeros = np.zeros(len(self.pairs), dtype=float)
            return RegressionLinkageFit(generation, len(X_bits), self.pairs, zeros, zeros, zeros, zeros.astype(int))

        y_reg = y - float(np.mean(y))
        alpha_g = self.theory_alpha(y_reg, n_samples=len(X_bits), p_columns=len(self.pairs))
        pair_coef = self._fit_with_current_alpha(X_bits, y_reg, alpha=alpha_g)
        pair_coef = np.nan_to_num(pair_coef, nan=0.0, posinf=0.0, neginf=0.0)

        stability = np.ones_like(pair_coef, dtype=float)
        selected_counts = (np.abs(pair_coef) > 1e-12).astype(int)
        if self.stability_subsamples > 0 and len(X_bits) >= 4:
            stability, selected_counts = self._stability_selection(X_bits, y_reg)

        weights = np.abs(pair_coef) * stability
        return RegressionLinkageFit(
            generation=int(generation),
            n_samples=int(len(X_bits)),
            pairs=self.pairs,
            coefficients=pair_coef,
            weights=weights,
            stability=stability,
            selected_counts=selected_counts,
        )

    def _stability_selection(self, X_bits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        m = max(2, min(n, int(round(self.stability_fraction * n))))
        counts = np.zeros(len(self.pairs), dtype=int)
        for _ in range(self.stability_subsamples):
            idx = self._rng.choice(n, size=m, replace=False)
            y_sub = y[idx]
            alpha_sub = self.theory_alpha(y_sub, n_samples=m, p_columns=len(self.pairs))
            coef = self._fit_with_current_alpha(X_bits[idx], y_sub, alpha=alpha_sub)
            counts += (np.abs(coef) > 1e-12).astype(int)
        stability = counts.astype(float) / max(1, self.stability_subsamples)
        return stability, counts

    # Helper kept for notebooks/debugging.
    def _design_matrix(self, X_bits: np.ndarray) -> tuple[np.ndarray, int]:
        builder = PairwiseDesignBuilder(
            self.n_features,
            include_main=self.include_main_effects,
        )
        design = builder.build(X_bits)
        return design.X, design.pair_start


def ga_type_to_regression_stage(ga_type: int) -> RegressionStage | None:
    mapping: dict[int, RegressionStage] = {
        2: "pairwise_lasso",
        3: "main_pairwise_lasso",
    }
    return mapping.get(int(ga_type))


def regression_stage_label(ga_type: int) -> str:
    stage = ga_type_to_regression_stage(ga_type)
    if stage is None:
        return "none"
    return stage
