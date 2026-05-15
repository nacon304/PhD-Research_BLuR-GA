from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

RegressionStage = Literal["pairwise_lr", "main_pairwise_lr", "sparse", "sparse_excess"]


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


class RegressionLinkageLearner:
    """Fit pairwise linkage weights from evaluated GA solutions."""

    def __init__(
        self,
        *,
        n_features: int,
        stage: RegressionStage,
        ridge_alpha: float = 1e-6,
        sparse_alpha: float = 0.001,
        l1_ratio: float = 0.95,
        stability_subsamples: int = 0,
        stability_fraction: float = 0.75,
        random_state: int = 1,
        excess_window: int = 5,
    ) -> None:
        self.n_features = int(n_features)
        self.stage = stage
        self.ridge_alpha = float(ridge_alpha)
        self.sparse_alpha = float(sparse_alpha)
        self.l1_ratio = float(l1_ratio)
        self.stability_subsamples = int(stability_subsamples)
        self.stability_fraction = float(stability_fraction)
        self.random_state = int(random_state)
        self.excess_window = int(excess_window)
        self.pairs = [(i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)]
        if self.pairs:
            self._pair_i = np.asarray([i for i, _ in self.pairs], dtype=int)
            self._pair_j = np.asarray([j for _, j in self.pairs], dtype=int)
        else:
            self._pair_i = np.empty(0, dtype=int)
            self._pair_j = np.empty(0, dtype=int)
        self._rng = np.random.default_rng(self.random_state)

    @property
    def include_main_effects(self) -> bool:
        return self.stage in {"main_pairwise_lr", "sparse", "sparse_excess"}

    @property
    def use_sparse_model(self) -> bool:
        return self.stage in {"sparse", "sparse_excess"}

    @property
    def use_excess_response(self) -> bool:
        return self.stage == "sparse_excess"

    def fit(self, chromosomes: np.ndarray, fitness: np.ndarray, generations: np.ndarray, *, generation: int) -> RegressionLinkageFit:
        X_bits = np.asarray(chromosomes, dtype=np.float64)
        y = np.asarray(fitness, dtype=np.float64)
        gens = np.asarray(generations, dtype=int)
        if X_bits.ndim != 2 or X_bits.shape[1] != self.n_features:
            raise ValueError("chromosomes must have shape (n_samples, n_features).")
        if len(X_bits) != len(y):
            raise ValueError("chromosomes and fitness must have the same length.")
        if len(X_bits) < 2:
            zeros = np.zeros(len(self.pairs), dtype=float)
            return RegressionLinkageFit(generation, len(X_bits), self.pairs, zeros, zeros, zeros, zeros.astype(int))

        if self.use_excess_response:
            y = self._baseline_standardized_excess(y, gens)
        else:
            y = y - float(np.mean(y))

        X_design, pair_start = self._design_matrix(X_bits)
        model = self._fit_model(X_design, y)
        coef = np.asarray(model.coef_, dtype=float)
        pair_coef = coef[pair_start : pair_start + len(self.pairs)]
        pair_coef = np.nan_to_num(pair_coef, nan=0.0, posinf=0.0, neginf=0.0)

        stability = np.ones_like(pair_coef, dtype=float)
        selected_counts = (np.abs(pair_coef) > 1e-12).astype(int)
        if self.use_sparse_model and self.stability_subsamples > 0 and len(X_bits) >= 4:
            stability, selected_counts = self._stability_selection(X_design, y, pair_start)

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

    def _design_matrix(self, X_bits: np.ndarray) -> tuple[np.ndarray, int]:
        if len(self.pairs) == 0:
            X_pairs = np.empty((X_bits.shape[0], 0), dtype=float)
        else:
            # Vectorized construction avoids thousands of temporary Python lists.
            X_pairs = X_bits[:, self._pair_i] * X_bits[:, self._pair_j]
        if self.include_main_effects:
            X_design = np.hstack([X_bits, X_pairs])
            pair_start = self.n_features
        else:
            X_design = X_pairs
            pair_start = 0
        return np.ascontiguousarray(X_design, dtype=np.float64), pair_start

    def _fit_model(self, X_design: np.ndarray, y: np.ndarray):
        if self.use_sparse_model:
            try:
                from sklearn.linear_model import ElasticNet
            except Exception as exc:  # pragma: no cover - environment-specific.
                raise ImportError("scikit-learn is required for sparse regression-linkage learning") from exc
            model = ElasticNet(
                alpha=self.sparse_alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                max_iter=5000,
                tol=1e-4,
                selection="cyclic",
                random_state=self.random_state,
            )
            return model.fit(X_design, y)
        return self._fit_dense_ridge_dual(X_design, y)

    def _fit_dense_ridge_dual(self, X_design: np.ndarray, y: np.ndarray) -> _DenseLinearModel:
        """Fast ridge/OLS-like fit for p >> n linkage design matrices.

        Stage 2/3 often has thousands of pairwise columns but only tens of
        archived GA samples. Solving in feature space can be very slow. The
        dual form solves an n x n system and then maps back to p coefficients:

            coef = X_c.T @ solve(X_c @ X_c.T + alpha I, y_c)
        """
        X = np.ascontiguousarray(X_design, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X_design and y length mismatch.")
        if X.shape[0] == 0:
            return _DenseLinearModel(np.zeros(X.shape[1], dtype=float), 0.0)
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y_arr))
        Xc = X - x_mean
        yc = y_arr - y_mean
        alpha = float(self.ridge_alpha)
        gram = Xc @ Xc.T
        if alpha > 0.0:
            gram = gram + alpha * np.eye(gram.shape[0], dtype=gram.dtype)
            try:
                dual = np.linalg.solve(gram, yc)
            except np.linalg.LinAlgError:
                dual = np.linalg.pinv(gram) @ yc
        else:
            dual = np.linalg.pinv(gram) @ yc
        coef = Xc.T @ dual
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        intercept = y_mean - float(x_mean @ coef)
        return _DenseLinearModel(coef_=coef, intercept_=intercept)

    def _stability_selection(self, X_design: np.ndarray, y: np.ndarray, pair_start: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        m = max(2, min(n, int(round(self.stability_fraction * n))))
        counts = np.zeros(len(self.pairs), dtype=int)
        for _ in range(self.stability_subsamples):
            idx = self._rng.choice(n, size=m, replace=False)
            model = self._fit_model(X_design[idx], y[idx])
            coef = np.asarray(model.coef_, dtype=float)[pair_start : pair_start + len(self.pairs)]
            counts += (np.abs(coef) > 1e-12).astype(int)
        stability = counts.astype(float) / max(1, self.stability_subsamples)
        return stability, counts

    def _baseline_standardized_excess(self, y: np.ndarray, generations: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y, dtype=float)
        unique_gens = np.unique(generations)
        global_med = float(np.median(y))
        global_mad = float(np.median(np.abs(y - global_med)))
        global_scale = max(1.4826 * global_mad, float(np.std(y)), 1e-8)
        for g in unique_gens:
            lo = int(g) - max(1, self.excess_window) + 1
            mask_window = (generations >= lo) & (generations <= int(g))
            yw = y[mask_window]
            if len(yw) >= 3:
                b = float(np.median(yw))
                mad = float(np.median(np.abs(yw - b)))
                scale = max(1.4826 * mad, 1e-8)
            else:
                b = global_med
                scale = global_scale
            mask_g = generations == int(g)
            out[mask_g] = (y[mask_g] - b) / scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def ga_type_to_regression_stage(ga_type: int) -> RegressionStage | None:
    mapping: dict[int, RegressionStage] = {
        2: "pairwise_lr",
        3: "main_pairwise_lr",
        4: "sparse",
        5: "sparse_excess",
    }
    return mapping.get(int(ga_type))


def regression_stage_label(ga_type: int) -> str:
    stage = ga_type_to_regression_stage(ga_type)
    if stage is None:
        return "none"
    return stage
